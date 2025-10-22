# pvl_gemini_api.py
# PVL - Gemini API (Google Developer API)
# Batch-wise selective retries + Gemini error detection
# Added: timeout handling as retryable + linear backoff (1-2-3-4...)

import os, io, time, base64, json, random
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

try:
    import torch, numpy as np
except Exception:
    torch = None
    np = None

GEMINI_BASE = "https://generativelanguage.googleapis.com"
PRIMARY_VER = "v1beta"
FALLBACK_VER = "v1"

HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

# -----------------------------
# Utils
# -----------------------------
def _log_debug(debug: bool, *args):
    if debug:
        print("[PVL_GEMINI]", *args, flush=True)

def _get_api_key(provided: str) -> str:
    if provided and provided.strip():
        return provided.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()

def _tensor_to_pil_first(image_tensor: Any) -> Optional[Image.Image]:
    if image_tensor is None:
        return None
    if isinstance(image_tensor, Image.Image):
        return image_tensor
    if torch is None or np is None:
        return None
    t = image_tensor
    if isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[0]
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        arr = t
    else:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        return None
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)

def _b64_from_pil(pil_img: Image.Image, mime: str = "image/png") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG" if mime == "image/png" else "JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _build_contents(prompt: Optional[str], pil_img: Optional[Image.Image]) -> list:
    parts = []
    if prompt and str(prompt).strip():
        parts.append({"text": str(prompt)})
    if pil_img is not None:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": _b64_from_pil(pil_img, mime="image/png")
            }
        })
    if not parts:
        parts = [{"text": ""}]
    return [{"role": "user", "parts": parts}]

def _extract_text(resp_json: Dict[str, Any]) -> str:
    cands = resp_json.get("candidates") or []
    if not cands:
        fb = resp_json.get("promptFeedback") or {}
        br = fb.get("blockReason") or fb.get("block_reason") or ""
        return f"[Gemini] No candidates returned{(' (blocked: ' + br + ')') if br else ''}."
    parts = (cands[0] or {}).get("content", {}).get("parts", []) or []
    out = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return out.strip()

# -----------------------------
# Error detection
# -----------------------------
_GOOD_FINISH = {"STOP", "FINISH_REASON_UNSPECIFIED", None}

def _is_blocked_prompt(resp_json: Dict[str, Any]) -> Optional[str]:
    fb = resp_json.get("promptFeedback") or {}
    br = fb.get("blockReason") or fb.get("block_reason")
    if br:
        return f"Prompt blocked (blockReason={br})"
    return None

def _finish_reason_error(resp_json: Dict[str, Any]) -> Optional[str]:
    cands = resp_json.get("candidates") or []
    if not cands:
        return "No candidates in response"
    fr = (cands[0] or {}).get("finishReason") or (cands[0] or {}).get("finish_reason")
    if fr not in _GOOD_FINISH:
        return f"Content generation stopped (finishReason={fr})"
    return None

def _gen_url(model: str, api_version: str) -> str:
    model_path = model if model.startswith("models/") else f"models/{model}"
    return f"{GEMINI_BASE}/{api_version}/{model_path}:generateContent"

# --- modified: now treats timeout as retryable ---
def _post(url: str, payload: Dict[str, Any], api_key: str, timeout: int, debug: bool, note: str):
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if debug:
            _log_debug(debug, f"HTTP {resp.status_code}")
        return resp
    except requests.exceptions.Timeout:
        if debug:
            print(f"[PVL_GEMINI] Request timeout after {timeout}s")
        class TimeoutResponse:
            status_code = 599
            text = "Client timeout"
            def json(self): return {"error": {"message": "Client timeout", "status": "TIMEOUT"}}
        return TimeoutResponse()
    except requests.exceptions.RequestException as e:
        if debug:
            print(f"[PVL_GEMINI] Request exception: {e}")
        raise

# ----- retry classification -----
_RETRY_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_NONRETRY_HTTP_CODES = {400, 401, 403, 404}

def _is_retryable_http(code: Optional[int], status_str: Optional[str]) -> bool:
    if code in _RETRY_HTTP_CODES or code == 599:  # include timeout sentinel
        return True
    if status_str:
        s = str(status_str).upper()
        if "RESOURCE_EXHAUSTED" in s or "UNAVAILABLE" in s:
            return True
    return False

def _is_nonretryable_http(code: Optional[int]) -> bool:
    return code in _NONRETRY_HTTP_CODES

# -----------------------------
# Core generation
# -----------------------------
def _generate_once(api_key: str, model: str, instructions: Optional[str], prompt: Optional[str],
                   pil_img: Optional[Image.Image], timeout: int, temperature: float,
                   top_p: float, top_k: int, debug: bool) -> Tuple[bool, str, int, bool]:

    sys_instr = {"parts": [{"text": instructions.strip()}]} if (instructions and instructions.strip()) else None
    contents = _build_contents(prompt=prompt, pil_img=pil_img)
    gen_cfg: Dict[str, Any] = {"temperature": float(temperature)}
    if isinstance(top_p, (int, float)) and top_p > 0:
        gen_cfg["topP"] = float(top_p)
    if isinstance(top_k, int) and top_k > 0:
        gen_cfg["topK"] = int(top_k)
    base_payload: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}

    def try_one_endpoint(api_ver: str, sys_key: str):
        url = _gen_url(model, api_ver)
        payload = json.loads(json.dumps(base_payload))
        if sys_instr is not None:
            payload[sys_key] = sys_instr
        resp = _post(url, payload, api_key, timeout, debug, sys_key)

        if resp.status_code != 200:
            try:
                j = resp.json()
            except Exception:
                retryable = _is_retryable_http(resp.status_code, None)
                return False, f"HTTP {resp.status_code}: {resp.text[:500]}", resp.status_code, retryable
            err = j.get("error") or {}
            code = err.get("code")
            msg = err.get("message") or resp.text[:400]
            status = err.get("status")
            if _is_nonretryable_http(code):
                return False, f"HTTP {resp.status_code} ({status or code}): {msg}", resp.status_code, False
            retryable = _is_retryable_http(code, status)
            return False, f"HTTP {resp.status_code} ({status or code}): {msg}", resp.status_code, retryable

        data = resp.json()
        block = _is_blocked_prompt(data)
        if block:
            return False, block, 200, False
        fr_err = _finish_reason_error(data)
        if fr_err:
            return False, fr_err, 200, False
        text = _extract_text(data)
        if not text:
            return False, "Empty text in response", 200, False
        return True, text, 200, False

    ok, res, code, retryable = try_one_endpoint(PRIMARY_VER, "system_instruction")
    if ok:
        return True, res, 200, False
    if code == 400 and ("system_instruction" in res or "systemInstruction" in res):
        ok2, res2, code2, retry2 = try_one_endpoint(PRIMARY_VER, "systemInstruction")
        if ok2:
            return True, res2, 200, False
        res, code, retryable = res2, code2, retry2
    if code == 404:
        ok3, res3, code3, retry3 = try_one_endpoint(FALLBACK_VER, "system_instruction")
        if ok3:
            return True, res3, 200, False
        return False, res3, code3 or 404, _is_retryable_http(code3, None)
    return False, res, code or 0, retryable

# -----------------------------
# Node
# -----------------------------
class PVL_Gemini_API:
    """PVL Gemini API with batch retry and linear backoff."""

    @classmethod
    def INPUT_TYPES(cls):
        default_model = "gemini-2.5-flash"
        return {
            "required": {
                "model": (HARDCODED_MODELS, {"default": default_model}),
                "tries": ("INT", {"default": 2, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 45, "min": 1, "max": 600}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 65, "min": 0, "max": 100}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 10}),
                "delimiter": ("STRING", {"default": "[++]"}),
                "append_variation_tag": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "instructions": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "PVL/LLM"

    def run(self, model, tries, timeout, temperature, top_p, top_k, batch,
            delimiter, append_variation_tag, debug,
            instructions="", prompt="", image=None, api_key="", seed=0):

        start_time = time.time()
        key = _get_api_key(api_key)
        if not key:
            raise RuntimeError("Missing API key.")

        pil_img = _tensor_to_pil_first(image) if image is not None else None
        if not ((instructions and instructions.strip()) or (prompt and str(prompt).strip()) or pil_img is not None):
            raise RuntimeError("Nothing to send: provide instructions, prompt, or image.")

        def make_variant(i: int) -> str:
            if append_variation_tag and batch > 1 and (prompt or "").strip():
                return f"{str(prompt).rstrip()}\n-----\nVariation {i}"
            return str(prompt or "")

        results, last_errors, hard_fail = {}, {}, []
        pending = list(range(batch))
        attempt = 0

        while pending and attempt < tries:
            attempt += 1
            if debug:
                _log_debug(debug, f"Round {attempt}/{tries}, retry indices: {[p+1 for p in pending]}")

            def _call(idx: int):
                ptxt = make_variant(idx + 1)
                return (idx, *_generate_once(key, model, instructions or "", ptxt,
                                             pil_img, timeout, temperature, top_p, top_k, debug))

            next_round = []
            with ThreadPoolExecutor(max_workers=min(batch, 8)) as ex:
                futs = {ex.submit(_call, idx): idx for idx in pending}
                for fut in as_completed(futs):
                    idx, ok, payload, retryable = fut.result()
                    if ok:
                        results[idx] = payload.strip()
                    else:
                        last_errors[idx] = payload
                        if retryable:
                            next_round.append(idx)
                        else:
                            hard_fail.append(idx)
                        print(f"[PVL_GEMINI] (item {idx+1}) error: {payload}", flush=True)

            pending = next_round
            if pending and attempt < tries:
                delay = attempt  # linear backoff
                if debug:
                    print(f"[PVL_GEMINI] Waiting {delay}s before next retry...")
                time.sleep(delay)

        failed = sorted(set(pending + hard_fail))
        if failed:
            msg = last_errors.get(failed[0], "Unknown error")
            raise RuntimeError(f"Gemini API failed for item(s) {', '.join(str(i+1) for i in failed)}: {msg}")

        combined = f" {delimiter} ".join(results[i] for i in range(batch))
        elapsed = time.time() - start_time
        print(f"[PVL_GEMINI] Completed in {elapsed:.2f}s (batch={batch}, tries={tries})", flush=True)
        return (combined,)


NODE_CLASS_MAPPINGS = {"PVL_Gemini_API": PVL_Gemini_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Gemini_API": "PVL - Gemini Api"}
