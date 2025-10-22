# pvl_gemini_api.py
# PVL - Gemini API (Google Developer API)
# Batch-wise selective retries + Gemini error detection
# Prints total execution time always, even when debug is off.

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
    """Best-effort extraction of the first candidate text."""
    cands = resp_json.get("candidates") or []
    if not cands:
        fb = resp_json.get("promptFeedback") or {}
        br = fb.get("blockReason") or fb.get("block_reason") or ""
        return f"[Gemini] No candidates returned{(' (blocked: ' + br + ')') if br else ''}."
    parts = (cands[0] or {}).get("content", {}).get("parts", []) or []
    out = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return out.strip()

# -----------------------------
# Error detection (aligned to Gemini docs)
# - Non-200: standard Google error model {"error":{code,message,status}}
# - 200 but blocked: promptFeedback.blockReason present  -> non-retryable
# - 200 but abnormal finishReason: treat non-retryable (e.g., SAFETY, RECITATION)
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

def _post(url: str, payload: Dict[str, Any], api_key: str, timeout: int, debug: bool, note: str) -> requests.Response:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    if debug:
        san = json.loads(json.dumps(payload))
        for msg in san.get("contents", []):
            for part in msg.get("parts", []):
                if "inline_data" in part and "data" in part["inline_data"]:
                    part["inline_data"]["data"] = f"<{len(part['inline_data']['data'])} base64 bytes>"
        _log_debug(debug, f"POST {url} ({note})")
        _log_debug(debug, "Request JSON:", json.dumps(san, ensure_ascii=False)[:10000])
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if debug:
        _log_debug(debug, f"HTTP {resp.status_code}")
        _log_debug(debug, f"Raw response (trunc 20k): {resp.text[:20000]}")
    return resp

# ----- retry classification helpers -----

_RETRY_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_NONRETRY_HTTP_CODES = {400, 401, 403, 404}

def _is_retryable_http(code: Optional[int], status_str: Optional[str]) -> bool:
    # Busy/high-traffic signals: 429/503 or status RESOURCE_EXHAUSTED / UNAVAILABLE
    if code in _RETRY_HTTP_CODES:
        return True
    if status_str:
        s = str(status_str).upper()
        if "RESOURCE_EXHAUSTED" in s or "UNAVAILABLE" in s:
            return True
    return False

def _is_nonretryable_http(code: Optional[int]) -> bool:
    return code in _NONRETRY_HTTP_CODES

def _jitter_delay(base: float) -> float:
    # small +/- 30% jitter to avoid stampede
    return base * (0.7 + 0.6 * random.random())

def _generate_once(api_key: str, model: str, instructions: Optional[str], prompt: Optional[str],
                   pil_img: Optional[Image.Image], timeout: int, temperature: float,
                   top_p: float, top_k: int, debug: bool) -> Tuple[bool, str, int, bool]:
    """
    Single request (no looping).
    Returns (ok, text_or_error, http_code_or_200, retryable_hint).
    """
    sys_instr = {"parts": [{"text": instructions.strip()}]} if (instructions and instructions.strip()) else None
    contents = _build_contents(prompt=prompt, pil_img=pil_img)

    gen_cfg: Dict[str, Any] = {"temperature": float(temperature)}
    if isinstance(top_p, (int, float)) and top_p > 0:
        gen_cfg["topP"] = float(top_p)
    if isinstance(top_k, int) and top_k > 0:
        gen_cfg["topK"] = int(top_k)

    base_payload: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}

    def try_one_endpoint(api_ver: str, sys_key: str) -> Tuple[bool, str, Optional[int], bool]:
        url = _gen_url(model, api_ver)
        payload = json.loads(json.dumps(base_payload))
        if sys_instr is not None:
            payload[sys_key] = sys_instr

        resp = _post(url, payload, api_key, timeout, debug, sys_key)

        # Non-200 => standard Google error envelope ({"error": {...}})
        if resp.status_code != 200:
            try:
                j = resp.json()
            except Exception:
                # treat unknown non-200 as retryable if it's a network-ish code
                retryable = _is_retryable_http(resp.status_code, None)
                return False, f"HTTP {resp.status_code} at {url}: {resp.text[:1000]}", resp.status_code, retryable
            err = (j.get("error") or {})
            code = err.get("code")
            msg = err.get("message") or resp.text[:800]
            status = err.get("status")

            if _is_nonretryable_http(code):
                return False, f"HTTP {resp.status_code} error ({status or code}): {msg}", resp.status_code, False

            retryable = _is_retryable_http(code, status)
            return False, f"HTTP {resp.status_code} error ({status or code}): {msg}", resp.status_code, retryable

        # 200 OK => still might be blocked or abnormal finishReason (both non-retryable)
        data = resp.json()
        block = _is_blocked_prompt(data)
        if block:
            return False, block, 200, False
        fr_err = _finish_reason_error(data)
        if fr_err:
            return False, fr_err, 200, False

        text = _extract_text(data)
        if not text:
            # treat as non-retryable: server said OK but gave nothing meaningful
            return False, "Empty text in successful response", 200, False
        return True, text, 200, False

    # First, try v1beta with "system_instruction", then fallback to "systemInstruction" if needed,
    # then try v1 similarly as a compatibility fallback.
    ok, res, code, retryable = try_one_endpoint(PRIMARY_VER, "system_instruction")
    if ok:
        return True, res, 200, False

    # Some deployments accept camelCase — only worth doing if the error suggested a system_instruction issue
    if code == 400 and ("system_instruction" in res or "systemInstruction" in res):
        ok2, res2, code2, retry2 = try_one_endpoint(PRIMARY_VER, "systemInstruction")
        if ok2:
            return True, res2, 200, False
        res, code, retryable = res2, code2, retry2

    # 404s and certain model/version mismatches: try fallback API version
    if code == 404:
        ok3, res3, code3, retry3 = try_one_endpoint(FALLBACK_VER, "system_instruction")
        if ok3:
            return True, res3, 200, False
        # if it's still a 404, it's non-retryable; otherwise inherit retry hint
        return False, res3, code3 or 404, _is_retryable_http(code3, None)

    return False, res, code or 0, retryable

# -----------------------------
# Node
# -----------------------------
class PVL_Gemini_API:
    """PVL Gemini API with batch, delimiter, '(variation N)' suffix toggle,
    per-item retry with Gemini-aware error detection, and time logging."""

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
                "delimiter": ("STRING", {"default": "[++]", "multiline": False}),
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

    def run(self, model: str, tries: int, timeout: int, temperature: float,
            top_p: float, top_k: int, batch: int, delimiter: str,
            append_variation_tag: bool, debug: bool,
            instructions: Optional[str] = "", prompt: Optional[str] = "",
            image: Any = None, api_key: str = "", seed: int = 0):

        start_time = time.time()

        key = _get_api_key(api_key)
        if not key:
            raise RuntimeError("Missing API key. Provide 'api_key' or set GEMINI_API_KEY.")

        pil_img = _tensor_to_pil_first(image) if image is not None else None
        if pil_img is None and image is not None and debug:
            _log_debug(debug, "IMAGE provided but could not be converted; continuing without image.")

        if not ((instructions and instructions.strip()) or (prompt and str(prompt).strip()) or pil_img is not None):
            raise RuntimeError("Nothing to send: provide at least one of instructions, prompt, or image.")

        # Prepare per-index prompt variants
        def make_variant(i: int) -> str:
            if append_variation_tag and batch > 1 and (prompt or "").strip():
                return f"{str(prompt).rstrip()}\n-----\nVariation {i}"
            return str(prompt or "")

        # Results and state
        results: Dict[int, str] = {}          # successful outputs
        last_errors: Dict[int, str] = {}      # last error message per index
        hard_fail: List[int] = []             # indices with non-retryable failures

        pending = list(range(batch))          # indices that still need a good answer
        attempt = 0
        max_workers = min(batch, 8)

        while pending and attempt < tries:
            attempt += 1
            if debug:
                _log_debug(debug, f"Retry round {attempt}/{tries} for indices: {[p+1 for p in pending]}")

            def _call(idx: int) -> Tuple[int, bool, str, bool]:
                ptxt = make_variant(idx + 1)
                ok, out_or_err, http_code, retryable = _generate_once(
                    api_key=key,
                    model=model,
                    instructions=instructions or "",
                    prompt=ptxt,
                    pil_img=pil_img,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    debug=debug,
                )
                return (idx, ok, out_or_err, retryable)

            next_round: List[int] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_call, idx): idx for idx in pending}
                for fut in as_completed(futs):
                    idx, ok, payload, retryable = fut.result()
                    if ok:
                        results[idx] = payload.strip()
                        last_errors.pop(idx, None)
                    else:
                        last_errors[idx] = payload
                        # retry only if retryable; otherwise mark hard fail
                        if retryable:
                            next_round.append(idx)
                        else:
                            hard_fail.append(idx)
                        print(f"[PVL_GEMINI] (batch item {idx+1}) API error: {payload}", flush=True)

            pending = next_round

            # Backoff only if there are retryable failures remaining
            if pending and attempt < tries:
                # linear backoff 2s, 4s, 6s with jitter to help when model is busy
                base = min(2 * attempt, 6)
                time.sleep(_jitter_delay(base))

        # If anything is left pending, they exhausted retries and still failed
        failed = sorted(set(pending + hard_fail))
        if failed:
            first_idx = failed[0]
            msg = last_errors.get(first_idx, "Unknown error")
            failed_str = ", ".join(str(i + 1) for i in failed)
            raise RuntimeError(f"Gemini API failed for batch item(s) {failed_str}: {msg}")

        combined = " {} ".format(delimiter).join(results[i] for i in range(batch))

        elapsed = time.time() - start_time
        print(f"[PVL_GEMINI] Completed in {elapsed:.2f} seconds (batch={batch}, tries={tries})", flush=True)

        return (combined,)


NODE_CLASS_MAPPINGS = {"PVL_Gemini_API": PVL_Gemini_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Gemini_API": "PVL - Gemini Api"}
