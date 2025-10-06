# pvl_gemini_api.py
# PVL - Gemini API (Google Developer API)
# Supports: batch parallel calls, delimiter output, optional "(variation N)" suffix toggle.
# Prints total execution time always, even when debug is off.

import os, io, time, base64, json
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

def _generate(api_key: str, model: str, instructions: Optional[str], prompt: Optional[str],
              pil_img: Optional[Image.Image], tries: int, timeout: int, temperature: float,
              top_p: float, top_k: int, debug: bool) -> Tuple[bool, str]:
    sys_instr = {"parts": [{"text": instructions.strip()}]} if (instructions and instructions.strip()) else None
    contents = _build_contents(prompt=prompt, pil_img=pil_img)

    gen_cfg: Dict[str, Any] = {"temperature": float(temperature)}
    if isinstance(top_p, (int, float)) and top_p > 0:
        gen_cfg["topP"] = float(top_p)
    if isinstance(top_k, int) and top_k > 0:
        gen_cfg["topK"] = int(top_k)

    base_payload: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}

    def try_one_endpoint(api_ver: str) -> Tuple[bool, str, Optional[int]]:
        url = _gen_url(model, api_ver)
        payload = json.loads(json.dumps(base_payload))
        if sys_instr is not None:
            payload["system_instruction"] = sys_instr
        resp = _post(url, payload, api_key, timeout, debug, "system_instruction")
        if resp.status_code == 200:
            data = resp.json()
            return True, _extract_text(data), 200
        if resp.status_code == 404:
            return False, f"HTTP 404 at {url}: {resp.text[:600]}", 404
        if resp.status_code == 400 and ("system_instruction" in resp.text):
            payload2 = json.loads(json.dumps(base_payload))
            if sys_instr is not None:
                payload2["systemInstruction"] = sys_instr
            resp2 = _post(url, payload2, api_key, timeout, debug, "systemInstruction")
            if resp2.status_code == 200:
                data2 = resp2.json()
                return True, _extract_text(data2), 200
            return False, f"HTTP {resp2.status_code} at {url}: {resp2.text[:1000]}", resp2.status_code
        return False, f"HTTP {resp.status_code} at {url}: {resp.text[:1000]}", resp.status_code

    last_err = ""
    for attempt in range(1, max(1, tries) + 1):
        ok, res, code = try_one_endpoint(PRIMARY_VER)
        if ok:
            return True, res
        last_err = res
        if code == 404:
            ok2, res2, _ = try_one_endpoint(FALLBACK_VER)
            if ok2:
                return True, res2
            last_err = res2
        time.sleep(min(2 * attempt, 6))

    return False, last_err or "Unknown error"


class PVL_Gemini_API:
    """PVL Gemini API with batch, delimiter, '(variation N)' suffix toggle, and time logging."""

    @classmethod
    def INPUT_TYPES(cls):
        default_model = "gemini-2.5-flash"
        return {
            "required": {
                "model": (HARDCODED_MODELS, {"default": default_model}),
                "tries": ("INT", {"default": 2, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 45, "min": 1, "max": 600}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 10}),
                "delimiter": ("STRING", {"default": "\\n-----\\n", "multiline": False}),
                "append_variation_tag": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "instructions": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
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
            image: Any = None, api_key: str = ""):

        start_time = time.time()  # <-- Start timer

        key = _get_api_key(api_key)
        if not key:
            raise RuntimeError("Missing API key. Provide 'api_key' or set GEMINI_API_KEY.")

        pil_img = _tensor_to_pil_first(image) if image is not None else None
        if pil_img is None and image is not None and debug:
            _log_debug(debug, "IMAGE provided but could not be converted; continuing without image.")

        if not ((instructions and instructions.strip()) or (prompt and str(prompt).strip()) or pil_img is not None):
            raise RuntimeError("Nothing to send: provide at least one of instructions, prompt, or image.")

        def single_call(i: int):
            prompt_variant = (
                f"{prompt.rstrip()}\n-----\nVariation {i}"
                if append_variation_tag and batch > 1 and prompt.strip()
                else prompt
            )
            ok, result = _generate(
                api_key=key, model=model, instructions=instructions or "",
                prompt=prompt_variant or "", pil_img=pil_img, tries=tries,
                timeout=timeout, temperature=temperature,
                top_p=top_p, top_k=top_k, debug=debug
            )
            if not ok:
                return f"[Error {i}] {result}"
            return result.strip()

        results: List[str] = []
        with ThreadPoolExecutor(max_workers=min(batch, 8)) as ex:
            futs = [ex.submit(single_call, i + 1) for i in range(batch)]
            for fut in as_completed(futs):
                results.append(fut.result())

        combined = f" {delimiter} ".join(results)

        # --- Time reporting ---
        elapsed = time.time() - start_time
        print(f"[PVL_GEMINI] Completed in {elapsed:.2f} seconds (batch={batch})")

        return (combined,)


NODE_CLASS_MAPPINGS = {"PVL_Gemini_API": PVL_Gemini_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Gemini_API": "PVL - Gemini Api"}
