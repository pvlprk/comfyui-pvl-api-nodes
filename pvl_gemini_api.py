# pvl_gemini_api.py
# PVL - Gemini Api (Google Developer API), **hard-coded model list**, no live fetch.
# - Inputs: instructions (optional), prompt (optional), image (optional)
# - Models: gemini-2.5-pro / -2.5-flash / -2.5-flash-lite / -2.0-flash / -1.5-pro / -1.5-flash
# - generationConfig: temperature, topP, topK
# - Retries, timeout, debug logging
# - API key via input or GEMINI_API_KEY env
#
# API contract per docs:
#   Primary: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
#   Fallback: retry once against /v1/ only on HTTP 404
#   System prompt field: "system_instruction" (snake_case)
#
# References:
#   - Generate Content endpoint + examples: ai.google.dev/api/generate-content
#   - Model catalogue & IDs: ai.google.dev/gemini-api/docs/models

import os, io, time, base64, json
from typing import List, Tuple, Any, Dict, Optional

import requests
from PIL import Image

try:
    import torch, numpy as np
except Exception:
    torch = None
    np = None

GEMINI_BASE = "https://generativelanguage.googleapis.com"
PRIMARY_VER = "v1beta"   # default per docs
FALLBACK_VER = "v1"      # only if 404 at endpoint

# ---- Hard-coded stable model list (text-only & multimodal) ----
HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

def _log_debug(debug: bool, *args):
    if debug:
        print("[PVL_GEMINI]", *args, flush=True)

def _get_api_key(provided: str) -> str:
    if provided and provided.strip():
        return provided.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()

def _tensor_to_pil_first(image_tensor: Any) -> Optional[Image.Image]:
    """ComfyUI IMAGE tensor/list -> PIL (first frame)."""
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
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)

def _b64_from_pil(pil_img: Image.Image, mime: str = "image/png") -> Tuple[str, str]:
    buf = io.BytesIO()
    fmt = "PNG" if mime == "image/png" else "JPEG"
    pil_img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return data, mime

def _build_contents(prompt: Optional[str], pil_img: Optional[Image.Image]) -> List[Dict[str, Any]]:
    parts = []
    if prompt and str(prompt).strip():
        parts.append({"text": str(prompt)})
    if pil_img is not None:
        b64, mime = _b64_from_pil(pil_img, mime="image/png")
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    # Ensure at least one Part so request is valid if only system instruction is sent
    if not parts:
        parts = [{"text": ""}]
    return [{"role": "user", "parts": parts}]

def _extract_text(response_json: Dict[str, Any]) -> str:
    cands = response_json.get("candidates") or []
    if not cands:
        pf = response_json.get("promptFeedback") or {}
        br = (pf.get("blockReason") or pf.get("block_reason") or "")
        return f"[Gemini] No candidates returned{f' (blocked: {br})' if br else ''}."
    content = (cands[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    out = []
    for p in parts:
        if "text" in p:
            out.append(p["text"])
    return "".join(out).strip()

def _gen_url(model: str, api_version: str) -> str:
    model_path = model if model.startswith("models/") else f"models/{model}"
    return f"{GEMINI_BASE}/{api_version}/{model_path}:generateContent"

def _request_once(url: str, payload: Dict[str, Any], api_key: str, timeout: int, debug: bool) -> requests.Response:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    if debug:
        sanitized = json.loads(json.dumps(payload))
        for msg in sanitized.get("contents", []):
            for part in msg.get("parts", []):
                if "inline_data" in part and "data" in part["inline_data"]:
                    part["inline_data"]["data"] = f"<{len(part['inline_data']['data'])} base64 bytes>"
        _log_debug(debug, f"POST {url}")
        _log_debug(debug, "Request JSON:", json.dumps(sanitized, ensure_ascii=False)[:10000])
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if debug:
        _log_debug(debug, f"HTTP {resp.status_code}")
        _log_debug(debug, f"Raw response (trunc 20k): {resp.text[:20000]}")
    return resp

def _generate_with_retries(
    api_key: str,
    model: str,
    instructions: Optional[str],
    prompt: Optional[str],
    pil_img: Optional[Image.Image],
    tries: int,
    timeout: int,
    temperature: float,
    top_p: float,
    top_k: int,
    debug: bool
) -> Tuple[bool, str]:
    sys_instr = {"parts": [{"text": instructions.strip()}]} if (instructions and instructions.strip()) else None
    contents = _build_contents(prompt=prompt, pil_img=pil_img)

    gen_cfg: Dict[str, Any] = {}
    if temperature is not None:
        gen_cfg["temperature"] = float(temperature)
    if isinstance(top_p, (int, float)) and top_p > 0:
        gen_cfg["topP"] = float(top_p)
    if isinstance(top_k, int) and top_k > 0:
        gen_cfg["topK"] = int(top_k)

    payload: Dict[str, Any] = {"contents": contents}
    if sys_instr is not None:
        # Per docs, prefer snake_case in REST examples; works across versions.
        payload["system_instruction"] = sys_instr
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    last_err = ""
    for attempt in range(1, max(1, tries) + 1):
        url_beta = _gen_url(model, PRIMARY_VER)
        try:
            resp = _request_once(url_beta, payload, api_key, timeout, debug)
            if resp.status_code == 200:
                data = resp.json()
                um = data.get("usageMetadata") or data.get("usage_metadata")
                if debug and um:
                    _log_debug(debug, f"usageMetadata: {json.dumps(um)}")
                return True, _extract_text(data)

            # If endpoint not found (rare), try v1 once.
            if resp.status_code == 404:
                url_v1 = _gen_url(model, FALLBACK_VER)
                _log_debug(debug, f"404 on {url_beta}. Retrying once with {url_v1} ...")
                resp2 = _request_once(url_v1, payload, api_key, timeout, debug)
                if resp2.status_code == 200:
                    data = resp2.json()
                    um = data.get("usageMetadata") or data.get("usage_metadata")
                    if debug and um:
                        _log_debug(debug, f"usageMetadata: {json.dumps(um)}")
                    return True, _extract_text(data)
                resp = resp2  # analyze below

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code} at {resp.request.url}: {resp.text[:600]}"
                _log_debug(debug, f"Retryable error attempt {attempt}: {last_err}")
                time.sleep(min(2 * attempt, 6))
                continue

            last_err = f"HTTP {resp.status_code} at {resp.request.url}: {resp.text[:1000]}"
            _log_debug(debug, f"Non-retryable: {last_err}")
            return False, last_err

        except requests.Timeout:
            last_err = f"Timeout after {timeout}s at {url_beta}"
            _log_debug(debug, f"Timeout attempt {attempt}: {last_err}")
            time.sleep(min(2 * attempt, 6))
            continue
        except Exception as e:
            last_err = f"Exception at {url_beta}: {e}"
            _log_debug(debug, f"Exception attempt {attempt}: {last_err}")
            time.sleep(min(2 * attempt, 6))
            continue

    return False, last_err or "Unknown error"

class PVL_Gemini_API:
    """
    PVL - Gemini Api
      - instructions (optional), prompt (optional), image (optional)
      - models: hard-coded list (1.5 / 2.0 / 2.5 ; pro / flash / flash-lite)
      - generationConfig: temperature, topP, topK
      - retries / timeout / debug logging
    """

    @classmethod
    def INPUT_TYPES(cls):
        default_model = "gemini-2.5-flash" if "gemini-2.5-flash" in HARDCODED_MODELS else HARDCODED_MODELS[0]
        return {
            "required": {
                "model": (HARDCODED_MODELS, {"default": default_model}),
                "tries": ("INT", {"default": 2, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 45, "min": 1, "max": 600}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),  # 0.0 => omit
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),                    # 0 => omit
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

    def run(
        self,
        model: str,
        tries: int,
        timeout: int,
        temperature: float,
        top_p: float,
        top_k: int,
        debug: bool,
        instructions: Optional[str] = "",
        prompt: Optional[str] = "",
        image: Any = None,
        api_key: str = "",
    ):
        key = _get_api_key(api_key)
        if not key:
            msg = ("[Gemini] Missing API key. Provide 'api_key' input or set GEMINI_API_KEY.")
            _log_debug(True, msg)
            raise RuntimeError(msg)

        pil_img = _tensor_to_pil_first(image) if image is not None else None
        if pil_img is None and image is not None and debug:
            _log_debug(debug, "IMAGE provided but could not be converted; continuing without image.")

        if not ((instructions and instructions.strip()) or (prompt and str(prompt).strip()) or pil_img is not None):
            raise RuntimeError("Nothing to send: provide at least one of instructions, prompt, or image.")

        ok, result = _generate_with_retries(
            api_key=key,
            model=model,
            instructions=instructions or "",
            prompt=(prompt or ""),
            pil_img=pil_img,
            tries=tries,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            debug=debug,
        )
        if not ok:
            raise RuntimeError(f"Gemini error: {result}")
        return (result,)

NODE_CLASS_MAPPINGS = {"PVL_Gemini_API": PVL_Gemini_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Gemini_API": "PVL - Gemini Api"}
