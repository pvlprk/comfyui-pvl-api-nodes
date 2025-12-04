# pvl_gemini_api_multi.py
# PVL - Gemini API (Google Developer API)
# Multi-image version: supports up to 6 optional image inputs.
# Batch-wise selective retries + Gemini error detection + linear timeout backoff
# Updated: optional OpenAI fallback / override for failed Gemini calls.
# All provided images are sent to OpenAI on fallback.

import os, io, time, base64, json
from typing import Any, Dict, Optional, Tuple, List, Callable
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

OPENAI_BASE = "https://api.openai.com/v1"

HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

OPENAI_MODEL_CHOICES = [
    "GPT-5.1",
    "GPT-5 pro",
    "GPT-5 mini",
    "GPT-5 nano",
]

# -----------------------------
# Utils
# -----------------------------
def _log_debug(debug: bool, *args):
    if debug:
        print("[PVL_GEMINI_MULTI]", *args, flush=True)


def _get_api_key(provided: str) -> str:
    if provided and provided.strip():
        return provided.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


def _get_openai_api_key(provided: str) -> str:
    if provided and provided.strip():
        return provided.strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


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


# ---- multi-image contents builder ----
def _build_contents(prompt: Optional[str], pil_imgs: List[Optional[Image.Image]]) -> list:
    parts = []
    if prompt and str(prompt).strip():
        parts.append({"text": str(prompt)})
    for pil_img in pil_imgs:
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
# Gemini error detection
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


# --- timeout treated as retryable (status 599) ---
def _post(url: str, payload: Dict[str, Any], api_key: str, timeout: int, debug: bool, note: str) -> requests.Response:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    try:
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
    except requests.exceptions.Timeout:
        if debug:
            print(f"[PVL_GEMINI_MULTI] Request timeout after {timeout}s")
        class TimeoutResponse:
            status_code = 599
            text = "Client timeout"
            def json(self): return {"error": {"message": "Client timeout", "status": "TIMEOUT"}}
        return TimeoutResponse()
    except requests.exceptions.RequestException as e:
        if debug:
            print(f"[PVL_GEMINI_MULTI] Request exception: {e}")
        raise


# -----------------------------
# OpenAI helpers (Responses API)
# -----------------------------

def _map_openai_model_name(choice: str) -> str:
    """
    Map UI-friendly model name to OpenAI model id.
    """
    mapping = {
        "GPT-5.1": "gpt-5.1",
        "GPT-5 pro": "gpt-5-pro",
        "GPT-5 mini": "gpt-5-mini",
        "GPT-5 nano": "gpt-5-nano",
    }
    return mapping.get(choice, "gpt-5.1")


def _openai_build_input(prompt: Optional[str], pil_images: Optional[List[Image.Image]]) -> List[Dict[str, Any]]:
    """
    Build OpenAI multimodal input. All provided images are sent as input_image blocks.
    """
    content: List[Dict[str, Any]] = []
    if prompt and str(prompt).strip():
        content.append({"type": "input_text", "text": str(prompt)})
    if pil_images:
        for img in pil_images:
            if img is None:
                continue
            data_b64 = _b64_from_pil(img, mime="image/png")
            data_url = f"data:image/png;base64,{data_b64}"
            content.append({"type": "input_image", "image_url": data_url})
    if not content:
        content = [{"type": "input_text", "text": ""}]
    return [{"role": "user", "content": content}]


def _openai_extract_text(resp_json: Dict[str, Any]) -> str:
    """
    Extract plain text from OpenAI Responses API JSON.

    We try, in order:
      - top-level "text" if it's a string
      - then search all output[i].content[j].text / .value
    """
    txt = resp_json.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    output = resp_json.get("output") or []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text_val = part.get("text") or part.get("value")
                if isinstance(text_val, str) and text_val.strip():
                    return text_val.strip()

    return ""


def _openai_post(url: str, payload: Dict[str, Any], api_key: str, timeout: int, debug: bool) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        if debug:
            san = json.loads(json.dumps(payload))
            # Mask potential data URLs so we don't dump huge base64 blobs.
            for item in san.get("input", []):
                for part in item.get("content", []):
                    if part.get("type") == "input_image":
                        url_val = part.get("image_url", "")
                        if isinstance(url_val, str) and url_val.startswith("data:image"):
                            part["image_url"] = "<data:image; base64 bytes masked>"
            _log_debug(debug, f"POST {url} (OpenAI Responses)")
            _log_debug(debug, "Request JSON:", json.dumps(san, ensure_ascii=False)[:10000])
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if debug:
            _log_debug(debug, f"OpenAI HTTP {resp.status_code}")
            _log_debug(debug, f"Raw OpenAI response (trunc 20k): {resp.text[:20000]}")
        return resp
    except requests.exceptions.Timeout:
        if debug:
            print(f"[PVL_GEMINI_MULTI] OpenAI request timeout after {timeout}s")
        class TimeoutResponse:
            status_code = 599
            text = "Client timeout"
            def json(self): return {"error": {"message": "Client timeout", "type": "timeout"}}
        return TimeoutResponse()
    except requests.exceptions.RequestException as e:
        if debug:
            print(f"[PVL_GEMINI_MULTI] OpenAI request exception: {e}")
        raise


def _openai_generate_once(
    api_key: str,
    model_choice: str,
    instructions: Optional[str],
    prompt: Optional[str],
    pil_images: Optional[List[Image.Image]],
    timeout: int,
    temperature: float,  # kept in signature for compatibility; not sent
    top_p: float,        # kept in signature for compatibility; not sent
    debug: bool,
) -> Tuple[bool, str]:
    """
    Single non-streaming call to OpenAI Responses API.
    NOTE: We do NOT send temperature or top_p because some models reject them.
    """
    model = _map_openai_model_name(model_choice)
    url = f"{OPENAI_BASE}/responses"

    input_block = _openai_build_input(prompt, pil_images)

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_block,
    }
    # Do NOT include temperature or top_p — some models don't support them.

    if instructions and str(instructions).strip():
        payload["instructions"] = str(instructions).strip()

    resp = _openai_post(url, payload, api_key, timeout, debug)

    if resp.status_code == 599:
        return False, "OpenAI client timeout"

    if resp.status_code != 200:
        try:
            j = resp.json()
        except Exception:
            return False, f"OpenAI HTTP {resp.status_code}: {resp.text[:1000]}"
        err = j.get("error") or {}
        msg = err.get("message") or resp.text[:800]
        etype = err.get("type") or err.get("code")
        return False, f"OpenAI HTTP {resp.status_code} error ({etype}): {msg}"

    data = resp.json()
    text = _openai_extract_text(data)
    if not text:
        return False, "Empty text in OpenAI response"
    return True, text


def _run_openai_for_indices(
    indices: List[int],
    per_call_prompt: Callable[[int], str],
    pil_images: Optional[List[Image.Image]],
    instructions: str,
    timeout: int,
    temperature: float,
    top_p: float,
    tries: int,
    openai_key: str,
    openai_model_choice: str,
    debug: bool,
    label: str,
) -> Tuple[Dict[int, str], Dict[int, str], List[int]]:
    """
    Run OpenAI Responses API for the provided 0-based indices with retry logic.
    Returns (results, last_errors, pending_indices).
    """
    results: Dict[int, str] = {}
    last_errors: Dict[int, str] = {}
    pending: List[int] = list(indices)
    attempt = 0
    max_workers = min(len(pending), 8) if pending else 1
    max_tries = max(1, tries)

    while pending and attempt < max_tries:
        attempt += 1
        if debug:
            _log_debug(debug, f"[OpenAI {label}] Retry round {attempt}/{max_tries} for indices: {[p+1 for p in pending]}")

        def _call(idx: int) -> Tuple[int, bool, str]:
            prompt_variant = per_call_prompt(idx + 1)
            ok, out_or_err = _openai_generate_once(
                api_key=openai_key,
                model_choice=openai_model_choice,
                instructions=instructions or "",
                prompt=prompt_variant or "",
                pil_images=pil_images,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                debug=debug,
            )
            return (idx, ok, out_or_err)

        next_round: List[int] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_call, idx): idx for idx in pending}
            for fut in as_completed(futs):
                idx, ok, payload = fut.result()
                if ok:
                    results[idx] = payload.strip()
                    last_errors.pop(idx, None)
                else:
                    last_errors[idx] = payload
                    lower = payload.lower()
                    if "timeout" in lower or "429" in lower or "503" in lower:
                        next_round.append(idx)
                    else:
                        print(f"[PVL_GEMINI_MULTI] (OpenAI {label} batch item {idx+1}) API error: {payload}", flush=True)

        pending = next_round
        if pending and attempt < max_tries:
            delay = attempt  # linear backoff
            if debug:
                print(f"[PVL_GEMINI_MULTI] [OpenAI {label}] Waiting {delay}s before retry...")
            time.sleep(delay)

    return results, last_errors, pending


# -----------------------------
# Core Gemini request (multi-image aware)
# -----------------------------
def _generate_once(api_key: str, model: str, instructions: Optional[str], prompt: Optional[str],
                   pil_imgs: List[Optional[Image.Image]], timeout: int, temperature: float,
                   top_p: float, top_k: int, debug: bool) -> Tuple[bool, str]:
    sys_instr = {"parts": [{"text": instructions.strip()}]} if (instructions and instructions.strip()) else None
    contents = _build_contents(prompt=prompt, pil_imgs=pil_imgs)

    gen_cfg: Dict[str, Any] = {"temperature": float(temperature)}
    if isinstance(top_p, (int, float)) and top_p > 0:
        gen_cfg["topP"] = float(top_p)
    if isinstance(top_k, int) and top_k > 0:
        gen_cfg["topK"] = int(top_k)
    base_payload: Dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}

    def try_one_endpoint(api_ver: str, sys_key: str) -> Tuple[bool, str, Optional[int]]:
        url = _gen_url(model, api_ver)
        payload = json.loads(json.dumps(base_payload))
        if sys_instr is not None:
            payload[sys_key] = sys_instr
        resp = _post(url, payload, api_key, timeout, debug, sys_key)
        # timeout or retryable-like 599
        if resp.status_code == 599:
            return False, "Client timeout", 599

        if resp.status_code != 200:
            try:
                j = resp.json()
            except Exception:
                return False, f"HTTP {resp.status_code} at {url}: {resp.text[:1000]}", resp.status_code
            err = (j.get("error") or {})
            code = err.get("code")
            msg = err.get("message") or resp.text[:800]
            status = err.get("status")
            return False, f"HTTP {resp.status_code} error ({status or code}): {msg}", resp.status_code

        data = resp.json()
        block = _is_blocked_prompt(data)
        if block:
            return False, block, 200
        fr_err = _finish_reason_error(data)
        if fr_err:
            return False, fr_err, 200

        text = _extract_text(data)
        if not text:
            return False, "Empty text in successful response", 200
        return True, text, 200

    ok, res, code = try_one_endpoint(PRIMARY_VER, "system_instruction")
    if ok:
        return True, res
    if code == 400 and ("system_instruction" in res or "systemInstruction" in res):
        ok2, res2, code2 = try_one_endpoint(PRIMARY_VER, "systemInstruction")
        if ok2:
            return True, res2
        res, code = res2, code2
    if code == 404:
        ok3, res3, _ = try_one_endpoint(FALLBACK_VER, "system_instruction")
        if ok3:
            return True, res3
        return False, res3
    return False, res


# -----------------------------
# Node
# -----------------------------
class PVL_Gemini_with_fallback_API_Multi:
    @classmethod
    def INPUT_TYPES(cls):
        default_model = "gemini-2.5-flash"
        return {
            "required": {
                "model": (HARDCODED_MODELS, {"default": default_model}),
                "tries": ("INT",   {"default": 2,  "min": 1,   "max": 10,  "step": 1}),
                "timeout": ("INT", {"default": 45, "min": 1,   "max": 600, "step": 5}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 65, "min": 1, "max": 1000, "step": 1}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "delimiter": ("STRING", {"default": "[++]"}),
                "append_variation_tag": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
                "instructions": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                # OpenAI-related optional controls
                "openai_fallback": ("BOOLEAN", {"default": False}),
                "force_openai": ("BOOLEAN", {"default": False}),
                "openai_model": (OPENAI_MODEL_CHOICES, {"default": "GPT-5 mini"}),
                "openai_api_key": ("STRING", {"default": ""}),
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
            seed: int = 0,
            image1: Any = None, image2: Any = None, image3: Any = None,
            image4: Any = None, image5: Any = None, image6: Any = None,
            api_key: str = "",
            openai_fallback: bool = False, force_openai: bool = False,
            openai_model: str = "GPT-5 mini", openai_api_key: str = ""):

        start_time = time.time()
        key = _get_api_key(api_key)

        # collect and compact provided images
        imgs_raw = [image1, image2, image3, image4, image5, image6]
        pil_imgs: List[Optional[Image.Image]] = [img for img in (_tensor_to_pil_first(i) for i in imgs_raw) if img is not None]
        if not ((instructions and instructions.strip()) or (prompt and str(prompt).strip()) or pil_imgs):
            raise RuntimeError("Nothing to send: provide at least one of instructions, prompt, or image.")

        def per_call_prompt(i: int) -> str:
            base = str(prompt or "")
            if append_variation_tag and batch > 1 and base.strip():
                return f"{base.rstrip()}\n-----\nVariation {i}"
            return base

        # 1) Force OpenAI: skip Gemini entirely.
        if force_openai:
            openai_key = _get_openai_api_key(openai_api_key)
            if not openai_key:
                raise RuntimeError("Missing OpenAI API key. Provide 'openai_api_key' or set OPENAI_API_KEY.")

            if debug:
                _log_debug(debug, f"Force OpenAI enabled, model={openai_model}. Skipping Gemini completely.")

            indices = list(range(batch))
            oa_results, oa_errors, oa_pending = _run_openai_for_indices(
                indices=indices,
                per_call_prompt=per_call_prompt,
                pil_images=pil_imgs,
                instructions=instructions or "",
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                tries=tries,
                openai_key=openai_key,
                openai_model_choice=openai_model,
                debug=debug,
                label="force",
            )

            failed_force = [i for i in indices if i not in oa_results]
            if failed_force:
                first_idx = failed_force[0]
                msg = oa_errors.get(first_idx, "Unknown error")
                failed_str = ", ".join(str(i + 1) for i in failed_force)
                raise RuntimeError(f"OpenAI API failed for batch item(s) {failed_str}: {msg}")

            combined_force = f" {delimiter} ".join(oa_results[i] for i in range(batch))
            elapsed = time.time() - start_time
            print(f"[PVL_GEMINI_MULTI] Completed via OpenAI only in {elapsed:.2f}s (batch={batch}, tries={tries})", flush=True)
            return (combined_force,)

        # 2) Gemini primary path (with optional OpenAI fallback).
        if not key:
            raise RuntimeError("Missing API key. Provide 'api_key' or set GEMINI_API_KEY.")

        results: Dict[int, str] = {}
        last_errors: Dict[int, str] = {}
        pending: List[int] = list(range(batch))
        attempt = 0
        max_workers = min(batch, 8)
        blocked_indices: List[int] = []

        while pending and attempt < tries:
            attempt += 1
            if debug:
                _log_debug(debug, f"Retry round {attempt}/{tries} for indices: {[p+1 for p in pending]}")

            def _call(idx: int) -> Tuple[int, bool, str]:
                prompt_variant = per_call_prompt(idx + 1)
                ok, out_or_err = _generate_once(
                    api_key=key,
                    model=model,
                    instructions=instructions or "",
                    prompt=prompt_variant or "",
                    pil_imgs=pil_imgs,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    debug=debug,
                )
                return (idx, ok, out_or_err)

            next_round: List[int] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_call, idx): idx for idx in pending}
                for fut in as_completed(futs):
                    idx, ok, payload = fut.result()
                    if ok:
                        results[idx] = payload.strip()
                        last_errors.pop(idx, None)
                    else:
                        last_errors[idx] = payload
                        lower = payload.lower()
                        if "prompt blocked" in lower or "blockreason" in lower:
                            # Prohibited content / moderation error -> do not retry with Gemini.
                            blocked_indices.append(idx)
                            print(f"[PVL_GEMINI_MULTI] (batch item {idx+1}) blocked by Gemini: {payload}", flush=True)
                        elif "timeout" in lower or "429" in lower or "503" in lower:
                            next_round.append(idx)
                        else:
                            print(f"[PVL_GEMINI_MULTI] (batch item {idx+1}) API error: {payload}", flush=True)

            # Only retry non-blocked indices.
            pending = [i for i in next_round if i not in blocked_indices]
            if pending and attempt < tries:
                delay = attempt  # linear backoff
                if debug:
                    print(f"[PVL_GEMINI_MULTI] Waiting {delay}s before retry...")
                time.sleep(delay)

        # Determine which indices still do not have a successful result.
        failed_indices = [i for i in range(batch) if i not in results]

        # 3) If there are failures and OpenAI fallback is enabled, send ONLY those to OpenAI.
        if failed_indices and openai_fallback:
            openai_key = _get_openai_api_key(openai_api_key)
            if not openai_key:
                first_idx = failed_indices[0]
                msg = last_errors.get(first_idx, "Unknown error")
                failed_str = ", ".join(str(i + 1) for i in failed_indices)
                raise RuntimeError(
                    f"Gemini API failed for batch item(s) {failed_str}: {msg} (and no OpenAI key available for fallback)"
                )

            remaining_tries = max(1, tries - attempt)
            if debug:
                _log_debug(
                    debug,
                    f"Using OpenAI fallback for indices {[i+1 for i in failed_indices]} "
                    f"with remaining_tries={remaining_tries}, model={openai_model}"
                )

            oa_results, oa_errors, oa_pending = _run_openai_for_indices(
                indices=failed_indices,
                per_call_prompt=per_call_prompt,
                pil_images=pil_imgs,
                instructions=instructions or "",
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                tries=remaining_tries,
                openai_key=openai_key,
                openai_model_choice=openai_model,
                debug=debug,
                label="fallback",
            )

            # Merge successful fallback results into main results.
            for idx, text in oa_results.items():
                results[idx] = text

            # Recompute failures after fallback.
            failed_after_fallback = [i for i in range(batch) if i not in results]
            if failed_after_fallback:
                first_idx = failed_after_fallback[0]
                msg = oa_errors.get(first_idx) or last_errors.get(first_idx, "Unknown error")
                failed_str = ", ".join(str(i + 1) for i in failed_after_fallback)
                raise RuntimeError(f"Gemini + OpenAI fallback both failed for batch item(s) {failed_str}: {msg}")

        # 4) If we still have failures and no fallback, raise as before.
        if not openai_fallback and failed_indices:
            first_idx = failed_indices[0]
            msg = last_errors.get(first_idx, "Unknown error")
            failed_str = ", ".join(str(i + 1) for i in failed_indices)
            raise RuntimeError(f"Gemini API failed for batch item(s) {failed_str}: {msg}")

        combined = f" {delimiter} ".join(results[i] for i in range(batch))
        elapsed = time.time() - start_time
        print(f"[PVL_GEMINI_MULTI] Completed in {elapsed:.2f}s (batch={batch}, tries={tries})", flush=True)
        return (combined,)


NODE_CLASS_MAPPINGS = {"PVL_Gemini_with_fallback_API_Multi": PVL_Gemini_with_fallback_API_Multi}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Gemini_with_fallback_API_Multi": "PVL Gemini with fallback Multi"}
