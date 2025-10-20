# pvl_google_nano_banana.py
# Node: PVL Google Nano-Banana API (Gemini + FAL) — with delimiter support + parallel prompts
# Author: PVL
# License: MIT
#
# Updated: Selective retry + robust error policy
# - Google pass for all.
# - Retry Google ONCE for retryable (408/429/5xx, overload/quota/timeout) failures only.
# - Do NOT retry Google on 4xx like 400 INVALID_ARGUMENT, 403 PERMISSION_DENIED.
# - If Google safety block (promptFeedback.blockReason or finishReason=SAFETY) -> NO Google retry; route to FAL.
# - Remaining failures -> FAL (once).
# - If FAL reports safety/policy error for any item -> raise immediately (halt workflow) with API error.
# - If none succeed overall -> raise error; else return only successful outputs and print errors for failed items.
#
# Requires:
#   pip install google-genai requests pillow numpy torch

import os
import io
import json
import base64
import typing as T
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
import torch

NODE_NAME = "PVL Google Nano-Banana API"
NODE_CATEGORY = "PVL/Google"
DEFAULT_MODEL = "gemini-2.5-flash-image-preview"
_TOP_P = 0.95
_TOP_K = 64
_MAX_TOKENS = 4096
_VALID_ASPECTS = {"21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16"}

# ----------------- image helpers -----------------

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, ...]
    return t

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")

def encode_pil_bytes(img: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()

def stack_images_same_size(tensors: T.List[torch.Tensor], debug: bool = False) -> torch.Tensor:
    """Concatenate (B,H,W,C) batches along B. If shapes mismatch, resize to the first image size."""
    if not tensors:
        raise RuntimeError("No images to stack.")
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError:
        if debug:
            print("[PVL NANO NODE] Mismatched sizes, resizing to match first image.")
        target_h, target_w = tensors[0].shape[1], tensors[0].shape[2]
        fixed = []
        for t in tensors:
            pil = tensor_to_pil(t)
            rp = pil.resize((target_w, target_h), Image.LANCZOS)
            fixed.append(pil_to_tensor(rp))
        return torch.cat(fixed, dim=0)

def _extract_image_bytes_from_part(part) -> T.Optional[bytes]:
    try:
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=False)
                except Exception:
                    return None
    except Exception:
        pass
    if isinstance(part, dict):
        blob = None
        if "inline_data" in part and isinstance(part["inline_data"], dict):
            blob = part["inline_data"].get("data")
        elif "inlineData" in part and isinstance(part["inlineData"], dict):
            blob = part["inlineData"].get("data")
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=False)
            except Exception:
                return None
    return None

def _extract_text_from_part(part) -> T.Optional[str]:
    try:
        txt = getattr(part, "text", None)
        if txt is not None:
            return str(txt)
    except Exception:
        pass
    if isinstance(part, dict) and "text" in part and part["text"] is not None:
        return str(part["text"])
    return None

def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")

# ----------------- 200-OK but blocked/aborted detection -----------------

_GOOD_FINISH = {None, "STOP", "FINISH_REASON_UNSPECIFIED"}

def _gemini_block_or_abort_msg(resp) -> T.Optional[str]:
    """
    Returns a human-readable error if the SDK response indicates a block or abnormal finish.
    Works with google-genai SDK objects (prompt_feedback, candidates[].finish_reason).
    """
    try:
        pf = getattr(resp, "prompt_feedback", None) or getattr(resp, "promptFeedback", None)
        if pf:
            br = getattr(pf, "block_reason", None) or getattr(pf, "blockReason", None)
            if br:
                return f"Prompt blocked (blockReason={br})"
    except Exception:
        pass

    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return "No candidates in response"
        fr = getattr(cands[0], "finish_reason", None) or getattr(cands[0], "finishReason", None)
        if fr not in _GOOD_FINISH:
            return f"Content generation stopped (finishReason={fr})"
    except Exception:
        pass
    return None

# ----------------- error classification helpers -----------------

def _parse_http_code_from_msg(msg: str) -> T.Optional[int]:
    import re as _re
    m = _re.search(r'\bHTTP\s+(\d{3})\b', msg, flags=_re.I)
    if m:
        return int(m.group(1))
    m = _re.search(r'\bstatus\s*[:=]?\s*(\d{3})\b', msg, flags=_re.I)
    if m:
        return int(m.group(1))
    return None

def _classify_google_error(msg: str) -> T.Tuple[bool, bool]:
    """
    Returns (retryable, safety_block)
    - retryable True for 408/429/5xx/quota/overload/transient
    - retryable False for 4xx like invalid argument / permission denied
    - safety_block True if policy/safety rejected (no Google retry; route to FAL)
    """
    s = msg.lower()

    # Safety
    if "blockreason" in s or "prompt blocked" in s:
        return (False, True)
    if "finishreason=safety" in s or "finish_reason=safety" in s:
        return (False, True)
    if "safety" in s and ("blocked" in s or "policy" in s):
        return (False, True)

    code = _parse_http_code_from_msg(msg)
    if code is not None:
        if code in (408, 429) or 500 <= code <= 599:
            return (True, False)
        if 400 <= code <= 499:
            return (False, False)

    # Keywords
    if "unavailable" in s or "overloaded" in s or "try again later" in s:
        return (True, False)
    if "quota" in s or "rate limit" in s or "exceeded" in s:
        return (True, False)
    if "invalid_argument" in s or "invalid argument" in s:
        return (False, False)
    if "permission_denied" in s or "permission denied" in s:
        return (False, False)

    # default: assume transient
    return (True, False)

def _classify_fal_error(msg: str) -> T.Tuple[bool, bool]:
    """
    Returns (retryable, safety_block) for FAL.
    - safety if queue status ERROR mentions safety/policy/blocked
    """
    s = msg.lower()
    code = _parse_http_code_from_msg(msg)
    if "status error" in s and ("safety" in s or "policy" in s or "blocked" in s):
        return (False, True)
    if "safety" in s and ("blocked" in s or "policy" in s):
        return (False, True)
    if code is not None:
        if code in (408, 429) or 500 <= code <= 599:
            return (True, False)
        if 400 <= code <= 499:
            return (False, False)
    return (True, False)

# ----------------- main node -----------------

class PVL_Google_NanoBanana_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A tiny banana spaceship over a neon city."}),
                "delimiter": ("STRING", {"default": "[++]", "multiline": False, "placeholder": "Regex delimiter e.g. [*], \\n, |"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "aspect_ratio": ("STRING", {"default": "1:1", "placeholder": "e.g. 16:9, 9:16, 3:2"}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "request_id": ("STRING", {"default": ""}),
                "debug_log": ("BOOLEAN", {"default": False}),
                # FAL flags
                "use_fal_fallback": ("BOOLEAN", {"default": True}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use FAL_KEY"}),
                "fal_route_img2img": ("STRING", {"default": "fal-ai/nano-banana/edit"}),
                "fal_route_txt2img": ("STRING", {"default": "fal-ai/nano-banana"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("text", "images")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    # -------- helpers --------

    def _make_client(self, api_key: str, endpoint_override: str):
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError("Google GenAI SDK not installed. Run: pip install google-genai") from e
        http_options = None
        if endpoint_override.strip():
            try:
                http_options = types.HttpOptions(base_url=endpoint_override.strip())
            except Exception:
                http_options = None
        if http_options is not None:
            client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            client = genai.Client(api_key=api_key)
        return client

    def _build_parts(self, prompt: str, images: T.Optional[torch.Tensor], mime: str):
        parts: T.List[dict] = []
        if prompt and prompt.strip():
            parts.append({"text": prompt})
        if images is not None and torch.is_tensor(images):
            batch = images if images.ndim == 4 else images.unsqueeze(0)
            for i in range(batch.shape[0]):
                pil = tensor_to_pil(batch[i:i+1])
                parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        return parts

    def _build_config(self, want_text: bool, aspect_ratio: str):
        """Build a config object for Gemini (google-genai SDK) or a fallback dict."""
        try:
            from google.genai import types
            cfg = types.GenerateContentConfig(
                top_p=float(_TOP_P),
                top_k=int(_TOP_K),
                max_output_tokens=int(_MAX_TOKENS),
                response_modalities=["IMAGE", "TEXT"] if want_text else ["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            return cfg
        except Exception:
            # Fallback dict config
            return {
                "top_p": float(_TOP_P),
                "top_k": int(_TOP_K),
                "max_output_tokens": int(_MAX_TOKENS),
                "response_modalities": ["IMAGE", "TEXT"] if want_text else ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

    def _build_call_prompts(self, base_prompts: T.List[str], num_images: int, debug: bool) -> T.List[str]:
        """
        Maps prompts to calls:
        - If len(prompts) >= num_images: take first num_images
        - If len(prompts) < num_images: repeat the last prompt to fill N
        """
        N = max(1, int(num_images))
        if not base_prompts:
            return [""]
        if len(base_prompts) >= N:
            call_prompts = base_prompts[:N]
        else:
            if debug:
                print(f"[PVL NANO NODE] Provided {len(base_prompts)} prompts but num_images={N}. "
                      f"Reusing the last prompt for remaining calls.")
                print(f"[PVL NANO WARNING] prompt list shorter than num_images ({len(base_prompts)} < {N}). "
                      f"Last entry will be reused for the remaining {N - len(base_prompts)} calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        if debug:
            for i, cp in enumerate(call_prompts, 1):
                show = cp if len(cp) <= 160 else (cp[:157] + "...")
                print(f"[PVL NANO NODE] Call {i} prompt: {show}")
        return call_prompts

    def _single_google_call(self, client, model: str, parts: list, cfg, request_id: str, debug: bool):
        kwargs = {}
        try:
            from google.genai import types
            if request_id.strip():
                kwargs["request_options"] = types.RequestOptions(request_id=request_id.strip())
        except Exception:
            pass
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config=cfg,
            **kwargs
        )
        # Treat 200-OK blocks/aborts as errors
        err = _gemini_block_or_abort_msg(resp)
        if err:
            raise RuntimeError(err)

        imgs, texts = [], []
        cands = getattr(resp, "candidates", None) or []
        for cand in cands:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            pparts = getattr(content, "parts", []) or []
            for p in pparts:
                blob = _extract_image_bytes_from_part(p)
                if blob:
                    try:
                        pil = Image.open(io.BytesIO(blob)).convert("RGB")
                        imgs.append(pil_to_tensor(pil))
                    except Exception as ex:
                        if debug:
                            print("[PVL NANO Debug] image decode error:", ex)
                else:
                    t = _extract_text_from_part(p)
                    if t:
                        texts.append(t)
        return imgs, texts, resp

    # -------- FAL Queue API - TWO PHASE EXECUTION --------

    def _fal_submit_only(self, route: str, prompt: str, image_tensor: T.Optional[torch.Tensor],
                         mime: str, fal_key: str, timeout: int, debug: bool,
                         output_format: str, aspect_ratio: str = "1:1", sync_mode: bool = False):
        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")

        data_urls: T.List[str] = []
        if image_tensor is not None and torch.is_tensor(image_tensor):
            batch = image_tensor if image_tensor.ndim == 4 else image_tensor.unsqueeze(0)
            for i in range(batch.shape[0]):
                pil = tensor_to_pil(batch[i:i+1])
                raw = encode_pil_bytes(pil, mime)
                data_urls.append(_data_url(mime, raw))

        base = "https://queue.fal.run"
        submit_url = f"{base}/{route.strip()}"
        headers = {"Authorization": f"Key {fal_key}"}

        payload = {
            "prompt": prompt or "",
            "num_images": 1,
            "output_format": "png" if str(output_format).lower() == "png" else "jpeg",
            "aspect_ratio": aspect_ratio,
            "sync_mode": sync_mode,
        }
        if data_urls:
            payload["image_urls"] = data_urls

        if debug:
            print(f"[FAL SUBMIT] prompt: {prompt[:60]}... sync_mode={sync_mode}")

        rr = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if rr.status_code != 200:
            raise RuntimeError(f"FAL submit error {rr.status_code}: {rr.text}")

        sub = rr.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        status_url = sub.get("status_url") or f"{base}/{route.strip()}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{route.strip()}/requests/{req_id}"

        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
            "prompt": prompt
        }

    def _fal_poll_and_fetch(self, request_info: dict, fal_key: str, timeout: int, debug: bool):
        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]
        req_id = request_info["request_id"]

        if debug:
            print(f"[FAL POLL] request_id={req_id[:16]}...")

        deadline = time.time() + timeout
        completed = False
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=min(10, timeout))
                if sr.ok:
                    js = sr.json()
                    st = js.get("status")
                    if st == "COMPLETED":
                        completed = True
                        break
                    if st == "ERROR":
                        msg = js.get("error") or "Unknown FAL error"
                        payload = js.get("payload")
                        if payload:
                            raise RuntimeError(f"FAL status ERROR: {msg} | details: {payload}")
                        raise RuntimeError(f"FAL status ERROR: {msg}")
            except Exception as e:
                if debug:
                    print(f"[FAL POLL] Status check error: {e}")
            time.sleep(0.6)

        if not completed:
            raise RuntimeError(f"FAL request {req_id[:16]} timed out after {timeout}s")

        rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout))
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

        data = rr.json()
        if debug:
            print(f"[FAL RESULT] request_id={req_id[:16]}... status=COMPLETED")

        resp = data.get("response") if isinstance(data, dict) else None
        if resp is None and isinstance(data, dict):
            resp = data

        images_out: T.List[torch.Tensor] = []
        buckets: T.List[T.Union[str, dict]] = []

        if isinstance(resp, dict):
            for key in ("images", "outputs", "artifacts"):
                val = resp.get(key)
                if isinstance(val, list):
                    buckets.extend(val)
            for key in ("image", "output", "result"):
                val = resp.get(key)
                if isinstance(val, (str, dict)):
                    buckets.append(val)

        for item in buckets:
            try:
                url_or_data = item if isinstance(item, str) else (item.get("url") or item.get("data") or item.get("image"))
                if not isinstance(url_or_data, str):
                    continue
                if url_or_data.startswith("data:image/"):
                    b64 = url_or_data.split(",", 1)[1]
                    blob = base64.b64decode(b64)
                else:
                    ir = requests.get(url_or_data, timeout=min(15, timeout))
                    if not ir.ok:
                        continue
                    blob = ir.content
                pil = Image.open(io.BytesIO(blob)).convert("RGB")
                images_out.append(pil_to_tensor(pil))
            except Exception as ex:
                if debug:
                    print("[FAL] image decode failed:", ex)

        description = resp.get("description", "") if isinstance(resp, dict) else ""

        if not images_out:
            raise RuntimeError(f"FAL returned no images for request_id={req_id}")

        return images_out[0], description

    # -------- parallel helpers with classification --------

    def _parallel_google_batch(self, indices: T.List[int], prompts: T.List[str],
                               client, model, cfg, request_id: str,
                               images, input_mime: str, timeout_sec: int, debug: bool):
        """
        Returns: (success_map, text_map, error_info)
            success_map: idx -> image_tensor
            text_map:    idx -> joined_text
            error_info:  idx -> {"msg": str, "retryable": bool, "safety": bool}
        """
        success_map: dict[int, torch.Tensor] = {}
        text_map: dict[int, str] = {}
        error_info: dict[int, dict] = {}

        def worker(idx: int, ptxt: str):
            try:
                parts = self._build_parts(ptxt, images, input_mime)
                g_imgs, g_texts, resp = self._single_google_call(client, model, parts, cfg, request_id, debug)
                if g_imgs:
                    success_map[idx] = g_imgs[0]
                    if g_texts:
                        text_map[idx] = "\n".join(g_texts)
                else:
                    raise RuntimeError("Gemini returned no images")
            except Exception as e:
                msg = str(e)
                retryable, safety = _classify_google_error(msg)
                error_info[idx] = {"msg": msg, "retryable": retryable, "safety": safety}
                print(f"[GOOGLE ERROR] (item {idx+1}) {msg} | retryable={retryable} safety={safety}", flush=True)

        with ThreadPoolExecutor(max_workers=min(len(indices), 6)) as ex:
            futs = [ex.submit(worker, i, prompts[i]) for i in indices]
            for _ in as_completed(futs):
                pass

        return success_map, text_map, error_info

    def _parallel_fal_batch(self, indices: T.List[int], prompts: T.List[str],
                            fal_key: str, route: str, images, input_mime: str,
                            timeout_sec: int, output_format: str,
                            aspect_ratio: str, sync_mode: bool, debug: bool):
        """
        Returns (success_map, text_map, error_info) analogous to Google batch.
        """
        success_map: dict[int, torch.Tensor] = {}
        text_map: dict[int, str] = {}
        error_info: dict[int, dict] = {}

        # Phase 1: submit
        submit_map: dict[int, dict] = {}

        def submit_worker(idx: int, ptxt: str):
            try:
                req = self._fal_submit_only(route, ptxt, images, input_mime, fal_key,
                                            timeout_sec, debug, output_format, aspect_ratio, sync_mode)
                submit_map[idx] = req
            except Exception as e:
                msg = f"FAL submit failed: {e}"
                retryable, safety = _classify_fal_error(msg)
                error_info[idx] = {"msg": msg, "retryable": retryable, "safety": safety}
                print(f"[FAL ERROR] (item {idx+1}) {msg} | retryable={retryable} safety={safety}", flush=True)

        with ThreadPoolExecutor(max_workers=min(len(indices), 6)) as ex:
            futs = [ex.submit(submit_worker, i, prompts[i]) for i in indices]
            for _ in as_completed(futs):
                pass

        # Phase 2: poll & fetch
        def poll_worker(idx: int, req_info: dict):
            if idx in error_info:
                return
            try:
                img, txt = self._fal_poll_and_fetch(req_info, fal_key, timeout_sec, debug)
                success_map[idx] = img
                if txt:
                    text_map[idx] = txt
            except Exception as e:
                msg = f"FAL poll failed: {e}"
                retryable, safety = _classify_fal_error(msg)
                error_info[idx] = {"msg": msg, "retryable": retryable, "safety": safety}
                print(f"[FAL ERROR] (item {idx+1}) {msg} | retryable={retryable} safety={safety}", flush=True)

        with ThreadPoolExecutor(max_workers=min(len(submit_map), 6)) as ex:
            futs = [ex.submit(poll_worker, idx, req) for idx, req in submit_map.items()]
            for _ in as_completed(futs):
                pass

        return success_map, text_map, error_info

    # -------- main --------

    def run(
        self,
        prompt: str,
        delimiter: str = "[*]",
        images: T.Optional[torch.Tensor] = None,
        aspect_ratio: str = "1:1",
        model: str = DEFAULT_MODEL,
        endpoint_override: str = "",
        api_key: str = "",
        seed: int = 0,  # accepted but not used; ComfyUI may pass it
        output_format: str = "png",
        capture_text_output: bool = False,
        num_images: int = 1,
        timeout_sec: int = 120,
        request_id: str = "",
        debug_log: bool = False,
        use_fal_fallback: bool = True,
        force_fal: bool = False,
        sync_mode: bool = False,
        fal_api_key: str = "",
        fal_route_img2img: str = "fal-ai/nano-banana/edit",
        fal_route_txt2img: str = "fal-ai/nano-banana",
    ):
        t0 = time.time()

        # Validate aspect_ratio
        if aspect_ratio.strip() not in _VALID_ASPECTS:
            print(f"[PVL NANO WARNING] Invalid or missing aspect_ratio '{aspect_ratio}', defaulting to 1:1.")
            aspect_ratio = "1:1"

        # Split prompts using regex delimiter
        try:
            base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
        except re.error:
            print(f"[PVL NANO WARNING] Invalid regex pattern '{delimiter}', using literal split.")
            base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]
        if not base_prompts:
            raise RuntimeError("No valid prompts provided.")

        # Map prompts to num_images calls
        call_prompts = self._build_call_prompts(base_prompts, num_images, debug_log)
        N = len(call_prompts)
        all_indices = list(range(N))

        input_mime = "image/png" if str(output_format).lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)

        # Decide FAL route based on presence of input images
        route_to_use = fal_route_img2img if (images is not None and torch.is_tensor(images) and images.numel() > 0) else fal_route_txt2img

        # FAL-only fast path
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but FAL_KEY missing.")
            fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
                all_indices, call_prompts, fal_key, route_to_use, images, input_mime,
                timeout_sec, output_format, aspect_ratio, sync_mode, debug_log
            )
            # FAL safety → stop immediately
            fal_safety = [i for i, info in fal_errs.items() if info.get("safety")]
            if fal_safety:
                i = fal_safety[0]
                raise RuntimeError(fal_errs[i]["msg"])
            if not fal_succ:
                any_err = next((v["msg"] for v in fal_errs.values()), "FAL failed with unknown error")
                raise RuntimeError(any_err)
            imgs_out = [fal_succ[i] for i in sorted(fal_succ.keys())]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                text_out = "\n".join(fal_texts[i] for i in sorted(fal_texts.keys()) if fal_texts.get(i))
            print(f"[PVL NANO INFO] FAL force mode: {len(fal_succ)}/{N} succeeded; returning successes only.")
            print(f"[PVL INFO] Completed in {time.time()-t0:.2f}s")
            return text_out, images_tensor

        # Google path
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("GEMINI_API_KEY missing and FAL_KEY missing.")
                # No Google key: go to FAL for ALL
                fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
                    all_indices, call_prompts, fal_key, route_to_use, images, input_mime,
                    timeout_sec, output_format, aspect_ratio, sync_mode, debug_log
                )
                fal_safety = [i for i, info in fal_errs.items() if info.get("safety")]
                if fal_safety:
                    i = fal_safety[0]
                    raise RuntimeError(fal_errs[i]["msg"])
                if not fal_succ:
                    any_err = next((v["msg"] for v in fal_errs.values()), "FAL failed with unknown error")
                    raise RuntimeError(any_err)
                imgs_out = [fal_succ[i] for i in sorted(fal_succ.keys())]
                images_tensor = stack_images_same_size(imgs_out, debug_log)
                text_out = ""
                if want_text:
                    text_out = "\n".join(fal_texts[i] for i in sorted(fal_texts.keys()) if fal_texts.get(i))
                print(f"[PVL NANO INFO] Google unavailable. FAL successes: {len(fal_succ)}/{N}. Completed in {time.time()-t0:.2f}s")
                return text_out, images_tensor
            else:
                raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")

        # Prepare Google client/config
        client = self._make_client(key, endpoint_override)
        cfg = self._build_config(want_text, aspect_ratio)
        if debug_log:
            print(f"[PVL NANO Debug] Google pass — {N} calls in parallel")

        # ROUND 1: Google for ALL
        g_succ1, g_texts1, g_errs1 = self._parallel_google_batch(
            all_indices, call_prompts, client, model, cfg, request_id, images, input_mime, timeout_sec, debug_log
        )
        succeeded = set(g_succ1.keys())
        failed_all = [i for i in all_indices if i not in succeeded]

        # Partition failures: retryable vs non-retryable vs safety
        retryable_idxs = [i for i in failed_all if g_errs1.get(i, {}).get("retryable", False) and not g_errs1.get(i, {}).get("safety", False)]
        safety_idxs    = [i for i in failed_all if g_errs1.get(i, {}).get("safety", False)]
        nonretry_idxs  = [i for i in failed_all if not g_errs1.get(i, {}).get("retryable", False) and not g_errs1.get(i, {}).get("safety", False)]

        # If ALL failed on Google
        if len(succeeded) == 0:
            if not use_fal_fallback:
                any_err = next((v["msg"] for v in g_errs1.values()), "Gemini failed with unknown error")
                raise RuntimeError(any_err)
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                any_err = next((v["msg"] for v in g_errs1.values()), "Gemini failed and FAL_KEY missing")
                raise RuntimeError(any_err)
            if debug_log:
                print("[PVL NANO Debug] All Google calls failed. Switching to FAL for all items.")
            fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
                all_indices, call_prompts, fal_key, route_to_use, images, input_mime,
                timeout_sec, output_format, aspect_ratio, sync_mode, debug_log
            )
            # Safety on FAL → immediate stop
            fal_safety = [i for i, info in fal_errs.items() if info.get("safety")]
            if fal_safety:
                i = fal_safety[0]
                raise RuntimeError(fal_errs[i]["msg"])
            if not fal_succ:
                any_err = next((v["msg"] for v in fal_errs.values()), next((v["msg"] for v in g_errs1.values()), "All providers failed"))
                raise RuntimeError(any_err)
            imgs_out = [fal_succ[i] for i in sorted(fal_succ.keys())]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                text_out = "\n".join(fal_texts[i] for i in sorted(fal_texts.keys()) if fal_texts.get(i))
            if len(fal_succ) < N:
                failures = sorted(set(all_indices) - set(fal_succ.keys()))
                for i in failures:
                    msg = fal_errs.get(i, {}).get("msg", "Unknown FAL error")
                    print(f"[PVL NANO ERROR] Item {i+1} failed after FAL: {msg}")
                print(f"[PVL NANO WARNING] Returning only {len(fal_succ)}/{N} successful results (FAL).")
            print(f"[PVL NANO INFO] Completed in {time.time()-t0:.2f}s")
            return text_out, images_tensor

        # PARTIAL success on Google: retry Google ONCE for retryable (non-safety)
        g_succ2, g_texts2, g_errs2 = ({}, {}, {})
        if retryable_idxs:
            if debug_log:
                print(f"[PVL NANO Debug] Google retry for retryable items: {[i+1 for i in retryable_idxs]}")
            g_succ2, g_texts2, g_errs2 = self._parallel_google_batch(
                retryable_idxs, call_prompts, client, model, cfg, request_id, images, input_mime, timeout_sec, debug_log
            )

        # Merge Google results
        g_succ_all = dict(g_succ1); g_succ_all.update(g_succ2)
        g_texts_all = dict(g_texts1); g_texts_all.update(g_texts2)
        still_failed = [i for i in all_indices if i not in g_succ_all]

        # Build FAL list: all safety + nonretry + retryable that still failed
        send_to_fal = set(safety_idxs + nonretry_idxs + [i for i in retryable_idxs if i in still_failed])
        if not send_to_fal:
            # All done on Google
            imgs_out = [g_succ_all[i] for i in sorted(g_succ_all.keys())]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text and g_texts_all:
                text_out = "\n".join(g_texts_all[i] for i in sorted(g_texts_all.keys()) if g_texts_all.get(i))
            print(f"[PVL NANO INFO] Google succeeded for all ({len(g_succ_all)}/{N}). Completed in {time.time()-t0:.2f}s")
            return text_out, images_tensor

        if not use_fal_fallback:
            if g_succ_all:
                imgs_out = [g_succ_all[i] for i in sorted(g_succ_all.keys())]
                images_tensor = stack_images_same_size(imgs_out, debug_log)
                text_out = ""
                if want_text and g_texts_all:
                    text_out = "\n".join(g_texts_all[i] for i in sorted(g_texts_all.keys()) if g_texts_all.get(i))
                for i in sorted(send_to_fal):
                    msg = (g_errs2.get(i, {}) or g_errs1.get(i, {})).get("msg", "Unknown Google error")
                    print(f"[PVL NANO ERROR] Item {i+1} failed after Google retry: {msg}")
                print(f"[PVL NANO WARNING] Returning only {len(g_succ_all)}/{N} successful results (Google only).")
                print(f"[PVL NANO INFO] Completed in {time.time()-t0:.2f}s")
                return text_out, images_tensor
            else:
                any_err = (next((v["msg"] for v in g_errs2.values()), None) or next((v["msg"] for v in g_errs1.values()), "Google failed"))
                raise RuntimeError(any_err)

        fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
        if not fal_key:
            if g_succ_all:
                imgs_out = [g_succ_all[i] for i in sorted(g_succ_all.keys())]
                images_tensor = stack_images_same_size(imgs_out, debug_log)
                text_out = ""
                if want_text and g_texts_all:
                    text_out = "\n".join(g_texts_all[i] for i in sorted(g_texts_all.keys()) if g_texts_all.get(i))
                for i in sorted(send_to_fal):
                    msg = (g_errs2.get(i, {}) or g_errs1.get(i, {})).get("msg", "Unknown Google error")
                    print(f"[PVL NANO ERROR] Item {i+1} failed after Google retry (no FAL key): {msg}")
                print(f"[PVL NANO WARNING] Returning only {len(g_succ_all)}/{N} successful results (partial).")
                print(f"[PVL NANO INFO] Completed in {time.time()-t0:.2f}s")
                return text_out, images_tensor
            else:
                any_err = (next((v["msg"] for v in g_errs2.values()), None) or next((v["msg"] for v in g_errs1.values()), "Google failed"))
                raise RuntimeError(any_err)

        # FAL for remaining
        send_list = sorted(send_to_fal)
        if debug_log:
            print(f"[PVL NANO Debug] Switching remaining failures to FAL: {[i+1 for i in send_list]}")

        fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
            send_list, call_prompts, fal_key, route_to_use, images, input_mime,
            timeout_sec, output_format, aspect_ratio, sync_mode, debug_log
        )

        # FAL safety → immediate stop
        fal_safety = [i for i, info in fal_errs.items() if info.get("safety")]
        if fal_safety:
            i = fal_safety[0]
            raise RuntimeError(fal_errs[i]["msg"])

        # Merge successes
        final_success_imgs: dict[int, torch.Tensor] = dict(g_succ_all)
        final_success_imgs.update(fal_succ)

        final_texts: dict[int, str] = dict(g_texts_all)
        for i, t in fal_texts.items():
            if t:
                prev = final_texts.get(i, "")
                final_texts[i] = (prev + ("\n" if prev else "") + t) if prev else t

        if not final_success_imgs:
            any_err = (next((v["msg"] for v in fal_errs.values()), None)
                       or next((v["msg"] for v in g_errs2.values()), None)
                       or next((v["msg"] for v in g_errs1.values()), "All providers failed"))
            raise RuntimeError(any_err)

        # Report residual failures
        ultimately_failed = sorted(set(all_indices) - set(final_success_imgs.keys()))
        for i in ultimately_failed:
            msg = (fal_errs.get(i, {}) or g_errs2.get(i, {}) or g_errs1.get(i, {})).get("msg", "Unknown error")
            print(f"[PVL NANO ERROR] Item {i+1} failed after Google + FAL: {msg}")

        # Return only successful outputs
        imgs_out = [final_success_imgs[i] for i in sorted(final_success_imgs.keys())]
        images_tensor = stack_images_same_size(imgs_out, debug_log)
        text_out = ""
        if want_text and final_texts:
            text_out = "\n".join(final_texts[i] for i in sorted(final_texts.keys()) if final_texts.get(i))

        if ultimately_failed:
            print(f"[PVL NANO WARNING] Returning only {len(final_success_imgs)}/{N} successful results (mixed Google/FAL).")
        print(f"[PVL NANO INFO] Completed in {time.time()-t0:.2f}s")

        return text_out, images_tensor


NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_API": PVL_Google_NanoBanana_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_API": NODE_NAME}
