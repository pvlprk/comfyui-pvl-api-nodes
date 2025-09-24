# pvl_google_nano_banana_multi_img.py
# Node: PVL Google Nano-Banana Multi API (multi-image inputs, Gemini + FAL fallback)
# Author: PVL
# License: MIT
#
# Requires:
#   pip install google-genai
#
# This version:
# - Adds 8 optional image inputs (image_1 ... image_8).
# - Skips missing inputs without error.
# - Preserves input order.
# - Supports mixed resolutions/aspect ratios.
# - Returns one image if num_images=1, or a stacked batch if num_images>1.
# - Adds force_fal toggle: if True, always use FAL API (ignore Google).
# - Fallback logic: only use FAL if Google fails AND use_fal_fallback is True.

import os, io, json, base64, typing as T, time
import requests
import numpy as np
from PIL import Image
import torch

NODE_NAME = "PVL Google Nano-Banana Multi API"
NODE_CATEGORY = "PVL/Google"
DEFAULT_MODEL = "gemini-2.5-flash-image-preview"

_TOP_P = 0.95
_TOP_K = 64
_MAX_TOKENS = 4096

# --- image helpers ---
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0,1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")

def encode_pil_bytes(img: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()

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
        if "inline_data" in part and isinstance(part["inline_data"], dict):
            blob = part["inline_data"].get("data")
        elif "inlineData" in part and isinstance(part["inlineData"], dict):
            blob = part["inlineData"].get("data")
        else:
            blob = None
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


class PVL_Google_NanoBanana_Multi_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A tiny banana spaceship over a neon city."}),
            },
            "optional": {
                # eight optional image inputs
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "output_format": (["png","jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "request_id": ("STRING", {"default": ""}),
                "debug_log": ("BOOLEAN", {"default": False}),
                # --- FAL controls ---
                "use_fal_fallback": ("BOOLEAN", {"default": False}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False,
                                           "placeholder": "Leave empty to use FAL_KEY"}),
                "fal_route": ("STRING", {"default": "fal-ai/nano-banana/edit"}),
            }
        }

    RETURN_TYPES = ("STRING","IMAGE",)
    RETURN_NAMES = ("text","images")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def _make_client(self, api_key: str, endpoint_override: str):
        from google import genai
        from google.genai import types

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

    def _build_parts(self, prompt: str, image_tensors: T.List[torch.Tensor], mime: str):
        parts: T.List[dict] = []
        if prompt and prompt.strip():
            parts.append({"text": prompt})
        for img in image_tensors:
            pil = tensor_to_pil(img)
            parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        return parts

    def _call_fal(self, prompt, image_tensors, input_mime, num_images, output_format, fal_route, fal_key, timeout_sec):
        # Prepare data URLs
        data_urls = []
        for t in image_tensors:
            pil = tensor_to_pil(t)
            raw = encode_pil_bytes(pil, input_mime)
            data_urls.append(_data_url(input_mime, raw))

        payload = {
            "prompt": prompt or "",
            "image_urls": data_urls,
            "num_images": int(max(1,num_images)),
            "output_format": ("png" if str(output_format).lower()=="png" else "jpeg"),
            "sync_mode": True,
        }

        r = requests.post(f"https://queue.fal.run/{fal_route.strip()}",
                          headers={"Authorization": f"Key {fal_key}"},
                          json=payload, timeout=timeout_sec)
        if r.status_code >= 400:
            raise RuntimeError(f"FAL error {r.status_code}: {r.text}")
        rdata = r.json()

        images_out = []
        urls = []
        if isinstance(rdata, dict):
            if "images" in rdata and isinstance(rdata["images"], list):
                urls = rdata["images"]
            elif "image" in rdata:
                urls = [rdata["image"]]

        for url in urls:
            try:
                ir = requests.get(url, timeout=timeout_sec)
                pil = Image.open(io.BytesIO(ir.content)).convert("RGB")
                images_out.append(pil_to_tensor(pil))
            except Exception:
                continue

        if not images_out:
            raise RuntimeError("FAL returned no images.")

        if len(images_out) == 1:
            return ("", images_out[0])
        else:
            return ("", torch.cat(images_out, dim=0))

    def run(self, prompt: str,
            image_1: T.Optional[torch.Tensor] = None,
            image_2: T.Optional[torch.Tensor] = None,
            image_3: T.Optional[torch.Tensor] = None,
            image_4: T.Optional[torch.Tensor] = None,
            image_5: T.Optional[torch.Tensor] = None,
            image_6: T.Optional[torch.Tensor] = None,
            image_7: T.Optional[torch.Tensor] = None,
            image_8: T.Optional[torch.Tensor] = None,
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "",
            temperature: float = 0.6, output_format: str = "png",
            capture_text_output: bool = False, num_images: int = 1,
            timeout_sec: int = 120, request_id: str = "",
            debug_log: bool = False,
            use_fal_fallback: bool = False, force_fal: bool = False,
            fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit"):

        key = (api_key or os.getenv("GEMINI_API_KEY","")).strip()
        input_mime = "image/png" if str(output_format).lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)

        # Collect all provided images into a list (skip None)
        image_tensors: T.List[torch.Tensor] = []
        for img in [image_1,image_2,image_3,image_4,image_5,image_6,image_7,image_8]:
            if img is not None and torch.is_tensor(img):
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                for j in range(img.shape[0]):
                    image_tensors.append(img[j:j+1])

        # ---- CASE 1: Force FAL ----
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but FAL_KEY missing.")
            return self._call_fal(prompt, image_tensors, input_mime, num_images, output_format, fal_route, fal_key, timeout_sec)

        # ---- CASE 2: Use Google ----
        if not key:
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")

        client = self._make_client(key, endpoint_override)
        parts = self._build_parts(prompt, image_tensors, input_mime)

        cfg = {
            "temperature": float(temperature),
            "top_p": _TOP_P,
            "top_k": _TOP_K,
            "max_output_tokens": _MAX_TOKENS,
            "response_modalities": ["IMAGE","TEXT"] if want_text else ["IMAGE"],
        }

        try:
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": parts}],
                config=cfg,
            )
        except Exception as e:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
                if not fal_key:
                    raise RuntimeError("Google failed and no FAL_KEY for fallback.")
                return self._call_fal(prompt, image_tensors, input_mime, num_images, output_format, fal_route, fal_key, timeout_sec)
            raise

        imgs, texts = [], []
        cands = getattr(resp, "candidates", None) or []
        for cand in cands:
            content = getattr(cand, "content", None)
            if not content: 
                continue
            for p in getattr(content, "parts", []) or []:
                blob = _extract_image_bytes_from_part(p)
                if blob:
                    pil = Image.open(io.BytesIO(blob)).convert("RGB")
                    imgs.append(pil_to_tensor(pil))
                else:
                    t = _extract_text_from_part(p)
                    if t: 
                        texts.append(t)

        if not imgs:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
                if not fal_key:
                    raise RuntimeError("Google returned no images and no FAL_KEY for fallback.")
                return self._call_fal(prompt, image_tensors, input_mime, num_images, output_format, fal_route, fal_key, timeout_sec)
            raise RuntimeError("Gemini returned no images.")

        if len(imgs) == 1:
            image_tensor = imgs[0]
        else:
            image_tensor = torch.cat(imgs, dim=0)

        text_out = ("\n".join(texts)) if (want_text and texts) else ""
        return (text_out, image_tensor)

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": PVL_Google_NanoBanana_Multi_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": "PVL Google NanoBanana API Multi"}
