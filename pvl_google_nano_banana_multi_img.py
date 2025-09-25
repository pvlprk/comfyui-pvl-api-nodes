# pvl_google_nano_banana_multi_img.py
# Node: PVL Google Nano-Banana Multi API (multi-image inputs, Gemini + FAL fallback)
# Author: PVL
# License: MIT
#
# Requires:
#   pip install google-genai
#
# Features:
# - 8 optional image inputs (image_1 ... image_8).
# - num_images supported for Google via parallel calls.
# - force_fal toggle to always use FAL API.
# - FAL Queue API integration for robustness.
# - If images differ in resolution -> return list of tensors.
# - Ensures IMAGE output is always 4D tensor or list of 4D tensors.
# - Debug mode prints payloads sent to API.

import os, io, json, base64, typing as T, time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            return genai.Client(api_key=api_key, http_options=http_options)
        return genai.Client(api_key=api_key)

    def _build_parts(self, prompt: str, image_tensors: T.List[torch.Tensor], mime: str):
        parts: T.List[dict] = []
        if prompt and prompt.strip():
            parts.append({"text": prompt})
        for img in image_tensors:
            pil = tensor_to_pil(img)
            parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        return parts

    # --- FAL Queue API ---
    def _fal_queue_call(self, route: str, prompt: str, image_tensors: T.List[torch.Tensor],
                        mime: str, fal_key: str, timeout: int, debug: bool,
                        num_images: int, output_format: str):

        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")

        # Build data URLs
        data_urls = []
        for t in image_tensors:
            pil = tensor_to_pil(t)
            raw = encode_pil_bytes(pil, mime)
            data_urls.append(_data_url(mime, raw))

        base = "https://queue.fal.run"
        submit_url = f"{base}/{route.strip()}"
        headers = {"Authorization": f"Key {fal_key}"}
        payload = {
            "prompt": prompt or "",
            "image_urls": data_urls,
            "num_images": int(max(1,num_images)),
            "output_format": ("png" if str(output_format).lower()=="png" else "jpeg"),
            "sync_mode": True,
        }
        if debug:
            print("[FAL SUBMIT]", json.dumps(payload)[:500])

        r = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        sub = r.json()
        req_id = sub.get("request_id")
        status_url = sub.get("status_url") or (f"{base}/{route.strip()}/requests/{req_id}/status" if req_id else None)
        resp_url = sub.get("response_url") or (f"{base}/{route.strip()}/requests/{req_id}" if req_id else None)
        if not req_id or not status_url or not resp_url:
            raise RuntimeError("FAL queue missing request_id/status/response")

        # Poll
        deadline = time.time() + timeout
        while time.time() < deadline:
            sr = requests.get(status_url, headers=headers, timeout=10)
            if sr.ok and sr.json().get("status") == "COMPLETED":
                break
            time.sleep(0.6)

        rr = requests.get(resp_url, headers=headers, timeout=15)
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
        rdata = rr.json()
        if debug:
            print("[FAL RAW]", json.dumps(rdata)[:800])

        # Collect outputs
        buckets = []
        resp = rdata.get("response") if isinstance(rdata, dict) else rdata
        if isinstance(resp, dict):
            for key in ("images","outputs","artifacts"):
                val = resp.get(key)
                if isinstance(val, list): buckets.extend(val)
            for key in ("image","output","result"):
                val = resp.get(key)
                if isinstance(val,(str,dict)): buckets.append(val)

        out=[]
        for item in buckets:
            try:
                url = item if isinstance(item,str) else (item.get("url") or item.get("data") or item.get("image"))
                if not url: continue
                if url.startswith("data:image/"):
                    blob=base64.b64decode(url.split(",",1)[1])
                else:
                    ir=requests.get(url,timeout=timeout)
                    if not ir.ok: continue
                    blob=ir.content
                pil=Image.open(io.BytesIO(blob)).convert("RGB")
                out.append(pil_to_tensor(pil))
            except Exception as ex:
                if debug: print("[FAL decode fail]",ex)

        if not out:
            raise RuntimeError("FAL returned no images")

        # return (text, images)
        return "", (out if len(out)>1 else out[0].unsqueeze(0))

    # --- main run ---
    def run(self, prompt: str,
            image_1=None,image_2=None,image_3=None,image_4=None,
            image_5=None,image_6=None,image_7=None,image_8=None,
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "", temperature: float = 0.6, output_format: str = "png",
            capture_text_output: bool = False, num_images: int = 1,
            timeout_sec: int = 120, request_id: str = "", debug_log: bool = False,
            use_fal_fallback: bool = False, force_fal: bool = False,
            fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit"):

        key = (api_key or os.getenv("GEMINI_API_KEY","")).strip()
        input_mime = "image/png" if output_format.lower()=="png" else "image/jpeg"
        want_text = bool(capture_text_output)

        # Gather images
        image_tensors=[]
        for img in [image_1,image_2,image_3,image_4,image_5,image_6,image_7,image_8]:
            if img is not None and torch.is_tensor(img):
                batch = img if img.ndim==4 else img.unsqueeze(0)
                for j in range(batch.shape[0]):
                    image_tensors.append(batch[j:j+1])

        # ---- CASE: FAL only ----
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
            if not fal_key: raise RuntimeError("force_fal=True but no FAL_KEY")
            return self._fal_queue_call(fal_route,prompt,image_tensors,input_mime,fal_key,timeout_sec,debug_log,num_images,output_format)

        # ---- CASE: Google ----
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

        if debug_log:
            print("[GOOGLE SUBMIT] Prompt:", (prompt or "")[:200])
            print("[GOOGLE SUBMIT] Num images:", num_images)
            print("[GOOGLE SUBMIT] Input images:", len(image_tensors))

        def google_call(i:int):
            resp = client.models.generate_content(
                model=model,
                contents=[{"role":"user","parts":parts}],
                config=cfg,
            )
            imgs,texts=[],[]
            for cand in getattr(resp,"candidates",[]) or []:
                for p in getattr(cand,"content",None).parts or []:
                    blob=_extract_image_bytes_from_part(p)
                    if blob:
                        pil=Image.open(io.BytesIO(blob)).convert("RGB")
                        imgs.append(pil_to_tensor(pil))
                    else:
                        t=_extract_text_from_part(p)
                        if t: texts.append(t)
            return imgs,texts

        N=max(1,int(num_images))
        out_imgs,out_texts=[],[]
        if N==1:
            imgs,texts=google_call(0)
            out_imgs.extend(imgs); out_texts.extend(texts)
        else:
            with ThreadPoolExecutor(max_workers=min(N,6)) as ex:
                futs={ex.submit(google_call,i):i for i in range(N)}
                for fut in as_completed(futs):
                    imgs,texts=fut.result()
                    out_imgs.extend(imgs); out_texts.extend(texts)

        if not out_imgs:
            if use_fal_fallback:
                fal_key=(fal_api_key or os.getenv("FAL_KEY","")).strip()
                return self._fal_queue_call(fal_route,prompt,image_tensors,input_mime,fal_key,timeout_sec,debug_log,num_images,output_format)
            raise RuntimeError("Gemini returned no images")

        # Handle size mismatch: return list if not stackable
        if len(out_imgs) > 1:
            try:
                images_tensor = torch.cat(out_imgs,dim=0)
            except RuntimeError:
                if debug_log: print("[GOOGLE] Mismatched sizes, returning list of tensors")
                images_tensor = out_imgs
        else:
            images_tensor = out_imgs[0]
            if images_tensor.ndim == 3:
                images_tensor = images_tensor.unsqueeze(0)

        text_out="\n".join(out_texts) if (want_text and out_texts) else ""
        return text_out, images_tensor

NODE_CLASS_MAPPINGS={"PVL_Google_NanoBanana_Multi_API":PVL_Google_NanoBanana_Multi_API}
NODE_DISPLAY_NAME_MAPPINGS={"PVL_Google_NanoBanana_Multi_API":NODE_NAME}
