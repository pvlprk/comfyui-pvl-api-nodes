# pvl_google_nano_banana_multi_img.py
# PVL Google Nano-Banana Multi API — with regex delimiter support
# Author: PVL
# License: MIT
#
# Features:
# - Text box for prompt input ("STRING").
# - Regex-based delimiter input (e.g., \n|\| or ;+).
# - num_images controls number of parallel API calls.
# - If prompts < num_images, reuses the *last* prompt to fill the rest, with a warning.
# - Parallel calls for Google; optional FAL-only mode and Google→FAL fallback.
# - Optionally capture text output and print to console.
# - Ensures IMAGE output is a 4D tensor (B, H, W, C).

import os, io, json, base64, typing as T, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types
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


# --------------------------- Image helpers ---------------------------

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Shape (H,W,C) -> (1,H,W,C) as ComfyUI image tensor
    return torch.from_numpy(arr)[None, ...]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # Accept (B,H,W,C) or (H,W,C)
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


def _extract_image_bytes_from_part(part) -> T.Optional[bytes]:
    """
    Enhanced to handle more image data formats from Gemini API response
    """
    # Debug: Print the part structure
    print(f"[EXTRACT DEBUG] Part type: {type(part)}")
    
    try:
        # Try to get inline_data in various formats
        inline = None
        if hasattr(part, 'inline_data'):
            inline = getattr(part, 'inline_data')
            print(f"[EXTRACT DEBUG] Found inline_data attribute")
        elif hasattr(part, 'inlineData'):
            inline = getattr(part, 'inlineData')
            print(f"[EXTRACT DEBUG] Found inlineData attribute")
        elif isinstance(part, dict) and 'inline_data' in part:
            inline = part['inline_data']
            print(f"[EXTRACT DEBUG] Found inline_data dict key")
        elif isinstance(part, dict) and 'inlineData' in part:
            inline = part['inlineData']
            print(f"[EXTRACT DEBUG] Found inlineData dict key")
        
        if inline is not None:
            # Try to get data from inline
            data = None
            if hasattr(inline, 'data'):
                data = getattr(inline, 'data')
                print(f"[EXTRACT DEBUG] Found data attribute, type: {type(data)}")
            elif isinstance(inline, dict) and 'data' in inline:
                data = inline['data']
                print(f"[EXTRACT DEBUG] Found data dict key, type: {type(data)}")
            
            if data is not None:
                if isinstance(data, str):
                    try:
                        result = base64.b64decode(data, validate=False)
                        print(f"[EXTRACT DEBUG] Successfully decoded base64 string of length {len(data)}")
                        return result
                    except Exception as e:
                        print(f"[EXTRACT DEBUG] Failed to decode base64: {e}")
                elif isinstance(data, bytes):
                    print(f"[EXTRACT DEBUG] Found raw bytes of length {len(data)}")
                    return data
    except Exception as e:
        print(f"[EXTRACT DEBUG] Exception during extraction: {e}")

    # Try alternative approaches
    try:
        # Check if the part itself has image data
        if hasattr(part, 'image_bytes'):
            image_bytes = getattr(part, 'image_bytes')
            print(f"[EXTRACT DEBUG] Found image_bytes attribute")
            if isinstance(image_bytes, bytes):
                return image_bytes
        
        # Check if the part is a dict with image data
        if isinstance(part, dict):
            for key in ['image_bytes', 'image_data', 'binary_data']:
                if key in part:
                    data = part[key]
                    print(f"[EXTRACT DEBUG] Found {key} dict key")
                    if isinstance(data, str):
                        try:
                            return base64.b64decode(data, validate=False)
                        except Exception:
                            pass
                    elif isinstance(data, bytes):
                        return data
    except Exception as e:
        print(f"[EXTRACT DEBUG] Exception during alternative extraction: {e}")
    
    # Last resort: try to find any base64 string in the part
    try:
        if isinstance(part, dict):
            for key, value in part.items():
                if isinstance(value, str) and len(value) > 100:  # Likely a base64 encoded image
                    try:
                        # Check if it looks like base64
                        if value.startswith('data:image/') or re.match(r'^[A-Za-z0-9+/]+={0,2}$', value):
                            # If it's a data URL, extract the base64 part
                            if value.startswith('data:image/'):
                                base64_data = value.split(',', 1)[1]
                            else:
                                base64_data = value
                            
                            result = base64.b64decode(base64_data, validate=False)
                            print(f"[EXTRACT DEBUG] Found and decoded base64 in dict key '{key}'")
                            return result
                    except Exception:
                        pass
    except Exception as e:
        print(f"[EXTRACT DEBUG] Exception during base64 search: {e}")
    
    print("[EXTRACT DEBUG] No image data found")
    return None


def _extract_text_from_part(part) -> T.Optional[str]:
    # Prefer dict, fallback to attribute getattr
    if isinstance(part, dict):
        if "text" in part and part["text"] is not None:
            return str(part["text"])
    txt = getattr(part, "text", None)
    return str(txt) if txt is not None else None


def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")


def _stack_images_same_size(tensors: T.List[torch.Tensor], debug: bool = False) -> torch.Tensor:
    """
    Concatenate (B,H,W,C) batches along B. If shapes mismatch, resize to the first image size.
    """
    if not tensors:
        raise RuntimeError("No images to stack.")
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError:
        if debug:
            print("[PVL NODE] Mismatched sizes, resizing to match first image.")
        target_h, target_w = tensors[0].shape[1], tensors[0].shape[2]
        fixed = []
        for t in tensors:
            pil = tensor_to_pil(t)
            rp = pil.resize((target_w, target_h), Image.LANCZOS)
            fixed.append(pil_to_tensor(rp))
        return torch.cat(fixed, dim=0)


# --------------------------- Main Node ---------------------------

class PVL_Google_NanoBanana_Multi_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter prompts separated by regex delimiter"}),
                "delimiter": ("STRING", {"default": "\\n-----\\n", "multiline": False, "placeholder": "Regex (e.g. \\n|\\| or ;+)"}),
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
                "aspect_ratio": ("STRING", {"default": "1:1", "placeholder": "e.g. 16:9, 9:16, 3:2 — works for both Google & FAL"}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
                "use_fal_fallback": ("BOOLEAN", {"default": False}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False,
                                           "placeholder": "Leave empty to use FAL_KEY"}),
                "fal_route": ("STRING", {"default": "fal-ai/nano-banana/edit"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("text", "images")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    # ------------------------- INTERNAL HELPERS -------------------------

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
        for img_tensor in image_tensors:
            if img_tensor.ndim == 4:
                for i in range(img_tensor.shape[0]):
                    single_img = img_tensor[i:i+1]
                    pil = tensor_to_pil(single_img)
                    parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
            else:
                pil = tensor_to_pil(img_tensor)
                parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        return parts

    def _build_call_prompts(self, base_prompts: T.List[str], num_images: int, debug: bool) -> T.List[str]:
        """
        Maps prompts to calls according to the agreed rule:
          - If len(prompts) >= num_images → take first num_images
          - If len(prompts) <  num_images → repeat the *last* prompt to fill
        """
        N = max(1, int(num_images))
        if not base_prompts:
            return []
        if len(base_prompts) >= N:
            call_prompts = base_prompts[:N]
        else:
            if debug:
                print(f"[PVL NODE] Provided {len(base_prompts)} prompts but num_images={N}. "
                      f"Reusing the last prompt for remaining calls.")
            print(f"[PVL WARNING] prompt list shorter than num_images: {len(base_prompts)} < {N}. "
                  f"Last entry will be reused for the remaining {N - len(base_prompts)} calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        if debug:
            for i, cp in enumerate(call_prompts, 1):
                show = cp if len(cp) <= 160 else (cp[:157] + "...")
                print(f"[PVL NODE] Call #{i} prompt: {show}")
        return call_prompts

    def _fal_queue_call(self, route: str, prompt: str, image_tensors: T.List[torch.Tensor],
                        mime: str, fal_key: str, timeout: int, debug: bool,
                        output_format: str, aspect_ratio: str = "1:1"):
        """
        Calls FAL queue endpoint in sync_mode, returns (image_tensor_batched1, description_text).
        """
        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")

        # Convert any input images to data URLs
        data_urls = []
        for t in image_tensors:
            if t.ndim == 4:
                for i in range(t.shape[0]):
                    single_img = t[i:i+1]
                    pil = tensor_to_pil(single_img)
                    raw = encode_pil_bytes(pil, mime)
                    data_urls.append(_data_url(mime, raw))
            else:
                pil = tensor_to_pil(t)
                raw = encode_pil_bytes(pil, mime)
                data_urls.append(_data_url(mime, raw))

        base = "https://queue.fal.run"
        submit_url = f"{base}/{route.strip()}"
        headers = {"Authorization": f"Key {fal_key}"}
        payload = {
            "prompt": prompt or "",
            "image_urls": data_urls,
            "num_images": 1,
            "output_format": ("png" if str(output_format).lower() == "png" else "jpeg"),
            "aspect_ratio": aspect_ratio,
            "sync_mode": True,
        }
        if debug:
            print("[FAL SUBMIT]", json.dumps(payload)[:1000])

        r = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        sub = r.json()
        req_id = sub.get("request_id")
        status_url = sub.get("status_url") or (f"{base}/{route.strip()}/requests/{req_id}/status" if req_id else None)
        resp_url = sub.get("response_url") or (f"{base}/{route.strip()}/requests/{req_id}" if req_id else None)
        if not req_id or not status_url or not resp_url:
            raise RuntimeError("FAL queue missing request_id/status/response")

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
            print("[FAL RAW]", json.dumps(rdata)[:2000])

        buckets = []
        resp = rdata.get("response") if isinstance(rdata, dict) else None
        if resp is None and isinstance(rdata, dict):
            resp = rdata
        if isinstance(resp, dict):
            for key in ("images", "outputs", "artifacts"):
                val = resp.get(key)
                if isinstance(val, list):
                    buckets.extend(val)
            for key in ("image", "output", "result"):
                val = resp.get(key)
                if isinstance(val, (str, dict)):
                    buckets.append(val)

        out = []
        for item in buckets:
            try:
                url = item if isinstance(item, str) \
                    else (item.get("url") or item.get("data") or item.get("image"))
                if not url:
                    continue
                if url.startswith("data:image/"):
                    blob = base64.b64decode(url.split(",", 1)[1])
                else:
                    ir = requests.get(url, timeout=timeout)
                    if not ir.ok:
                        continue
                    blob = ir.content
                pil = Image.open(io.BytesIO(blob)).convert("RGB")
                out.append(pil_to_tensor(pil))
            except Exception as ex:
                if debug:
                    print("[FAL decode fail]", ex)

        if not out:
            raise RuntimeError("FAL returned no images")

        return out[0], (resp.get("description", "") if isinstance(resp, dict) else "")

    # --------------------------- RUN MAIN -----------------------------

    def run(self, prompt: str, delimiter: str,
            image_1=None, image_2=None, image_3=None, image_4=None,
            image_5=None, image_6=None, image_7=None, image_8=None,
            aspect_ratio: str = "1:1",
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "", temperature: float = 0.6, output_format: str = "png",
            capture_text_output: bool = False, num_images: int = 1,
            timeout_sec: int = 120, debug_log: bool = False,
            use_fal_fallback: bool = False, force_fal: bool = False,
            fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit"):
        # --- Validate aspect ratio (for Google & FAL) ---
        ar_original = aspect_ratio.strip()
        valid_ratios = {"21:9","1:1","4:3","3:2","2:3","5:4","4:5","3:4","16:9","9:16"}
        if not ar_original or ar_original not in valid_ratios:
            print(f"[PVL WARNING] Invalid or missing aspect_ratio '{ar_original}', using 1:1.")
            aspect_ratio = "1:1"
        else:
            aspect_ratio = ar_original

        # --- Regex-based prompt splitting (with safe fallback to literal split) ---
        try:
            base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
        except re.error:
            print(f"[PVL WARNING] Invalid regex pattern '{delimiter}', using literal split.")
            base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]

        if not base_prompts:
            raise RuntimeError("No valid prompts provided.")

        # Collect any provided images (tensors) in order
        image_tensors: T.List[torch.Tensor] = []
        for img in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]:
            if img is not None and torch.is_tensor(img):
                image_tensors.append(img)

        # Map provided prompts to num_images calls
        call_prompts = self._build_call_prompts(base_prompts, num_images, debug_log)

        input_mime = "image/png" if output_format.lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)

        # ---------------- FAL-only path ----------------
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but no FAL_KEY provided.")
            results, texts = [], []
            with ThreadPoolExecutor(max_workers=min(len(call_prompts), 6)) as ex:
                futs = {
                    ex.submit(self._fal_queue_call, fal_route, p, image_tensors,
                              input_mime, fal_key, timeout_sec, debug_log, output_format, aspect_ratio): p
                    for p in call_prompts
                }
                for fut in as_completed(futs):
                    img, t = fut.result()
                    results.append(img)  # Fixed: removed .unsqueeze(0)
                    if t:
                        texts.append(t)
            images_tensor = _stack_images_same_size(results, debug_log)
            text_out = "\n".join(texts) if want_text else ""
            return text_out, images_tensor

        # ---------------- Google path (with optional FAL fallback) ----------------
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")
        client = self._make_client(key, endpoint_override)

        def google_call(p: str, debug: bool):
            # Prepend explicit image generation instruction
            image_prompt = f"{p}"
            if debug:
                print(f"[GEMINI PROMPT] {image_prompt}")
            
            parts = self._build_parts(image_prompt, image_tensors, input_mime)

            # Correct configuration for image generation
            cfg = types.GenerateContentConfig(
                temperature=float(temperature),
                top_p=_TOP_P,
                top_k=_TOP_K,
                max_output_tokens=_MAX_TOKENS,
                response_modalities=["Image"],  # Request image generation
                # No response_mime_type - model returns PNG by default
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )

            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": parts}],
                config=cfg,
            )
            
            if debug:
                print("[GEMINI RESPONSE STRUCTURE]")
                print(f"  Candidates: {len(getattr(resp, 'candidates', []))}")
                for i, cand in enumerate(getattr(resp, 'candidates', []) or []):
                    content = getattr(cand, 'content', None)
                    parts_out = getattr(content, 'parts', []) if content else []
                    print(f"  Candidate {i}: {len(parts_out)} parts")
                    for j, prt in enumerate(parts_out):
                        # Enhanced debugging
                        print(f"    Part {j}: Type = {type(prt)}")
                        if hasattr(prt, '__dict__'):
                            print(f"      Attributes: {list(prt.__dict__.keys())}")
                        elif isinstance(prt, dict):
                            print(f"      Keys: {list(prt.keys())}")
            
            imgs, texts = [], []
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts_out = getattr(content, "parts", []) if content else []
                for prt in parts_out:
                    blob = _extract_image_bytes_from_part(prt)
                    if blob:
                        if debug:
                            print(f"    Extracted image blob of size {len(blob)} bytes")
                        try:
                            pil = Image.open(io.BytesIO(blob)).convert("RGB")
                            imgs.append(pil_to_tensor(pil))
                        except Exception as e:
                            print(f"    Failed to convert blob to image: {e}")
                    else:
                        t = _extract_text_from_part(prt)
                        if t:
                            if debug:
                                print(f"    Extracted text: {t[:100]}...")
                            texts.append(t)
            
            if debug:
                print(f"[GEMINI RESULT] Found {len(imgs)} images and {len(texts)} text parts")
            
            return imgs, texts

        out_imgs: T.List[torch.Tensor] = []
        out_texts: T.List[str] = []
        with ThreadPoolExecutor(max_workers=min(len(call_prompts), 6)) as ex:
            futs = {ex.submit(google_call, p, debug_log): p for p in call_prompts}
            for fut in as_completed(futs):
                imgs, texts = fut.result()
                if imgs:
                    # Fixed: removed .unsqueeze(0) since pil_to_tensor already adds batch dimension
                    out_imgs.append(imgs[0])
                if texts:
                    out_texts.extend(texts)

        if not out_imgs:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("Google returned no images and FAL fallback has no FAL_KEY.")
                results, texts = [], []
                with ThreadPoolExecutor(max_workers=min(len(call_prompts), 6)) as ex:
                    futs = {
                        ex.submit(self._fal_queue_call, fal_route, p, image_tensors,
                              input_mime, fal_key, timeout_sec, debug_log, output_format, aspect_ratio): p
                        for p in call_prompts
                    }
                    for fut in as_completed(futs):
                        img, t = fut.result()
                        results.append(img)  # Fixed: removed .unsqueeze(0)
                        if t:
                            texts.append(t)
                images_tensor = _stack_images_same_size(results, debug_log)
                combo_texts = out_texts + texts
                text_out = "\n".join(combo_texts) if want_text else ""
                return text_out, images_tensor
            raise RuntimeError("Gemini returned no images")

        images_tensor = _stack_images_same_size(out_imgs, debug_log)
        text_out = "\n".join(out_texts) if (want_text and out_texts) else ""
        if text_out:
            print(f"[PVL Google NanoBanana Output]:\n{text_out}\n")
        return text_out, images_tensor


NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": PVL_Google_NanoBanana_Multi_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": NODE_NAME}