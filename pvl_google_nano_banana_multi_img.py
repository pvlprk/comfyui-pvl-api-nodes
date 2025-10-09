# pvl_google_nano_banana_multi_img.py

# PVL Google Nano-Banana Multi API — with regex delimiter support

# Author: PVL
# License: MIT

# Features:
# - Text box for prompt input ("STRING").
# - Regex-based delimiter input (e.g., \n|\| or ;+).
# - num_images controls number of parallel API calls.
# - If prompts < num_images, reuses the *last* prompt to fill the rest, with a warning.
# - Parallel calls for Google; optional FAL-only mode and Google→FAL fallback.
# - TRUE PARALLEL FAL execution: submit all requests first, then poll for results.
# - Optionally capture text output and print to console.
# - Ensures IMAGE output is a 4D tensor (B, H, W, C).
# - sync_mode toggle for FAL API (default: False)

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
    return torch.from_numpy(arr)[None, ...]

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

def _extract_image_bytes_from_part(part) -> T.Optional[bytes]:
    """Extract image bytes from Gemini API response part."""
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
                "delimiter": ("STRING", {"default": "[*]", "multiline": False, "placeholder": "Regex (e.g. \\n|\\| or ;+)"}),
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["fixed", "randomize"], {"default": "randomize"}),
                                      "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
                "use_fal_fallback": ("BOOLEAN", {"default": True}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False,
                                          "placeholder": "Leave empty to use FAL_KEY"}),
                "fal_route": ("STRING", {"default": "fal-ai/nano-banana/edit"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        - If len(prompts) < num_images → repeat the *last* prompt to fill
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
    
    # -------- FAL API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, route: str, prompt: str, image_tensors: T.List[torch.Tensor],
                         mime: str, fal_key: str, timeout: int, debug: bool,
                         output_format: str, aspect_ratio: str = "1:1", sync_mode: bool = False):
        """
        Phase 1: Submit request to FAL queue and return request info immediately.
        Does NOT poll for completion.
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
            "sync_mode": sync_mode,
        }
        
        if debug:
            print(f"[FAL SUBMIT] prompt: {prompt[:60]}... sync_mode={sync_mode}")
        
        r = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        
        sub = r.json()
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
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns (image_tensor, description_text).
        """
        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]
        req_id = request_info["request_id"]
        
        if debug:
            print(f"[FAL POLL] request_id={req_id[:16]}...")
        
        # Poll for completion
        deadline = time.time() + timeout
        while time.time() < deadline:
            sr = requests.get(status_url, headers=headers, timeout=10)
            if sr.ok and sr.json().get("status") == "COMPLETED":
                break
            time.sleep(0.6)
        
        # Fetch result
        rr = requests.get(resp_url, headers=headers, timeout=15)
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
        
        rdata = rr.json()
        if debug:
            print(f"[FAL RESULT] request_id={req_id[:16]}... status=COMPLETED")
        
        # Extract response data
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
                    print(f"[FAL decode fail] {ex}")
        
        if not out:
            raise RuntimeError(f"FAL returned no images for request_id={req_id}")
        
        description = resp.get("description", "") if isinstance(resp, dict) else ""
        return out[0], description
    
    # --------------------------- RUN MAIN -----------------------------
    
    def run(self, prompt: str, delimiter: str,
            image_1=None, image_2=None, image_3=None, image_4=None,
            image_5=None, image_6=None, image_7=None, image_8=None,
            aspect_ratio: str = "1:1",
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            capture_text_output: bool = False, num_images: int = 1,
            timeout_sec: int = 120, debug_log: bool = False,
            use_fal_fallback: bool = False, force_fal: bool = False,
            sync_mode: bool = False,
            fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit",
                seed: int = 0,
                control_after_generate: str = "randomize"):
        
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
        
        # ---------------- FAL-only path (TRUE PARALLEL) ----------------
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but no FAL_KEY provided.")
            
            if debug_log:
                print(f"[FAL] Submitting {len(call_prompts)} requests...")
            
            # PHASE 1: Submit all requests (fast, non-blocking)
            request_infos = []
            for p in call_prompts:
                req_info = self._fal_submit_only(fal_route, p, image_tensors,
                                                 input_mime, fal_key, timeout_sec, 
                                                 debug_log, output_format, 
                                                 aspect_ratio, sync_mode)
                request_infos.append(req_info)
            
            if debug_log:
                print(f"[FAL] All {len(request_infos)} requests submitted. Polling for results...")
            
            # PHASE 2: Poll all requests in parallel
            results, texts = [], []
            with ThreadPoolExecutor(max_workers=min(len(request_infos), 6)) as ex:
                futs = {
                    ex.submit(self._fal_poll_and_fetch, req_info, fal_key, 
                             timeout_sec, debug_log): req_info
                    for req_info in request_infos
                }
                for fut in as_completed(futs):
                    img, t = fut.result()
                    results.append(img)
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
            parts = self._build_parts(p, image_tensors, input_mime)
            
            cfg = types.GenerateContentConfig(
                top_p=_TOP_P,
                top_k=_TOP_K,
                max_output_tokens=_MAX_TOKENS,
                response_modalities=["Image"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )
            
            if debug:
                print(f"[GOOGLE SUBMIT] prompt: {p[:100]}...")
            
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": parts}],
                config=cfg,
            )
            
            imgs, texts = [], []
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts_out = getattr(content, "parts", []) if content else []
                
                for prt in parts_out:
                    blob = _extract_image_bytes_from_part(prt)
                    if blob:
                        try:
                            pil = Image.open(io.BytesIO(blob)).convert("RGB")
                            imgs.append(pil_to_tensor(pil))
                        except Exception as e:
                            if debug:
                                print(f"[Decode fail] {e}")
                    else:
                        t = _extract_text_from_part(prt)
                        if t:
                            texts.append(t)
            
            if debug:
                print(f"[GOOGLE RESULT] Found {len(imgs)} images")
            
            return imgs, texts
        
        out_imgs: T.List[torch.Tensor] = []
        out_texts: T.List[str] = []
        
        with ThreadPoolExecutor(max_workers=min(len(call_prompts), 6)) as ex:
            futs = {ex.submit(google_call, p, debug_log): p for p in call_prompts}
            for fut in as_completed(futs):
                imgs, texts = fut.result()
                if imgs:
                    out_imgs.append(imgs[0])
                if texts:
                    out_texts.extend(texts)
        
        if not out_imgs:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("Google returned no images and FAL fallback has no FAL_KEY.")
                
                if debug_log:
                    print(f"[FAL FALLBACK] Submitting {len(call_prompts)} requests...")
                
                # PHASE 1: Submit all requests
                request_infos = []
                for p in call_prompts:
                    req_info = self._fal_submit_only(fal_route, p, image_tensors,
                                                     input_mime, fal_key, timeout_sec,
                                                     debug_log, output_format,
                                                     aspect_ratio, sync_mode)
                    request_infos.append(req_info)
                
                if debug_log:
                    print(f"[FAL FALLBACK] All requests submitted. Polling for results...")
                
                # PHASE 2: Poll all requests in parallel
                results, texts = [], []
                with ThreadPoolExecutor(max_workers=min(len(request_infos), 6)) as ex:
                    futs = {
                        ex.submit(self._fal_poll_and_fetch, req_info, fal_key,
                                 timeout_sec, debug_log): req_info
                        for req_info in request_infos
                    }
                    for fut in as_completed(futs):
                        img, t = fut.result()
                        results.append(img)
                        if t:
                            texts.append(t)
                
                images_tensor = _stack_images_same_size(results, debug_log)
                combo_texts = out_texts + texts
                text_out = "\n".join(combo_texts) if want_text else ""
                return text_out, images_tensor
            
            raise RuntimeError("Gemini returned no images")
        
        images_tensor = _stack_images_same_size(out_imgs, debug_log)
        text_out = "\n".join(out_texts) if (want_text and out_texts) else ""
        
        if text_out and debug_log:
            print(f"[PVL Google NanoBanana Output]:\n{text_out}\n")
        
        return text_out, images_tensor

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": PVL_Google_NanoBanana_Multi_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_Multi_API": NODE_NAME}
