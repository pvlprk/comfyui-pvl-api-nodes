# pvl_google_nano_banana.py
# Node: PVL Google Nano-Banana API (Gemini + FAL) — with delimiter support + parallel prompts
# Author: PVL
# License: MIT
#
# Features:
# - Regex-based delimiter input for splitting prompts (default: [*])
# - Parallel API calls for multiple prompts
# - TRUE PARALLEL FAL execution: submit all requests first, then poll for results
# - If prompts < num_images, reuses the last prompt to fill remaining calls
# - Single optional ComfyUI IMAGE input (can be batched).
# - aspect_ratio supported for both Google and FAL.
# - use_fal_fallback default True (fallback to FAL when Google returns no images).
# - force_fal toggle to always use FAL (bypass Google).
# - Dual FAL routes: img2img (edit) vs txt2img (generate) chosen automatically.
# - Individual request error handling with partial results support.
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
            print("[PVL NODE] Mismatched sizes, resizing to match first image.")
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

# ----------------- main node -----------------

class PVL_Google_NanoBanana_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A tiny banana spaceship over a neon city."}),
                "delimiter": ("STRING", {"default": "[*]", "multiline": False, "placeholder": "Regex delimiter e.g. [*], \\n, |"}),
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
        Maps prompts to calls according to the agreed rule:
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
                print(f"[PVL NODE] Provided {len(base_prompts)} prompts but num_images={N}. "
                      f"Reusing the last prompt for remaining calls.")
                print(f"[PVL WARNING] prompt list shorter than num_images ({len(base_prompts)} < {N}). "
                      f"Last entry will be reused for the remaining {N - len(base_prompts)} calls.")
            call_prompts = base_prompts + [base_prompts[-1] * 1] * (N - len(base_prompts))
        
        if debug:
            for i, cp in enumerate(call_prompts, 1):
                show = cp if len(cp) <= 160 else (cp[:157] + "...")
                print(f"[PVL NODE] Call {i} prompt: {show}")
        
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
        
        imgs, texts = [], []
        cands = getattr(resp, "candidates", None) or []
        
        for cand in cands:
            content = getattr(cand, "content", None)
            finish_reason = getattr(cand, "finish_reason", None)
            
            if debug:
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        print("[PVL Debug] usage_metadata:", getattr(um, "__dict__", str(um)))
                    print(f"[PVL Debug] finish_reason: {finish_reason}")
                except Exception:
                    pass
            
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
                            print("[PVL Debug] image decode error:", ex)
                else:
                    t = _extract_text_from_part(p)
                    if t:
                        texts.append(t)
        
        return imgs, texts, resp
    
    # -------- FAL Queue API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, route: str, prompt: str, image_tensors: T.Optional[torch.Tensor],
                         mime: str, fal_key: str, timeout: int, debug: bool,
                         output_format: str, aspect_ratio: str = "1:1", sync_mode: bool = False):
        """
        Phase 1: Submit request to FAL queue and return request info immediately.
        Does NOT poll for completion.
        """
        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")
        
        # Build data URLs only if image provided
        data_urls: T.List[str] = []
        if image_tensors is not None and torch.is_tensor(image_tensors):
            batch = image_tensors if image_tensors.ndim == 4 else image_tensors.unsqueeze(0)
            for i in range(batch.shape[0]):
                pil = tensor_to_pil(batch[i:i + 1])
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
        
        # Only add image_urls for img2img route
        if data_urls:
            payload["image_urls"] = data_urls
        
        if debug:
            print(f"[FAL SUBMIT] prompt: {prompt[:60]}... sync_mode={sync_mode}")
        
        # Submit request
        rr = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if rr.status_code != 200:
            raise RuntimeError(f"FAL submit error {rr.status_code}: {rr.text}")
        
        sub = rr.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")
        
        # Get status and result URLs
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
        
        # Poll for completion with timeout check
        deadline = time.time() + timeout
        completed = False
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=min(10, timeout))
                if sr.ok and sr.json().get("status") == "COMPLETED":
                    completed = True
                    break
            except Exception as e:
                if debug:
                    print(f"[FAL POLL] Status check error: {e}")
            time.sleep(0.6)
        
        # Check if we timed out
        if not completed:
            raise RuntimeError(f"FAL request {req_id[:16]} timed out after {timeout}s")
        
        # Fetch result
        rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout))
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
        
        data = rr.json()
        if debug:
            print(f"[FAL RESULT] request_id={req_id[:16]}... status=COMPLETED")
        
        # Extract response data
        resp = data.get("response") if isinstance(data, dict) else None
        if resp is None and isinstance(data, dict):
            resp = data
        
        # Parse images from various possible locations
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
        # Validate aspect_ratio
        if aspect_ratio.strip() not in _VALID_ASPECTS:
            print(f"[PVL WARNING] Invalid or missing aspect_ratio '{aspect_ratio}', defaulting to 1:1.")
            aspect_ratio = "1:1"
        
        # Split prompts using regex delimiter
        try:
            base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
        except re.error:
            print(f"[PVL WARNING] Invalid regex pattern '{delimiter}', using literal split.")
            base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]
        
        if not base_prompts:
            raise RuntimeError("No valid prompts provided.")
        
        # Map prompts to num_images calls
        call_prompts = self._build_call_prompts(base_prompts, num_images, debug_log)
        
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        input_mime = "image/png" if str(output_format).lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)
        
        # Decide FAL route based on presence of input images
        route_to_use = fal_route_img2img if (images is not None and torch.is_tensor(images) and images.numel() > 0) else fal_route_txt2img
        
        # Helper function for parallel FAL submission + polling
        def parallel_fal_execution(prompts_list, fal_key_str, route, debug):
            """Submit all FAL requests in parallel, then poll all in parallel"""
            if debug:
                print(f"[FAL] Submitting {len(prompts_list)} requests in parallel...")
            
            # PHASE 1: Submit all requests IN PARALLEL
            submit_results = []
            with ThreadPoolExecutor(max_workers=min(len(prompts_list), 6)) as ex:
                submit_futs = {
                    ex.submit(self._fal_submit_only, route, p, images, input_mime, 
                             fal_key_str, timeout_sec, debug, output_format, 
                             aspect_ratio, sync_mode): p
                    for p in prompts_list
                }
                for fut in as_completed(submit_futs):
                    try:
                        req_info = fut.result()
                        submit_results.append(req_info)
                    except Exception as e:
                        if debug:
                            print(f"[FAL SUBMIT ERROR] {e}")
            
            if not submit_results:
                raise RuntimeError("All FAL submission requests failed")
            
            if debug:
                print(f"[FAL] {len(submit_results)} requests submitted successfully. Polling for results...")
            
            # PHASE 2: Poll all requests IN PARALLEL
            results, texts = [], []
            failed_count = 0
            with ThreadPoolExecutor(max_workers=min(len(submit_results), 6)) as ex:
                poll_futs = {
                    ex.submit(self._fal_poll_and_fetch, req_info, fal_key_str, 
                             timeout_sec, debug): req_info
                    for req_info in submit_results
                }
                for fut in as_completed(poll_futs):
                    try:
                        img, t = fut.result()
                        results.append(img)
                        if t:
                            texts.append(t)
                    except Exception as e:
                        failed_count += 1
                        if debug:
                            print(f"[FAL POLL ERROR] {e}")
            
            if not results:
                raise RuntimeError(f"All FAL requests failed during polling ({failed_count} failures)")
            
            if failed_count > 0:
                print(f"[PVL WARNING] {failed_count}/{len(submit_results)} FAL requests failed, continuing with {len(results)} successful results")
            
            return results, texts
        
        # ---- CASE: FAL only (TRUE PARALLEL) ----
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but FAL_KEY missing.")
            
            results, texts = parallel_fal_execution(call_prompts, fal_key, route_to_use, debug_log)
            
            images_tensor = stack_images_same_size(results, debug_log)
            text_out = "\n".join(texts) if want_text else ""
            return text_out, images_tensor
        
        # ---- Google path ----
        if not key:
            if use_fal_fallback:
                # no Google key, try FAL directly (TRUE PARALLEL)
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("GEMINI_API_KEY missing and FAL_KEY missing.")
                
                results, texts = parallel_fal_execution(call_prompts, fal_key, route_to_use, debug_log)
                
                images_tensor = stack_images_same_size(results, debug_log)
                text_out = "\n".join(texts) if want_text else ""
                return text_out, images_tensor
            
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")
        
        client = self._make_client(key, endpoint_override)
        cfg = self._build_config(want_text, aspect_ratio)
        
        if debug_log:
            print(f"[PVL Debug] Processing {len(call_prompts)} prompts in parallel")
        
        def google_call(p: str, debug: bool):
            parts = self._build_parts(p, images, input_mime)
            g_imgs, g_texts, _resp = self._single_google_call(client, model, parts, cfg, request_id, debug)
            return g_imgs, g_texts
        
        out_imgs: T.List[torch.Tensor] = []
        out_texts: T.List[str] = []
        failed_google = 0
        
        # Google API calls with error handling
        with ThreadPoolExecutor(max_workers=min(len(call_prompts), 6)) as ex:
            futs = {ex.submit(google_call, p, debug_log): p for p in call_prompts}
            for fut in as_completed(futs):
                try:
                    imgs, texts = fut.result()
                    if imgs:
                        out_imgs.append(imgs[0])
                    out_texts.extend(texts)
                except Exception as e:
                    failed_google += 1
                    if debug_log:
                        print(f"[GOOGLE ERROR] {e}")
        
        if failed_google > 0:
            print(f"[PVL WARNING] {failed_google}/{len(call_prompts)} Google requests failed")
        
        # If Google failed to produce images, optionally fallback to FAL (TRUE PARALLEL)
        if not out_imgs:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("Google returned no images and FAL_KEY is missing for fallback.")
                
                results, texts = parallel_fal_execution(call_prompts, fal_key, route_to_use, debug_log)
                
                images_tensor = stack_images_same_size(results, debug_log)
                
                if want_text:
                    combined_text = ""
                    if out_texts:
                        combined_text += "\n\n--- Google ---\n\n" + "\n".join(out_texts)
                    if texts:
                        combined_text += "\n\n--- FAL ---\n\n" + "\n".join(texts)
                    text_out = combined_text
                else:
                    text_out = ""
                
                return text_out, images_tensor
            else:
                raise RuntimeError("Gemini returned no images")
        
        # Merge Google images to a 4D tensor
        images_tensor = stack_images_same_size(out_imgs, debug_log)
        
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        
        text_out = "\n".join(out_texts) if (want_text and out_texts) else ""
        
        if text_out and debug_log:
            print("[PVL Google NanoBanana Output]:\n" + text_out)
        
        return text_out, images_tensor

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_API": PVL_Google_NanoBanana_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_API": NODE_NAME}
