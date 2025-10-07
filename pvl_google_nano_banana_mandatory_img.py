# pvl_google_nano_banana_mandatory_img.py

# Node: PVL Google Nano-Banana API (Mandatory IMG + Delimiter + Aspect Ratio + Parallel + FAL)

# Author: PVL
# License: MIT

# Features in this build:
# - Mandatory image input.
# - Literal delimiter input (default "[*]") to split a long prompt into multiple prompts.
# - Prompt reuse rule to match num_images
# - Aspect ratio support (STRING) via new SDK
# - TRUE PARALLEL execution for both Google and FAL
# - FAL fallback and Force-FAL modes with sync_mode toggle
# - Error handling for individual requests with partial results support
# - Execution time is printed at the end

import os, io, json, base64, typing as T, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from PIL import Image
import torch
from google.genai import types

NODE_NAME = "PVL Google Nano-Banana API mandatory IMG"
NODE_CATEGORY = "PVL/Google"
DEFAULT_MODEL = "gemini-2.5-flash-image-preview"
_TOP_P = 0.95
_TOP_K = 64
_MAX_TOKENS = 4096

# --------------------------- Image/Tensor helpers ---------------------------

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # (1,H,W,3)

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
        inline = part.get("inline_data") or part.get("inlineData")
        if isinstance(inline, dict):
            data = inline.get("data")
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=False)
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

class PVL_Google_NanoBanana_API_mandatory_IMG:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter prompts separated by delimiter"}),
                "images": ("IMAGE",),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[*]", "placeholder": "Delimiter string"}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
                "use_fal_fallback": ("BOOLEAN", {"default": True}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use FAL_KEY"}),
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
        from google.genai import types as gtypes
        
        http_options = None
        if endpoint_override.strip():
            try:
                http_options = gtypes.HttpOptions(base_url=endpoint_override.strip())
            except Exception:
                http_options = None
        
        if http_options is not None:
            return genai.Client(api_key=api_key, http_options=http_options)
        return genai.Client(api_key=api_key)
    
    def _build_parts(self, prompt: str, images: torch.Tensor, mime: str):
        parts: T.List[dict] = []
        
        if prompt and prompt.strip():
            parts.append({"text": prompt})
        
        batch = images if images.ndim == 4 else images.unsqueeze(0)
        for i in range(batch.shape[0]):
            pil = tensor_to_pil(batch[i:i+1])
            parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        
        return parts
    
    # ---- FAL Queue API - TWO PHASE EXECUTION ----
    
    def _fal_submit_only(self, route: str, prompt: str, image_tensor: torch.Tensor,
                         mime: str, fal_key: str, timeout: int, debug: bool,
                         output_format: str, aspect_ratio: str = "1:1", sync_mode: bool = False):
        """
        Phase 1: Submit request to FAL queue and return request info immediately.
        Does NOT poll for completion.
        """
        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")
        
        # Build data URLs for all frames
        data_urls: T.List[str] = []
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
            "image_urls": data_urls,
            "num_images": 1,
            "output_format": ("png" if str(output_format).lower() == "png" else "jpeg"),
            "aspect_ratio": aspect_ratio,
            "sync_mode": sync_mode,
        }
        
        if debug:
            print(f"[FAL SUBMIT] prompt: {prompt[:60]}... sync_mode={sync_mode}")
        
        # Submit request
        r = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        
        sub = r.json()
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
                sr = requests.get(status_url, headers=headers, timeout=10)
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
        rr = requests.get(resp_url, headers=headers, timeout=15)
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
        
        rdata = rr.json()
        if debug:
            print(f"[FAL RESULT] request_id={req_id[:16]}... status=COMPLETED")
        
        # Extract response data
        resp = rdata.get("response", rdata) if isinstance(rdata, dict) else rdata
        
        # Parse images from various possible locations
        buckets = []
        if isinstance(resp, dict):
            for key in ("images", "outputs", "artifacts"):
                val = resp.get(key)
                if isinstance(val, list):
                    buckets.extend(val)
            
            for key in ("image", "output", "result"):
                val = resp.get(key)
                if isinstance(val, (str, dict)):
                    buckets.append(val)
        
        out: T.List[torch.Tensor] = []
        for item in buckets:
            try:
                url = item if isinstance(item, str) else (item.get("url") or item.get("data") or item.get("image"))
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
            raise RuntimeError(f"FAL returned no images for request_id={req_id}")
        
        description = ""
        if isinstance(resp, dict):
            description = resp.get("description") or resp.get("output_text") or ""
        
        images_tensor = _stack_images_same_size(out, debug)
        return images_tensor, description
    
    # --------------------------- RUN MAIN -----------------------------
    
    def run(self, prompt: str, images: torch.Tensor,
            delimiter: str = "[*]", aspect_ratio: str = "1:1",
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "", temperature: float = 0.6,
            output_format: str = "png", capture_text_output: bool = False,
            num_images: int = 1, timeout_sec: int = 120,
            debug_log: bool = False, use_fal_fallback: bool = True,
            force_fal: bool = False, sync_mode: bool = False,
            fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit"):
        
        _t0 = time.time()
        
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        input_mime = "image/png" if output_format.lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)
        N = max(1, int(num_images))
        
        # Split prompts by literal delimiter and expand to N using reuse rule
        raw_parts = [p.strip() for p in str(prompt).split(delimiter) if p.strip()]
        if not raw_parts:
            raise RuntimeError("Prompt is empty after splitting by delimiter.")
        
        if len(raw_parts) >= N:
            prompts = raw_parts[:N]
        else:
            prompts = raw_parts + [raw_parts[-1]] * (N - len(raw_parts))
        
        if debug_log:
            print(f"[PVL Debug] delimiter='{delimiter}' aspect_ratio='{aspect_ratio}' num_images={N} sync_mode={sync_mode}")
            for i, pr in enumerate(prompts, 1):
                preview = pr if len(pr) <= 160 else (pr[:157] + "...")
                print(f"[PVL Debug] Call #{i} prompt: {preview}")
        
        # Helper function for parallel FAL submission + polling
        def parallel_fal_execution(prompts_list, fal_key_str, debug):
            """Submit all FAL requests in parallel, then poll all in parallel"""
            if debug:
                print(f"[FAL] Submitting {len(prompts_list)} requests in parallel...")
            
            # PHASE 1: Submit all requests IN PARALLEL
            submit_results = []
            with ThreadPoolExecutor(max_workers=min(len(prompts_list), 6)) as ex:
                submit_futs = {
                    ex.submit(self._fal_submit_only, fal_route, p, images, input_mime,
                             fal_key_str, int(timeout_sec), debug, output_format,
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
            images_list, texts = [], []
            failed_count = 0
            with ThreadPoolExecutor(max_workers=min(len(submit_results), 6)) as ex:
                poll_futs = {
                    ex.submit(self._fal_poll_and_fetch, req_info, fal_key_str,
                             int(timeout_sec), debug): req_info
                    for req_info in submit_results
                }
                for fut in as_completed(poll_futs):
                    try:
                        it, txt = fut.result()
                        images_list.append(it)
                        if want_text and txt:
                            texts.append(txt)
                    except Exception as e:
                        failed_count += 1
                        if debug:
                            print(f"[FAL POLL ERROR] {e}")
            
            if not images_list:
                raise RuntimeError(f"All FAL requests failed during polling ({failed_count} failures)")
            
            if failed_count > 0:
                print(f"[PVL WARNING] {failed_count}/{len(submit_results)} FAL requests failed, continuing with {len(images_list)} successful results")
            
            return images_list, texts
        
        # Force-FAL path (TRUE PARALLEL)
        if force_fal:
            fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
            if not fal_key:
                raise RuntimeError("force_fal=True but FAL_KEY missing.")
            
            images_list, texts = parallel_fal_execution(prompts, fal_key, debug_log)
            
            images_tensor = _stack_images_same_size(images_list, debug_log)
            text_out = "\n".join(texts) if (want_text and texts) else ""
            
            _t1 = time.time()
            print(f"[PVL Google NanoBanana mandatory IMG] Execution time: {(_t1 - _t0):.2f}s")
            return text_out, images_tensor
        
        # No API key? Try fallback if enabled (TRUE PARALLEL)
        if not key:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("GEMINI_API_KEY missing and FAL_KEY missing.")
                
                images_list, texts = parallel_fal_execution(prompts, fal_key, debug_log)
                
                images_tensor = _stack_images_same_size(images_list, debug_log)
                text_out = "\n".join(texts) if (want_text and texts) else ""
                
                _t1 = time.time()
                print(f"[PVL Google NanoBanana mandatory IMG] Execution time: {(_t1 - _t0):.2f}s")
                return text_out, images_tensor
            
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")
        
        # Google SDK path
        from google import genai
        client = self._make_client(key, endpoint_override)
        
        def google_call(p: str, idx: int):
            parts = self._build_parts(p, images, input_mime)
            
            cfg = types.GenerateContentConfig(
                temperature=float(temperature),
                top_p=_TOP_P,
                top_k=_TOP_K,
                max_output_tokens=_MAX_TOKENS,
                response_modalities=["Image"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )
            
            if debug_log:
                print(f"[GOOGLE SUBMIT] Call #{idx+1} model={model}")
            
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
                            if debug_log:
                                print("[Decode fail]", e)
                    else:
                        t = _extract_text_from_part(prt)
                        if t:
                            texts.append(t)
            
            return imgs, texts
        
        out_imgs: T.List[torch.Tensor] = []
        out_texts: T.List[str] = []
        failed_google = 0
        
        with ThreadPoolExecutor(max_workers=min(N, 6)) as ex:
            futs = {ex.submit(google_call, p, i): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                try:
                    imgs, texts = fut.result()
                    if imgs:
                        out_imgs.extend(imgs)
                    if texts:
                        out_texts.extend(texts)
                except Exception as e:
                    failed_google += 1
                    if debug_log:
                        print(f"[GOOGLE ERROR] {e}")
        
        if failed_google > 0:
            print(f"[PVL WARNING] {failed_google}/{N} Google requests failed")
        
        if not out_imgs:
            if use_fal_fallback:
                # per-prompt FAL fallback (TRUE PARALLEL)
                fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()
                if not fal_key:
                    raise RuntimeError("Google returned no images and FAL_KEY missing for fallback.")
                
                images_list, texts = parallel_fal_execution(prompts, fal_key, debug_log)
                
                images_tensor = _stack_images_same_size(images_list, debug_log)
                text_out = "\n".join(out_texts + texts) if want_text else ""
                
                _t1 = time.time()
                print(f"[PVL Google NanoBanana mandatory IMG] Execution time: {(_t1 - _t0):.2f}s")
                return text_out, images_tensor
            
            raise RuntimeError("Gemini returned no images")
        
        images_tensor = _stack_images_same_size(out_imgs, debug_log)
        text_out = "\n".join(out_texts) if (want_text and out_texts) else ""
        
        if text_out:
            print(f"[PVL Google NanoBanana Output]:\n{text_out}\n")
        
        _t1 = time.time()
        print(f"[PVL Google NanoBanana mandatory IMG] Execution time: {(_t1 - _t0):.2f}s")
        return text_out, images_tensor

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_API_mandatory_IMG": PVL_Google_NanoBanana_API_mandatory_IMG}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_API_mandatory_IMG": NODE_NAME}
