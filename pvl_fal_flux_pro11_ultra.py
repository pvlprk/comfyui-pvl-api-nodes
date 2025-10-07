import os
import re
import torch
import numpy as np
import json
import time
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxPro_v1_1_Ultra_API:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "aspect_ratio": ("STRING", {"default": "16:9", "defaultInput": True}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[*]", "multiline": False, "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "raw": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools"
    
    _ALLOWED_AR = {"21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"}
    
    def _raise(self, msg):
        raise RuntimeError(msg)
    
    def _normalize_aspect_ratio(self, ar_value: str) -> str:
        """
        Accepts common synonyms and separators and returns a canonical w:h string.
        Examples accepted: '16:9', '16x9', '16-9', '16by9', 'landscape', 'portrait', 'square'
        """
        if not isinstance(ar_value, str) or not ar_value.strip():
            self._raise("Aspect ratio must be a non-empty string.")
        
        s = ar_value.strip().lower()
        
        # Simple synonyms
        if s in ("landscape",):
            s = "16:9"
        elif s in ("portrait",):
            s = "9:16"
        elif s in ("square",):
            s = "1:1"
        else:
            # Normalize separators: x, -, by -> :
            s = s.replace("by", ":").replace(" ", "")
            s = s.replace("x", ":").replace("-", ":")
            # collapse accidental doubles like '16::9'
            parts = [p for p in s.split(":") if p]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                s = f"{int(parts[0])}:{int(parts[1])}"
        
        if s not in self._ALLOWED_AR:
            self._raise(
                f"Invalid aspect_ratio '{ar_value}'. "
                f"Allowed: {', '.join(sorted(self._ALLOWED_AR))}."
            )
        
        return s
    
    def _build_call_prompts(self, base_prompts, num_images):
        """
        Maps prompts to calls according to the rule:
        - If len(prompts) >= num_images → take first num_images
        - If len(prompts) < num_images → Use each prompt in order. For remaining calls, reuse the last prompt.
        """
        N = max(1, int(num_images))
        if not base_prompts:
            return []
        
        if len(base_prompts) >= N:
            call_prompts = base_prompts[:N]
        else:
            print(f"[PVL WARNING] Provided {len(base_prompts)} prompts but num_images={N}. "
                  f"Reusing the last prompt for remaining calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        
        return call_prompts
    
    # -------- FAL Queue API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, prompt_text, seed, output_format, sync_mode,
                         safety_tolerance, aspect_ratio, enable_safety_checker, raw):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        arguments = {
            "prompt": prompt_text,
            "num_images": 1,  # Each call generates 1 image
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance,
            "aspect_ratio": aspect_ratio,
            "enable_safety_checker": enable_safety_checker,
            "raw": raw
        }
        
        if seed != -1:
            arguments["seed"] = seed
        
        # Check if ApiHandler supports async submission
        if hasattr(ApiHandler, 'submit_only'):
            return ApiHandler.submit_only("fal-ai/flux-pro/v1.1-ultra", arguments)
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/flux-pro/v1.1-ultra", arguments)
    
    def _direct_fal_submit(self, endpoint, arguments):
        """Direct FAL queue API submission when ApiHandler doesn't support async."""
        fal_key = os.getenv("FAL_KEY", "")
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")
        
        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}
        
        r = requests.post(submit_url, headers=headers, json=arguments, timeout=120)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        
        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")
        
        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"
        
        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }
    
    def _fal_poll_and_fetch(self, request_info, timeout=120):
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns image tensor.
        """
        # Check if ApiHandler supports async polling
        if hasattr(ApiHandler, 'poll_and_get_result'):
            result = ApiHandler.poll_and_get_result(request_info, timeout)
        else:
            # Fallback to direct polling
            fal_key = os.getenv("FAL_KEY", "")
            headers = {"Authorization": f"Key {fal_key}"}
            
            status_url = request_info["status_url"]
            resp_url = request_info["response_url"]
            
            # Poll for completion
            deadline = time.time() + timeout
            completed = False
            while time.time() < deadline:
                try:
                    sr = requests.get(status_url, headers=headers, timeout=10)
                    if sr.ok and sr.json().get("status") == "COMPLETED":
                        completed = True
                        break
                except Exception:
                    pass
                time.sleep(0.6)
            
            if not completed:
                raise RuntimeError(f"FAL request timed out after {timeout}s")
            
            # Fetch result
            rr = requests.get(resp_url, headers=headers, timeout=15)
            if not rr.ok:
                raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
            
            result = rr.json().get("response", rr.json())
        
        # Validate result
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        
        if "images" not in result or not result["images"]:
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")
        
        # NSFW detection via official field
        has_nsfw = result.get("has_nsfw_concepts")
        if isinstance(has_nsfw, list) and any(bool(x) for x in has_nsfw):
            self._raise("FAL: NSFW content detected by safety system (has_nsfw_concepts).")
        
        # Process images using ResultProcessor
        processed_result = ResultProcessor.process_image_result(result)
        
        # Check for black/empty image(s)
        if processed_result and len(processed_result) > 0:
            img_tensor = processed_result[0]
            if not isinstance(img_tensor, torch.Tensor):
                self._raise("FAL: internal error — processed image is not a tensor.")
            if torch.all(img_tensor == 0) or (img_tensor.mean() < 1e-6):
                self._raise("FAL: received an all-black image (likely filtered/failed).")
        
        return processed_result[0] if processed_result else None
    
    def generate_image(self, prompt, seed, num_images, output_format,
                      sync_mode, safety_tolerance, aspect_ratio,
                      delimiter="[*]",
                      enable_safety_checker=True, raw=False):
        
        _t0 = time.time()
        
        try:
            # Validate/normalize aspect ratio string
            aspect_ratio = self._normalize_aspect_ratio(aspect_ratio)
            
            # Split prompts using delimiter with regex support
            try:
                base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
            except re.error:
                print(f"[PVL WARNING] Invalid regex pattern '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]
            
            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")
            
            # Map prompts to num_images calls
            call_prompts = self._build_call_prompts(base_prompts, num_images)
            print(f"[PVL INFO] Processing {len(call_prompts)} prompts")
            
            # Single call: process directly (less overhead)
            if len(call_prompts) == 1:
                req_info = self._fal_submit_only(
                    call_prompts[0], seed, output_format, sync_mode,
                    safety_tolerance, aspect_ratio, enable_safety_checker, raw
                )
                result = self._fal_poll_and_fetch(req_info)
                
                if result.ndim == 3:
                    result = result.unsqueeze(0)
                
                _t1 = time.time()
                print(f"[PVL INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                return (result,)
            
            # Multiple calls: TRUE PARALLEL execution with seed increment
            print(f"[PVL INFO] Submitting {len(call_prompts)} requests in parallel...")
            
            # PHASE 1: Submit all requests in parallel
            submit_results = []
            max_workers = min(len(call_prompts), 6)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                submit_futs = {
                    executor.submit(
                        self._fal_submit_only,
                        call_prompts[i],
                        seed if seed == -1 else (seed + i) % 4294967296,  # Increment seed with overflow protection
                        output_format, sync_mode,
                        safety_tolerance, aspect_ratio, enable_safety_checker, raw
                    ): i
                    for i in range(len(call_prompts))
                }
                
                for fut in as_completed(submit_futs):
                    idx = submit_futs[fut]
                    try:
                        req_info = fut.result()
                        submit_results.append((idx, req_info))
                    except Exception as e:
                        print(f"[PVL ERROR] Submit failed for prompt {idx}: {e}")
            
            if not submit_results:
                raise RuntimeError("All FAL submission requests failed")
            
            print(f"[PVL INFO] {len(submit_results)} requests submitted. Polling for results...")
            
            # PHASE 2: Poll all requests in parallel
            results = {}
            failed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                poll_futs = {
                    executor.submit(self._fal_poll_and_fetch, req_info): idx
                    for idx, req_info in submit_results
                }
                
                for fut in as_completed(poll_futs):
                    idx = poll_futs[fut]
                    try:
                        result = fut.result()
                        if result is not None:
                            results[idx] = result
                    except Exception as e:
                        failed_count += 1
                        print(f"[PVL ERROR] Poll failed for prompt {idx}: {e}")
            
            if not results:
                raise RuntimeError(f"All FAL requests failed during polling ({failed_count} failures)")
            
            if failed_count > 0:
                print(f"[PVL WARNING] {failed_count}/{len(call_prompts)} requests failed, continuing with {len(results)} successful results")
            
            # Combine all image tensors in order
            all_images = []
            for i in range(len(call_prompts)):
                if i in results:
                    img_tensor = results[i]
                    
                    if torch.is_tensor(img_tensor):
                        # Handle both 3D (H,W,C) and 4D (B,H,W,C) tensors
                        if img_tensor.ndim == 3:
                            img_tensor = img_tensor.unsqueeze(0)
                        all_images.append(img_tensor)
            
            if not all_images:
                raise RuntimeError("No images were generated from API calls")
            
            # Stack all images into single batch
            final_tensor = torch.cat(all_images, dim=0)
            
            _t1 = time.time()
            print(f"[PVL INFO] Successfully generated {final_tensor.shape[0]} images in {(_t1 - _t0):.2f}s")
            
            # Print seed info if seed was manually set
            if seed != -1:
                seed_list = [seed + i for i in range(len(all_images))]
                print(f"[PVL INFO] Seeds used: {seed_list}")
            
            return (final_tensor,)
            
        except Exception as e:
            print(f"Error generating image with FLUX Pro 1.1 Ultra: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {"PVL_fal_FluxPro_v1_1_Ultra_API": PVL_fal_FluxPro_v1_1_Ultra_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_FluxPro_v1_1_Ultra_API": "PVL Flux Pro 1.1 Ultra (fal.ai)"}
