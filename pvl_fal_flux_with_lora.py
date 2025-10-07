import os
import re
import torch
import time
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxWithLora_API:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1392, "min": 256, "max": 1440}),
                "height": ("INT", {"default": 752, "min": 256, "max": 1440}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[*]", "multiline": False, "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"}),
                "lora1_name": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_name": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_name": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"
    
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
    
    def _fal_submit_only(self, prompt_text, width, height, steps, CFG, seed,
                         enable_safety_checker, output_format, sync_mode,
                         lora1_name, lora1_scale, lora2_name, lora2_scale,
                         lora3_name, lora3_scale):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        arguments = {
            "prompt": prompt_text,
            "num_inference_steps": steps,
            "guidance_scale": CFG,
            "num_images": 1,  # Each call generates 1 image
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "image_size": {
                "width": width,
                "height": height
            }
        }
        
        if seed != -1:
            arguments["seed"] = seed
        
        # Handle LoRAs
        loras = []
        if lora1_name.strip():
            loras.append({"path": lora1_name.strip(), "scale": lora1_scale})
        if lora2_name.strip():
            loras.append({"path": lora2_name.strip(), "scale": lora2_scale})
        if lora3_name.strip():
            loras.append({"path": lora3_name.strip(), "scale": lora3_scale})
        if loras:
            arguments["loras"] = loras
        
        # Check if ApiHandler supports async submission
        if hasattr(ApiHandler, 'submit_only'):
            return ApiHandler.submit_only("fal-ai/flux-lora", arguments)
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/flux-lora", arguments)
    
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
        
        # Process result using ResultProcessor
        return ResultProcessor.process_image_result(result)
    
    def generate_image(self, prompt, width, height, steps, CFG, seed,
                      num_images, enable_safety_checker, output_format, sync_mode,
                      delimiter="[*]",
                      lora1_name="", lora1_scale=1.0,
                      lora2_name="", lora2_scale=1.0,
                      lora3_name="", lora3_scale=1.0):
        
        _t0 = time.time()
        
        try:
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
                    call_prompts[0], width, height, steps, CFG, seed,
                    enable_safety_checker, output_format, sync_mode,
                    lora1_name, lora1_scale, lora2_name, lora2_scale,
                    lora3_name, lora3_scale
                )
                result = self._fal_poll_and_fetch(req_info)
                
                img_tensor = result[0] if isinstance(result, tuple) else result
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                _t1 = time.time()
                print(f"[PVL INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                return (img_tensor,)
            
            # Multiple calls: TRUE PARALLEL execution with seed increment
            print(f"[PVL INFO] Submitting {len(call_prompts)} requests in parallel...")
            
            # PHASE 1: Submit all requests in parallel
            submit_results = []
            max_workers = min(len(call_prompts), 6)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                submit_futs = {
                    executor.submit(
                        self._fal_submit_only,
                        call_prompts[i], width, height, steps, CFG,
                        seed if seed == -1 else (seed + i) % 4294967296,  # FIXED: Increment seed with overflow protection
                        enable_safety_checker, output_format, sync_mode,
                        lora1_name, lora1_scale, lora2_name, lora2_scale,
                        lora3_name, lora3_scale
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
                    result = results[i]
                    img_tensor = result[0] if isinstance(result, tuple) else result
                    
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
                seed_list = [(seed + i) % 4294967296 for i in range(len(all_images))]
                print(f"[PVL INFO] Seeds used: {seed_list}")
            
            return (final_tensor,)
            
        except Exception as e:
            print(f"Error generating image with FLUX: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX", e)

NODE_CLASS_MAPPINGS = {"PVL_fal_FluxWithLora_API": PVL_fal_FluxWithLora_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_FluxWithLora_API": "PVL FAL Flux with LoRA"}
