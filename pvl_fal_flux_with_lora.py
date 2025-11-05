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
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440}),
                "height": ("INT", {"default":1024, "min": 256, "max": 1440}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                # Retry + timeout + debug controls
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[++]", "multiline": False, "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"}),
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

    # -------- Local fallback error handler (replaces missing ApiHandler.handle_image_generation_error) --------
    def _handle_error(self, api_name: str, error: Exception):
        """
        Generic fallback error handler for image generation nodes that expect (IMAGE,) output.
        Returns a 1×256×256×3 black tensor to keep ComfyUI graphs running.
        """
        print(f"[{api_name} ERROR] {str(error)}")
        try:
            blank_image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        except Exception as e:
            # If torch allocation truly fails, re-raise original context
            raise RuntimeError(f"{api_name} fallback image creation failed: {e}") from error
        return (blank_image,)

    # -------- Detect FAL prohibited-content error so we don't retry it --------
    def _is_content_policy_violation(self, message_or_json) -> bool:
        """
        Detect FAL's prohibited-content error (non-retryable).
        We check for JSON {"type": "content_policy_violation"} or the phrase in a stringified message.
        """
        try:
            if isinstance(message_or_json, dict):
                et = str(message_or_json.get("type", "")).lower()
                if "content_policy_violation" in et:
                    return True
                # sometimes error may be nested
                err = message_or_json.get("error")
                if isinstance(err, dict):
                    et2 = str(err.get("type", "")).lower()
                    if "content_policy_violation" in et2:
                        return True
                if "content_policy_violation" in str(message_or_json).lower():
                    return True
            elif isinstance(message_or_json, str):
                s = message_or_json.lower()
                if "content_policy_violation" in s:
                    return True
        except Exception:
            pass
        return False

    def _build_call_prompts(self, base_prompts, num_images, debug=False):
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
            if debug:
                print(f"[PVL Flux Lora Fal] Provided {len(base_prompts)} prompts but num_images={N}. Reusing the last prompt for remaining calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        
        if debug:
            for i, p in enumerate(call_prompts):
                show = p if len(p) <= 160 else (p[:157] + "...")
                print(f"[PVL Flux Lora Fal] Call {i+1} prompt: {show}")
        return call_prompts
    
    # -------- FAL Queue API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, prompt_text, width, height, steps, CFG, seed,
                         enable_safety_checker, output_format, sync_mode,
                         lora1_name, lora1_scale, lora2_name, lora2_scale,
                         lora3_name, lora3_scale, timeout_sec=120, debug=False):
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

        if debug:
            print(f"[FAL SUBMIT] payload: {arguments}")
        
        # Check if ApiHandler supports async submission
        if hasattr(ApiHandler, 'submit_only'):
            try:
                return ApiHandler.submit_only("fal-ai/flux-lora", arguments, timeout=timeout_sec, debug=debug) \
                    if 'timeout' in ApiHandler.submit_only.__code__.co_varnames else ApiHandler.submit_only("fal-ai/flux-lora", arguments)
            except Exception as e:
                raise RuntimeError(f"FAL submit_only failed: {e}")
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/flux-lora", arguments, timeout_sec, debug)
    
    def _direct_fal_submit(self, endpoint, arguments, timeout_sec=120, debug=False):
        """Direct FAL queue API submission when ApiHandler doesn't support async."""
        fal_key = os.getenv("FAL_KEY", "")
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")
        
        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}
        
        r = requests.post(submit_url, headers=headers, json=arguments, timeout=timeout_sec)
        if not r.ok:
            # include server text for visibility & detect policy violation
            try:
                js = r.json()
                if self._is_content_policy_violation(js):
                    raise RuntimeError(f"FAL content_policy_violation: {js}")
            except Exception:
                pass
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        
        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")
        
        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"
        
        if debug:
            print(f"[FAL SUBMIT OK] request_id={req_id}")
        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }
    
    def _fal_poll_and_fetch(self, request_info, timeout_sec=120, debug=False, item_idx=None, attempt=None, started_at=None):
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns image tensor.
        """
        # Check if ApiHandler supports async polling
        if hasattr(ApiHandler, 'poll_and_get_result'):
            try:
                if 'timeout' in ApiHandler.poll_and_get_result.__code__.co_varnames:
                    result = ApiHandler.poll_and_get_result(request_info, timeout=timeout_sec, debug=debug)
                else:
                    result = ApiHandler.poll_and_get_result(request_info)
            except Exception as e:
                raise RuntimeError(f"FAL poll_and_get_result failed: {e}")
        else:
            # Fallback to direct polling
            fal_key = os.getenv("FAL_KEY", "")
            headers = {"Authorization": f"Key {fal_key}"}
            
            status_url = request_info["status_url"]
            resp_url = request_info["response_url"]
            req_id = request_info.get("request_id", "")[:16]

            # Poll for completion
            deadline = time.time() + timeout_sec
            completed = False
            while time.time() < deadline:
                try:
                    sr = requests.get(status_url, headers=headers, timeout=min(10, timeout_sec))
                    if sr.ok:
                        js = sr.json()
                        st = js.get("status")
                        if debug:
                            elapsed = (time.time() - (started_at or 0.0)) if started_at else 0.0
                            print(f"[FAL POLL] item={item_idx} attempt={attempt} req={req_id} status={st} elapsed={elapsed:.1f}s")
                        if st == "COMPLETED":
                            completed = True
                            break
                        if st == "ERROR":
                            # include payload when present
                            msg = js.get("error") or "Unknown FAL error"
                            payload = js.get("payload")
                            if payload:
                                raise RuntimeError(f"FAL status ERROR: {msg} | details: {payload}")
                            raise RuntimeError(f"FAL status ERROR: {msg}")
                except Exception as e:
                    if debug:
                        print(f"[FAL POLL] item={item_idx} attempt={attempt} req={req_id} status_check_error: {e}")
                time.sleep(0.6)
            
            if not completed:
                raise RuntimeError(f"FAL request {req_id} timed out after {timeout_sec}s")
            
            # Fetch result
            rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout_sec))
            if not rr.ok:
                # detect policy violation if server returns json body
                try:
                    js = rr.json()
                    if self._is_content_policy_violation(js):
                        raise RuntimeError(f"FAL content_policy_violation: {js}")
                except Exception:
                    pass
                raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
            
            result = rr.json().get("response", rr.json())
        
        # Process result using ResultProcessor
        return ResultProcessor.process_image_result(result)

    # Worker that performs submission + polling with retry policy for a single item
    def _run_one_with_retries(
        self,
        item_index: int,
        prompt_text: str,
        width: int, height: int, steps: int, CFG: float,
        seed_base: int,
        enable_safety_checker: bool, output_format: str, sync_mode: bool,
        lora1_name: str, lora1_scale: float,
        lora2_name: str, lora2_scale: float,
        lora3_name: str, lora3_scale: float,
        retries: int, timeout_sec: int, debug: bool
    ):
        """
        Returns tuple (success: bool, image_tensor or None, last_error_message or '')
        Retries only current item up to `retries` times. Each retry performs a fresh submit+poll.
        """
        # seed increment per item (keeps original behavior with wrap)
        seed_for_item = seed_base if seed_base == -1 else ((seed_base + item_index) % 4294967296)

        last_err = ""
        for attempt in range(1, int(retries) + 2):  # attempts = 1 + retries
            t0 = time.time()
            try:
                if debug:
                    print(f"[FAL ITEM] item={item_index} attempt={attempt}/{retries+1} seed={seed_for_item}")
                req_info = self._fal_submit_only(
                    prompt_text, width, height, steps, CFG, seed_for_item,
                    enable_safety_checker, output_format, sync_mode,
                    lora1_name, lora1_scale, lora2_name, lora2_scale,
                    lora3_name, lora3_scale, timeout_sec=timeout_sec, debug=debug
                )
                result = self._fal_poll_and_fetch(
                    req_info, timeout_sec=timeout_sec, debug=debug,
                    item_idx=item_index, attempt=attempt, started_at=t0
                )
                img_tensor = result[0] if isinstance(result, tuple) else result
                if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                if debug:
                    print(f"[FAL ITEM OK] item={item_index} attempt={attempt} dt={time.time()-t0:.2f}s")
                return True, img_tensor, ""
            except Exception as e:
                last_err = str(e)
                # Dump server response text if it's embedded in our error string already
                print(f"[FAL ITEM ERROR] item={item_index} attempt={attempt} -> {last_err}")
                # Stop retrying if it's a content policy violation
                if self._is_content_policy_violation(last_err):
                    if debug:
                        print(f"[FAL ITEM INFO] item={item_index} content_policy_violation detected — stopping retries.")
                    break
                # continue to next attempt if any remain
        return False, None, last_err

    def generate_image(self, prompt, width, height, steps, CFG, seed,
                      num_images, enable_safety_checker, output_format, sync_mode,
                      retries=2, timeout_sec=120, debug_log=False,
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
                print(f"[PVL Flux Lora Fal WARNING] Invalid regex pattern '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]
            
            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")
            
            # Map prompts to num_images calls
            call_prompts = self._build_call_prompts(base_prompts, num_images, debug=debug_log)
            N = len(call_prompts)
            print(f"[PVL Flux Lora Fal INFO] Processing {N} prompt(s) with retries={retries}, timeout={timeout_sec}s")

            # Single call path (now with retry)
            if N == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    prompt_text=call_prompts[0],
                    width=width, height=height, steps=steps, CFG=CFG,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker, output_format=output_format, sync_mode=sync_mode,
                    lora1_name=lora1_name, lora1_scale=lora1_scale,
                    lora2_name=lora2_name, lora2_scale=lora2_scale,
                    lora3_name=lora3_name, lora3_scale=lora3_scale,
                    retries=retries, timeout_sec=timeout_sec, debug=debug_log
                )
                if ok and torch.is_tensor(img_tensor):
                    _t1 = time.time()
                    print(f"[PVL Flux Lora Fal INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                    return (img_tensor,)
                # else: if all attempts failed
                raise RuntimeError(last_err or "All attempts failed for single request")

            # Multiple calls: per-item parallel execution with per-item retries
            print(f"[PVL Flux Lora Fal INFO] Submitting {N} requests in parallel...")

            results_map = {}
            errors_map = {}
            max_workers = min(N, 6)

            def worker(i):
                ptxt = call_prompts[i]
                return i, *self._run_one_with_retries(
                    item_index=i,
                    prompt_text=ptxt,
                    width=width, height=height, steps=steps, CFG=CFG,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker, output_format=output_format, sync_mode=sync_mode,
                    lora1_name=lora1_name, lora1_scale=lora1_scale,
                    lora2_name=lora2_name, lora2_scale=lora2_scale,
                    lora3_name=lora3_name, lora3_scale=lora3_scale,
                    retries=retries, timeout_sec=timeout_sec, debug=debug_log
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = [executor.submit(worker, i) for i in range(N)]
                for fut in as_completed(futs):
                    i, ok, img_tensor, last_err = fut.result()
                    if ok and torch.is_tensor(img_tensor):
                        results_map[i] = img_tensor
                    else:
                        errors_map[i] = last_err or "Unknown error"

            if not results_map:
                # All failed
                sample_err = next(iter(errors_map.values()), "All FAL requests failed")
                raise RuntimeError(sample_err)

            # Prepare ordered stack of successful images
            all_images = [results_map[i] for i in sorted(results_map.keys()) if torch.is_tensor(results_map[i])]
            if not all_images:
                raise RuntimeError("No images were generated from API calls")

            # Stack into single batch
            try:
                final_tensor = torch.cat(all_images, dim=0)
            except RuntimeError:
                # Rare shape mismatch; normalize to first image size
                first = all_images[0]
                H, W = int(first.shape[1]), int(first.shape[2])
                fixed = []
                for t in all_images:
                    if t.shape[1] == H and t.shape[2] == W:
                        fixed.append(t)
                    else:
                        # Use ImageUtils if available; otherwise simple PIL fallback via tensor<->pil is in ResultProcessor
                        pil = ImageUtils.tensor_to_pil(t)
                        rp = pil.resize((W, H))
                        fixed.append(ImageUtils.pil_to_tensor(rp))
                final_tensor = torch.cat(fixed, dim=0)

            _t1 = time.time()
            print(f"[PVL Flux Lora Fal INFO] Successfully generated {final_tensor.shape[0]}/{N} images in {(_t1 - _t0):.2f}s")

            # Report failures but do not raise if we had at least one success
            failed_idxs = sorted(set(range(N)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(f"[PVL Flux Lora Fal ERROR] Item {i+1} failed after {retries+1} attempt(s): {errors_map.get(i,'Unknown error')}")
                print(f"[PVL Flux Lora Fal WARNING] Returning only {final_tensor.shape[0]}/{N} successful results.")

            # Print seed info if seed was manually set
            if seed != -1:
                seed_list = [(seed + i) % 4294967296 for i in range(N)]
                print(f"[PVL Flux Lora Fal INFO] Seeds used: {seed_list}")

            return (final_tensor,)
            
        except Exception as e:
            print(f"Error generating image with FLUX: {str(e)}")
            return self._handle_error("FLUX", e)

NODE_CLASS_MAPPINGS = {"PVL_fal_FluxWithLora_API": PVL_fal_FluxWithLora_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_FluxWithLora_API": "PVL FAL Flux with LoRA"}
