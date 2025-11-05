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
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "aspect_ratio": ("STRING", {"default": "1:1", "defaultInput": True}),
                # NEW controls:
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 50, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[++]", "multiline": False, "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "raw": ("BOOLEAN", {"default": True}),
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
            print(f"[PVL Flux Pro1.1 WARNING] Provided {len(base_prompts)} prompts but num_images={N}. "
                  f"Reusing the last prompt for remaining calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        
        return call_prompts

    # -------- prohibited-content detector (no-retry) --------
    def _is_content_policy_violation(self, msg_or_json) -> bool:
        """
        Detect FAL's prohibited-content error (non-retryable).
        Looks for JSON {"type": "content_policy_violation"} or the phrase in string messages.
        """
        try:
            if isinstance(msg_or_json, dict):
                et = str(msg_or_json.get("type", "")).lower()
                if "content_policy_violation" in et:
                    return True
                err = msg_or_json.get("error")
                if isinstance(err, dict):
                    et2 = str(err.get("type", "")).lower()
                    if "content_policy_violation" in et2:
                        return True
                if "content_policy_violation" in str(msg_or_json).lower():
                    return True
            elif isinstance(msg_or_json, str):
                s = msg_or_json.lower()
                if "content_policy_violation" in s:
                    return True
        except Exception:
            pass
        return False
    
    # -------- FAL Queue API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, prompt_text, seed, output_format, sync_mode,
                         safety_tolerance, aspect_ratio, enable_safety_checker, raw,
                         timeout_sec=120, debug=False):
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
            try:
                if 'timeout' in ApiHandler.submit_only.__code__.co_varnames:
                    return ApiHandler.submit_only("fal-ai/flux-pro/v1.1-ultra", arguments, timeout=timeout_sec, debug=debug)
                else:
                    return ApiHandler.submit_only("fal-ai/flux-pro/v1.1-ultra", arguments)
            except Exception as e:
                raise RuntimeError(f"FAL submit_only failed: {e}")
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/flux-pro/v1.1-ultra", arguments, timeout_sec, debug)
    
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
            # Try to parse JSON to detect policy violation
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
        Returns a single IMAGE tensor (B,H,W,C).
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
                # Attempt to detect content policy violation from JSON body
                try:
                    js = rr.json()
                    if self._is_content_policy_violation(js):
                        raise RuntimeError(f"FAL content_policy_violation: {js}")
                except Exception:
                    pass
                raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
            
            result = rr.json().get("response", rr.json())
        
        # Validate + NSFW + image decoding via ResultProcessor
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        
        if "images" not in result or not result["images"]:
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")
        
        has_nsfw = result.get("has_nsfw_concepts")
        if isinstance(has_nsfw, list) and any(bool(x) for x in has_nsfw):
            self._raise("FAL: NSFW content detected by safety system (has_nsfw_concepts).")
        
        processed = ResultProcessor.process_image_result(result)  # -> (IMAGE,) or similar
        if not processed or not isinstance(processed[0], torch.Tensor):
            self._raise("FAL: internal error — processed image is not a tensor.")
        
        img_tensor = processed[0]
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        # sanity for black/empty
        if torch.all(img_tensor == 0) or (img_tensor.mean() < 1e-6):
            self._raise("FAL: received an all-black image (likely filtered/failed).")
        
        return img_tensor

    # -------- per-item worker with retry logic --------
    def _run_one_with_retries(
        self,
        item_index: int,
        prompt_text: str,
        seed_base: int,
        output_format: str,
        sync_mode: bool,
        safety_tolerance: str,
        aspect_ratio: str,
        enable_safety_checker: bool,
        raw: bool,
        retries: int,
        timeout_sec: int,
        debug: bool,
    ):
        """
        Returns: (success: bool, image_tensor or None, last_error_message or '')
        Retries only this item up to `retries` times (total attempts = retries+1).
        Stops retrying immediately on content_policy_violation.
        """
        seed_for_item = seed_base if seed_base == -1 else ((seed_base + item_index) % 4294967296)
        last_err = ""
        for attempt in range(1, int(retries) + 2):
            t0 = time.time()
            try:
                if debug:
                    print(f"[FAL ITEM] item={item_index} attempt={attempt}/{retries+1} seed={seed_for_item} ar={aspect_ratio} tol={safety_tolerance}")
                req_info = self._fal_submit_only(
                    prompt_text,
                    seed_for_item,
                    output_format,
                    sync_mode,
                    safety_tolerance,
                    aspect_ratio,
                    enable_safety_checker,
                    raw,
                    timeout_sec=timeout_sec,
                    debug=debug,
                )
                img_tensor = self._fal_poll_and_fetch(
                    req_info,
                    timeout_sec=timeout_sec,
                    debug=debug,
                    item_idx=item_index,
                    attempt=attempt,
                    started_at=t0,
                )
                if debug:
                    print(f"[FAL ITEM OK] item={item_index} attempt={attempt} dt={time.time()-t0:.2f}s")
                return True, img_tensor, ""
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                print(f"[FAL ITEM ERROR] item={item_index} attempt={attempt} -> {err_str}")
                # Do not retry prohibited-content errors
                if self._is_content_policy_violation(err_str):
                    if debug:
                        print(f"[FAL ITEM INFO] item={item_index} content_policy_violation detected — stopping retries.")
                    break
                # otherwise, continue to next attempt
        return False, None, last_err
    
    def generate_image(self, prompt, seed, num_images, output_format,
                      sync_mode, safety_tolerance, aspect_ratio,
                      retries=2, timeout_sec=120, debug_log=False,
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
                print(f"[PVL Flux Pro1.1 WARNING] Invalid regex pattern '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]
            
            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")
            
            # Map prompts to num_images calls
            call_prompts = self._build_call_prompts(base_prompts, num_images)
            N = len(call_prompts)
            print(f"[PVL Flux Pro1.1 INFO] Processing {N} prompt(s) with retries={retries}, timeout={timeout_sec}s")

            # Single call: use per-item retry worker
            if N == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    prompt_text=call_prompts[0],
                    seed_base=seed,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    safety_tolerance=safety_tolerance,
                    aspect_ratio=aspect_ratio,
                    enable_safety_checker=enable_safety_checker,
                    raw=raw,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    _t1 = time.time()
                    print(f"[PVL Flux Pro1.1 INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                    return (img_tensor,)
                raise RuntimeError(last_err or "All attempts failed for single request")
            
            # Multiple calls: TRUE PARALLEL with per-item retries
            print(f"[PVL Flux Pro1.1 INFO] Submitting {N} requests in parallel...")
            results_map: dict[int, torch.Tensor] = {}
            errors_map: dict[int, str] = {}
            max_workers = min(N, 6)

            def worker(i: int):
                return i, *self._run_one_with_retries(
                    item_index=i,
                    prompt_text=call_prompts[i],
                    seed_base=seed,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    safety_tolerance=safety_tolerance,
                    aspect_ratio=aspect_ratio,
                    enable_safety_checker=enable_safety_checker,
                    raw=raw,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
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
            
            # Combine all image tensors in order, normalizing size if needed
            ordered = [results_map[i] for i in sorted(results_map.keys())]
            try:
                final_tensor = torch.cat(ordered, dim=0)
            except RuntimeError:
                # shape mismatch; normalize to first
                first = ordered[0]
                H, W = int(first.shape[1]), int(first.shape[2])
                fixed = []
                for t in ordered:
                    if t.shape[1] == H and t.shape[2] == W:
                        fixed.append(t)
                    else:
                        pil = ImageUtils.tensor_to_pil(t)
                        rp = pil.resize((W, H))
                        fixed.append(ImageUtils.pil_to_tensor(rp))
                final_tensor = torch.cat(fixed, dim=0)
            
            _t1 = time.time()
            print(f"[PVL Flux Pro1.1 INFO] Successfully generated {final_tensor.shape[0]}/{N} images in {(_t1 - _t0):.2f}s")
            
            # Report partial failures without raising
            failed_idxs = sorted(set(range(N)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(f"[PVL Flux Pro1.1 ERROR] Item {i+1} failed after {retries+1} attempt(s): {errors_map.get(i,'Unknown error')}")
                print(f"[PVL Flux Pro1.1 WARNING] Returning only {final_tensor.shape[0]}/{N} successful results.")
            
            # Print seed info if seed was manually set
            if seed != -1:
                seed_list = [(seed + i) % 4294967296 for i in range(N)]
                print(f"[PVL Flux Pro1.1 INFO] Seeds used: {seed_list}")
            
            return (final_tensor,)
            
        except Exception as e:
            print(f"Error generating image with FLUX Pro 1.1 Ultra: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {"PVL_fal_FluxPro_v1_1_Ultra_API": PVL_fal_FluxPro_v1_1_Ultra_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_FluxPro_v1_1_Ultra_API": "PVL Flux Pro 1.1 Ultra (fal.ai)"}
