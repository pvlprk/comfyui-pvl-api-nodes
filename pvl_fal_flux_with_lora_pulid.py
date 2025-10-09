import os
import re
import torch
import time
import requests
import io
import base64
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxWithLoraPulID_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a woman holding sign with glowing green text 'PuLID for FLUX'"}),
                "reference_image": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[*]", "multiline": False, "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], {"default": "landscape_4_3"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "bad quality, worst quality, text, signature, watermark, extra limbs"}),
                "lora_path": ("STRING", {"default": "", "placeholder": "Optional LoRA path"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100}),
                "true_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "id_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "max_sequence_length": (["128", "256", "512"], {"default": "128"}),
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
    def _fal_submit_only(self, prompt_text, reference_image_url, image_size, custom_width, custom_height,
                        num_inference_steps, seed, guidance_scale, negative_prompt, sync_mode,
                        enable_safety_checker, lora_path, lora_strength, start_step, true_cfg,
                        id_weight, max_sequence_length, debug=False):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        # Build image_size argument
        if custom_width > 0 and custom_height > 0:
            image_size_arg = {"width": custom_width, "height": custom_height}
        else:
            image_size_arg = image_size

        arguments = {
            "prompt": prompt_text,
            "reference_image_url": reference_image_url,
            "image_size": image_size_arg,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "sync_mode": sync_mode,
            "enable_safety_checker": enable_safety_checker,
            "true_cfg": true_cfg,
            "id_weight": id_weight,
            "max_sequence_length": max_sequence_length,
        }

        if seed != -1:
            arguments["seed"] = seed

        if lora_path.strip():
            arguments["lora_path"] = lora_path.strip()
            arguments["lora_strength"] = lora_strength

        if start_step > 0:
            arguments["start_step"] = start_step

        if debug:
            print(f"[PVL DEBUG] Submit arguments: {arguments}")

        # Check if ApiHandler supports async submission
        if hasattr(ApiHandler, 'submit_only'):
            result = ApiHandler.submit_only("fal-ai/flux-pulid-lora", arguments)
            if debug:
                print(f"[PVL DEBUG] ApiHandler.submit_only result: {result}")
            return result
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/flux-pulid-lora", arguments, debug)

    def _direct_fal_submit(self, endpoint, arguments, debug=False):
        """Direct FAL queue API submission when ApiHandler doesn't support async."""
        fal_key = os.getenv("FAL_KEY", "")
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        if debug:
            print(f"[PVL DEBUG] Submitting to: {submit_url}")
            print(f"[PVL DEBUG] Headers: {headers}")
            print(f"[PVL DEBUG] Arguments: {arguments}")

        r = requests.post(submit_url, headers=headers, json=arguments, timeout=120)

        if debug:
            print(f"[PVL DEBUG] Submit response status: {r.status_code}")
            print(f"[PVL DEBUG] Submit response body: {r.text}")

        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"

        result = {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }

        if debug:
            print(f"[PVL DEBUG] Request info: {result}")

        return result

    def _fal_poll_and_fetch(self, request_info, timeout=120, debug=False):
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns image tensor.
        """
        if debug:
            print(f"[PVL DEBUG] Polling request: {request_info}")
            print(f"[PVL DEBUG] Timeout: {timeout}s")

        # Check if ApiHandler supports async polling
        if hasattr(ApiHandler, 'poll_and_get_result'):
            result = ApiHandler.poll_and_get_result(request_info, timeout)
            if debug:
                print(f"[PVL DEBUG] ApiHandler.poll_and_get_result returned: {result}")
        else:
            # Fallback to direct polling
            fal_key = os.getenv("FAL_KEY", "")
            headers = {"Authorization": f"Key {fal_key}"}
            status_url = request_info["status_url"]
            resp_url = request_info["response_url"]

            if debug:
                print(f"[PVL DEBUG] Polling status_url: {status_url}")

            # Poll for completion
            deadline = time.time() + timeout
            completed = False
            poll_count = 0
            while time.time() < deadline:
                try:
                    poll_count += 1
                    sr = requests.get(status_url, headers=headers, timeout=10)
                    status_data = sr.json() if sr.ok else {}

                    if debug:
                        print(f"[PVL DEBUG] Poll #{poll_count}: status={status_data.get('status')}, response={status_data}")

                    if sr.ok and status_data.get("status") == "COMPLETED":
                        completed = True
                        if debug:
                            print(f"[PVL DEBUG] Request completed after {poll_count} polls")
                        break
                except Exception as e:
                    if debug:
                        print(f"[PVL DEBUG] Poll error: {e}")
                    pass
                time.sleep(0.6)

            if not completed:
                raise RuntimeError(f"FAL request timed out after {timeout}s ({poll_count} polls)")

            # Fetch result
            if debug:
                print(f"[PVL DEBUG] Fetching result from: {resp_url}")

            rr = requests.get(resp_url, headers=headers, timeout=15)

            if debug:
                print(f"[PVL DEBUG] Result response status: {rr.status_code}")
                print(f"[PVL DEBUG] Result response body: {rr.text[:500]}...")  # First 500 chars

            if not rr.ok:
                raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

            result = rr.json().get("response", rr.json())

            if debug:
                print(f"[PVL DEBUG] Parsed result: {result}")

        # Process result using ResultProcessor
        processed = ResultProcessor.process_image_result(result)

        if debug:
            print(f"[PVL DEBUG] Processed result type: {type(processed)}")
            if torch.is_tensor(processed):
                print(f"[PVL DEBUG] Tensor shape: {processed.shape}")
            elif isinstance(processed, tuple):
                print(f"[PVL DEBUG] Tuple length: {len(processed)}")
                if len(processed) > 0 and torch.is_tensor(processed[0]):
                    print(f"[PVL DEBUG] First tensor shape: {processed[0].shape}")

        return processed

    def generate_image(self, prompt, reference_image, num_images, num_inference_steps,
                      guidance_scale, seed, enable_safety_checker, sync_mode, timeout, debug,
                      delimiter="[*]",
                      image_size="landscape_4_3", custom_width=0, custom_height=0,
                      negative_prompt="bad quality, worst quality, text, signature, watermark, extra limbs",
                      lora_path="", lora_strength=1.0, start_step=0, true_cfg=1.0,
                      id_weight=1.0, max_sequence_length="128"):
        _t0 = time.time()
        try:
            if debug:
                print(f"[PVL DEBUG] ===== STARTING GENERATION =====")
                print(f"[PVL DEBUG] Timeout: {timeout}s")
                print(f"[PVL DEBUG] Debug mode: {debug}")
                print(f"[PVL DEBUG] Num images: {num_images}")
                print(f"[PVL DEBUG] Seed: {seed}")
                print(f"[PVL DEBUG] Prompt: {prompt[:100]}...")

            # Convert reference image to base64 data URI (FIXED: replaces ImageUtils.upload_image)
            print("[PVL INFO] Converting reference image to base64...")

            # Convert tensor to PIL Image
            if torch.is_tensor(reference_image):
                # Handle batch dimension
                if reference_image.ndim == 4:
                    img_tensor = reference_image[0]  # Take first image
                else:
                    img_tensor = reference_image

                if debug:
                    print(f"[PVL DEBUG] Input image tensor shape: {img_tensor.shape}")
                    print(f"[PVL DEBUG] Input image dtype: {img_tensor.dtype}")
                    print(f"[PVL DEBUG] Input image range: [{img_tensor.min()}, {img_tensor.max()}]")

                # Convert from torch tensor (0-1 float) to PIL Image
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)

                if debug:
                    print(f"[PVL DEBUG] PIL image size: {pil_image.size}")
                    print(f"[PVL DEBUG] PIL image mode: {pil_image.mode}")

                # Convert to base64 data URI
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                reference_image_url = f"data:image/png;base64,{img_base64}"

                if debug:
                    print(f"[PVL DEBUG] Base64 length: {len(img_base64)} chars")
                    print(f"[PVL DEBUG] Data URI prefix: {reference_image_url[:100]}")

                print("[PVL INFO] Reference image converted to base64 data URI")
            else:
                raise RuntimeError("reference_image must be a torch tensor")

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

            if debug:
                for i, p in enumerate(call_prompts):
                    print(f"[PVL DEBUG] Prompt {i}: {p[:100]}...")

            # Single call: process directly (less overhead)
            if len(call_prompts) == 1:
                if debug:
                    print(f"[PVL DEBUG] Single call mode - processing directly")

                req_info = self._fal_submit_only(
                    call_prompts[0], reference_image_url, image_size, custom_width, custom_height,
                    num_inference_steps, seed, guidance_scale, negative_prompt, sync_mode,
                    enable_safety_checker, lora_path, lora_strength, start_step, true_cfg,
                    id_weight, max_sequence_length, debug
                )
                result = self._fal_poll_and_fetch(req_info, timeout, debug)
                img_tensor = result[0] if isinstance(result, tuple) else result
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                _t1 = time.time()
                print(f"[PVL INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                return (img_tensor,)

            # Multiple calls: TRUE PARALLEL execution with seed increment
            print(f"[PVL INFO] Submitting {len(call_prompts)} requests in parallel...")

            if debug:
                print(f"[PVL DEBUG] Multi-call mode - parallel execution")

            # PHASE 1: Submit all requests in parallel
            submit_results = []
            max_workers = min(len(call_prompts), 6)

            if debug:
                print(f"[PVL DEBUG] PHASE 1: Submitting with {max_workers} workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                submit_futs = {
                    executor.submit(
                        self._fal_submit_only,
                        call_prompts[i], reference_image_url, image_size, custom_width, custom_height,
                        num_inference_steps,
                        seed if seed == -1 else (seed + i) % 4294967296,
                        guidance_scale, negative_prompt, sync_mode,
                        enable_safety_checker, lora_path, lora_strength, start_step, true_cfg,
                        id_weight, max_sequence_length, debug
                    ): i
                    for i in range(len(call_prompts))
                }

                for fut in as_completed(submit_futs):
                    idx = submit_futs[fut]
                    try:
                        req_info = fut.result()
                        submit_results.append((idx, req_info))
                        if debug:
                            print(f"[PVL DEBUG] Submit completed for prompt {idx}")
                    except Exception as e:
                        print(f"[PVL ERROR] Submit failed for prompt {idx}: {e}")

            if not submit_results:
                raise RuntimeError("All FAL submission requests failed")

            print(f"[PVL INFO] {len(submit_results)} requests submitted. Polling for results...")

            if debug:
                print(f"[PVL DEBUG] PHASE 2: Polling with {max_workers} workers")

            # PHASE 2: Poll all requests in parallel
            results = {}
            failed_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                poll_futs = {
                    executor.submit(self._fal_poll_and_fetch, req_info, timeout, debug): idx
                    for idx, req_info in submit_results
                }

                for fut in as_completed(poll_futs):
                    idx = poll_futs[fut]
                    try:
                        result = fut.result()
                        results[idx] = result
                        if debug:
                            print(f"[PVL DEBUG] Poll completed for prompt {idx}")
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

                        if debug:
                            print(f"[PVL DEBUG] Image {i} tensor shape: {img_tensor.shape}")

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

            if debug:
                print(f"[PVL DEBUG] Final tensor shape: {final_tensor.shape}")
                print(f"[PVL DEBUG] ===== GENERATION COMPLETE =====")

            return (final_tensor,)

        except Exception as e:
            print(f"Error generating image with FLUX PuLID LoRA: {str(e)}")
            if debug:
                import traceback
                print(f"[PVL DEBUG] Full traceback:")
                traceback.print_exc()
            # Fallback error handling - return empty tensor
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_tensor,)

NODE_CLASS_MAPPINGS = {"PVL_fal_FluxWithLoraPulID_API": PVL_fal_FluxWithLoraPulID_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_FluxWithLoraPulID_API": "PVL Flux Lora PulID (fal.ai)"}
