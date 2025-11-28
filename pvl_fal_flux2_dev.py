import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_Flux2_Dev_API:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpg", "png", "webp"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                # Retry + timeout + debug controls
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {
                    "default": "[++]",
                    "multiline": False,
                    "placeholder": "Delimiter for splitting prompts (e.g., [*], \\n, |)"
                }),
                # LoRA inputs
                "lora1_name": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_name": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_name": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                # Up to 8 optional image inputs
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_error(self, api_name: str, error: Exception):
        """
        Generic fallback error handler for image generation nodes that expect (IMAGE,) output.
        Returns a 1×256×256×3 black tensor to keep ComfyUI graphs running.
        """
        print(f"[{api_name} ERROR] {str(error)}")
        try:
            blank_image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"{api_name} fallback image creation failed: {e}") from error
        return (blank_image,)

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
        - If len(prompts) < num_images → reuse last prompt for remaining calls.
        """
        N = max(1, int(num_images))
        if not base_prompts:
            return []

        if len(base_prompts) >= N:
            call_prompts = base_prompts[:N]
        else:
            if debug:
                print(f"[PVL Flux2 Dev Fal] Provided {len(base_prompts)} prompts but num_images={N}. "
                      f"Reusing the last prompt for remaining calls.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))

        if debug:
            for i, p in enumerate(call_prompts):
                show = p if len(p) <= 160 else (p[:157] + "...")
                print(f"[PVL Flux2 Dev Fal] Call {i + 1} prompt: {show}")
        return call_prompts

    # ------------------------------------------------------------------
    # FAL Queue API - submit + poll
    # ------------------------------------------------------------------

    def _fal_submit_only(
        self,
        endpoint: str,
        prompt_text: str,
        width: int,
        height: int,
        steps: int,
        CFG: float,
        seed: int,
        acceleration: str,
        enable_prompt_expansion: bool,
        enable_safety_checker: bool,
        output_format: str,
        sync_mode: bool,
        image_urls,
        loras,
        timeout_sec: int = 120,
        debug: bool = False,
    ):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        # Map UI "jpg" → API "jpeg"
        fmt = "jpeg" if output_format == "jpg" else output_format

        arguments = {
            "prompt": prompt_text,
            "num_inference_steps": steps,
            "guidance_scale": CFG,
            "num_images": 1,  # each call generates 1 image
            "image_size": {
                "width": width,
                "height": height,
            },
            "acceleration": acceleration,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
            "output_format": fmt,
            "sync_mode": sync_mode,
        }

        if seed != -1:
            arguments["seed"] = seed

        # Image endpoints
        if image_urls and endpoint in ("fal-ai/flux-2/edit", "fal-ai/flux-2/lora/edit"):
            arguments["image_urls"] = image_urls[:3]

        # LoRA endpoints
        if loras and endpoint in ("fal-ai/flux-2/lora", "fal-ai/flux-2/lora/edit"):
            arguments["loras"] = loras

        if debug:
            print(
                f"[FAL SUBMIT] endpoint={endpoint} "
                f"images={len(image_urls) if image_urls else 0} "
                f"loras={len(loras) if loras else 0}"
            )

        # If ApiHandler gets extended in future, we can use it;
        # otherwise we fall back to direct queue REST.
        if hasattr(ApiHandler, "submit_only"):
            try:
                if "timeout" in ApiHandler.submit_only.__code__.co_varnames:
                    return ApiHandler.submit_only(endpoint, arguments, timeout=timeout_sec, debug=debug)
                else:
                    return ApiHandler.submit_only(endpoint, arguments)
            except Exception as e:
                raise RuntimeError(f"FAL submit_only failed: {e}")
        else:
            return self._direct_fal_submit(endpoint, arguments, timeout_sec, debug)

    def _direct_fal_submit(self, endpoint, arguments, timeout_sec=120, debug=False):
        """Direct FAL queue API submission when ApiHandler doesn't support async."""
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        r = requests.post(submit_url, headers=headers, json=arguments, timeout=timeout_sec)
        if not r.ok:
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
            print(f"[FAL SUBMIT OK] endpoint={endpoint} request_id={req_id}")
        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }

    def _fal_poll_and_fetch(
        self,
        request_info,
        timeout_sec=120,
        debug=False,
        item_idx=None,
        attempt=None,
        started_at=None,
    ):
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns image tensor (or tuple processed by ResultProcessor).
        """
        # If ApiHandler grows polling support, we can use it.
        if hasattr(ApiHandler, "poll_and_get_result"):
            try:
                if "timeout" in ApiHandler.poll_and_get_result.__code__.co_varnames:
                    result = ApiHandler.poll_and_get_result(request_info, timeout=timeout_sec, debug=debug)
                else:
                    result = ApiHandler.poll_and_get_result(request_info)
            except Exception as e:
                raise RuntimeError(f"FAL poll_and_get_result failed: {e}")
        else:
            fal_key = FalConfig.get_api_key()
            if not fal_key:
                raise RuntimeError("FAL_KEY environment variable not set")

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
                            print(
                                f"[FAL POLL] item={item_idx} attempt={attempt} "
                                f"req={req_id} status={st} elapsed={elapsed:.1f}s"
                            )
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
                        print(
                            f"[FAL POLL] item={item_idx} attempt={attempt} "
                            f"req={req_id} status_check_error: {e}"
                        )
                time.sleep(0.6)

            if not completed:
                raise RuntimeError(f"FAL request {req_id} timed out after {timeout_sec}s")

            rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout_sec))
            if not rr.ok:
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

    # ------------------------------------------------------------------
    # Per-item worker with retries
    # ------------------------------------------------------------------

    def _run_one_with_retries(
        self,
        item_index: int,
        endpoint: str,
        prompt_text: str,
        width: int,
        height: int,
        steps: int,
        CFG: float,
        seed_base: int,
        acceleration: str,
        enable_prompt_expansion: bool,
        enable_safety_checker: bool,
        output_format: str,
        sync_mode: bool,
        image_urls,
        loras,
        retries: int,
        timeout_sec: int,
        debug: bool,
    ):
        """
        Returns tuple (success: bool, image_tensor or None, last_error_message or '')
        Retries only current item up to `retries` times. Each retry performs a fresh submit+poll.
        """
        seed_for_item = seed_base if seed_base == -1 else ((seed_base + item_index) % 4294967296)
        total_attempts = int(retries) + 1

        last_err = ""
        for attempt in range(1, total_attempts + 1):  # attempts = 1 + retries
            t0 = time.time()
            try:
                if debug:
                    print(
                        f"[FAL ITEM] endpoint={endpoint} item={item_index} "
                        f"attempt={attempt}/{total_attempts} "
                        f"seed={seed_for_item} "
                        f"images={len(image_urls) if image_urls else 0} "
                        f"loras={len(loras) if loras else 0}"
                    )
                req_info = self._fal_submit_only(
                    endpoint,
                    prompt_text,
                    width,
                    height,
                    steps,
                    CFG,
                    seed_for_item,
                    acceleration,
                    enable_prompt_expansion,
                    enable_safety_checker,
                    output_format,
                    sync_mode,
                    image_urls,
                    loras,
                    timeout_sec=timeout_sec,
                    debug=debug,
                )
                result = self._fal_poll_and_fetch(
                    req_info,
                    timeout_sec=timeout_sec,
                    debug=debug,
                    item_idx=item_index,
                    attempt=attempt,
                    started_at=t0,
                )
                img_tensor = result[0] if isinstance(result, tuple) else result
                if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                if debug:
                    print(
                        f"[FAL ITEM OK] endpoint={endpoint} item={item_index} "
                        f"attempt={attempt}/{total_attempts} "
                        f"dt={time.time() - t0:.2f}s"
                    )
                return True, img_tensor, ""
            except Exception as e:
                last_err = str(e)
                print(
                    f"[FAL ITEM ERROR] endpoint={endpoint} item={item_index} "
                    f"attempt={attempt}/{total_attempts} -> {last_err}"
                )
                if self._is_content_policy_violation(last_err):
                    if debug:
                        print(
                            f"[FAL ITEM INFO] endpoint={endpoint} item={item_index} "
                            f"content_policy_violation detected — stopping retries."
                        )
                    break

        print(
            f"[FAL ITEM FAILED] endpoint={endpoint} item={item_index} "
            f"after {total_attempts} attempt(s). Last error: {last_err}"
        )
        return False, None, last_err

    # ------------------------------------------------------------------
    # LoRA + image collection
    # ------------------------------------------------------------------

    def _collect_loras(self, lora1_name, lora1_scale, lora2_name, lora2_scale, lora3_name, lora3_scale):
        loras = []
        if str(lora1_name).strip():
            loras.append({"path": str(lora1_name).strip(), "scale": float(lora1_scale)})
        if str(lora2_name).strip():
            loras.append({"path": str(lora2_name).strip(), "scale": float(lora2_scale)})
        if str(lora3_name).strip():
            loras.append({"path": str(lora3_name).strip(), "scale": float(lora3_scale)})
        return loras

    def _collect_image_urls(self, images, debug=False):
        """
        Convert connected image tensors to Base64 PNG data URIs for FAL.
        We do NOT log the data URI itself, only the count.
        """
        urls = []
        for idx, img in enumerate(images):
            if img is None:
                continue
            if not isinstance(img, torch.Tensor):
                continue
            try:
                data_uri = ImageUtils.image_to_data_uri(img)
                urls.append(data_uri)
                if debug:
                    print(f"[FAL IMAGE] found image input at slot {idx + 1}")
            except Exception as e:
                print(f"[FAL IMAGE ENCODE ERROR] image_{idx + 1}: {e}")

        if debug:
            print(f"[FAL IMAGE] total encoded images for request: {len(urls)}")

        if len(urls) > 3:
            if debug:
                print(f"[FAL IMAGE INFO] {len(urls)} images provided, truncating to first 3.")
            urls = urls[:3]
        return urls

    def _select_endpoint(self, has_image: bool, has_lora: bool):
        if not has_image and not has_lora:
            return "fal-ai/flux-2"
        if has_image and not has_lora:
            return "fal-ai/flux-2/edit"
        if not has_image and has_lora:
            return "fal-ai/flux-2/lora"
        return "fal-ai/flux-2/lora/edit"

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt,
        width,
        height,
        steps,
        CFG,
        seed,
        num_images,
        acceleration,
        enable_prompt_expansion,
        enable_safety_checker,
        output_format,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        delimiter="[++]",
        lora1_name="",
        lora1_scale=1.0,
        lora2_name="",
        lora2_scale=1.0,
        lora3_name="",
        lora3_scale=1.0,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
    ):
        t0 = time.time()

        try:
            # Split prompts with regex delimiter
            try:
                base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
            except re.error:
                print(f"[PVL Flux2 Dev Fal WARNING] Invalid regex pattern '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]

            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")

            # Map prompts to calls
            call_prompts = self._build_call_prompts(base_prompts, num_images, debug=debug_log)
            N = len(call_prompts)
            print(
                f"[PVL Flux2 Dev Fal INFO] Processing {N} prompt(s) with retries={retries}, "
                f"timeout={timeout_sec}s, num_images={num_images}"
            )

            # LoRAs: determine presence from inputs (names), then build payload list.
            has_lora_input = any(str(x).strip() for x in (lora1_name, lora2_name, lora3_name))
            loras = self._collect_loras(
                lora1_name,
                lora1_scale,
                lora2_name,
                lora2_scale,
                lora3_name,
                lora3_scale,
            )

            # Images: presence is based on any non-None tensor, then we encode to data URIs.
            images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
            has_image_input = any(img is not None for img in images)
            image_urls = self._collect_image_urls(images, debug=debug_log)

            endpoint = self._select_endpoint(has_image=has_image_input, has_lora=has_lora_input)
            print(
                f"[PVL Flux2 Dev Fal INFO] Selected endpoint='{endpoint}' "
                f"(has_image_input={has_image_input}, has_lora_input={has_lora_input}, "
                f"num_image_uris={len(image_urls)}, num_loras={len(loras)})"
            )

            if endpoint in ("fal-ai/flux-2/edit", "fal-ai/flux-2/lora/edit") and not image_urls:
                raise RuntimeError(
                    "Edit endpoint selected but no valid image data URIs were produced from the inputs."
                )

            # Single-call path
            if N == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    endpoint=endpoint,
                    prompt_text=call_prompts[0],
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed_base=seed,
                    acceleration=acceleration,
                    enable_prompt_expansion=enable_prompt_expansion,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    image_urls=image_urls,
                    loras=loras,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    t1 = time.time()
                    print(
                        f"[PVL Flux2 Dev Fal INFO] Successfully generated 1 image "
                        f"in {t1 - t0:.2f}s using endpoint='{endpoint}'"
                    )
                    return (img_tensor,)
                raise RuntimeError(last_err or "All attempts failed for single request")

            # Multi-call parallel path
            print(
                f"[PVL Flux2 Dev Fal INFO] Submitting {N} requests in parallel "
                f"using endpoint='{endpoint}'..."
            )

            results_map = {}
            errors_map = {}
            max_workers = min(N, 6)

            def worker(i):
                ptxt = call_prompts[i]
                return i, *self._run_one_with_retries(
                    item_index=i,
                    endpoint=endpoint,
                    prompt_text=ptxt,
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed_base=seed,
                    acceleration=acceleration,
                    enable_prompt_expansion=enable_prompt_expansion,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    image_urls=image_urls,
                    loras=loras,
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
                sample_err = next(iter(errors_map.values()), "All FAL requests failed")
                raise RuntimeError(sample_err)

            all_images = [
                results_map[i] for i in sorted(results_map.keys()) if torch.is_tensor(results_map[i])
            ]
            if not all_images:
                raise RuntimeError("No images were generated from API calls")

            # Stack into single batch
            try:
                final_tensor = torch.cat(all_images, dim=0)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to stack images (mismatched shapes?): {e}")

            t1 = time.time()
            print(
                f"[PVL Flux2 Dev Fal INFO] Successfully generated "
                f"{final_tensor.shape[0]}/{N} images in {t1 - t0:.2f}s "
                f"using endpoint='{endpoint}'"
            )

            failed_idxs = sorted(set(range(N)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(
                        f"[PVL Flux2 Dev Fal ERROR] Item {i + 1} failed after "
                        f"{retries + 1} attempt(s): {errors_map.get(i, 'Unknown error')}"
                    )
                print(
                    f"[PVL Flux2 Dev Fal WARNING] Returning only "
                    f"{final_tensor.shape[0]}/{N} successful results."
                )

            if seed != -1:
                seed_list = [(seed + i) % 4294967296 for i in range(N)]
                print(f"[PVL Flux2 Dev Fal INFO] Seeds used: {seed_list}")

            return (final_tensor,)

        except Exception as e:
            print(f"Error generating image with FLUX.2: {str(e)}")
            return self._handle_error("FLUX.2", e)


NODE_CLASS_MAPPINGS = {
    "PVL_fal_Flux2_Dev_API": PVL_fal_Flux2_Dev_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_Flux2_Dev_API": "PVL FAL Flux.2 Dev (fal.ai)",
}
