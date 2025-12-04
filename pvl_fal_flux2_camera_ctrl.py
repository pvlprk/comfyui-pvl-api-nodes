import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_Flux2_Camera_Ctrl_API:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Single required image input
                "image": ("IMAGE",),

                # Camera controls
                "horizontal_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0
                }),
                "vertical_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -90.0,
                    "max": 90.0,
                    "step": 1.0
                }),
                "zoom": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),

                # Image size: renamed presets + custom width/height
                "image_size": (
                    [
                        "auto",    # omit image_size -> use input resolution
                        "1:1",     # square_hd
                        "3:4",     # portrait_4_3
                        "9:16",    # portrait_16_9
                        "4:3",     # landscape_4_3
                        "16:9",    # landscape_16_9
                        "custom",  # uses width/height below
                    ],
                    {"default": "auto"},
                ),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048}),

                # Guidance / steps / LoRA / acceleration
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "num_inference_steps": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 200,
                }),
                "acceleration": (
                    ["none", "regular"],
                    {"default": "regular"},
                ),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),

                # Seed / batch / safety / format
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (
                    ["png", "jpg", "webp"],
                    {"default": "png"},
                ),

                # Retry + timeout + debug controls
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 600,
                    "step": 5
                }),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {},
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

    # ------------------------------------------------------------------
    # FAL Queue API - submit + poll
    # ------------------------------------------------------------------

    def _fal_submit_only(
        self,
        endpoint: str,
        image_urls,
        horizontal_angle: float,
        vertical_angle: float,
        zoom: float,
        image_size,
        guidance_scale: float,
        num_inference_steps: int,
        acceleration: str,
        lora_scale: float,
        seed: int,
        enable_safety_checker: bool,
        output_format: str,
        sync_mode: bool,
        timeout_sec: int = 120,
        debug: bool = False,
    ):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        if not image_urls:
            raise RuntimeError(
                "flux-2-lora-gallery/multiple-angles: at least one input image is required."
            )

        # Map UI "jpg" → API "jpeg"
        if output_format == "jpg":
            fmt = "jpeg"
        else:
            fmt = output_format  # "png" or "webp"

        arguments = {
            "image_urls": image_urls,
            "horizontal_angle": float(horizontal_angle),
            "vertical_angle": float(vertical_angle),
            "zoom": float(zoom),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "acceleration": acceleration,
            "lora_scale": float(lora_scale),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": fmt,
            "sync_mode": bool(sync_mode),
        }

        # image_size handling
        if isinstance(image_size, dict):
            arguments["image_size"] = image_size
        else:
            if image_size == "auto":
                # omit image_size → FAL will use input resolution
                pass
            else:
                arguments["image_size"] = image_size  # enum string

        if seed != -1:
            arguments["seed"] = int(seed)

        if debug:
            print(
                f"[FAL SUBMIT] endpoint={endpoint} "
                f"images={len(image_urls)} "
                f"img_size={arguments.get('image_size', 'auto (input resolution)')} "
                f"h_angle={horizontal_angle} v_angle={vertical_angle} zoom={zoom}"
            )

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
        image_urls,
        horizontal_angle: float,
        vertical_angle: float,
        zoom: float,
        image_size,
        guidance_scale: float,
        num_inference_steps: int,
        acceleration: str,
        lora_scale: float,
        seed_base: int,
        enable_safety_checker: bool,
        output_format: str,
        sync_mode: bool,
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
                        f"images={len(image_urls)}"
                    )
                req_info = self._fal_submit_only(
                    endpoint=endpoint,
                    image_urls=image_urls,
                    horizontal_angle=horizontal_angle,
                    vertical_angle=vertical_angle,
                    zoom=zoom,
                    image_size=image_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    acceleration=acceleration,
                    lora_scale=lora_scale,
                    seed=seed_for_item,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    timeout_sec=timeout_sec,
                    debug=debug,
                )
                result = self._fal_poll_and_fetch(
                    request_info=req_info,
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
    # Image collection
    # ------------------------------------------------------------------

    def _collect_image_urls(self, image, debug=False):
        """
        Convert a single image tensor to a Base64 PNG data URI for FAL.
        We do NOT log the data URI itself, only the fact it's present.
        """
        if image is None:
            raise RuntimeError("No image provided to camera control node.")
        if not isinstance(image, torch.Tensor):
            raise RuntimeError("Camera control image must be a torch.Tensor.")

        try:
            data_uri = ImageUtils.image_to_data_uri(image)
            if debug:
                print("[FAL IMAGE] encoded 1 image for request")
            return [data_uri]
        except Exception as e:
            raise RuntimeError(f"FAL IMAGE ENCODE ERROR (camera control): {e}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_image(
        self,
        image,
        horizontal_angle,
        vertical_angle,
        zoom,
        image_size,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        acceleration,
        lora_scale,
        seed,
        num_images,
        enable_safety_checker,
        output_format,
        retries=2,
        timeout_sec=30,
        debug_log=False,
    ):
        t0 = time.time()
        sync_mode = False  # always store outputs in FAL history

        try:
            # Encode required single image
            image_urls = self._collect_image_urls(image, debug=debug_log)

            endpoint = "fal-ai/flux-2-lora-gallery/multiple-angles"
            print(
                f"[PVL Flux2 LoraGallery MA Fal INFO] Selected endpoint='{endpoint}' "
                f"(num_image_uris={len(image_urls)})"
            )

            # Map UI image_size labels to FAL enums
            size_map = {
                "1:1": "square_hd",
                "3:4": "portrait_4_3",
                "9:16": "portrait_16_9",
                "4:3": "landscape_4_3",
                "16:9": "landscape_16_9",
            }

            # Prepare image_size payload used for all items
            if image_size == "custom":
                image_size_payload = {"width": int(width), "height": int(height)}
            elif image_size == "auto":
                image_size_payload = "auto"  # sentinel → handled in _fal_submit_only
            else:
                image_size_payload = size_map.get(image_size, image_size)

            N = max(1, int(num_images))
            print(
                f"[PVL Flux2 LoraGallery MA Fal INFO] Processing {N} image(s) "
                f"with retries={retries}, timeout={timeout_sec}s"
            )

            # Single-call path
            if N == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    endpoint=endpoint,
                    image_urls=image_urls,
                    horizontal_angle=horizontal_angle,
                    vertical_angle=vertical_angle,
                    zoom=zoom,
                    image_size=image_size_payload,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    acceleration=acceleration,
                    lora_scale=lora_scale,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    t1 = time.time()
                    print(
                        f"[PVL Flux2 LoraGallery MA Fal INFO] Successfully generated 1 image "
                        f"in {t1 - t0:.2f}s using endpoint='{endpoint}'"
                    )
                    return (img_tensor,)
                raise RuntimeError(last_err or "All attempts failed for single request")

            # Multi-call parallel path
            print(
                f"[PVL Flux2 LoraGallery MA Fal INFO] Submitting {N} requests in parallel "
                f"using endpoint='{endpoint}'..."
            )

            results_map = {}
            errors_map = {}
            max_workers = min(N, 6)

            def worker(i):
                return i, *self._run_one_with_retries(
                    item_index=i,
                    endpoint=endpoint,
                    image_urls=image_urls,
                    horizontal_angle=horizontal_angle,
                    vertical_angle=vertical_angle,
                    zoom=zoom,
                    image_size=image_size_payload,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    acceleration=acceleration,
                    lora_scale=lora_scale,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
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
                f"[PVL Flux2 LoraGallery MA Fal INFO] Successfully generated "
                f"{final_tensor.shape[0]}/{N} images in {t1 - t0:.2f}s "
                f"using endpoint='{endpoint}'"
            )

            failed_idxs = sorted(set(range(N)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(
                        f"[PVL Flux2 LoraGallery MA Fal ERROR] Item {i + 1} failed after "
                        f"{retries + 1} attempt(s): {errors_map.get(i, 'Unknown error')}"
                    )
                print(
                    f"[PVL Flux2 LoraGallery MA Fal WARNING] Returning only "
                    f"{final_tensor.shape[0]}/{N} successful results."
                )

            if seed != -1:
                seed_list = [(seed + i) % 4294967296 for i in range(N)]
                print(f"[PVL Flux2 LoraGallery MA Fal INFO] Seeds used: {seed_list}")

            return (final_tensor,)

        except Exception as e:
            print(f"Error generating image with FLUX.2 Multiple Angles: {str(e)}")
            return self._handle_error("FLUX.2 Multiple Angles", e)


NODE_CLASS_MAPPINGS = {
    "PVL_fal_Flux2_Camera_Ctrl_API": PVL_fal_Flux2_Camera_Ctrl_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_Flux2_Camera_Ctrl_API": "PVL Flux.2 Camera Control (fal.ai)",
}
