import time
from typing import List, Optional

import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_SeeDream_45_API:

    """
    PVL SeeDream 4.5 (fal.ai)

    - If any image input is connected -> use edit endpoint:
        "fal-ai/bytedance/seedream/v4.5/edit"
      and send images as Base64 data URIs via image_urls.

    - If no image is connected -> use text-to-image endpoint:
        "fal-ai/bytedance/seedream/v4.5/text-to-image"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                # Image size: selector + custom width/height
                # These map to SeeDream enums:
                #   "1:1"   -> square_hd
                #   "3:4"   -> portrait_4_3
                #   "9:16"  -> portrait_16_9
                #   "4:3"   -> landscape_4_3
                #   "16:9"  -> landscape_16_9
                #   "auto_2K" / "auto_4K" -> passed through
                #   "custom" -> width/height object
                "image_size": (
                    [
                        "1:1",     # square_hd
                        "3:4",     # portrait_4_3
                        "9:16",    # portrait_16_9
                        "4:3",     # landscape_4_3
                        "16:9",    # landscape_16_9
                        "auto_2K",
                        "auto_4K",
                        "custom",
                    ],
                    {"default": "custom"},
                ),
                # SeeDream wants widths/heights between 1920 and 4096 (per docs).
                # We don't hard-enforce, but we hint via min/max.
                "width": ("INT", {"default": 2048, "min": 512, "max": 4096}),
                "height": ("INT", {"default": 2048, "min": 512, "max": 4096}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                # Retry + timeout + debug controls
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 60, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Up to 8 optional image inputs (for edit endpoint).
                # We will send them as Base64 data URIs in image_urls.
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

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

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

    def _build_image_size_payload(self, image_size, width: int, height: int, debug: bool = False):
        """
        Map UI image_size selector to SeeDream's API payload.
        """
        size_map = {
            "1:1": "square_hd",
            "3:4": "portrait_4_3",
            "9:16": "portrait_16_9",
            "4:3": "landscape_4_3",
            "16:9": "landscape_16_9",
        }

        if isinstance(image_size, dict):
            payload = image_size
        else:
            if image_size == "custom":
                payload = {
                    "width": int(width),
                    "height": int(height),
                }
            elif image_size in ("auto_2K", "auto_4K"):
                payload = image_size
            else:
                payload = size_map.get(image_size, image_size)

        if debug:
            print(f"[SeeDream 4.5] image_size payload={payload}")
        return payload

    def _collect_image_urls(self, images: List[Optional[torch.Tensor]], debug: bool = False):
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
                    print(f"[SeeDream 4.5] found image input at slot {idx + 1}")
            except Exception as e:
                print(f"[SeeDream 4.5] IMAGE ENCODE ERROR image_{idx + 1}: {e}")

        if debug:
            print(f"[SeeDream 4.5] total encoded images for request: {len(urls)}")

        # SeeDream allows up to 10 image inputs; node provides 8, so no truncation needed.
        return urls

    def _select_endpoint(self, has_image: bool) -> str:
        if has_image:
            return "fal-ai/bytedance/seedream/v4.5/edit"
        return "fal-ai/bytedance/seedream/v4.5/text-to-image"

    def _call_seedream(
        self,
        endpoint: str,
        arguments: dict,
        timeout_sec: int,
        retries: int,
        debug: bool,
    ):
        """
        Simple retry loop around ApiHandler.submit_and_get_result for SeeDream.
        Returns a tensor batch (B,H,W,C) on success; raises if all attempts fail.
        """
        last_err = ""
        total_attempts = int(retries) + 1
        for attempt in range(1, total_attempts + 1):
            t0 = time.time()
            try:
                if debug:
                    print(
                        f"[SeeDream 4.5] endpoint={endpoint} attempt "
                        f"{attempt}/{total_attempts} args.keys={list(arguments.keys())}"
                    )
                result = ApiHandler.submit_and_get_result(endpoint, arguments)
                out = ResultProcessor.process_image_result(result)
                img_tensor = out[0] if isinstance(out, tuple) else out
                if debug:
                    dt = time.time() - t0
                    print(
                        f"[SeeDream 4.5] endpoint={endpoint} attempt "
                        f"{attempt}/{total_attempts} OK in {dt:.2f}s"
                    )
                return img_tensor
            except Exception as e:
                last_err = str(e)
                print(
                    f"[SeeDream 4.5 ERROR] endpoint={endpoint} attempt "
                    f"{attempt}/{total_attempts} -> {last_err}"
                )
                # If we hit explicit content policy violation, don't keep retrying.
                if self._is_content_policy_violation(last_err):
                    if debug:
                        print(
                            "[SeeDream 4.5 INFO] content_policy_violation detected — "
                            "stopping retries."
                        )
                    break

        raise RuntimeError(last_err or "SeeDream 4.5: all attempts failed")

    # ---------------------------------------------------------
    # Main
    # ---------------------------------------------------------

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_images,
        max_images,
        seed,
        enable_safety_checker,
        sync_mode,
        retries=2,
        timeout_sec=60,
        debug_log=False,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
    ):
        """
        Main ComfyUI entrypoint.
        """
        t0 = time.time()
        api_name = "SeeDream 4.5 (fal.ai)"

        try:
            # Collect images & decide endpoint.
            images = [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                image_7,
                image_8,
            ]
            has_image_input = any(img is not None for img in images)
            image_urls = self._collect_image_urls(images, debug=debug_log) if has_image_input else []

            endpoint = self._select_endpoint(has_image=has_image_input)
            print(
                f"[SeeDream 4.5 INFO] Selected endpoint='{endpoint}' "
                f"(has_image_input={has_image_input}, num_image_uris={len(image_urls)})"
            )

            if "edit" in endpoint and not image_urls:
                raise RuntimeError(
                    "SeeDream edit endpoint selected but no valid image data URIs were produced "
                    "from the inputs."
                )

            # Build image_size payload
            img_size_payload = self._build_image_size_payload(
                image_size=image_size,
                width=width,
                height=height,
                debug=debug_log,
            )

            # Build arguments as per SeeDream docs
            arguments = {
                "prompt": str(prompt),
                "image_size": img_size_payload,
                "num_images": int(num_images),
                "max_images": int(max_images),
                "enable_safety_checker": bool(enable_safety_checker),
                "sync_mode": bool(sync_mode),
            }

            if seed != -1:
                arguments["seed"] = int(seed)

            if image_urls:
                arguments["image_urls"] = image_urls

            if debug_log:
                safe_args_preview = {k: v for k, v in arguments.items() if k != "image_urls"}
                print(f"[SeeDream 4.5 DEBUG] arguments (without image_urls): {safe_args_preview}")
                print(f"[SeeDream 4.5 DEBUG] image_urls count: {len(image_urls)}")

            # Call FAL with retry logic
            img_tensor = self._call_seedream(
                endpoint=endpoint,
                arguments=arguments,
                timeout_sec=timeout_sec,
                retries=retries,
                debug=debug_log,
            )

            # Ensure batch dimension
            if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)

            t1 = time.time()
            print(
                f"[SeeDream 4.5 INFO] Successfully generated {img_tensor.shape[0]} image(s) "
                f"in {t1 - t0:.2f}s using endpoint='{endpoint}'"
            )

            if seed != -1:
                print(f"[SeeDream 4.5 INFO] Seed used: {seed}")

            return (img_tensor,)

        except Exception as e:
            print(f"Error generating image with {api_name}: {str(e)}")
            return self._handle_error(api_name, e)

NODE_CLASS_MAPPINGS = {
    "PVL_fal_SeeDream_45_API": PVL_fal_SeeDream_45_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_SeeDream_45_API": "PVL SeeDream 4.5 (fal.ai)",
}