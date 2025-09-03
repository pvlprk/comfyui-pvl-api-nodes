import os
import torch
import numpy as np
import json
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
                # CHANGED: make connectable string input (socket by default)
                "aspect_ratio": ("STRING", {"default": "16:9", "defaultInput": True}),
            },
            "optional": {
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

    def generate_image(self, prompt, seed, num_images, output_format,
                       sync_mode, safety_tolerance, aspect_ratio,
                       enable_safety_checker=True, raw=False):
        # Validate/normalize aspect ratio string
        aspect_ratio = self._normalize_aspect_ratio(aspect_ratio)

        # Prepare the arguments for the API call
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance,
            "aspect_ratio": aspect_ratio,
            "enable_safety_checker": enable_safety_checker,
            "raw": raw
        }
        if seed != -1:
            arguments["seed"] = seed

        # Submit the request and get the result (ApiHandler re-raises on failures)
        result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/v1.1-ultra", arguments)

        # Basic structural validations
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

        # Process images (may raise)
        processed_result = ResultProcessor.process_image_result(result)

        # Check for black/empty image(s) and abort
        if processed_result and len(processed_result) > 0:
            img_tensor = processed_result[0]
            if not isinstance(img_tensor, torch.Tensor):
                self._raise("FAL: internal error — processed image is not a tensor.")
            if torch.all(img_tensor == 0) or (img_tensor.mean() < 1e-6):
                self._raise("FAL: received an all-black image (likely filtered/failed).")

        return processed_result
