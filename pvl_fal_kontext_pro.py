import os
import torch
import numpy as np
import json
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_KontextPro_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], {"default": "1:1"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "PVL_tools"

    def _raise(self, msg):
        # Helper to standardize raising: ComfyUI will stop the workflow
        raise RuntimeError(msg)

    def edit_image(self, prompt, image, seed, CFG, num_images, output_format, 
                  sync_mode, safety_tolerance, aspect_ratio):
        # Upload the input image to get a URL
        image_url = ImageUtils.image_to_data_uri(image)
        if not image_url:
            self._raise("FAL: failed to upload input image.")

        # Prepare the arguments for the API call
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "guidance_scale": CFG,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance,
            "aspect_ratio": aspect_ratio
        }
        if seed != -1:
            arguments["seed"] = seed

        # Submit the request and get the result (ApiHandler re-raises on failures)
        result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/kontext", arguments)

        # Basic structural validations
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            # Some errors come under an 'error' key; surface that if present
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
            # Consider a frame 'black' if all pixels are exactly zero OR mean is extremely low
            if torch.all(img_tensor == 0) or (img_tensor.mean() < 1e-6):
                self._raise("FAL: received an all-black image (likely filtered/failed).")

        return processed_result