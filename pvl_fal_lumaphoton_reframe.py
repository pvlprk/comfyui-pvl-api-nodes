import os
import torch
import numpy as np
import json
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_LumaPhoton_Reframe_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], {"default": "16:9"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "grid_position_x": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "grid_position_y": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "x_start": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "x_end": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "y_start": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "y_end": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reframe_image"
    CATEGORY = "PVL_tools"

    def _raise(self, msg):
        # Helper to standardize raising: ComfyUI will stop the workflow
        raise RuntimeError(msg)

    def reframe_image(self, image, aspect_ratio, prompt="", grid_position_x=-1, 
                     grid_position_y=-1, x_start=-1, x_end=-1, y_start=-1, y_end=-1):
        # Upload the input image to get a URL
        image_url = ImageUtils.upload_image(image)
        if not image_url:
            self._raise("FAL: failed to upload input image.")

        # Prepare the arguments for the API call
        arguments = {
            "image_url": image_url,
            "aspect_ratio": aspect_ratio
        }
        
        # Add optional parameters if they are provided (not -1)
        if prompt:
            arguments["prompt"] = prompt
        if grid_position_x != -1:
            arguments["grid_position_x"] = grid_position_x
        if grid_position_y != -1:
            arguments["grid_position_y"] = grid_position_y
        if x_start != -1:
            arguments["x_start"] = x_start
        if x_end != -1:
            arguments["x_end"] = x_end
        if y_start != -1:
            arguments["y_start"] = y_start
        if y_end != -1:
            arguments["y_end"] = y_end

        # Submit the request and get the result (ApiHandler re-raises on failures)
        result = ApiHandler.submit_and_get_result("fal-ai/luma-photon/reframe", arguments)

        # Basic structural validations
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            # Some errors come under an 'error' key; surface that if present
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")

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