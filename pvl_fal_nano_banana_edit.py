import io
from typing import List, Tuple

import numpy as np
import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_NanoBanana_API:
    """
    ComfyUI node for FAL 'fal-ai/nano-banana/edit' — batch-friendly image edit using a prompt.

    Inputs:
      - prompt (STRING): Edit instruction for the model.
      - image (IMAGE): One or more input images (batch supported).
      - num_images (INT): Number of output images to generate.
      - output_format (CHOICE): "jpeg" or "png" (API default is "jpeg").
      - sync_mode (BOOLEAN): If true, API can return data-URIs instead of hosted URLs.

    Outputs:
      - IMAGE: Edited image tensor (B, H, W, C) in [0, 1].
      - STRING: Optional text/description returned by the endpoint, if any.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "edit_images"
    CATEGORY = "PVL_tools"

    # ------------------------- helpers -------------------------
    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image) -> List[torch.Tensor]:
        """
        Ensure we always upload individual frames as 3D tensors to avoid 4D->PIL issues.
        Returns a list of per-frame tensors in shape (H, W, C) or (C, H, W).
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return [image[i] for i in range(image.shape[0])]
            elif image.ndim == 3:
                return [image]
            else:
                self._raise("FAL: unsupported image tensor dimensionality.")
        elif isinstance(image, np.ndarray):
            t = torch.from_numpy(image)
            return self._split_image_batch(t)
        else:
            self._raise("FAL: unsupported image type (expected torch Tensor or numpy.ndarray).")

    # ------------------------- main -------------------------
    def edit_images(self, prompt, image, num_images, output_format, sync_mode):
        if not isinstance(prompt, str) or not prompt.strip():
            self._raise("FAL: prompt is required.")

        # Split batch into individual frames and upload each to get a URL
        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")

        image_urls = []
        for idx, frame in enumerate(frames):
            url = ImageUtils.upload_image(frame)
            if not url:
                self._raise(f"FAL: failed to upload input image at index {idx}.")
            image_urls.append(url)

        # Prepare arguments per model schema
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
            "output_format": output_format,        # <-- new
            "sync_mode": bool(sync_mode),          # <-- new
        }

        # Submit request and wait for result
        result = ApiHandler.submit_and_get_result("fal-ai/nano-banana/edit", arguments)

        # Validate basic shape
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")

        # Process images (supports URLs and data: URIs)
        processed = ResultProcessor.process_image_result(result)
        if not processed or not isinstance(processed, tuple):
            self._raise("FAL: internal error — failed to process output images.")

        # Optional: surface 'description' string returned by the model
        description = result.get("description", "") or ""

        # Return IMAGE tensor and description
        return (processed[0], description)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_fal_NanoBanana_API": PVL_fal_NanoBanana_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_NanoBanana_API": "FAL • Nano Banana Edit",
}
