import io
from typing import List, Tuple
import numpy as np
import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_SeeDream4_API:
    """
    ComfyUI node for FAL 'fal-ai/bytedance/seedream/v4/edit' — image editing using prompt and uploaded input images.

    Inputs:
      - prompt (STRING): Edit instruction for the model.
      - image (IMAGE): One or more input images (batch supported).
      - image_size (CHOICE): Only used for UI preset, width & height are always passed to API.
      - width / height (INT): Pixel dimensions passed to the API.
      - seed (INT): Optional seed for deterministic results.
      - num_images (INT): Number of outputs to generate.
      - sync_mode (BOOLEAN): Whether to wait for results directly (may increase latency).

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
                "image_size": (
                    ["square_hd", "square", "portrait_3_4", "portrait_9_16", "landscape_4_3", "landscape_16_9", "custom"],
                    {"default": "square_hd"},
                ),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "edit_seedream_images"
    CATEGORY = "PVL_tools"

    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image) -> List[torch.Tensor]:
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

    def edit_seedream_images(self, prompt, image, num_images, image_size, width, height, seed, sync_mode):
        if not isinstance(prompt, str) or not prompt.strip():
            self._raise("FAL: prompt is required.")

        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")

        image_urls = []
        for idx, frame in enumerate(frames):
            url = ImageUtils.image_to_data_uri(frame)
            if not url:
                self._raise(f"FAL: failed to upload input image at index {idx}.")
            image_urls.append(url)

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": {"width": width, "height": height},
            "num_images": int(num_images),
            "seed": int(seed),
            "sync_mode": bool(sync_mode),
        }

        result = ApiHandler.submit_and_get_result("fal-ai/bytedance/seedream/v4/edit", arguments)

        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")

        processed = ResultProcessor.process_image_result(result)
        if not processed or not isinstance(processed, tuple):
            self._raise("FAL: internal error — failed to process output images.")

        description = result.get("description", "") or ""
        return (processed[0], description)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_fal_SeeDream4_API": PVL_fal_SeeDream4_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_SeeDream4_API": "FAL • SeeDream4 Edit",
}
