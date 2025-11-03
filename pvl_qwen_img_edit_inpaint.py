import io
from typing import List, Tuple

import numpy as np
import torch

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_Qwen_Img_Edit_Inpaint:
    """
    ComfyUI node for FAL 'fal-ai/qwen-image-edit/inpaint' — image editing with mask & prompt.

    Inputs:
      - prompt (STRING): Edit instruction for the model.
      - image (IMAGE): Input image tensor (batch or single).
      - mask (IMAGE, optional): Mask image for inpainting (white=edit, black=keep).
      - negative_prompt (STRING, optional): Negative prompt for generation.
      - num_inference_steps (INT): Diffusion steps (default: 30).
      - guidance_scale (FLOAT): CFG scale (default: 4.0).
      - num_images (INT): Number of output images to generate.
      - enable_safety_checker (BOOLEAN): Enable safety filtering (default: True).
      - output_format (CHOICE): jpeg / png (default: png).
      - acceleration (CHOICE): none / regular / high (default: regular).
      - strength (FLOAT): Inpainting noise strength (0–1, default: 0.93).
      - sync_mode (BOOLEAN): Whether to request data URIs instead of URLs (default: False).
      - seed (INT): Optional random seed (default: 0 -> random).

    Outputs:
      - IMAGE: Generated edited image tensor (B, H, W, C) in [0, 1].
      - STRING: Prompt used (for reference).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": ""}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "strength": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt")
    FUNCTION = "edit_images"
    CATEGORY = "PVL_tools"

    # ------------------------- helpers -------------------------
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

    # ------------------------- main -------------------------
    def edit_images(
        self,
        prompt,
        image,
        mask=None,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=4.0,
        num_images=1,
        enable_safety_checker=True,
        output_format="png",
        acceleration="regular",
        strength=0.93,
        sync_mode=False,
        seed=0,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            self._raise("FAL: prompt is required.")

        # Split batch into individual frames
        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")

        # Encode primary image as data URI or uploaded URL
        image_uri = ImageUtils.image_to_data_uri(frames[0])

        # Encode mask if provided
        mask_uri = None
        if mask is not None:
            mask_frames = self._split_image_batch(mask)
            mask_uri = ImageUtils.image_to_data_uri(mask_frames[0])

        # Prepare request body
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or " ",
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "num_images": int(num_images),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
            "acceleration": acceleration,
            "strength": float(strength),
            "sync_mode": bool(sync_mode),
        }

        if seed > 0:
            args["seed"] = int(seed)
        if image_uri:
            args["image_url"] = image_uri
        if mask_uri:
            args["mask_url"] = mask_uri

        # Submit and wait for result
        result = ApiHandler.submit_and_get_result("fal-ai/qwen-image-edit/inpaint", args)

        # Validate response
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")

        # Convert to torch tensor
        processed = ResultProcessor.process_image_result(result)
        if not processed or not isinstance(processed, tuple):
            self._raise("FAL: internal error — failed to process output images.")

        # Return (image, prompt)
        return (processed[0], prompt)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_Qwen_Img_Edit_Inpaint": PVL_Qwen_Img_Edit_Inpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_Qwen_Img_Edit_Inpaint": "PVL Qwen Image Edit Inpaint (fal.ai)",
}
