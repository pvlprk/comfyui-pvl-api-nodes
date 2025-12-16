import base64
import io
import torch
import numpy as np
from PIL import Image

from .fal_utils import ApiHandler, ResultProcessor


class PVL_fal_FluxPro_Fill_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_image"
    CATEGORY = "PVL_tools"

    # ------------------------------------------------------------
    # IMAGE → BASE64
    # ------------------------------------------------------------
    def _image_to_base64(self, image_tensor: torch.Tensor) -> str:
        img = image_tensor.detach().cpu()

        if img.ndim == 4:
            img = img[0]

        img = img.clamp(0, 1)
        img_np = (img.numpy() * 255).astype(np.uint8)

        pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------
    # MASK → BASE64 (GRAYSCALE)
    # ------------------------------------------------------------
    def _mask_to_base64(self, mask_tensor: torch.Tensor) -> str:
        m = mask_tensor.detach().cpu()

        # Accept: (H,W), (1,H,W), (B,1,H,W)
        while m.ndim > 2:
            m = m[0]

        m = m.float().clamp(0, 1)

        mask_np = (m.numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(mask_np, mode="L")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def fill_image(
        self,
        prompt,
        image,
        mask,
        seed,
        num_images,
        output_format,
        sync_mode,
        safety_tolerance,
    ):
        image_b64 = self._image_to_base64(image)
        mask_b64 = self._mask_to_base64(mask)

        args = {
            "prompt": prompt,
            "image_url": image_b64,
            "mask_url": mask_b64,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance,
        }

        if seed != -1:
            args["seed"] = seed

        result = ApiHandler.submit_and_get_result(
            "fal-ai/flux-pro/v1/fill",
            args,
        )

        if not result or "images" not in result or not result["images"]:
            raise RuntimeError(f"FAL: no images returned ({result})")

        processed = ResultProcessor.process_image_result(result)
        return processed
