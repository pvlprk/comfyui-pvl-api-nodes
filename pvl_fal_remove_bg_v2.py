import torch
import numpy as np

from .fal_utils import ImageUtils, ApiHandler


class PVL_fal_RemoveBackground_API:
    """
    ComfyUI node for FAL 'fal-ai/birefnet/v2' — Remove Background V2.

    Inputs:
      - image (IMAGE): One input image (batch NOT supported).
      - model (CHOICE): Which model variant to use.
      - operating_resolution (CHOICE): Resolution for inference ("1024x1024" or "2048x2048").
      - output_format (CHOICE): "png" or "webp".
      - output_mask (BOOLEAN): Whether to also return the mask.
      - refine_foreground (BOOLEAN): Whether to refine the foreground (default: True).

    Outputs:
      - IMAGE: Foreground with background removed (RGB or RGBA if provided).
      - MASK: Optional mask (1-channel, float, shape [1,H,W]) if output_mask=True.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "General Use (Light)",
                        "General Use (Light 2K)",
                        "General Use (Heavy)",
                        "Matting",
                        "Portrait",
                    ],
                    {"default": "General Use (Light)"},
                ),
                "operating_resolution": (
                    ["1024x1024", "2048x2048"],
                    {"default": "1024x1024"},
                ),
                "output_format": (
                    ["png", "webp"],
                    {"default": "png"},
                ),
                "output_mask": ("BOOLEAN", {"default": False}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "PVL_tools"

    # ------------------------- helpers -------------------------
    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image):
        import numpy as np
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

    def _download_pil(self, url, mode=None):
        import requests, io
        from PIL import Image
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        pil = Image.open(io.BytesIO(resp.content))
        if mode is not None:
            pil = pil.convert(mode)
        return pil

    # ------------------------- main -------------------------
    def remove_background(
        self,
        image,
        model,
        operating_resolution,
        output_format,
        output_mask,
        refine_foreground,
    ):
        # Split batch into frames
        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")
        if len(frames) > 1:
            self._raise("FAL: batch >1 not supported for Remove Background API.")

        # Inline image as base64 data URI (avoids storage upload)
        image_url = ImageUtils.image_to_data_uri(frames[0])
        if not image_url:
            self._raise("FAL: failed to convert input image.")

        arguments = {
            "image_url": image_url,
            "model": model,
            "operating_resolution": operating_resolution,
            "output_format": output_format,
            "output_mask": bool(output_mask),
            "refine_foreground": bool(refine_foreground),
        }

        # Submit request
        result = ApiHandler.submit_and_get_result("fal-ai/birefnet/v2", arguments)
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")

        # ---- Foreground image (keep alpha if present) ----
        if "image" not in result or not isinstance(result["image"], dict):
            self._raise("FAL: response missing foreground image.")
        fg_url = result["image"].get("url")
        if not fg_url:
            self._raise("FAL: foreground image has no URL.")

        pil_fg = self._download_pil(fg_url, mode=None)  # keep native mode
        fg_arr = np.array(pil_fg).astype(np.float32) / 255.0  # (H,W,3) or (H,W,4)
        if fg_arr.ndim == 2:  # grayscale fallback
            fg_arr = np.expand_dims(fg_arr, axis=-1)
        fg_tensor = torch.from_numpy(fg_arr).unsqueeze(0)  # (1,H,W,C)

        # ---- Mask (shape [1,H,W], float 0..1) ----
        mask_tensor = torch.zeros((1, fg_arr.shape[0], fg_arr.shape[1]), dtype=torch.float32)

        if bool(output_mask):
            mask_url = None
            if isinstance(result.get("mask_image"), dict):
                mask_url = result["mask_image"].get("url")

            if mask_url:
                pil_mask_rgba = self._download_pil(mask_url, mode=None)  # keep native
                if "A" in pil_mask_rgba.getbands():
                    alpha = pil_mask_rgba.getchannel("A")
                    mask_arr = np.array(alpha).astype(np.float32) / 255.0  # (H,W)
                else:
                    pil_mask_L = pil_mask_rgba.convert("L")
                    mask_arr = np.array(pil_mask_L).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # (1,H,W)
            else:
                # fallback: try to read alpha channel from foreground if present
                if "A" in pil_fg.getbands():
                    alpha = pil_fg.getchannel("A")
                    mask_arr = np.array(alpha).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return (fg_tensor, mask_tensor)

# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_fal_RemoveBackground_API": PVL_fal_RemoveBackground_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_RemoveBackground_API": "PVL Remove Background V2 (fal.ai)",
}
