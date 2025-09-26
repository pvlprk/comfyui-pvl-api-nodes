import torch
import numpy as np

from .fal_utils import ImageUtils, ApiHandler


class PVL_fal_DepthAnythingV2_API:
    """
    ComfyUI node for FAL 'fal-ai/image-preprocessors/depth-anything/v2' — Depth Map Estimation.

    Inputs:
      - image (IMAGE): One input image (batch NOT supported).

    Outputs:
      - IMAGE: Depth map image tensor (B,H,W,C).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "depth_anything"
    CATEGORY = "PVL_tools"

    # ------------------------- helpers -------------------------
    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image):
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
    def depth_anything(self, image):
        # Split and validate
        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")

        if len(frames) > 1:
            self._raise("FAL: batch >1 not supported for depth-anything API.")

        # Upload
        image_url = ImageUtils.image_to_data_uri(frames[0])
        if not image_url:
            self._raise("FAL: failed to upload input image.")

        arguments = {
            "image_url": image_url,
        }

        # Submit request
        result = ApiHandler.submit_and_get_result(
            "fal-ai/image-preprocessors/depth-anything/v2",
            arguments,
        )

        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")

        if "image" not in result or not isinstance(result["image"], dict):
            self._raise("FAL: response missing depth map image.")

        out_url = result["image"].get("url")
        if not out_url:
            self._raise("FAL: depth map image has no URL.")

        # Download result
        import requests, io
        from PIL import Image
        resp = requests.get(out_url, timeout=120)
        resp.raise_for_status()
        pil_out = Image.open(io.BytesIO(resp.content)).convert("RGB")

        out_arr = np.array(pil_out).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_arr).unsqueeze(0)  # (1,H,W,C)

        return (out_tensor,)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_fal_DepthAnythingV2_API": PVL_fal_DepthAnythingV2_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_DepthAnythingV2_API": "PVL Depth Anything V2 (fal.ai)",
}
