import base64
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image


# -------------------------
# FalConfig: API key lookup
# -------------------------
@dataclass
class FalConfig:
    """Holds configuration and helpers for fal.ai REST API access."""
    api_key_env_vars: Tuple[str, ...] = ("FAL_KEY", "FAL_API_KEY", "FAL_CLIENT_KEY")

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        for key in cls.api_key_env_vars:
            val = os.environ.get(key)
            if val:
                return val
        return None


# -------------------------
# Image utilities
# -------------------------
class ImageUtils:
    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a torch Tensor into an RGB PIL.Image.
        Accepts:
          - 4D: (B,H,W,C) or (B,C,H,W) -> uses the first frame
          - 3D: (H,W,C) or (C,H,W)
          - 2D: (H,W) -> expands to (H,W,1)
        Dtypes: float (0..1) or uint8.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError("expected torch.Tensor")

        img = image_tensor.detach().cpu()

        # Handle 4D batches: pick first sample
        if img.ndim == 4:
            img = img[0]

        # Handle 2D grayscale
        if img.ndim == 2:
            img = img.unsqueeze(-1)  # (H,W,1)

        if img.ndim != 3:
            raise RuntimeError(f"image tensor must be 2D/3D/4D, got shape {tuple(img.shape)}.")

        # If CHW -> permute to HWC
        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = img.permute(1, 2, 0)

        # Normalize dtype -> uint8
        if img.dtype in (torch.float16, torch.float32, torch.float64):
            img = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        np_img = img.numpy()

        # Expand/drop channels to RGB
        if np_img.shape[2] == 1:
            np_img = np.repeat(np_img, 3, axis=2)
        elif np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]

        return Image.fromarray(np_img, mode="RGB")

    @staticmethod
    def upload_image(image_tensor: torch.Tensor) -> str:
        """
        Save a tensor as PNG and upload to fal.storage.
        Returns a URL usable in API requests.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pil_image.save(tmp_path, format="PNG")

            fal_key = FalConfig.get_api_key()
            if not fal_key:
                raise RuntimeError("FAL_KEY environment variable not set")

            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    "https://fal.run/storage/upload",
                    headers={"Authorization": f"Key {fal_key}"},
                    files={"file": f},
                    timeout=120,
                )

            if resp.status_code != 200:
                raise RuntimeError(f"FAL upload failed: {resp.status_code} {resp.text}")

            data = resp.json()
            url = data.get("url") or data.get("file_url") or data.get("signed_url")
            if not url:
                raise RuntimeError("FAL: upload returned no URL")

            return url
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def image_to_data_uri(image_tensor: torch.Tensor) -> str:
        """
        Convert a torch.Tensor image into a Base64 PNG data URI.
        Useful for FAL endpoints that accept inline images instead of URLs.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# -------------------------
# Result processor
# -------------------------
class ResultProcessor:
    @staticmethod
    def _pil_from_data_uri(data_uri: str) -> Image.Image:
        header, b64 = data_uri.split(",", 1)
        if ";base64" not in header:
            raise RuntimeError("non-base64 data URI not supported")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def process_image_result(result: Dict) -> Tuple[torch.Tensor]:
        if "images" not in result or not isinstance(result["images"], list) or not result["images"]:
            raise RuntimeError("FAL: response contained no images.")

        images_np = []
        for img_info in result["images"]:
            img_url = img_info.get("url")
            content = img_info.get("content")

            if isinstance(img_url, str) and img_url.startswith("data:"):
                pil = ResultProcessor._pil_from_data_uri(img_url)
            elif isinstance(content, str) and content.startswith("data:"):
                pil = ResultProcessor._pil_from_data_uri(content)
            elif isinstance(img_url, str):
                resp = requests.get(img_url, timeout=120)
                resp.raise_for_status()
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                raise RuntimeError("FAL: image entry missing URL/content")

            arr = np.array(pil).astype(np.float32) / 255.0
            images_np.append(arr)

        stacked = np.stack(images_np, axis=0)  # (B, H, W, C)
        tensor = torch.from_numpy(stacked)
        return (tensor,)


# -------------------------
# API handler
# -------------------------
class ApiHandler:
    @staticmethod
    def submit_and_get_result(model_id: str, arguments: Dict) -> Dict:
        """
        Submit request to fal.run REST API and return JSON result.
        """
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        url = f"https://fal.run/{model_id}"

        resp = requests.post(
            url,
            headers={
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json"
            },
            json=arguments,
            timeout=300,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"FAL API error {resp.status_code}: {resp.text}")

        return resp.json()
