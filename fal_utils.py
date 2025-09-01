
import base64
import configparser
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image

# Quiet noisy HTTP/client logs
for _name in ("httpx", "urllib3", "fal_client"):
    try:
        logging.getLogger(_name).setLevel(logging.WARNING)
    except Exception:
        pass


@dataclass
class FalConfig:
    """Holds configuration and lazy client creation for fal.ai."""
    api_key_env_vars: Tuple[str, ...] = ("FAL_KEY", "FAL_API_KEY", "FAL_CLIENT_KEY")

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        for key in cls.api_key_env_vars:
            val = os.environ.get(key)
            if val:
                return val
        return None

    @classmethod
    def get_client(cls):
        """Return fal_client module if installed, configured with API key if present."""
        try:
            import fal_client  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FAL: 'fal_client' package is not installed. Please 'pip install fal-client'."
            ) from e

        api_key = cls.get_api_key()
        if api_key:
            try:
                fal_client.api_key = api_key  # newer versions
            except Exception:
                os.environ["FAL_KEY"] = api_key
        return fal_client


class ImageUtils:
    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a (H,W,C) float[0..1] or (C,H,W) float[0..1] tensor into RGB PIL.Image.
        Also accepts uint8.
        """
        try:
            if not isinstance(image_tensor, torch.Tensor):
                raise RuntimeError("expected torch.Tensor")

            img = image_tensor.detach().cpu()
            if img.ndim != 3:
                raise RuntimeError("image tensor must be 3D (HWC or CHW).")

            # Convert CHW -> HWC
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = img.permute(1, 2, 0)

            # Normalize to uint8
            if img.dtype in (torch.float16, torch.float32, torch.float64):
                img = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
            elif img.dtype != torch.uint8:
                img = img.to(torch.uint8)

            np_img = img.numpy()
            if np_img.shape[2] == 1:
                np_img = np.repeat(np_img, 3, axis=2)
            elif np_img.shape[2] == 4:
                # drop alpha (FAL generally expects 3 channels)
                np_img = np_img[:, :, :3]

            return Image.fromarray(np_img, mode="RGB")
        except Exception as e:
            raise RuntimeError(f"FAL: error converting tensor to PIL: {str(e)}")

    @staticmethod
    def upload_image(image_tensor: torch.Tensor) -> str:
        """
        Save to a temporary PNG and upload via fal_client.upload_file().
        Returns an HTTPS URL that can be passed to the FAL model.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pil_image.save(tmp_path, format="PNG")
            fal_client = FalConfig.get_client()
            uploaded = fal_client.upload_file(tmp_path)
            # fal_client returns either a dict or a URL string depending on version
            if isinstance(uploaded, dict):
                url = uploaded.get("url") or uploaded.get("file_url") or uploaded.get("signed_url")
            else:
                url = str(uploaded)
            if not url or not isinstance(url, str):
                raise RuntimeError("FAL: upload_file returned no URL.")
            return url
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class ResultProcessor:
    @staticmethod
    def _pil_from_data_uri(data_uri: str) -> Image.Image:
        """
        Decode a base64 data URI and return a PIL.Image (RGB).
        """
        try:
            header, b64 = data_uri.split(",", 1)
            if ";base64" not in header:
                raise RuntimeError("non-base64 data URI not supported")
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"FAL: error decoding data URI image: {str(e)}")

    @staticmethod
    def process_image_result(result: Dict) -> Tuple[torch.Tensor]:
        """
        Parse result['images'] which may contain http(s) URLs or data: URIs.
        Returns a tuple with a single torch tensor of shape (B, H, W, C) in [0,1].
        """
        if "images" not in result or not isinstance(result["images"], list) or not result["images"]:
            raise RuntimeError("FAL: response contained no images.")

        images_np = []
        for img_info in result["images"]:
            # Most FAL models return items like {"url": "...", "content_type": "..."}.
            # Some return {"content": "data:image/png;base64,..."} in sync mode.
            img_url = img_info.get("url")
            content = img_info.get("content")

            try:
                if isinstance(img_url, str) and img_url.startswith("data:"):
                    pil = ResultProcessor._pil_from_data_uri(img_url)
                elif isinstance(content, str) and content.startswith("data:"):
                    pil = ResultProcessor._pil_from_data_uri(content)
                elif isinstance(img_url, str):
                    parts = urlsplit(img_url)
                    if parts.scheme in ("http", "https"):
                        resp = requests.get(img_url, timeout=120)
                        resp.raise_for_status()
                        pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    else:
                        raise RuntimeError(f"unsupported image URL scheme: {parts.scheme}")
                else:
                    raise RuntimeError("image entry missing URL/content")
            except Exception as e:
                raise RuntimeError(f"FAL: error downloading/decoding image: {str(e)}")

            arr = np.array(pil).astype(np.float32) / 255.0
            images_np.append(arr)

        stacked = np.stack(images_np, axis=0)  # (B, H, W, C)
        tensor = torch.from_numpy(stacked)
        return (tensor,)


class ApiHandler:
    @staticmethod
    def submit_and_get_result(model_id: str, arguments: Dict) -> Dict:
        """
        Submits a request and waits synchronously for the final JSON result.
        Prefers fal_client if installed. Raises RuntimeError on failure.
        """
        fal_client = FalConfig.get_client()

        # Try the common submit/result pair
        try:
            task = fal_client.submit(model_id, arguments=arguments)
            # Some versions return a dict with 'request_id'; both forms are supported
            try:
                res = fal_client.result(task)
            except Exception:
                req_id = task.get("request_id") if isinstance(task, dict) else task
                res = fal_client.result(req_id)
            if not isinstance(res, dict):
                raise RuntimeError("unexpected result type")
            return res
        except Exception as e_submit:
            # Fallback: try subscribe (streaming) and capture the final result
            try:
                final = None
                for event in fal_client.subscribe(model_id, arguments=arguments):
                    # event may be {'type': 'result', 'result': {...}} or {'event': 'completed', 'data': {...}}
                    if isinstance(event, dict):
                        if event.get("type") == "result" and isinstance(event.get("result"), dict):
                            final = event["result"]
                        elif event.get("event") in ("completed", "result") and isinstance(event.get("data"), dict):
                            final = event["data"]
                if isinstance(final, dict):
                    return final
            except Exception:
                pass

            raise RuntimeError(f"FAL: submit failed: {str(e_submit)}")
