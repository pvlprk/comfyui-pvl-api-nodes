
import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
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
    """Holds configuration and helpers for fal.ai client/module access."""
    api_key_env_vars: Tuple[str, ...] = ("FAL_KEY", "FAL_API_KEY", "FAL_CLIENT_KEY")

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        for key in cls.api_key_env_vars:
            val = os.environ.get(key)
            if val:
                return val
        return None

    @classmethod
    def get_module(cls):
        """Return the 'fal_client' module, configured with API key if possible."""
        try:
            import fal_client  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FAL: 'fal_client' package is not installed. Please 'pip install fal-client'."
            ) from e

        api_key = cls.get_api_key()
        if api_key:
            # Newer fal-client reads it from env; some expose top-level api_key attr
            try:
                fal_client.api_key = api_key  # type: ignore[attr-defined]
            except Exception:
                os.environ["FAL_KEY"] = api_key
        return fal_client

    @classmethod
    def make_sync_client(cls):
        """Instantiate fal_client.SyncClient() if available, with API key configured."""
        fal_client = cls.get_module()
        SyncClient = getattr(fal_client, "SyncClient", None)
        if SyncClient is None:
            return None
        try:
            return SyncClient()
        except Exception:
            # Some versions require api_key arg explicitly
            api_key = cls.get_api_key()
            if api_key:
                try:
                    return SyncClient(api_key=api_key)
                except Exception:
                    pass
        return None


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
        try:
            if not isinstance(image_tensor, torch.Tensor):
                raise RuntimeError("expected torch.Tensor")

            img = image_tensor.detach().cpu()

            # Handle 4D batches: pick first sample
            if img.ndim == 4:
                # (B,H,W,C) or (B,C,H,W)
                img = img[0]

            # Handle 2D grayscale (H,W) -> (H,W,1)
            if img.ndim == 2:
                img = img.unsqueeze(-1)  # (H,W,1)

            if img.ndim != 3:
                raise RuntimeError(f"image tensor must be 2D/3D/4D, got shape {tuple(img.shape)}.")

            # If CHW -> permute to HWC
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = img.permute(1, 2, 0)  # (H,W,C)

            # Validate/force HWC
            if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
                # If still ambiguous but channels at end look valid, assume HWC
                if img.shape[0] not in (1, 3, 4) and img.shape[2] in (1, 3, 4):
                    pass  # already HWC
                elif img.shape[0] in (1, 3, 4):
                    img = img.permute(1, 2, 0)
                else:
                    raise RuntimeError(f"unrecognized tensor layout {tuple(img.shape)} (expected HWC or CHW).")

            # Normalize dtype -> uint8
            if img.dtype in (torch.float16, torch.float32, torch.float64):
                # Clamp to [0,1]
                img = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
            elif img.dtype != torch.uint8:
                img = img.to(torch.uint8)

            np_img = img.numpy()

            # Expand or drop channels to RGB
            if np_img.shape[2] == 1:
                np_img = np.repeat(np_img, 3, axis=2)
            elif np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]  # drop alpha

            return Image.fromarray(np_img, mode="RGB")
        except Exception as e:
            raise RuntimeError(f"FAL: error converting tensor to PIL: {str(e)}")

    @staticmethod
    def upload_image(image_tensor: torch.Tensor) -> str:
        """
        Save to a temporary PNG and upload via fal_client upload method.
        Returns an HTTPS URL that can be passed to the FAL model.
        Supports both module-level and SyncClient instance APIs.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pil_image.save(tmp_path, format="PNG")
            fal_module = FalConfig.get_module()
            url: Optional[str] = None

            # 1) module-level upload_file
            uploader = getattr(fal_module, "upload_file", None)
            if callable(uploader):
                uploaded = uploader(tmp_path)
                url = (uploaded.get("url") if isinstance(uploaded, dict) else str(uploaded))  # type: ignore[assignment]

            # 2) instance method upload_file, e.g., SyncClient.upload_file
            if not url:
                sc = FalConfig.make_sync_client()
                if sc is not None and hasattr(sc, "upload_file"):
                    uploaded = sc.upload_file(tmp_path)
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
            # Common forms:
            #   {"url": "https://..."} or {"url": "data:image/png;base64,..."} or {"content": "data:..."}
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
    def _submit_module_style(fal_module: Any, model_id: str, arguments: Dict) -> Optional[Dict]:
        """
        Path A: module-level API where submit() returns a handler with .get(),
        or a dict containing request_id. Attempts both.
        """
        submit_fn = getattr(fal_module, "submit", None)
        if not callable(submit_fn):
            return None

        handler = submit_fn(model_id, arguments=arguments)

        # 1) If handler has .get(), prefer that.
        if hasattr(handler, "get") and callable(getattr(handler, "get")):
            try:
                res = handler.get()
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

        # 2) If handler is dict with request_id, try result(request_id)
        req_id = None
        if isinstance(handler, dict):
            req_id = handler.get("request_id") or handler.get("id") or handler.get("task_id")

        if req_id:
            result_fn = getattr(fal_module, "result", None)
            if callable(result_fn):
                try:
                    res = result_fn(req_id)
                    if isinstance(res, dict):
                        return res
                except TypeError:
                    # In some versions result() is a SyncClient method (bound), not module-level.
                    pass

            # Try SyncClient instance as a fallback
            sc = FalConfig.make_sync_client()
            if sc is not None and hasattr(sc, "result"):
                res = sc.result(req_id)
                if isinstance(res, dict):
                    return res

        return None

    @staticmethod
    def _submit_instance_style(fal_module: Any, model_id: str, arguments: Dict) -> Optional[Dict]:
        """
        Path B: instance API using SyncClient() with submit()/result() methods.
        """
        sc = FalConfig.make_sync_client()
        if sc is None:
            return None
        submit_m = getattr(sc, "submit", None)
        if not callable(submit_m):
            return None

        handler = submit_m(model_id, arguments=arguments)

        # 1) handler.get()
        if hasattr(handler, "get") and callable(getattr(handler, "get")):
            try:
                res = handler.get()
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

        # 2) handler dict -> result(req_id)
        req_id = None
        if isinstance(handler, dict):
            req_id = handler.get("request_id") or handler.get("id") or handler.get("task_id")
        elif hasattr(handler, "request_id"):
            req_id = getattr(handler, "request_id")

        if req_id:
            res = sc.result(req_id)
            if isinstance(res, dict):
                return res

        return None

    @staticmethod
    def _subscribe_stream(fal_module: Any, model_id: str, arguments: Dict) -> Optional[Dict]:
        """
        Streaming fallback: iterate subscribe() either from module-level or SyncClient instance.
        """
        # Module-level subscribe
        subscribe_fn = getattr(fal_module, "subscribe", None)
        if callable(subscribe_fn):
            try:
                final = None
                for event in subscribe_fn(model_id, arguments=arguments):
                    if isinstance(event, dict):
                        if event.get("type") == "result" and isinstance(event.get("result"), dict):
                            final = event["result"]
                        elif event.get("event") in ("completed", "result") and isinstance(event.get("data"), dict):
                            final = event["data"]
                if isinstance(final, dict):
                    return final
            except Exception:
                pass

        # Instance subscribe
        sc = FalConfig.make_sync_client()
        if sc is not None and hasattr(sc, "subscribe"):
            try:
                final = None
                for event in sc.subscribe(model_id, arguments=arguments):
                    if isinstance(event, dict):
                        if event.get("type") == "result" and isinstance(event.get("result"), dict):
                            final = event["result"]
                        elif event.get("event") in ("completed", "result") and isinstance(event.get("data"), dict):
                            final = event["data"]
                if isinstance(final, dict):
                    return final
            except Exception:
                pass

        return None

    @staticmethod
    def submit_and_get_result(model_id: str, arguments: Dict) -> Dict:
        """
        Submits a request and waits synchronously for the final JSON result.
        Compatible with multiple fal_client versions (module-level and SyncClient).
        """
        fal_module = FalConfig.get_module()

        # Try module-level submit flows
        res = ApiHandler._submit_module_style(fal_module, model_id, arguments)
        if isinstance(res, dict):
            return res

        # Try instance-based flow
        res = ApiHandler._submit_instance_style(fal_module, model_id, arguments)
        if isinstance(res, dict):
            return res

        # Fallback to streaming
        res = ApiHandler._subscribe_stream(fal_module, model_id, arguments)
        if isinstance(res, dict):
            return res

        raise RuntimeError("FAL: submit failed: no compatible fal_client interface found or call did not return a result.")
