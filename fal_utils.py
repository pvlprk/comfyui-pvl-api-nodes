
import configparser
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import torch
from PIL import Image

# Quiet noisy HTTP/client logs (“HTTP Request: GET … 202 Accepted” etc.)
for _name in ("httpx", "urllib3", "fal_client"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Try to import the official FAL Python client
try:
    from fal_client.client import SyncClient  # type: ignore
except Exception:  # pragma: no cover
    SyncClient = None  # type: ignore


@dataclass
class FalConfig:
    """
    Backward-compatible config that mirrors the old behavior (env or config.ini),
    and builds a SyncClient in a way that works across fal_client versions.
    """
    _instance: "FalConfig" = None  # type: ignore
    _client: Optional["SyncClient"] = None  # type: ignore
    _key: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_singleton"):
            cls._singleton = super().__new__(cls)
            cls._singleton._initialize()
        return cls._singleton

    def _initialize(self):
        # 1) Try environment first
        key = os.environ.get("FAL_KEY")

        # 2) Fallback to config.ini (../config.ini relative to this file)
        if not key:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            config_path = os.path.join(parent_dir, "config.ini")

            config = configparser.ConfigParser()
            config.read(config_path)
            try:
                key = config["API"]["FAL_KEY"]
                # also place into env so clients that read env directly can see it
                os.environ["FAL_KEY"] = key
            except KeyError:
                pass

        if not key or key == "<your_fal_api_key_here>":
            raise RuntimeError("FAL: API key not found. Set FAL_KEY env or config.ini [API] FAL_KEY.")

        self._key = key

    def get_client(self):
        if SyncClient is None:
            raise RuntimeError("FAL: 'fal_client' package not installed. pip install fal-client")
        if self._client is None:
            # Old fal_client expects key=..., newer may accept credentials=...
            try:
                self._client = SyncClient(key=self._key)  # old working signature
            except TypeError:
                try:
                    self._client = SyncClient(credentials=self._key)  # newer signature
                except TypeError:
                    # Ultimate fallback: rely on env var only
                    self._client = SyncClient()
        return self._client

    def get_key(self):
        return self._key


class ImageUtils:
    @staticmethod
    def tensor_to_pil(image):
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.dtype in [np.float32, np.float64]:
                image_np = (image_np * 255).astype(np.uint8)

            return Image.fromarray(image_np).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"FAL: error converting tensor to PIL: {str(e)}")

    @staticmethod
    def upload_image(image):
        """
        Preserve old, proven behavior: write PNG to temp file and use client.upload_file().
        (Reliable across fal_client versions and avoids oversized data URIs.)
        """
        pil_image = ImageUtils.tensor_to_pil(image)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name
            client = FalConfig().get_client()
            image_url = client.upload_file(temp_file_path)
            if not image_url:
                raise RuntimeError("FAL: upload_file returned no URL.")
            return image_url
        except Exception as e:
            raise RuntimeError(f"FAL: error uploading image: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

    @staticmethod
    def mask_to_image(mask):
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result


class ResultProcessor:
    @staticmethod
    def process_image_result(result):
        if "images" not in result or not isinstance(result["images"], list) or not result["images"]:
            raise RuntimeError("FAL: response contained no images.")
        images = []
        try:
            for img_info in result["images"]:
                img_url = img_info.get("url")
                if not img_url:
                    raise RuntimeError("FAL: image entry missing URL.")
                img_response = requests.get(img_url, timeout=120)
                img_response.raise_for_status()
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)
        except Exception as e:
            raise RuntimeError(f"FAL: error downloading/decoding image(s): {str(e)}")

        stacked_images = np.stack(images, axis=0)  # (B, H, W, C)
        img_tensor = torch.from_numpy(stacked_images)
        return (img_tensor,)

    @staticmethod
    def create_blank_image():
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (img_tensor,)


class ApiHandler:
    @staticmethod
    def _print_queue_update(update: Dict):
        # Pretty, safe prints for queue state
        status = (update or {}).get("status")
        if status == "IN_QUEUE":
            pos = update.get("queue_position")
            total = update.get("queue_size")
            if pos is not None:
                print(f"[FAL] queued… position {pos}" + (f" of {total}" if total is not None else ""))
            else:
                print("[FAL] queued…")
        elif status == "IN_PROGRESS":
            logs = update.get("logs") or []
            for log in logs:
                msg = (log or {}).get("message")
                if msg:
                    print(f"[FAL] {msg}")
        elif status == "COMPLETED":
            print("[FAL] completed")
        elif status == "FAILED":
            print("[FAL] failed")

    @staticmethod
    def submit_and_get_result(endpoint, arguments):
        client = FalConfig().get_client()
        try:
            # Try modern submit with streaming logs + queue updates
            handler = client.submit(
                endpoint,
                arguments=arguments,
                logs=True,
                on_queue_update=ApiHandler._print_queue_update,
            )
        except TypeError:
            # Fallback for older clients: no logs/callback args
            handler = client.submit(endpoint, arguments=arguments)

        try:
            return handler.get()
        except Exception as e:
            raise RuntimeError(f"FAL: request failed for {endpoint}: {str(e)}")

    @staticmethod
    def handle_image_generation_error(model_name, error):
        # Keep, but raise instead of returning a blank image
        raise RuntimeError(f"FAL: error generating image with {model_name}: {str(error)}")
