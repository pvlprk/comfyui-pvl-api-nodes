
from __future__ import annotations
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import numpy as np
from PIL import Image

# Try relative import first (when packaged as a module), then fallback.
try:
    from .fal_utils import ImageUtils  # type: ignore
except Exception:
    from fal_utils import ImageUtils  # type: ignore

COMFYDEPLOY_QUEUE_URL = "https://api.comfydeploy.com/api/run/deployment/queue"
COMFYDEPLOY_RUN_URL = "https://api.comfydeploy.com/api/run/{run_id}"
POLL_INTERVAL = 3.0           # seconds between polls
MAX_POLL_SECONDS = 200        # total poll duration

class PVL_ComfyDeploy_LoraPrep:
    """
    Single-image in → single-image out ComfyDeploy node.
    - Exactly one IMAGE input (named `image`).
    - Exactly one IMAGE output ([1, H, W, 3]).
    - No batch parameter (internally enforces batch=1 on the API).
    - Removed text inputs (style, prompt).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "deployment_id": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647, "step": 1}),
                "timeout": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 3600.0, "step": 1.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "queue_and_fetch_single"
    CATEGORY = "PVL/API"
    OUTPUT_NODE = False

    # -------------------------- Helpers -------------------------- #

    @staticmethod
    def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Any, str]:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        text = resp.text or ""
        try:
            data = resp.json()
        except ValueError:
            data = None
        return resp.status_code, data, text

    @staticmethod
    def _get_json(url: str, headers: Dict[str, str]) -> Tuple[int, Any, str]:
        resp = requests.get(url, headers=headers, timeout=60)
        text = resp.text or ""
        try:
            data = resp.json()
        except ValueError:
            data = None
        return resp.status_code, data, text

    @staticmethod
    def _tensor_to_data_uri(image_tensor: torch.Tensor) -> str:
        # Use the same robust conversion as fal_utils.ImageUtils.image_to_data_uri
        return ImageUtils.image_to_data_uri(image_tensor)

    @staticmethod
    def _pil_to_tensor_rgb(pil_img: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a ComfyUI IMAGE tensor: [1, H, W, 3] float in [0,1].
        """
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr)[None, ...]              # (1, H, W, 3)

    @staticmethod
    def _extract_image_urls_from_list(lst: List[Any]) -> List[str]:
        urls: List[str] = []
        if not isinstance(lst, list):
            return urls
        for it in lst:
            if isinstance(it, dict):
                u = it.get("url") or it.get("src") or it.get("signed_url")
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
        return urls

    @staticmethod
    def _extract_image_urls_from_outputs(run_json: Dict[str, Any]) -> List[str]:
        """
        Handle both ComfyDeploy list-style outputs and dict-style (FAL-like) outputs.
        """
        urls: List[str] = []
        outputs = run_json.get("outputs")
        if isinstance(outputs, list):  # ComfyDeploy style
            for out in outputs:
                if not isinstance(out, dict):
                    continue
                data = out.get("data") or {}
                urls += PVL_ComfyDeploy_LoraPrep._extract_image_urls_from_list(data.get("images") or [])
                urls += PVL_ComfyDeploy_LoraPrep._extract_image_urls_from_list(data.get("files") or [])
        elif isinstance(outputs, dict):  # FAL-like style
            for _, node_data in outputs.items():
                if isinstance(node_data, dict):
                    urls += PVL_ComfyDeploy_LoraPrep._extract_image_urls_from_list(node_data.get("images") or [])
                    urls += PVL_ComfyDeploy_LoraPrep._extract_image_urls_from_list(node_data.get("files") or [])
        return urls

    # -------------------------- Main -------------------------- #

    def queue_and_fetch_single(
        self,
        image,
        deployment_id: str,
        api_key: str,
        seed: int,
        timeout: float,
        debug: bool = False,
    ):
        t0 = time.time()

        # Default 1x1 black image
        empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        def _fail(msg: str):
            if debug:
                print(f"[PVL_ComfyDeploy_LoraPrep] {msg}")
            print(f"[PVL_ComfyDeploy_LoraPrep] Error after {time.time() - t0:.2f}s: {msg}")
            return (empty,)

        if image is None or not isinstance(image, torch.Tensor):
            return _fail("Missing image tensor")
        if not deployment_id or not api_key:
            return _fail("Missing deployment_id or api_key")

        # Convert input image to data URI using the same method as the example.
        try:
            data_uri = self._tensor_to_data_uri(image)
            if debug:
                print(f"[PVL_ComfyDeploy_LoraPrep] Converted input image to data URI ({len(data_uri)} chars)")
        except Exception as e:
            return _fail(f"Failed to convert input image: {e}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "deployment_id": deployment_id,
            "inputs": {
                "input_image": data_uri,
                "batch": 1,              # always single
                "seed": int(seed),
            },
        }

        if debug:
            print(f"[PVL_ComfyDeploy_LoraPrep] Queueing run to {COMFYDEPLOY_QUEUE_URL}")

        # Queue
        try:
            code, qdata, raw = self._post_json(COMFYDEPLOY_QUEUE_URL, headers, payload)
        except requests.RequestException as e:
            return _fail(f"Queue request failed: {e}")

        if code < 200 or code >= 300 or not isinstance(qdata, dict):
            return _fail(f"Queue error {code}: {raw[:250]}")

        run_id = qdata.get("run_id") or qdata.get("id") or qdata.get("queue_id")
        if not run_id:
            return _fail("No run_id returned from queue")

        # Poll until we have at least one image URL or terminal state
        collected_urls: List[str] = []
        latest_json: Optional[Dict[str, Any]] = None
        if debug:
            print(f"[PVL_ComfyDeploy_LoraPrep] Polling run_id={run_id}")

        while True:
            if (time.time() - t0) >= MAX_POLL_SECONDS:
                break
            try:
                code, rdata, raw = self._get_json(COMFYDEPLOY_RUN_URL.format(run_id=run_id), headers)
            except requests.RequestException as e:
                if debug:
                    print(f"[PVL_ComfyDeploy_LoraPrep] Poll error: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            if code < 200 or code >= 300 or not isinstance(rdata, dict):
                time.sleep(POLL_INTERVAL)
                continue

            latest_json = rdata

            image_urls = self._extract_image_urls_from_outputs(rdata)
            if image_urls:
                # Keep order, drop duplicates
                seen = set()
                collected_urls = [u for u in image_urls if not (u in seen or seen.add(u))]

            status = str(rdata.get("status", "")).lower()
            if debug and status:
                print(f"[PVL_ComfyDeploy_LoraPrep] Status: {status}; found {len(collected_urls)} image url(s)")
            if len(collected_urls) >= 1:
                break
            if status in {"succeeded", "completed", "finished", "failed", "error", "canceled", "cancelled"}:
                break

            time.sleep(POLL_INTERVAL)

        if not collected_urls:
            return _fail("No image URLs found in outputs")

        first_url = collected_urls[0]

        # Download first image -> tensor [1,H,W,3]
        try:
            if debug:
                print(f"[PVL_ComfyDeploy_LoraPrep] Downloading: {first_url}")
            resp = requests.get(first_url, timeout=60)
            resp.raise_for_status()
            with Image.open(io.BytesIO(resp.content)) as pil_img:
                pil_img.load()
                tensor = self._pil_to_tensor_rgb(pil_img)  # (1,H,W,3)
                if debug:
                    print(f"[PVL_ComfyDeploy_LoraPrep] Loaded image size: {pil_img.size}")
        except Exception as e:
            return _fail(f"Download failed for {first_url}: {e}")

        elapsed = time.time() - t0
        print(f"[PVL_ComfyDeploy_LoraPrep] Generated 1 image in {elapsed:.2f}s")

        return (tensor,)

# ComfyUI discovery mappings
NODE_CLASS_MAPPINGS = {
    "PVL_ComfyDeploy_LoraPrep": PVL_ComfyDeploy_LoraPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_ComfyDeploy_LoraPrep": "PVL - PVL_ComfyDeploy_LoraPrep",
}
