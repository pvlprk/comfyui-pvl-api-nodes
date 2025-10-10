
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
MAX_POLL_SECONDS = 150        # total poll duration

class PVL_Comfydeploy_Avatar_PulID_API:
    """
    Queue a ComfyDeploy run, poll for results, download images, and return
    them as a ComfyUI IMAGE tensor—using the SAME image handling patterns as
    the FAL example node (tensor->dataURI for input, PIL->np->tensor for output).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "deployment_id": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "batch": ("INT", {"default": 1, "min": 0, "max": 999999, "step": 1}),
                "style": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647, "step": 1}),
                # kept for graph compatibility
                "timeout": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 3600.0, "step": 1.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "queue_and_fetch"
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
        Mirrors the example node's PIL->np->tensor pathway.
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
                urls += PVL_Comfydeploy_Avatar_PulID_API._extract_image_urls_from_list(data.get("images") or [])
                urls += PVL_Comfydeploy_Avatar_PulID_API._extract_image_urls_from_list(data.get("files") or [])
        elif isinstance(outputs, dict):  # FAL-like style
            for node_id, node_data in outputs.items():
                if isinstance(node_data, dict):
                    urls += PVL_Comfydeploy_Avatar_PulID_API._extract_image_urls_from_list(node_data.get("images") or [])
                    urls += PVL_Comfydeploy_Avatar_PulID_API._extract_image_urls_from_list(node_data.get("files") or [])
        return urls

    # -------------------------- Main -------------------------- #

    def queue_and_fetch(
        self,
        ref_image,
        deployment_id: str,
        api_key: str,
        batch: int,
        style: str,
        prompt: str,
        seed: int,
        timeout: float,
        debug: bool = False,
    ):
        t0 = time.time()

        # Default 1x1 black image (tuple return, like the example node)
        empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        def _done_failed(msg: str):
            if debug:
                print(f"[PVL_ComfyDeploy_Avatar] {msg}")
            print(f"[PVL_ComfyDeploy_Avatar] Error after {time.time() - t0:.2f}s: {msg}")
            return (empty,)

        if ref_image is None or not isinstance(ref_image, torch.Tensor):
            return _done_failed("Missing ref_image tensor")
        if not deployment_id or not api_key:
            return _done_failed("Missing deployment_id or api_key")

        # Convert input image to data URI using the same method as the FAL example.
        try:
            data_uri = self._tensor_to_data_uri(ref_image)
            if debug:
                print(f"[PVL_ComfyDeploy_Avatar] Converted ref image to data URI ({len(data_uri)} chars)")
        except Exception as e:
            return _done_failed(f"Failed to convert input image: {e}")

        expected_images = int(batch) if int(batch) > 0 else 1

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "deployment_id": deployment_id,
            "inputs": {
                "input_image": data_uri,
                "batch": int(batch),
                "style": style or "",
                "prompt": prompt or "",
                "seed": int(seed),
            },
        }

        if debug:
            print(f"[PVL_ComfyDeploy_Avatar] Queueing run to {COMFYDEPLOY_QUEUE_URL}")

        # Queue
        try:
            code, qdata, raw = self._post_json(COMFYDEPLOY_QUEUE_URL, headers, payload)
        except requests.RequestException as e:
            return _done_failed(f"Queue request failed: {e}")

        if code < 200 or code >= 300 or not isinstance(qdata, dict):
            return _done_failed(f"Queue error {code}: {raw[:250]}")

        run_id = qdata.get("run_id") or qdata.get("id") or qdata.get("queue_id")
        if not run_id:
            return _done_failed("No run_id returned from queue")

        # Poll
        collected_urls: List[str] = []
        latest_json: Optional[Dict[str, Any]] = None
        if debug:
            print(f"[PVL_ComfyDeploy_Avatar] Polling run_id={run_id}")

        while True:
            if (time.time() - t0) >= MAX_POLL_SECONDS:
                break
            try:
                code, rdata, raw = self._get_json(COMFYDEPLOY_RUN_URL.format(run_id=run_id), headers)
            except requests.RequestException as e:
                if debug:
                    print(f"[PVL_ComfyDeploy_Avatar] Poll error: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            if code < 200 or code >= 300 or not isinstance(rdata, dict):
                time.sleep(POLL_INTERVAL)
                continue

            latest_json = rdata

            image_urls = self._extract_image_urls_from_outputs(rdata)
            if image_urls:
                # keep order, drop duplicates
                seen = set()
                collected_urls = [u for u in image_urls if not (u in seen or seen.add(u))]

            status = str(rdata.get("status", "")).lower()
            if debug and status:
                print(f"[PVL_ComfyDeploy_Avatar] Status: {status}; found {len(collected_urls)} image url(s)")
            if len(collected_urls) >= expected_images:
                break
            if status in {"succeeded", "completed", "finished", "failed", "error", "canceled", "cancelled"}:
                break

            time.sleep(POLL_INTERVAL)

        if not collected_urls:
            return _done_failed("No image URLs found in outputs")

        collected_urls = collected_urls[:expected_images]

        # Download using the same pathway as the example: requests -> PIL -> np/255 -> tensor
        tensors: List[torch.Tensor] = []
        for url in collected_urls:
            try:
                if debug:
                    print(f"[PVL_ComfyDeploy_Avatar] Downloading: {url}")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with Image.open(io.BytesIO(resp.content)) as pil_img:
                    pil_img.load()
                    tensor = self._pil_to_tensor_rgb(pil_img)
                    tensors.append(tensor)
                    if debug:
                        print(f"[PVL_ComfyDeploy_Avatar] Loaded image size: {pil_img.size}")
            except Exception as e:
                print(f"[PVL_ComfyDeploy_Avatar] Download failed for {url}: {e}")

        if not tensors:
            return _done_failed("Images failed to download/convert")

        # Stack along batch dimension -> (B,H,W,3); return as tuple, like the example node.
        try:
            out = torch.cat(tensors, dim=0)
        except Exception:
            # Fallback to first if shapes mismatch
            out = tensors[0]

        elapsed = time.time() - t0
        print(f"[PVL_ComfyDeploy_Avatar] Successfully generated {out.shape[0]} image(s) in {elapsed:.2f}s")

        return (out,)

# ComfyUI discovery mappings
NODE_CLASS_MAPPINGS = {
    "PVL_Comfydeploy_Avatar_PulID_API": PVL_Comfydeploy_Avatar_PulID_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_Comfydeploy_Avatar_PulID_API": "PVL — ComfyDeploy Avatar PulID API",
}
