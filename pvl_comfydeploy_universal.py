from __future__ import annotations
import io
import time
import base64
from typing import Any, Dict, List, Tuple
import requests
import torch
import numpy as np
from PIL import Image

# Optional helper from your repo; used for exact parity with LoraPrep.
try:
    from .fal_utils import ImageUtils  # type: ignore
except Exception:
    ImageUtils = None  # We'll fall back gracefully if not present.

COMFYDEPLOY_QUEUE_URL = "https://api.comfydeploy.com/api/run/deployment/queue"
COMFYDEPLOY_RUN_URL = "https://api.comfydeploy.com/api/run/{run_id}"
POLL_INTERVAL = 3.0  # seconds between polls

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

# -------------------------- Image helpers -------------------------- #

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Accepts Comfy-style tensors:
      - [B,H,W,3] (take first) float32 0..1
      - [H,W,3] float32 0..1
      - [B,3,H,W] or [3,H,W] -> CHW -> HWC
      - [H,W] -> grayscale
    Returns RGB PIL.Image.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    tt = t.detach().cpu()

    # Drop batch if present
    if tt.dim() == 4:
        tt = tt[0]

    # CHW -> HWC
    if tt.dim() == 3 and tt.shape[0] in (1, 3):
        tt = tt.permute(1, 2, 0)

    if tt.dim() == 2:
        arr = tt.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) * 255.0
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    if tt.dim() != 3 or tt.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported tensor shape for image: {tuple(tt.shape)}")

    arr = tt.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    if arr.shape[-1] == 1:  # expand grayscale to RGB
        arr = np.repeat(arr, 3, axis=-1)

    return Image.fromarray(arr, mode="RGB")

def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def _tensor_to_data_uri_via_imageutils(t: torch.Tensor) -> str:
    """
    Exact LoraPrep parity path: prefer ImageUtils.image_to_data_uri if available.
    Fallback to internal PNG->base64.
    """
    if ImageUtils is not None and hasattr(ImageUtils, "image_to_data_uri"):
        try:
            return ImageUtils.image_to_data_uri(t)
        except Exception:
            pass  # fallback below

    pil = _tensor_to_pil(t)
    png = _pil_to_png_bytes(pil)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"

# -------------------------- JSON conversion -------------------------- #

def _to_jsonable(name: str, val: Any, upload_tensors: bool) -> Any:
    """
    Convert arbitrary values into JSON-safe types.
    For tensors/images: default to LoraPrep behavior (data URI).
    If upload_tensors=True and an uploader is available, return a URL instead.
    """
    # Primitives
    if val is None or isinstance(val, (bool, int, float, str)):
        return val

    # NumPy
    if isinstance(val, (np.bool_, np.integer, np.floating)):
        return val.item()
    if isinstance(val, np.ndarray):
        if val.size <= 64:
            return val.tolist()
        # Try treat as image
        try:
            pil = Image.fromarray(val.astype(np.uint8)) if val.dtype != np.uint8 else Image.fromarray(val)
            png = _pil_to_png_bytes(pil.convert("RGB"))
            if upload_tensors and ImageUtils is not None:
                for attr in ("upload_bytes_return_url", "upload_image_bytes"):
                    if hasattr(ImageUtils, attr):
                        try:
                            fn = getattr(ImageUtils, attr)
                            return fn(png, mime="image/png")
                        except Exception:
                            pass
            b64 = base64.b64encode(png).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return f"[{name}]_ndarray(shape={tuple(val.shape)}, dtype={val.dtype})"

    # Torch tensors
    if isinstance(val, torch.Tensor):
        # scalar -> number
        if val.dim() == 0:
            return val.item()
        # small -> list
        if val.numel() <= 64:
            return val.detach().cpu().tolist()

        # If uploads explicitly enabled and available, prefer URL upload
        if upload_tensors and ImageUtils is not None:
            try:
                pil = _tensor_to_pil(val)
                png = _pil_to_png_bytes(pil)
                if hasattr(ImageUtils, "upload_bytes_return_url"):
                    return ImageUtils.upload_bytes_return_url(png, mime="image/png")
                if hasattr(ImageUtils, "upload_image_bytes"):
                    return ImageUtils.upload_image_bytes(png, mime="image/png")
            except Exception:
                pass

        # Default/Parity: data URI via ImageUtils.image_to_data_uri (or fallback)
        try:
            return _tensor_to_data_uri_via_imageutils(val)
        except Exception:
            return f"[{name}]_tensor(shape={tuple(val.shape)}, dtype={val.dtype})"

    # Containers
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(f"{name}[{i}]", v, upload_tensors) for i, v in enumerate(val)]
    if isinstance(val, dict):
        return {str(k): _to_jsonable(f"{name}.{k}", v, upload_tensors) for k, v in val.items()}

    # Fallback
    return str(val)

# -------------------------- Node -------------------------- #

class PVL_ComfyDeploy_Universal:
    """
    Flexible ComfyDeploy node with LoraPrep-style image handling:
    - 10 (parameter_name_i, value_i) pairs. Values are OPTIONAL and ANY type.
    - If a value is an IMAGE tensor and you set the corresponding name to 'input_image',
      it will be serialized to a data URI exactly like your LoraPrep node.
    - Set upload_tensors=True to try uploading PNG bytes and pass a URL instead.
    - Collects all produced image URLs and returns a single IMAGE batch [B,H,W,3].
    """

    @classmethod
    def INPUT_TYPES(cls):
        required: Dict[str, Tuple[Any, Dict[str, Any]]] = {
            "deployment_id": ("STRING", {"default": "", "multiline": False}),
            "api_key": ("STRING", {"default": "", "multiline": False}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647, "step": 1}),
            "timeout": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 3600.0, "step": 1.0}),
            # Parity default: use data-URI, not upload
            "upload_tensors": ("BOOLEAN", {"default": False}),
            "debug": ("BOOLEAN", {"default": False}),
        }
        optional: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        for i in range(1, 11):
            required[f"parameter_name_{i}"] = ("STRING", {"default": "", "multiline": False})
            optional[f"value_{i}"] = (any, {"default": None})  # optional & connectable
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "queue_and_fetch_batch"
    CATEGORY = "PVL/API"
    OUTPUT_NODE = False

    # -------------------------- HTTP -------------------------- #

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
    def _pil_to_tensor_rgb(pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0  # (H,W,3)
        return torch.from_numpy(arr)[None, ...]              # (1,H,W,3)

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

    @classmethod
    def _extract_image_urls_from_outputs(cls, run_json: Dict[str, Any]) -> List[str]:
        """
        Supports ComfyDeploy list-style outputs and dict-style (FAL-like) outputs.
        """
        urls: List[str] = []
        outputs = run_json.get("outputs")
        if isinstance(outputs, list):  # ComfyDeploy style
            for out in outputs:
                if not isinstance(out, dict):
                    continue
                data = out.get("data") or {}
                urls += cls._extract_image_urls_from_list(data.get("images") or [])
                urls += cls._extract_image_urls_from_list(data.get("files") or [])
        elif isinstance(outputs, dict):  # FAL-like style
            for _, node_data in outputs.items():
                if isinstance(node_data, dict):
                    urls += cls._extract_image_urls_from_list(node_data.get("images") or [])
                    urls += cls._extract_image_urls_from_list(node_data.get("files") or [])
        return urls

    # -------------------------- Main -------------------------- #

    def queue_and_fetch_batch(
        self,
        deployment_id: str,
        api_key: str,
        seed: int,
        timeout: float,
        upload_tensors: bool = False,
        debug: bool = False,
        **kwargs
    ):
        t0 = time.time()
        empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        def _fail(msg: str):
            print(f"[PVL_ComfyDeploy_Universal] Error after {time.time() - t0:.2f}s: {msg}")
            return (empty,)

        if not deployment_id or not api_key:
            return _fail("Missing deployment_id or api_key")

        # Build inputs payload: add seed and all non-empty name/value pairs, JSON-serialized.
        inputs_dict: Dict[str, Any] = {"seed": int(seed)}
        for i in range(1, 11):
            name = kwargs.get(f"parameter_name_{i}")
            if isinstance(name, str) and name.strip():
                raw_val = kwargs.get(f"value_{i}")  # may be absent (optional)
                try:
                    inputs_dict[name.strip()] = _to_jsonable(name.strip(), raw_val, upload_tensors)
                except Exception as ex:
                    if debug:
                        print(f"[PVL_ComfyDeploy_Universal] JSONify failed for {name}: {ex}")
                    inputs_dict[name.strip()] = str(raw_val)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"deployment_id": deployment_id, "inputs": inputs_dict}

        if debug:
            print(f"[PVL_ComfyDeploy_Universal] Payload keys: {list(inputs_dict.keys())}")

        # Queue the job
        try:
            code, qdata, raw = self._post_json(COMFYDEPLOY_QUEUE_URL, headers, payload)
        except requests.RequestException as e:
            return _fail(f"Queue request failed: {e}")

        if code < 200 or code >= 300 or not isinstance(qdata, dict):
            return _fail(f"Queue error {code}: {str(raw)[:250]}")

        run_id = qdata.get("run_id") or qdata.get("id") or qdata.get("queue_id")
        if not run_id:
            return _fail("No run_id returned from queue")

        if debug:
            print(f"[PVL_ComfyDeploy_Universal] Polling run_id={run_id}")

        collected_urls: List[str] = []
        while True:
            if (time.time() - t0) >= float(timeout):
                break
            try:
                code, rdata, _ = self._get_json(COMFYDEPLOY_RUN_URL.format(run_id=run_id), headers)
            except requests.RequestException:
                time.sleep(POLL_INTERVAL)
                continue

            if code < 200 or code >= 300 or not isinstance(rdata, dict):
                time.sleep(POLL_INTERVAL)
                continue

            image_urls = self._extract_image_urls_from_outputs(rdata)
            if image_urls:
                collected_urls = list(dict.fromkeys(image_urls))  # unique, ordered

            status = str(rdata.get("status", "")).lower()
            if debug:
                print(f"[PVL_ComfyDeploy_Universal] Status={status} URLs={len(collected_urls)}")
            if collected_urls or status in {"succeeded", "completed", "finished", "failed", "error", "canceled", "cancelled"}:
                break

            time.sleep(POLL_INTERVAL)

        if not collected_urls:
            return _fail("No image URLs found in outputs")

        # Download all images and stack into a batch
        tensors = []
        for url in collected_urls:
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with Image.open(io.BytesIO(resp.content)) as pil_img:
                    pil_img.load()
                    tensors.append(self._pil_to_tensor_rgb(pil_img))
            except Exception as e:
                if debug:
                    print(f"[PVL_ComfyDeploy_Universal] Failed to download {url}: {e}")

        if not tensors:
            return _fail("Failed to download any images")

        batch = torch.cat(tensors, dim=0)  # [B,H,W,3]
        print(f"[PVL_ComfyDeploy_Universal] Retrieved {batch.shape[0]} image(s) in {time.time()-t0:.2f}s")
        return (batch,)

NODE_CLASS_MAPPINGS = {"PVL_ComfyDeploy_Universal": PVL_ComfyDeploy_Universal}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_ComfyDeploy_Universal": "PVL - PVL_ComfyDeploy_Universal"}
