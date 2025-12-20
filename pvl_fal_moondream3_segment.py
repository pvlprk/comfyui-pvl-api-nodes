import json
import time
import io
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from PIL import Image

from .fal_utils import ImageUtils


class PVL_fal_Moondream3_Segment_API:
    """
    ComfyUI node for FAL 'fal-ai/moondream3-preview/segment' — Segmentation (Moondream 3 preview).

    - Uses Base64 Data URI for image_url (no upload)
    - TRUE PARALLEL batch:
        submit all -> poll all -> pack results
    - Supports:
        object (string), spatial_references (JSON), preview (bool),
        settings {temperature, top_p, max_tokens}

    Returns:
      mask_preview: IMAGE   (mask image as RGB for easy viewing; black if not returned)
      mask:         MASK    (mask alpha or luminance; zeros if not returned)
      path:         STRING  (SVG path data; JSON array aligned to outputs)
      bbox_json:    STRING  (bbox object; JSON array aligned to outputs)
      finish_json:  STRING  (finish_reason; JSON array aligned to outputs)
      usage_json:   STRING  (usage_info; JSON array aligned to outputs)

    spatial_references_json format (normalized coords 0..1), examples:
      - Points:
        [{"x": 0.64, "y": 0.40}]
      - Arrays of 2 floats (x,y):
        [[0.64, 0.40]]
      - Arrays of 4 floats (x1,y1,x2,y2):
        [[0.10, 0.20, 0.40, 0.50]]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "object": ("STRING", {"default": "mango", "multiline": False}),
                "preview": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "spatial_references_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("mask_preview", "mask", "path", "bbox_json", "finish_json", "usage_json")
    FUNCTION = "segment"
    CATEGORY = "PVL_tools"

    # ------------------------- helpers -------------------------

    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return [image[i] for i in range(image.shape[0])]
            if image.ndim == 3:
                return [image]
            self._raise(f"Moondream3 Segment: unsupported image tensor shape {tuple(image.shape)}")
        if isinstance(image, np.ndarray):
            return self._split_image_batch(torch.from_numpy(image))
        self._raise("Moondream3 Segment: unsupported image type (expected torch.Tensor or np.ndarray).")

    def _parse_spatial_refs(self, s: Optional[str]) -> List[Any]:
        if s is None:
            return []
        s = s.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except Exception as e:
            self._raise(f"Moondream3 Segment: invalid JSON in spatial_references_json: {e}")
        if data is None:
            return []
        if not isinstance(data, list):
            self._raise("Moondream3 Segment: spatial_references_json must be a JSON list.")
        # Can be list[dict] or list[list]
        return data

    def _pil_from_data_uri(self, data_uri: str) -> Image.Image:
        header, b64 = data_uri.split(",", 1)
        if ";base64" not in header:
            self._raise("Moondream3 Segment: non-base64 data URI not supported.")
        raw = io.BytesIO(__import__("base64").b64decode(b64))
        raw.seek(0)
        return Image.open(raw)

    def _download_pil_any(self, file_obj_or_url: Any, mode: Optional[str] = None) -> Image.Image:
        url = None
        if isinstance(file_obj_or_url, dict):
            url = file_obj_or_url.get("url") or file_obj_or_url.get("content") or file_obj_or_url.get("file_data")
        elif isinstance(file_obj_or_url, str):
            url = file_obj_or_url

        if not isinstance(url, str) or not url:
            self._raise("Moondream3 Segment: missing URL/content for image file.")

        if url.startswith("data:"):
            pil = self._pil_from_data_uri(url)
        else:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            pil = Image.open(io.BytesIO(r.content))

        if mode is not None:
            pil = pil.convert(mode)
        return pil

    def _pil_to_image_tensor_rgb(self, pil: Image.Image) -> torch.Tensor:
        pil = pil.convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _pil_to_mask_tensor(self, pil: Image.Image) -> torch.Tensor:
        # Prefer alpha if present, else luminance
        if "A" in pil.getbands():
            a = pil.getchannel("A")
            arr = np.array(a).astype(np.float32) / 255.0
            return torch.from_numpy(arr)
        pil_l = pil.convert("L")
        arr = np.array(pil_l).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    # ------------------------- FAL queue API (two-phase) -------------------------

    def _direct_fal_submit(self, endpoint: str, arguments: Dict[str, Any], sync_mode: bool) -> Dict[str, str]:
        import os
        fal_key = os.getenv("FAL_KEY", "") or os.getenv("FAL_API_KEY", "") or os.getenv("FAL_CLIENT_KEY", "")
        if not fal_key:
            self._raise("Moondream3 Segment: FAL_KEY environment variable not set.")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        payload = dict(arguments)
        payload["sync_mode"] = bool(sync_mode)

        r = requests.post(submit_url, headers=headers, json=payload, timeout=120)
        if not r.ok:
            self._raise(f"Moondream3 Segment: submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            self._raise("Moondream3 Segment: did not return request_id")

        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"

        return {"request_id": req_id, "status_url": status_url, "response_url": resp_url}

    def _direct_fal_poll_and_fetch(self, request_info: Dict[str, str], timeout: float = 30.0) -> Dict[str, Any]:
        import os
        fal_key = os.getenv("FAL_KEY", "") or os.getenv("FAL_API_KEY", "") or os.getenv("FAL_CLIENT_KEY", "")
        if not fal_key:
            self._raise("Moondream3 Segment: FAL_KEY environment variable not set.")

        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]

        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=10)
                if sr.ok:
                    js = sr.json()
                    if js.get("status") == "COMPLETED":
                        break
                    if js.get("status") in ("FAILED", "CANCELED", "CANCELLED"):
                        self._raise(f"Moondream3 Segment: request failed: {js}")
            except Exception:
                pass
            time.sleep(0.25)

        if time.time() >= deadline:
            self._raise(f"Moondream3 Segment: request timed out after {timeout}s")

        rr = requests.get(resp_url, headers=headers, timeout=30)
        if not rr.ok:
            self._raise(f"Moondream3 Segment: result fetch error {rr.status_code}: {rr.text}")

        rdata = rr.json()
        return rdata.get("response", rdata)

    # ------------------------- per-frame work -------------------------

    def _submit_only(
        self,
        frame: torch.Tensor,
        obj: str,
        spatial_refs: List[Any],
        preview: bool,
        temperature: float,
        top_p: float,
        max_tokens: int,
        sync_mode: bool,
    ) -> Dict[str, str]:
        image_url = ImageUtils.image_to_data_uri(frame)
        if not image_url:
            self._raise("Moondream3 Segment: failed to convert input image to data URI.")

        args: Dict[str, Any] = {
            "image_url": image_url,
            "object": str(obj) if obj is not None else "",
            "spatial_references": spatial_refs or [],
            "preview": bool(preview),
            "settings": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
            }
        }

        return self._direct_fal_submit("fal-ai/moondream3-preview/segment", args, sync_mode)

    def _process_result(
        self,
        result: Dict[str, Any],
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, Any]:
        """
        Returns:
          mask_preview_rgb (H,W,3)
          mask (H,W)
          path (str|None)
          bbox (dict|None)
          finish_reason (str|None)
          usage_info (dict|None)
        """
        if not isinstance(result, dict):
            self._raise("Moondream3 Segment: unexpected response type (expected dict).")

        finish_reason = result.get("finish_reason", None)
        usage_info = result.get("usage_info", None)
        path = result.get("path", None)
        bbox = result.get("bbox", None)

        img = result.get("image", None)  # may be null
        if isinstance(img, dict) and (img.get("url") or img.get("content") or img.get("file_data")):
            pil_mask = self._download_pil_any(img, mode=None)
            mask_t = self._pil_to_mask_tensor(pil_mask).clamp(0, 1)
            # preview as RGB grayscale
            m = mask_t.cpu().numpy()
            rgb = np.stack([m, m, m], axis=-1).astype(np.float32)
            prev_t = torch.from_numpy(rgb)
            return prev_t, mask_t, path, bbox, finish_reason, usage_info

        # Not detected or preview=False -> null image
        h, w = fallback_hw
        mask_t = torch.zeros((h, w), dtype=torch.float32)
        prev_t = torch.zeros((h, w, 3), dtype=torch.float32)
        return prev_t, mask_t, path, bbox, finish_reason, usage_info

    # ------------------------- main -------------------------

    def segment(
        self,
        image,
        object,
        preview,
        temperature,
        top_p,
        max_tokens,
        sync_mode=False,
        spatial_references_json="[]",
    ):
        t0 = time.time()

        frames = self._split_image_batch(image)
        if not frames:
            self._raise("Moondream3 Segment: no input image frames provided.")

        spatial_refs = self._parse_spatial_refs(spatial_references_json)

        batch_size = len(frames)
        print(f"[PVL Moondream3 Segment] Processing {batch_size} image(s) | preview={bool(preview)} | sync_mode={bool(sync_mode)}")

        # Single image fast path
        if batch_size == 1:
            req_info = self._submit_only(
                frames[0],
                object,
                spatial_refs,
                preview,
                temperature,
                top_p,
                max_tokens,
                sync_mode,
            )
            # API doc says timeout ~20s; we allow a bit more.
            result = self._direct_fal_poll_and_fetch(req_info, timeout=40.0)

            h = int(frames[0].shape[0]) if frames[0].ndim == 3 else int(frames[0].shape[1])
            w = int(frames[0].shape[1]) if frames[0].ndim == 3 else int(frames[0].shape[2])

            prev, m, path, bbox, finish, usage = self._process_result(result, (h, w))

            print(f"[PVL Moondream3 Segment] Done in {time.time() - t0:.2f}s")
            return (
                prev.unsqueeze(0),
                m.unsqueeze(0),
                json.dumps([path], ensure_ascii=False),
                json.dumps([bbox], ensure_ascii=False),
                json.dumps([finish], ensure_ascii=False),
                json.dumps([usage], ensure_ascii=False),
            )

        # Batch: TRUE PARALLEL
        max_workers = min(batch_size, 6)
        print(f"[PVL Moondream3 Segment] Submitting {batch_size} request(s) in parallel (workers={max_workers})...")

        submit_results: List[Tuple[int, Dict[str, str]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    self._submit_only,
                    frames[i],
                    object,
                    spatial_refs,
                    preview,
                    temperature,
                    top_p,
                    max_tokens,
                    sync_mode,
                ): i
                for i in range(batch_size)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    req_info = fut.result()
                    submit_results.append((idx, req_info))
                except Exception as e:
                    print(f"[PVL Moondream3 Segment] Submit failed for image {idx}: {e}")

        if not submit_results:
            self._raise("[PVL Moondream3 Segment] All submissions failed.")

        print(f"[PVL Moondream3 Segment] {len(submit_results)}/{batch_size} submitted. Polling in parallel...")

        results: Dict[int, Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, Any]] = {}
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            poll_futs = {
                ex.submit(self._direct_fal_poll_and_fetch, req_info, 40.0): idx
                for idx, req_info in submit_results
            }
            for fut in as_completed(poll_futs):
                idx = poll_futs[fut]
                try:
                    result = fut.result()

                    h = int(frames[idx].shape[0]) if frames[idx].ndim == 3 else int(frames[idx].shape[1])
                    w = int(frames[idx].shape[1]) if frames[idx].ndim == 3 else int(frames[idx].shape[2])

                    results[idx] = self._process_result(result, (h, w))
                except Exception as e:
                    failed += 1
                    print(f"[PVL Moondream3 Segment] Poll failed for image {idx}: {e}")

        if not results:
            self._raise(f"[PVL Moondream3 Segment] All requests failed during polling ({failed} failures).")

        if failed > 0:
            print(f"[PVL Moondream3 Segment WARNING] {failed}/{batch_size} failed; returning only successful outputs.")

        # Pack outputs in original order, skipping failed indices
        preview_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        path_list: List[Any] = []
        bbox_list: List[Any] = []
        finish_list: List[Any] = []
        usage_list: List[Any] = []

        for i in range(batch_size):
            if i in results:
                prev, m, path, bbox, finish, usage = results[i]
                preview_list.append(prev)
                mask_list.append(m)
                path_list.append(path)
                bbox_list.append(bbox)
                finish_list.append(finish)
                usage_list.append(usage)

        preview_batch = torch.stack(preview_list, dim=0)
        mask_batch = torch.stack(mask_list, dim=0)

        print(f"[PVL Moondream3 Segment] Done {len(preview_list)}/{batch_size} in {time.time() - t0:.2f}s")

        return (
            preview_batch,
            mask_batch,
            json.dumps(path_list, ensure_ascii=False),
            json.dumps(bbox_list, ensure_ascii=False),
            json.dumps(finish_list, ensure_ascii=False),
            json.dumps(usage_list, ensure_ascii=False),
        )

