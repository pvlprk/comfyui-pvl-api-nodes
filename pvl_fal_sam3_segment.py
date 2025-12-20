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

class PVL_fal_Sam3_Segmentation_API:
    """
    ComfyUI node for FAL 'fal-ai/sam-3/image' — Segment Image (SAM-3).

    Key features:
    - Base64 Data URI submission (no storage upload)
    - TRUE PARALLEL batch execution:
        Phase 1: Submit all -> get request_ids immediately
        Phase 2: Poll all -> fetch results in parallel
    - Supports prompt, point prompts, box prompts, apply_mask, output_format, sync_mode
    - Returns:
        preview: IMAGE (the API's primary preview image)
        mask:    MASK  (the FIRST returned mask per input image; if none, zeros)
        scores_json: STRING (JSON array aligned to input batch)
        boxes_json:  STRING (JSON array aligned to input batch)

    Notes:
    - The API can return multiple masks. This node returns only the *first* mask as MASK output
      (because ComfyUI MASK output is typically one mask per input image). The other masks'
      metadata (scores/boxes) are included in JSON outputs.
    - point_prompts_json and box_prompts_json are optional JSON strings:
        point_prompts_json example:
          [{"x": 120, "y": 220, "label": 1, "object_id": 0}, {"x": 10, "y": 20, "label": 0}]
        box_prompts_json example:
          [{"x_min": 10, "y_min": 20, "x_max": 400, "y_max": 500, "object_id": 0}]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "wheel", "multiline": False}),
                "apply_mask": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "return_multiple_masks": ("BOOLEAN", {"default": False}),
                "max_masks": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "include_scores": ("BOOLEAN", {"default": False}),
                "include_boxes": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # JSON strings for advanced prompts
                "point_prompts_json": ("STRING", {"default": "[]", "multiline": True}),
                "box_prompts_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("preview", "mask", "scores_json", "boxes_json")
    FUNCTION = "segment"
    CATEGORY = "PVL_tools"

    # ------------------------- small helpers -------------------------

    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return [image[i] for i in range(image.shape[0])]
            if image.ndim == 3:
                return [image]
            self._raise(f"FAL SAM-3: unsupported image tensor shape {tuple(image.shape)}")
        if isinstance(image, np.ndarray):
            return self._split_image_batch(torch.from_numpy(image))
        self._raise("FAL SAM-3: unsupported image type (expected torch.Tensor or np.ndarray).")

    def _parse_json_list(self, s: Optional[str], name: str) -> List[Dict[str, Any]]:
        if s is None:
            return []
        s = s.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except Exception as e:
            self._raise(f"FAL SAM-3: invalid JSON in {name}: {e}")
        if data is None:
            return []
        if not isinstance(data, list):
            self._raise(f"FAL SAM-3: {name} must be a JSON list (got {type(data).__name__}).")
        # Ensure dict entries
        out = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self._raise(f"FAL SAM-3: {name}[{i}] must be an object/dict.")
            out.append(item)
        return out

    def _pil_from_data_uri(self, data_uri: str) -> Image.Image:
        header, b64 = data_uri.split(",", 1)
        if ";base64" not in header:
            self._raise("FAL SAM-3: non-base64 data URI not supported.")
        raw = io.BytesIO()
        raw.write(__import__("base64").b64decode(b64))
        raw.seek(0)
        return Image.open(raw)

    def _download_pil_any(self, file_obj_or_url: Any, mode: Optional[str] = None) -> Image.Image:
        """
        Accepts:
          - dict with {url: "..."} (possibly data URI)
          - string URL (possibly data URI)
        """
        url = None
        if isinstance(file_obj_or_url, dict):
            url = file_obj_or_url.get("url") or file_obj_or_url.get("content") or file_obj_or_url.get("file_data")
        elif isinstance(file_obj_or_url, str):
            url = file_obj_or_url

        if not isinstance(url, str) or not url:
            self._raise("FAL SAM-3: missing URL/content for image file.")

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
        arr = np.array(pil).astype(np.float32) / 255.0  # (H,W,3)
        return torch.from_numpy(arr)

    def _pil_to_mask_tensor(self, pil: Image.Image) -> torch.Tensor:
        # Prefer alpha if present, else luminance
        if "A" in pil.getbands():
            a = pil.getchannel("A")
            arr = np.array(a).astype(np.float32) / 255.0
            return torch.from_numpy(arr)  # (H,W)
        pil_l = pil.convert("L")
        arr = np.array(pil_l).astype(np.float32) / 255.0
        return torch.from_numpy(arr)  # (H,W)

    # ------------------------- FAL queue API (two-phase) -------------------------

    def _direct_fal_submit(self, endpoint: str, arguments: Dict[str, Any], sync_mode: bool) -> Dict[str, str]:
        import os
        fal_key = os.getenv("FAL_KEY", "") or os.getenv("FAL_API_KEY", "") or os.getenv("FAL_CLIENT_KEY", "")
        if not fal_key:
            self._raise("FAL SAM-3: FAL_KEY environment variable not set.")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        payload = dict(arguments)
        payload["sync_mode"] = bool(sync_mode)

        r = requests.post(submit_url, headers=headers, json=payload, timeout=120)
        if not r.ok:
            self._raise(f"FAL SAM-3: submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            self._raise("FAL SAM-3: did not return request_id")

        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"

        return {"request_id": req_id, "status_url": status_url, "response_url": resp_url}

    def _direct_fal_poll_and_fetch(self, request_info: Dict[str, str], timeout: float = 120.0) -> Dict[str, Any]:
        import os
        fal_key = os.getenv("FAL_KEY", "") or os.getenv("FAL_API_KEY", "") or os.getenv("FAL_CLIENT_KEY", "")
        if not fal_key:
            self._raise("FAL SAM-3: FAL_KEY environment variable not set.")

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
                        self._raise(f"FAL SAM-3: request failed: {js}")
            except Exception:
                pass
            time.sleep(0.6)

        if time.time() >= deadline:
            self._raise(f"FAL SAM-3: request timed out after {timeout}s")

        rr = requests.get(resp_url, headers=headers, timeout=30)
        if not rr.ok:
            self._raise(f"FAL SAM-3: result fetch error {rr.status_code}: {rr.text}")

        rdata = rr.json()
        return rdata.get("response", rdata)

    # ------------------------- per-frame work -------------------------

    def _submit_only(
        self,
        frame: torch.Tensor,
        prompt: str,
        point_prompts: List[Dict[str, Any]],
        box_prompts: List[Dict[str, Any]],
        apply_mask: bool,
        output_format: str,
        return_multiple_masks: bool,
        max_masks: int,
        include_scores: bool,
        include_boxes: bool,
        sync_mode: bool,
    ) -> Dict[str, str]:
        image_url = ImageUtils.image_to_data_uri(frame)
        if not image_url:
            self._raise("FAL SAM-3: failed to convert input image to data URI.")

        args: Dict[str, Any] = {
            "image_url": image_url,
            "prompt": str(prompt) if prompt is not None else "wheel",
            "point_prompts": point_prompts or [],
            "box_prompts": box_prompts or [],
            "apply_mask": bool(apply_mask),
            "output_format": output_format,
            "return_multiple_masks": bool(return_multiple_masks),
            "max_masks": int(max_masks),
            "include_scores": bool(include_scores),
            "include_boxes": bool(include_boxes),
        }

        return self._direct_fal_submit("fal-ai/sam-3/image", args, sync_mode)

    def _process_result(
        self,
        result: Dict[str, Any],
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """
        Returns:
          preview_rgb (H,W,3)
          mask_1 (H,W) -- first mask (or zeros)
          scores_entry (any) -- for JSON output
          boxes_entry (any) -- for JSON output
        """
        if not isinstance(result, dict):
            self._raise("FAL SAM-3: unexpected response type (expected dict).")

        # Preview image: result["image"] (optional in some outputs, but documented)
        preview_tensor: Optional[torch.Tensor] = None
        if isinstance(result.get("image"), dict) and (result["image"].get("url") or result["image"].get("content")):
            pil_prev = self._download_pil_any(result["image"], mode=None)
            preview_tensor = self._pil_to_image_tensor_rgb(pil_prev)

        # Masks list
        masks_list = result.get("masks", [])
        first_mask_tensor: Optional[torch.Tensor] = None

        if isinstance(masks_list, list) and len(masks_list) > 0:
            # first mask
            pil_mask0 = self._download_pil_any(masks_list[0], mode=None)
            first_mask_tensor = self._pil_to_mask_tensor(pil_mask0)

            # If preview missing, create a visual preview by compositing mask over original is not possible here;
            # we keep preview as provided by API. If missing, fallback to showing the mask as grayscale RGB.
            if preview_tensor is None:
                # Convert mask to RGB preview
                m = first_mask_tensor.clamp(0, 1).cpu().numpy()
                rgb = np.stack([m, m, m], axis=-1).astype(np.float32)
                preview_tensor = torch.from_numpy(rgb)
        else:
            # No mask returned -> zeros
            h, w = fallback_hw
            first_mask_tensor = torch.zeros((h, w), dtype=torch.float32)
            if preview_tensor is None:
                # black image fallback
                preview_tensor = torch.zeros((h, w, 3), dtype=torch.float32)

        # Scores / boxes (may or may not be present; include_* toggles affect it)
        scores_entry = result.get("scores", None)
        boxes_entry = result.get("boxes", None)

        return preview_tensor, first_mask_tensor, scores_entry, boxes_entry

    # ------------------------- main -------------------------

    def segment(
        self,
        image,
        prompt,
        apply_mask,
        output_format,
        return_multiple_masks,
        max_masks,
        include_scores,
        include_boxes,
        sync_mode=False,
        point_prompts_json="[]",
        box_prompts_json="[]",
    ):
        t0 = time.time()

        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL SAM-3: no input image frames provided.")

        point_prompts = self._parse_json_list(point_prompts_json, "point_prompts_json")
        box_prompts = self._parse_json_list(box_prompts_json, "box_prompts_json")

        batch_size = len(frames)
        print(f"[PVL SAM-3] Processing {batch_size} image(s) | sync_mode={bool(sync_mode)}")

        # Single image fast path
        if batch_size == 1:
            req_info = self._submit_only(
                frames[0],
                prompt,
                point_prompts,
                box_prompts,
                apply_mask,
                output_format,
                return_multiple_masks,
                max_masks,
                include_scores,
                include_boxes,
                sync_mode,
            )
            result = self._direct_fal_poll_and_fetch(req_info, timeout=180.0)
            h = int(frames[0].shape[0]) if frames[0].ndim == 3 else int(frames[0].shape[1])
            w = int(frames[0].shape[1]) if frames[0].ndim == 3 else int(frames[0].shape[2])
            prev, m0, sc, bx = self._process_result(result, (h, w))

            print(f"[PVL SAM-3] Done in {time.time() - t0:.2f}s")
            return (
                prev.unsqueeze(0),                 # IMAGE batch
                m0.unsqueeze(0),                   # MASK batch
                json.dumps([sc], ensure_ascii=False),
                json.dumps([bx], ensure_ascii=False),
            )

        # Batch: TRUE PARALLEL
        max_workers = min(batch_size, 6)
        print(f"[PVL SAM-3] Submitting {batch_size} request(s) in parallel (workers={max_workers})...")

        submit_results: List[Tuple[int, Dict[str, str]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    self._submit_only,
                    frames[i],
                    prompt,
                    point_prompts,
                    box_prompts,
                    apply_mask,
                    output_format,
                    return_multiple_masks,
                    max_masks,
                    include_scores,
                    include_boxes,
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
                    print(f"[PVL SAM-3] Submit failed for image {idx}: {e}")

        if not submit_results:
            self._raise("[PVL SAM-3] All submissions failed.")

        print(f"[PVL SAM-3] {len(submit_results)}/{batch_size} submitted. Polling in parallel...")

        results: Dict[int, Tuple[torch.Tensor, torch.Tensor, Any, Any]] = {}
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            poll_futs = {
                ex.submit(self._direct_fal_poll_and_fetch, req_info, 180.0): idx
                for idx, req_info in submit_results
            }
            for fut in as_completed(poll_futs):
                idx = poll_futs[fut]
                try:
                    result = fut.result()
                    # fallback H/W from input frame idx
                    h = int(frames[idx].shape[0]) if frames[idx].ndim == 3 else int(frames[idx].shape[1])
                    w = int(frames[idx].shape[1]) if frames[idx].ndim == 3 else int(frames[idx].shape[2])
                    results[idx] = self._process_result(result, (h, w))
                except Exception as e:
                    failed += 1
                    print(f"[PVL SAM-3] Poll failed for image {idx}: {e}")

        if not results:
            self._raise(f"[PVL SAM-3] All requests failed during polling ({failed} failures).")

        if failed > 0:
            print(f"[PVL SAM-3 WARNING] {failed}/{batch_size} failed; returning only successful outputs.")

        # Pack outputs in original order, skipping failed indices
        preview_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        scores_out: List[Any] = []
        boxes_out: List[Any] = []

        for i in range(batch_size):
            if i in results:
                prev, m0, sc, bx = results[i]
                preview_list.append(prev)
                mask_list.append(m0)
                scores_out.append(sc)
                boxes_out.append(bx)

        preview_batch = torch.stack(preview_list, dim=0)  # (B,H,W,3) but H/W must match across results
        mask_batch = torch.stack(mask_list, dim=0)        # (B,H,W)

        print(f"[PVL SAM-3] Done {len(preview_list)}/{batch_size} in {time.time() - t0:.2f}s")

        return (
            preview_batch,
            mask_batch,
            json.dumps(scores_out, ensure_ascii=False),
            json.dumps(boxes_out, ensure_ascii=False),
        )
