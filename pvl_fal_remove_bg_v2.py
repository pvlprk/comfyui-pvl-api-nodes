import torch
import numpy as np
import time
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import ImageUtils, ApiHandler

class PVL_fal_RemoveBackground_API:
    """
    ComfyUI node for FAL 'fal-ai/birefnet/v2' — Remove Background V2.
    
    Features:
    - TRUE PARALLEL execution: submits all requests first, then polls all in parallel
    - Error handling for individual requests with partial results support
    - Batch processing with worker cap to prevent thread explosion
    - Optional sync_mode toggle
    
    Inputs:
    - image (IMAGE): Input image(s). Batch processing is supported with parallel API calls.
    - model (CHOICE): Which model variant to use.
    - operating_resolution (CHOICE): Resolution for inference ("1024x1024" or "2048x2048").
    - output_format (CHOICE): "png" or "webp".
    - output_mask (BOOLEAN): Whether to also return the mask.
    - refine_foreground (BOOLEAN): Whether to refine the foreground (default: True).
    - sync_mode (BOOLEAN): Use synchronous mode for FAL API (default: False).
    
    Outputs:
    - IMAGE: Foreground with background removed (RGB or RGBA if provided).
    - MASK: Optional mask (1-channel, float, shape [B,H,W]) if output_mask=True.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "General Use (Light)",
                        "General Use (Light 2K)",
                        "General Use (Heavy)",
                        "Matting",
                        "Portrait",
                    ],
                    {"default": "General Use (Light)"},
                ),
                "operating_resolution": (
                    ["1024x1024", "2048x2048"],
                    {"default": "1024x1024"},
                ),
                "output_format": (
                    ["png", "webp"],
                    {"default": "png"},
                ),
                "output_mask": ("BOOLEAN", {"default": False}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "mask")
    FUNCTION = "remove_background"
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
    
    def _download_pil(self, url, mode=None):
        from PIL import Image
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        pil = Image.open(io.BytesIO(resp.content))
        if mode is not None:
            pil = pil.convert(mode)
        return pil
    
    # -------- FAL Queue API - TWO PHASE EXECUTION --------
    
    def _fal_submit_only(self, frame, model, operating_resolution, output_format, 
                         output_mask, refine_foreground, sync_mode):
        """
        Phase 1: Submit request to FAL and return request info immediately.
        Does NOT wait for completion.
        """
        # Convert image to data URI
        image_url = ImageUtils.image_to_data_uri(frame)
        if not image_url:
            raise RuntimeError("FAL: failed to convert input image.")
        
        arguments = {
            "image_url": image_url,
            "model": model,
            "operating_resolution": operating_resolution,
            "output_format": output_format,
            "output_mask": bool(output_mask),
            "refine_foreground": bool(refine_foreground),
        }
        
        # Check if ApiHandler supports async submission
        # If it has submit_only method, use it; otherwise use direct API call
        if hasattr(ApiHandler, 'submit_only'):
            return ApiHandler.submit_only("fal-ai/birefnet/v2", arguments, sync_mode)
        else:
            # Fallback to direct FAL queue API
            return self._direct_fal_submit("fal-ai/birefnet/v2", arguments, sync_mode)
    
    def _direct_fal_submit(self, endpoint, arguments, sync_mode):
        """Direct FAL queue API submission when ApiHandler doesn't support async."""
        import os
        
        fal_key = os.getenv("FAL_KEY", "")
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")
        
        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}
        
        payload = dict(arguments)
        payload["sync_mode"] = sync_mode
        
        r = requests.post(submit_url, headers=headers, json=payload, timeout=120)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        
        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")
        
        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"
        
        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }
    
    def _fal_poll_and_fetch(self, request_info, timeout=120):
        """
        Phase 2: Poll a single FAL request until complete and fetch the result.
        Returns (fg_tensor, mask_tensor).
        """
        import os
        
        fal_key = os.getenv("FAL_KEY", "")
        headers = {"Authorization": f"Key {fal_key}"}
        
        # Check if ApiHandler supports async polling
        if hasattr(ApiHandler, 'poll_and_get_result'):
            result = ApiHandler.poll_and_get_result(request_info, timeout)
        else:
            # Fallback to direct polling
            status_url = request_info["status_url"]
            resp_url = request_info["response_url"]
            
            # Poll for completion
            deadline = time.time() + timeout
            completed = False
            while time.time() < deadline:
                try:
                    sr = requests.get(status_url, headers=headers, timeout=10)
                    if sr.ok and sr.json().get("status") == "COMPLETED":
                        completed = True
                        break
                except Exception:
                    pass
                time.sleep(0.6)
            
            if not completed:
                raise RuntimeError(f"FAL request timed out after {timeout}s")
            
            # Fetch result
            rr = requests.get(resp_url, headers=headers, timeout=15)
            if not rr.ok:
                raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
            
            rdata = rr.json()
            result = rdata.get("response", rdata)
        
        # Process result
        return self._process_result(result)
    
    def _process_result(self, result):
        """Process FAL API result and return tensors."""
        if not isinstance(result, dict):
            raise RuntimeError("FAL: unexpected response type (expected dict).")
        
        # ---- Foreground image (keep alpha if present) ----
        if "image" not in result or not isinstance(result["image"], dict):
            raise RuntimeError("FAL: response missing foreground image.")
        
        fg_url = result["image"].get("url")
        if not fg_url:
            raise RuntimeError("FAL: foreground image has no URL.")
        
        pil_fg = self._download_pil(fg_url, mode=None)  # keep native mode
        fg_arr = np.array(pil_fg).astype(np.float32) / 255.0  # (H,W,3) or (H,W,4)
        
        if fg_arr.ndim == 2:  # grayscale fallback
            fg_arr = np.expand_dims(fg_arr, axis=-1)
        
        fg_tensor = torch.from_numpy(fg_arr)  # (H,W,C)
        
        # ---- Mask (shape [H,W], float 0..1) ----
        mask_tensor = torch.zeros((fg_arr.shape[0], fg_arr.shape[1]), dtype=torch.float32)
        
        # Try to get mask from response
        mask_url = None
        if isinstance(result.get("mask_image"), dict):
            mask_url = result["mask_image"].get("url")
        
        if mask_url:
            pil_mask_rgba = self._download_pil(mask_url, mode=None)
            if "A" in pil_mask_rgba.getbands():
                alpha = pil_mask_rgba.getchannel("A")
                mask_arr = np.array(alpha).astype(np.float32) / 255.0
            else:
                pil_mask_L = pil_mask_rgba.convert("L")
                mask_arr = np.array(pil_mask_L).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_arr)
        else:
            # Fallback: extract alpha channel from foreground if present
            if "A" in pil_fg.getbands():
                alpha = pil_fg.getchannel("A")
                mask_arr = np.array(alpha).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_arr)
        
        return fg_tensor, mask_tensor
    
    # ------------------------- main -------------------------
    
    def remove_background(
        self,
        image,
        model,
        operating_resolution,
        output_format,
        output_mask,
        refine_foreground,
        sync_mode=False,
    ):
        _t0 = time.time()
        
        # Split batch into frames
        frames = self._split_image_batch(image)
        if not frames:
            self._raise("FAL: no input image frames provided.")
        
        batch_size = len(frames)
        print(f"[FAL RemoveBG] Processing {batch_size} images...")
        
        # Single image: process directly (less overhead)
        if batch_size == 1:
            try:
                req_info = self._fal_submit_only(
                    frames[0], model, operating_resolution, output_format,
                    output_mask, refine_foreground, sync_mode
                )
                fg_tensor, mask_tensor = self._fal_poll_and_fetch(req_info)
                
                _t1 = time.time()
                print(f"[FAL RemoveBG] Completed in {(_t1 - _t0):.2f}s")
                return (fg_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
            except Exception as e:
                self._raise(f"FAL: processing failed: {e}")
        
        # Multiple images: TRUE PARALLEL execution
        print(f"[FAL RemoveBG] Submitting {batch_size} requests in parallel...")
        
        # PHASE 1: Submit all requests in parallel
        submit_results = []
        max_workers = min(batch_size, 6)  # Cap at 6 workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            submit_futs = {
                executor.submit(
                    self._fal_submit_only,
                    frames[i],
                    model,
                    operating_resolution,
                    output_format,
                    output_mask,
                    refine_foreground,
                    sync_mode
                ): i
                for i in range(batch_size)
            }
            
            for fut in as_completed(submit_futs):
                idx = submit_futs[fut]
                try:
                    req_info = fut.result()
                    submit_results.append((idx, req_info))
                except Exception as e:
                    print(f"[FAL RemoveBG] Submit failed for image {idx}: {e}")
        
        if not submit_results:
            self._raise("All FAL submission requests failed")
        
        print(f"[FAL RemoveBG] {len(submit_results)} requests submitted. Polling for results...")
        
        # PHASE 2: Poll all requests in parallel
        results = {}
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            poll_futs = {
                executor.submit(self._fal_poll_and_fetch, req_info): idx
                for idx, req_info in submit_results
            }
            
            for fut in as_completed(poll_futs):
                idx = poll_futs[fut]
                try:
                    fg_tensor, mask_tensor = fut.result()
                    results[idx] = (fg_tensor, mask_tensor)
                except Exception as e:
                    failed_count += 1
                    print(f"[FAL RemoveBG] Poll failed for image {idx}: {e}")
        
        if not results:
            self._raise(f"All FAL requests failed during polling ({failed_count} failures)")
        
        if failed_count > 0:
            print(f"[FAL RemoveBG WARNING] {failed_count}/{batch_size} requests failed, continuing with {len(results)} successful results")
        
        # Stack results in original order (with None for failed images)
        fg_list = []
        mask_list = []
        
        for i in range(batch_size):
            if i in results:
                fg_tensor, mask_tensor = results[i]
                fg_list.append(fg_tensor)
                mask_list.append(mask_tensor)
        
        if not fg_list:
            self._raise("No successful results to return")
        
        # Stack into batched tensors
        fg_batch = torch.stack(fg_list, dim=0)  # (B,H,W,C)
        mask_batch = torch.stack(mask_list, dim=0)  # (B,H,W)
        
        _t1 = time.time()
        print(f"[FAL RemoveBG] Completed {len(fg_list)}/{batch_size} images in {(_t1 - _t0):.2f}s")
        
        return (fg_batch, mask_batch)

# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {
    "PVL_fal_RemoveBackground_API": PVL_fal_RemoveBackground_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_RemoveBackground_API": "PVL Remove Background V2 (fal.ai) [Batch]",
}
