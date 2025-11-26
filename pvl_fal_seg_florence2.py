import os
import io
import time
import base64
import torch
import requests
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig

class PVL_Seg_Florence2_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_color": ("STRING", {"default": "255,255,255"}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "timeout_per_retry": ("INT", {"default": 180, "min": 30, "max": 600, "step": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt_1": ("STRING", {"multiline": True, "default": ""}),
                "prompt_2": ("STRING", {"multiline": True, "default": ""}),
                "prompt_3": ("STRING", {"multiline": True, "default": ""}),
                "prompt_4": ("STRING", {"multiline": True, "default": ""}),
                "prompt_5": ("STRING", {"multiline": True, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "IMAGE","IMAGE","IMAGE","IMAGE","IMAGE",
        "MASK","MASK","MASK","MASK","MASK",
    )
    RETURN_NAMES = (
        "masked_1","masked_2","masked_3","masked_4","masked_5",
        "mask_1","mask_2","mask_3","mask_4","mask_5",
    )
    FUNCTION = "process_images"
    CATEGORY = "PVL_tools_FAL"

    def _tensor_to_data_url(self, image_tensor, mime="image/png"):
        """Convert ComfyUI IMAGE tensor [B,H,W,C] to data URL"""
        if not torch.is_tensor(image_tensor):
            raise RuntimeError("Expected torch.Tensor for image input")
        img = image_tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]  # Get first batch
        # img is now [H, W, C] - already in correct format for ComfyUI
        arr = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _rasterize_polygons(self, polygons, w, h):
        """Create mask tensor from polygon data - returns [1, H, W]"""
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for poly in polygons:
            pts = [(p["x"], p["y"]) for p in poly.get("points", []) if "x" in p and "y" in p]
            if pts:
                draw.polygon(pts, fill=255)
        
        arr = np.array(mask, dtype=np.float32) / 255.0
        
        # Should already be (H, W) from PIL "L" mode
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
        
        # Convert to tensor (1, H, W) for MASK output
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor

    def _submit_single(self, image_tensor, prompt, debug):
        image_url = self._tensor_to_data_url(image_tensor)
        endpoint = "fal-ai/florence-2-large/referring-expression-segmentation"
        base = "https://queue.fal.run"
        headers = {"Authorization": f"Key {FalConfig.get_api_key()}"}
        args = {"image_url": image_url, "text_input": prompt or ""}
        if debug:
            print(f"[PVL Florence2 DEBUG] Submitting prompt: '{prompt[:80]}'")
        r = requests.post(f"{base}/{endpoint}", headers=headers, json=args, timeout=120)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")
        j = r.json()
        rid = j.get("request_id")
        if not rid:
            raise RuntimeError("No request_id returned from FAL")
        return {
            "status_url": j.get("status_url") or f"{base}/{endpoint}/requests/{rid}/status",
            "response_url": j.get("response_url") or f"{base}/{endpoint}/requests/{rid}",
        }

    def _poll_and_fetch(self, req, timeout=180, debug=False):
        if req is None:
            return None
        headers = {"Authorization": f"Key {FalConfig.get_api_key()}"}
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise RuntimeError("FAL request timeout")
            s = requests.get(req["status_url"], headers=headers, timeout=15)
            if s.ok:
                st = s.json()
                if st.get("status") == "COMPLETED":
                    break
                if st.get("status") == "FAILED":
                    raise RuntimeError(f"FAL failed: {st}")
            time.sleep(0.6)
        r = requests.get(req["response_url"], headers=headers, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Result fetch failed: {r.text}")
        data = r.json().get("response", r.json())
        return data

    def _apply_mask(self, image_tensor, mask_tensor, mask_color):
        """Blend image with mask color - returns [1, H, W, C] for ComfyUI"""
        try:
            rgb = [int(c.strip()) for c in mask_color.split(",")]
            rgb = np.clip(np.array(rgb) / 255.0, 0, 1)
        except Exception:
            rgb = np.array([1.0, 1.0, 1.0])
        
        # Input is already [B, H, W, C] from ComfyUI
        img = image_tensor.detach().cpu()
        if img.ndim == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        
        # Get dimensions
        B, H, W, C = img.shape
        
        # Ensure mask is (1, H, W)
        mask = mask_tensor.detach().cpu()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != 1:
            mask = mask[0:1]
        
        # Resize mask if needed
        if mask.shape[-2:] != (H, W):
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=(H, W), mode="nearest"
            )[0]
        
        mask = mask.clamp(0, 1)
        
        # Expand mask to match image: (1, H, W) -> (1, H, W, 1) -> (1, H, W, 3)
        mask_3d = mask.unsqueeze(-1).expand(1, H, W, 3)
        
        # Create color tensor in [1, 1, 1, 3] format
        color_tensor = torch.tensor(rgb, dtype=img.dtype).view(1, 1, 1, 3)
        
        # Blend: image * mask + color * (1 - mask)
        blended = img[0:1] * mask_3d + color_tensor * (1 - mask_3d)
        
        # Return as (1, H, W, C) - ComfyUI IMAGE format
        return blended.clamp(0, 1)

    def _try_request(self, image_tensor, prompt, timeout, retries, debug):
        last_err = None
        for attempt in range(retries + 1):
            try:
                req = self._submit_single(image_tensor, prompt, debug)
                result = self._poll_and_fetch(req, timeout=timeout, debug=debug)
                return result
            except Exception as e:
                last_err = e
                if debug:
                    print(f"[PVL Florence2 RETRY] attempt {attempt+1}/{retries+1} failed: {e}")
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        if debug:
            print(f"[PVL Florence2 ERROR] all retries failed: {last_err}")
        return None

    def process_images(self, mask_color, retries, timeout_per_retry, seed, debug, **kwargs):
        pairs = []
        for i in range(1, 6):
            img = kwargs.get(f"image_{i}")
            prm = kwargs.get(f"prompt_{i}", "")
            if img is not None and torch.is_tensor(img):
                pairs.append((i, img, prm))
            elif debug:
                print(f"[PVL Florence2 INFO] missing image_{i}")

        if not pairs:
            if debug:
                print("[PVL Florence2 WARN] no valid images")
            return tuple([None]*10)

        results = {}
        with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
            futs = {
                ex.submit(self._try_request, img, prm, timeout_per_retry, retries, debug): idx
                for idx, img, prm in pairs
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[PVL Florence2 ERROR] request image_{idx}: {e}")

        masked_out = [None]*5
        mask_out = [None]*5

        for idx, data in results.items():
            try:
                if not data:
                    raise RuntimeError("no data returned after retries")
                
                polygons = data.get("results", {}).get("polygons", [])
                img_meta = data.get("image", {})
                url = img_meta.get("url")
                
                if not url:
                    raise RuntimeError("no image url in result")

                # Fetch result image to determine dimensions
                resp = requests.get(url, timeout=60)
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                w, h = pil.size

                # Create mask tensor (1, H, W)
                mask_t = self._rasterize_polygons(polygons, w, h)
                
                # Get original input image
                img_t = next((img for i, img, _ in pairs if i == idx), None)
                if img_t is None:
                    raise RuntimeError(f"source image for idx {idx} not found")
                
                # Apply mask and get blended output (1, H, W, C)
                masked_t = self._apply_mask(img_t, mask_t, mask_color)
                
                if debug:
                    print(f"[PVL Florence2 DEBUG] image_{idx}: mask shape {mask_t.shape}, output shape {masked_t.shape}")
                
                masked_out[idx-1] = masked_t
                mask_out[idx-1] = mask_t

            except Exception as e:
                print(f"[PVL Florence2 ERROR] postproc image_{idx}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()

        # Ensure all outputs are proper tensors in correct format
        for i in range(5):
            # Check image output - must be [B, H, W, C]
            if masked_out[i] is not None:
                t = masked_out[i]
                if not torch.is_tensor(t):
                    masked_out[i] = None
                    continue
                
                # Ensure [B, H, W, C] format
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                
                # Verify it's in the right format
                if t.ndim != 4 or t.shape[-1] != 3:
                    if debug:
                        print(f"[PVL Florence2 ERROR] Invalid image output shape {t.shape}, expected [B, H, W, 3]")
                    masked_out[i] = None
                    continue
                
                masked_out[i] = t.clamp(0, 1).float()
            
            # Check mask output - must be [B, H, W]
            if mask_out[i] is not None:
                t = mask_out[i]
                if not torch.is_tensor(t):
                    mask_out[i] = None
                    continue
                
                # Ensure [B, H, W] format for MASK
                if t.ndim == 2:
                    t = t.unsqueeze(0)
                if t.ndim != 3:
                    mask_out[i] = None
                    continue
                
                mask_out[i] = t.clamp(0, 1).float()

        if debug:
            done = len([m for m in masked_out if m is not None])
            print(f"[PVL Florence2 INFO] completed {done} outputs (some may have failed)")

        return tuple(masked_out + mask_out)

NODE_CLASS_MAPPINGS = {"PVL_Seg_Florence2_API": PVL_Seg_Florence2_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Seg_Florence2_API": "PVL Seg Florence2 (fal.ai)"}
