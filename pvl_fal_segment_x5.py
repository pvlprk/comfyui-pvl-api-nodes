import os
import io
import time
import base64
import torch
import requests
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig


class PVL_fal_EvfSam_X5_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "semantic_type": ("BOOLEAN", {"default": False}),
                "mask_only": ("BOOLEAN", {"default": True}),
                "use_grounding_dino": ("BOOLEAN", {"default": False}),
                "revert_mask": ("BOOLEAN", {"default": False}),
                "blur_mask": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "expand_mask": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # optional prompts
                "prompt_1": ("STRING", {"multiline": True, "default": ""}),
                "prompt_2": ("STRING", {"multiline": True, "default": ""}),
                "prompt_3": ("STRING", {"multiline": True, "default": ""}),
                "prompt_4": ("STRING", {"multiline": True, "default": ""}),
                "prompt_5": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                # optional images
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1_out", "image_2_out", "image_3_out", "image_4_out", "image_5_out")
    FUNCTION = "process_images"
    CATEGORY = "PVL_tools_FAL"

    # -----------------------------
    # Helper: encode tensor → Base64 Data URI
    # -----------------------------
    def _tensor_to_data_url(self, image_tensor, mime="image/png"):
        if not torch.is_tensor(image_tensor):
            raise RuntimeError("Expected torch.Tensor for image input")
        img = image_tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
            img = img.permute(1, 2, 0)
        img = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    # -----------------------------
    # SINGLE SUBMISSION
    # -----------------------------
    def _submit_single(
        self,
        image_tensor,
        prompt,
        negative_prompt,
        semantic_type,
        mask_only,
        use_grounding_dino,
        revert_mask,
        blur_mask,
        expand_mask,
        fill_holes,
        seed,
        debug,
    ):
        if image_tensor is None:
            return None

        mime = "image/png"
        image_url = self._tensor_to_data_url(image_tensor, mime)

        arguments = {
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "semantic_type": semantic_type,
            "image_url": image_url,
            "mask_only": mask_only,
            "use_grounding_dino": use_grounding_dino,
            "revert_mask": revert_mask,
            "blur_mask": blur_mask if blur_mask % 2 == 1 or blur_mask == 0 else blur_mask + 1,
            "expand_mask": expand_mask,
            "fill_holes": fill_holes,
        }
        if seed != -1:
            arguments["seed"] = seed

        endpoint = "fal-ai/evf-sam"
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        if debug:
            print(f"[PVL Segment X5 DEBUG] Submitting with prompt len={len(prompt or '')}")

        r = requests.post(submit_url, headers=headers, json=arguments, timeout=120)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        return {
            "request_id": req_id,
            "status_url": sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status",
            "response_url": sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}",
        }

    # -----------------------------
    # POLL & FETCH
    # -----------------------------
    def _poll_and_fetch(self, request_info, debug=False, timeout=180):
        if request_info is None:
            return None
        fal_key = FalConfig.get_api_key()
        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]

        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise RuntimeError(f"FAL request timed out after {timeout}s")

            try:
                sr = requests.get(status_url, headers=headers, timeout=10)
                if sr.ok:
                    sdata = sr.json()
                    if sdata.get("status") == "COMPLETED":
                        break
                    if sdata.get("status") == "FAILED":
                        raise RuntimeError(f"FAL request failed: {sdata}")
            except Exception as e:
                if debug:
                    print(f"[PVL Segment X5 DEBUG] Poll error: {e}")
            time.sleep(0.6)

        rr = requests.get(resp_url, headers=headers, timeout=30)
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

        result = rr.json().get("response", rr.json())
        if "image" in result and isinstance(result["image"], dict):
            url = result["image"].get("url")
            if not url:
                return None
            resp = requests.get(url, timeout=60)
            if not resp.ok:
                return None
            pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(pil).astype("float32") / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        return None

    # -----------------------------
    # MAIN PROCESS
    # -----------------------------
    def process_images(
        self,
        semantic_type,
        mask_only,
        use_grounding_dino,
        revert_mask,
        blur_mask,
        expand_mask,
        fill_holes,
        seed,
        debug,
        **kwargs,
    ):
        # extract optional images and prompts
        image_1 = kwargs.get("image_1")
        image_2 = kwargs.get("image_2")
        image_3 = kwargs.get("image_3")
        image_4 = kwargs.get("image_4")
        image_5 = kwargs.get("image_5")

        prompt_1 = kwargs.get("prompt_1", "")
        prompt_2 = kwargs.get("prompt_2", "")
        prompt_3 = kwargs.get("prompt_3", "")
        prompt_4 = kwargs.get("prompt_4", "")
        prompt_5 = kwargs.get("prompt_5", "")
        negative_prompt = kwargs.get("negative_prompt", "")

        pairs = []
        for idx, (img, prm) in enumerate([
            (image_1, prompt_1),
            (image_2, prompt_2),
            (image_3, prompt_3),
            (image_4, prompt_4),
            (image_5, prompt_5),
        ], start=1):
            if img is not None and torch.is_tensor(img):
                pairs.append((idx, img, prm))
            elif debug:
                print(f"[PVL Segment X5 INFO] Skipping missing image_{idx}")

        if not pairs:
            if debug:
                print("[PVL Segment X5 WARNING] No valid images provided. Nothing to process.")
            return (None, None, None, None, None)

        if debug:
            print(f"[PVL Segment X5 INFO] Processing {len(pairs)} image(s) in parallel")

        submit_info = {}
        results = {}

        # ---- PHASE 1: Submit all ----
        with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
            futs = {
                ex.submit(
                    self._submit_single,
                    img,
                    prm,
                    negative_prompt,
                    semantic_type,
                    mask_only,
                    use_grounding_dino,
                    revert_mask,
                    blur_mask,
                    expand_mask,
                    fill_holes,
                    (seed + i) % 4294967296 if seed != -1 else -1,
                    debug,
                ): idx
                for i, (idx, img, prm) in enumerate(pairs)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    submit_info[idx] = fut.result()
                except Exception as e:
                    print(f"[PVL Segment X5 ERROR] Submission failed for image_{idx}: {e}")

        if debug:
            print(f"[PVL Segment X5 DEBUG] Submitted {len(submit_info)} requests successfully")

        # ---- PHASE 2: Poll & Fetch ----
        with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
            futs = {
                ex.submit(self._poll_and_fetch, info, debug): idx
                for idx, info in submit_info.items()
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[PVL Segment X5 ERROR] Poll failed for image_{idx}: {e}")

        outputs = [None] * 5
        for idx, tensor in results.items():
            outputs[idx - 1] = tensor

        if debug:
            print(f"[PVL Segment X5 INFO] Completed {len([o for o in outputs if o is not None])} outputs")

        return tuple(outputs)


NODE_CLASS_MAPPINGS = {"PVL_fal_EvfSam_X5_API": PVL_fal_EvfSam_X5_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_fal_EvfSam_X5_API": "PVL Segment X5 (fal.ai)"}
