import io
import re
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from .fal_utils import ImageUtils, ResultProcessor, ApiHandler


class PVL_Qwen_Img_Edit_Plus:
    """
    ComfyUI node for FAL 'fal-ai/qwen-image-edit-plus' — multi-image edit using parallel API calls.

    Features:
      • Parallel API calls (num_images determines number of requests)
      • All input images submitted to each call (combined context)
      • Retry with linear backoff (1–2–3 s)
      • Timeout control
      • Aspect ratio selector ('custom' + presets)
      • Safety checker toggle (default: off)
      • Negative prompt support (sent to API)
      • Delimiter-based prompt splitting
      • Debug logging
      • Seed auto-increment per call: seed, seed+1, …
    """

    _AR_TO_API = {
        "1:1": "square_hd",
        "3:4": "portrait_4_3",
        "9:16": "portrait_16_9",
        "4:3": "landscape_4_3",
        "16:9": "landscape_16_9",
    }

    @classmethod
    def INPUT_TYPES(cls):
        image_inputs = {f"image{i}": ("IMAGE",) for i in range(1, 7)}
        required_inputs = {"prompt": ("STRING", {"multiline": True})}
        optional_inputs = {
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "aspect_ratio": (
                ["custom", "1:1", "3:4", "9:16", "4:3", "16:9"],
                {"default": "custom"},
            ),
            "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
            "delimiter": ("STRING", {"default": "[++]", "multiline": False}),
            "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 12}),
            "enable_safety_checker": ("BOOLEAN", {"default": False}),
            "output_format": (["jpeg", "png"], {"default": "png"}),
            "acceleration": (["none", "regular"], {"default": "regular"}),
            "sync_mode": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0}),
            "debug": ("BOOLEAN", {"default": False}),
            "timeout_sec": ("INT", {"default": 180, "min": 10, "max": 600}),
            "max_retries": ("INT", {"default": 3, "min": 0, "max": 10}),
        }

        return {"required": required_inputs, "optional": {**image_inputs, **optional_inputs}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt")
    FUNCTION = "edit_images"
    CATEGORY = "PVL_tools"

    # ---------- helpers ----------
    def _raise(self, msg: str):
        raise RuntimeError(msg)

    def _split_image_batch(self, image) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return [image[i] for i in range(image.shape[0])]
            elif image.ndim == 3:
                return [image]
            else:
                self._raise("Unsupported image tensor dimensionality.")
        elif isinstance(image, np.ndarray):
            return self._split_image_batch(torch.from_numpy(image))
        else:
            self._raise("Unsupported image type.")

    def _prepare_image_urls(self, imgs: List[torch.Tensor]) -> List[str]:
        urls = []
        for idx, img in enumerate(imgs):
            frames = self._split_image_batch(img)
            if frames:
                try:
                    urls.append(ImageUtils.image_to_data_uri(frames[0]))
                except Exception as e:
                    print(f"[WARN] Failed to encode image {idx+1}: {e}")
        return urls

    def _stack_images(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            raise RuntimeError("No images to stack.")
        try:
            return torch.cat(tensors, dim=0)
        except RuntimeError:
            from PIL import Image
            h, w = tensors[0].shape[1:3]
            fixed = []
            for t in tensors:
                arr = (t.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                if t.ndim == 4:
                    arr = arr[0]
                pil = Image.fromarray(arr)
                rp = pil.resize((w, h), Image.LANCZOS)
                fixed.append(torch.from_numpy(np.array(rp, dtype=np.float32) / 255.0)[None, ...])
            return torch.cat(fixed, dim=0)

    def _map_aspect_ratio(self, ar_value, width, height, debug):
        if ar_value == "custom":
            if debug:
                print(f"[QWEN NODE] Custom aspect ratio: {width}x{height}")
            return {"width": int(width), "height": int(height)}
        return self._AR_TO_API.get(ar_value, "square_hd")

    # ---------- main ----------
    def edit_images(
        self,
        prompt,
        aspect_ratio="custom",
        width=1024,
        height=1024,
        delimiter="[++]",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=4.0,
        num_images=1,
        enable_safety_checker=False,
        output_format="png",
        acceleration="regular",
        sync_mode=False,
        seed=0,
        debug=False,
        timeout_sec=180,
        max_retries=3,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            self._raise("Prompt is required.")

        imgs = [i for i in [image1, image2, image3, image4, image5, image6] if i is not None]
        if not imgs:
            self._raise("At least one input image is required.")

        # Split prompts by delimiter
        try:
            parts = [p.strip() for p in re.split(delimiter, prompt) if p.strip()]
        except re.error:
            parts = [p.strip() for p in prompt.split(delimiter) if p.strip()]
        if not parts:
            parts = [prompt.strip()]

        # Adjust number of prompts
        prompts = (parts + [parts[-1]] * (num_images - len(parts)))[:num_images]

        img_urls = self._prepare_image_urls(imgs)
        if not img_urls:
            self._raise("Failed to encode any input images.")

        image_size_value = self._map_aspect_ratio(aspect_ratio, width, height, debug)
        call_count = num_images

        if debug:
            print(f"[QWEN NODE] Launching {call_count} parallel API calls (each using {len(img_urls)} image(s))")
            print(f"[QWEN NODE] Aspect ratio: {aspect_ratio}, safety checker: {enable_safety_checker}")
            print(f"[QWEN NODE] Negative prompt: {negative_prompt or '(empty)'}")

        # ---------- worker ----------
        def worker(idx, ptxt, effective_seed: int):
            for attempt in range(1, max_retries + 1):
                args = {
                    "prompt": ptxt,
                    "negative_prompt": negative_prompt or " ",
                    "image_urls": img_urls,  # all input images
                    "image_size": image_size_value,
                    "num_inference_steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
                    "num_images": 1,
                    "enable_safety_checker": bool(enable_safety_checker),
                    "output_format": output_format,
                    "acceleration": acceleration,
                    "sync_mode": bool(sync_mode),
                }
                if effective_seed > 0:
                    args["seed"] = int(effective_seed)

                try:
                    if debug:
                        show_seed = args.get("seed", "random")
                        print(f"[QWEN {idx+1}] Attempt {attempt}/{max_retries} | seed={show_seed}")
                    res = ApiHandler.submit_and_get_result("fal-ai/qwen-image-edit-plus", args)
                    imgs = ResultProcessor.process_image_result(res)[0]
                    if debug:
                        print(f"[QWEN {idx+1}] OK ({imgs.shape})")
                    return imgs
                except Exception as e:
                    if attempt < max_retries:
                        wait = attempt
                        if debug:
                            print(f"[QWEN {idx+1}] Retry {attempt}: {e} (wait {wait}s)")
                        time.sleep(wait)
                    else:
                        if debug:
                            print(f"[QWEN {idx+1}] Failed after {max_retries} retries: {e}")
                        return None

        # ---------- seeds ----------
        if seed > 0:
            per_call_seeds = [seed + i for i in range(call_count)] if num_images > 1 else [seed]
        else:
            per_call_seeds = [0 for _ in range(call_count)]

        # ---------- parallel execution ----------
        results: Dict[int, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=min(call_count, 6)) as ex:
            futs = [ex.submit(worker, i, prompts[i], per_call_seeds[i]) for i in range(call_count)]
            for i, fut in enumerate(as_completed(futs, timeout=timeout_sec)):
                res = fut.result()
                if res is not None:
                    results[i] = res

        if not results:
            self._raise("All parallel API calls failed.")

        stacked = self._stack_images([results[i] for i in sorted(results.keys())])

        if debug:
            print(f"[QWEN NODE] Completed {len(results)}/{call_count} successful calls")

        return (stacked, prompt)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {"PVL_Qwen_Img_Edit_Plus": PVL_Qwen_Img_Edit_Plus}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Qwen_Img_Edit_Plus": "PVL Qwen Image Edit Plus (Parallel)"}
