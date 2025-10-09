import os
import time
import torch
import requests
import io
from PIL import Image
import numpy as np
from .fal_utils import ImageUtils, ApiHandler

class PVL_fal_FluxDevPulidAvatar_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "man with pink hair wearing shorts", "multiline": True}),
                "style": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "image": ("IMAGE",),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "PVL_tools_FAL"

    def generate(self, prompt, style, batch, image, debug=False, seed=0):
        _t0 = time.time()

        try:
            img_data_uri = ImageUtils.image_to_data_uri(image)

            if debug:
                print(f"[PVL_FAL_AVATAR] Image converted to data URI ({len(img_data_uri)} chars)")

            arguments = {
                "prompt": prompt,
                "style": style,
                "batch": batch,
                "ref image": img_data_uri,
            }

            if seed and seed > 0:
                arguments["seed"] = seed
                if debug:
                    print(f"[PVL_FAL_AVATAR] Using seed: {seed}")

            if debug:
                print(f"[PVL_FAL_AVATAR] Submitting request")

            result = ApiHandler.submit_and_get_result("comfy/Mini-Studio/flux-dev-pulid-avatar", arguments)

            if debug:
                print(f"[PVL_FAL_AVATAR] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

            final_tensor = self._download_images_from_result(result, debug)

            _t1 = time.time()
            elapsed = _t1 - _t0
            print(f"[PVL_FAL_AVATAR] Successfully generated {batch} image(s) in {elapsed:.2f}s")

            return final_tensor

        except Exception as e:
            _t1 = time.time()
            elapsed = _t1 - _t0
            print(f"[PVL_FAL_AVATAR] Error after {elapsed:.2f}s: {str(e)}")

            if debug:
                import traceback
                traceback.print_exc()

            return (torch.zeros((1, 512, 512, 3)),)

    def _download_images_from_result(self, result, debug=False):
        output_images = []

        if not isinstance(result, dict) or "outputs" not in result:
            raise RuntimeError("Invalid result structure")

        outputs = result["outputs"]

        for node_id, node_data in outputs.items():
            if not isinstance(node_data, dict) or "images" not in node_data:
                continue

            images_list = node_data["images"]
            if not isinstance(images_list, list):
                continue

            if debug:
                print(f"[PVL_FAL_AVATAR] Node {node_id}: Processing {len(images_list)} image(s)")

            for img_info in images_list:
                if not isinstance(img_info, dict) or "url" not in img_info:
                    continue

                img_url = img_info["url"]

                if debug:
                    print(f"[PVL_FAL_AVATAR] Downloading: {img_url}")

                try:
                    response = requests.get(img_url, timeout=30)
                    response.raise_for_status()

                    pil_img = Image.open(io.BytesIO(response.content))
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    img_array = np.array(pil_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)[None,]
                    output_images.append(img_tensor)

                    if debug:
                        print(f"[PVL_FAL_AVATAR] Loaded: {pil_img.size}")

                except Exception as e:
                    print(f"[PVL_FAL_AVATAR] Download failed: {e}")

        if not output_images:
            raise RuntimeError("No images downloaded")

        return (torch.cat(output_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "PVL_fal_FluxDevPulidAvatar_API": PVL_fal_FluxDevPulidAvatar_API
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_FluxDevPulidAvatar_API": "PVL FAL Flux Dev Pulid Avatar"
}
