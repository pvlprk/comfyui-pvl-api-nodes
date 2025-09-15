import os
import torch
import numpy as np
import json
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxPro_Fill_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_image"
    CATEGORY = "PVL_tools"

    def _raise(self, msg):
        # Helper to standardize raising: ComfyUI will stop the workflow
        raise RuntimeError(msg)

    def _upload_mask(self, mask):
        # Convert mask tensor to PIL in grayscale
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        if mask_np.ndim == 4:
            mask_np = mask_np.squeeze(0)
        if mask_np.ndim == 3:
            # If it's 3 channels, take the first channel
            mask_np = mask_np[:, :, 0]
        # Now mask_np is 2D

        if mask_np.dtype in [np.float32, np.float64]:
            mask_np = (mask_np * 255).astype(np.uint8)

        mask_pil = Image.fromarray(mask_np, mode='L')

        # Now upload the mask_pil
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                mask_pil.save(temp_file, format="PNG")
                temp_file_path = temp_file.name
            client = FalConfig().get_client()
            mask_url = client.upload_file(temp_file_path)
            if not mask_url:
                raise RuntimeError("FAL: upload_file returned no URL for mask.")
            return mask_url
        except Exception as e:
            raise RuntimeError(f"FAL: error uploading mask: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def fill_image(self, prompt, image, mask, seed, num_images, output_format, 
                  sync_mode, safety_tolerance):
        # Upload the input image to get a URL
        image_url = ImageUtils.image_to_data_uri(image)
        if not image_url:
            self._raise("FAL: failed to upload input image.")

        # Upload the mask to get a URL
        mask_url = self._upload_mask(mask)
        if not mask_url:
            self._raise("FAL: failed to upload mask.")

        # Prepare the arguments for the API call
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance
        }
        if seed != -1:
            arguments["seed"] = seed

        # Submit the request and get the result (ApiHandler re-raises on failures)
        result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/v1/fill", arguments)

        # Basic structural validations
        if not isinstance(result, dict):
            self._raise("FAL: unexpected response type (expected dict).")
        if "images" not in result or not result["images"]:
            # Some errors come under an 'error' key; surface that if present
            err_msg = None
            if isinstance(result.get("error"), dict):
                err_msg = result["error"].get("message") or result["error"].get("detail")
            self._raise(f"FAL: no images returned{f' ({err_msg})' if err_msg else ''}.")

        # NSFW detection via official field
        has_nsfw = result.get("has_nsfw_concepts")
        if isinstance(has_nsfw, list) and any(bool(x) for x in has_nsfw):
            self._raise("FAL: NSFW content detected by safety system (has_nsfw_concepts).")

        # Process images (may raise)
        processed_result = ResultProcessor.process_image_result(result)

        # Check for black/empty image(s) and abort
        if processed_result and len(processed_result) > 0:
            img_tensor = processed_result[0]
            if not isinstance(img_tensor, torch.Tensor):
                self._raise("FAL: internal error — processed image is not a tensor.")
            # Consider a frame 'black' if all pixels are exactly zero OR mean is extremely low
            if torch.all(img_tensor == 0) or (img_tensor.mean() < 1e-6):
                self._raise("FAL: received an all-black image (likely filtered/failed).")

        return processed_result