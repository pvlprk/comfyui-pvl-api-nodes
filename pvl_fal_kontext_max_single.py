import os
import torch
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_KontextMaxSingle_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], {"default": "1:1"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "Pavel/FalAI"

    def edit_image(self, prompt, image, seed, CFG, num_images, output_format, 
                  sync_mode, safety_tolerance, aspect_ratio):
        try:
            # Upload the input image to get a URL
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                raise Exception("Failed to upload image")
            
            # Prepare the arguments for the API call
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "guidance_scale": CFG,  # Using the renamed parameter
                "num_images": num_images,
                "output_format": output_format,
                "sync_mode": sync_mode,
                "safety_tolerance": safety_tolerance,
                "aspect_ratio": aspect_ratio
            }
            
            # Add seed if provided (not -1)
            if seed != -1:
                arguments["seed"] = seed
            
            # Submit the request and get the result
            result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/kontext/max", arguments)
            
            # Process the result and return the image tensor
            return ResultProcessor.process_image_result(result)
            
        except Exception as e:
            print(f"Error editing image with FLUX Kontext Max: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX Kontext Max", e)