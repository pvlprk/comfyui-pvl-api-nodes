import os
import torch
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxDev_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1392, "min": 256, "max": 1440}),
                "height": ("INT", {"default": 752, "min": 256, "max": 1440}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def generate_image(self, prompt, width, height, steps, CFG, seed, 
                      num_images, enable_safety_checker, output_format, sync_mode,
                      acceleration):
        try:
            # Prepare the arguments for the API call
            arguments = {
                "prompt": prompt,
                "num_inference_steps": steps,  # Using the renamed parameter
                "guidance_scale": CFG,  # Using the renamed parameter
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "sync_mode": sync_mode,
                "image_size": {  # Always use custom dimensions
                    "width": width,
                    "height": height
                },
                "acceleration": acceleration  # FLUX DEV specific parameter
            }
            
            # Add seed if provided (not -1)
            if seed != -1:
                arguments["seed"] = seed
            
            # Submit the request and get the result
            result = ApiHandler.submit_and_get_result("fal-ai/flux-1/dev", arguments)
            
            # Process the result and return the image tensor
            return ResultProcessor.process_image_result(result)
            
        except Exception as e:
            print(f"Error generating image with FLUX DEV: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX DEV", e)