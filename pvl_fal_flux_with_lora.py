import os
import torch
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxWithLora_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1392, "min": 256, "max": 1440}),
                "height": ("INT", {"default": 752, "min": 256, "max": 1440}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora1_name": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_name": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_name": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Pavel/FalAI"

    def generate_image(self, prompt, width, height, steps, CFG, seed, 
                      num_images, enable_safety_checker, output_format, sync_mode,
                      lora1_name="", lora1_scale=1.0,
                      lora2_name="", lora2_scale=1.0,
                      lora3_name="", lora3_scale=1.0):
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
                "image_size": {  # Always use custom dimensions now
                    "width": width,
                    "height": height
                }
            }
            
            # Add seed if provided (not -1)
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle LoRAs
            loras = []
            
            # Add LoRA 1 if provided
            if lora1_name.strip():
                loras.append({
                    "path": lora1_name.strip(),
                    "scale": lora1_scale
                })
            
            # Add LoRA 2 if provided
            if lora2_name.strip():
                loras.append({
                    "path": lora2_name.strip(),
                    "scale": lora2_scale
                })
            
            # Add LoRA 3 if provided
            if lora3_name.strip():
                loras.append({
                    "path": lora3_name.strip(),
                    "scale": lora3_scale
                })
            
            if loras:
                arguments["loras"] = loras
            
            # Submit the request and get the result
            result = ApiHandler.submit_and_get_result("fal-ai/flux-lora", arguments)
            
            # Process the result and return the image tensor
            return ResultProcessor.process_image_result(result)
            
        except Exception as e:
            print(f"Error generating image with FLUX: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX", e)