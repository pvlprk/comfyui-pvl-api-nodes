import os
import torch
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_KontextDevInpaint_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "lora1_name": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_name": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_name": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inpaint"
    CATEGORY = "Pavel/FalAI"

    def inpaint(self, image, mask, prompt, steps, CFG, strength, num_images, 
                enable_safety_checker, output_format, sync_mode, acceleration,
                reference_image=None, seed=-1,
                lora1_name="", lora1_scale=1.0,
                lora2_name="", lora2_scale=1.0,
                lora3_name="", lora3_scale=1.0):
        try:
            # Upload the input image to get a URL
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                raise Exception("Failed to upload image")
            
            # Upload the mask to get a URL
            mask_url = ImageUtils.upload_image(ImageUtils.mask_to_image(mask))
            if not mask_url:
                raise Exception("Failed to upload mask")
            
            # Upload the reference image if provided
            reference_image_url = None
            if reference_image is not None:
                reference_image_url = ImageUtils.upload_image(reference_image)
                if not reference_image_url:
                    raise Exception("Failed to upload reference image")
            
            # Prepare the arguments for the API call
            arguments = {
                "image_url": image_url,
                "mask_url": mask_url,
                "prompt": prompt,
                "num_inference_steps": steps,  # Using the renamed parameter
                "guidance_scale": CFG,  # Using the renamed parameter
                "strength": strength,
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "sync_mode": sync_mode,
                "acceleration": acceleration
            }
            
            # Add reference image URL if provided
            if reference_image_url:
                arguments["reference_image_url"] = reference_image_url
            
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
            result = ApiHandler.submit_and_get_result("fal-ai/flux-kontext-lora/inpaint", arguments)
            
            # Process the result and return the image tensor
            return ResultProcessor.process_image_result(result)
            
        except Exception as e:
            print(f"Error inpainting with FLUX Kontext Dev: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX Kontext Dev Inpaint", e)