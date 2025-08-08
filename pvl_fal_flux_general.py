import os
import torch
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler

class PVL_fal_FluxGeneral_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1392, "min": 256, "max": 1440}),
                "height": ("INT", {"default": 752, "min": 256, "max": 1440}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "negative_prompt": ("STRING", {"default": ""}),
                "scheduler": (["euler", "dpmpp_2m"], {"default": "euler"}),
                "use_beta_schedule": ("BOOLEAN", {"default": False}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "nag_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nag_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image": ("IMAGE",),
                "reference_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def generate_image(self, prompt, width, height, steps, CFG, num_images, 
                      enable_safety_checker, output_format, sync_mode,
                      seed=-1, negative_prompt="", scheduler="euler",
                      use_beta_schedule=False, base_shift=0.5, max_shift=1.15,
                      nag_scale=3.0, nag_tau=2.5, nag_alpha=0.25, nag_end=0.25,
                      reference_image=None, reference_strength=0.65,
                      reference_start=0.0, reference_end=1.0,
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
                "image_size": {  # Always use custom dimensions
                    "width": width,
                    "height": height
                },
                "scheduler": scheduler,
                "base_shift": base_shift,
                "max_shift": max_shift,
                "nag_scale": nag_scale,
                "nag_tau": nag_tau,
                "nag_alpha": nag_alpha,
                "nag_end": nag_end
            }
            
            # Add negative prompt if provided
            if negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt
            
            # Add use_beta_schedule if True
            if use_beta_schedule:
                arguments["use_beta_schedule"] = use_beta_schedule
            
            # Add seed if provided (not -1)
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle reference image if provided
            if reference_image is not None:
                reference_image_url = ImageUtils.upload_image(reference_image)
                if reference_image_url:
                    arguments["reference_image_url"] = reference_image_url
                    arguments["reference_strength"] = reference_strength
                    arguments["reference_start"] = reference_start
                    arguments["reference_end"] = reference_end
            
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
            result = ApiHandler.submit_and_get_result("fal-ai/flux-general", arguments)
            
            # Process the result and return the image tensor
            return ResultProcessor.process_image_result(result)
            
        except Exception as e:
            print(f"Error generating image with FLUX General: {str(e)}")
            return ApiHandler.handle_image_generation_error("FLUX General", e)