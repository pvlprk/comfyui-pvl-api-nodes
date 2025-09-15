import os
import torch
from .fal_utils import ResultProcessor, ApiHandler

class PVL_fal_QwenImage_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                # list of LoRAs: [{"path": "...", "scale": 1.0}, {...}]
                "loras": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def generate_image(self, prompt, width, height, steps, CFG, seed,
                      num_images, enable_safety_checker, output_format,
                      sync_mode, acceleration, negative_prompt, loras=""):
        try:
            arguments = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": CFG,
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "sync_mode": sync_mode,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "acceleration": acceleration,
                "negative_prompt": negative_prompt,
            }

            if seed != -1:
                arguments["seed"] = seed

            # Parse LoRAs if provided as JSON text
            if loras.strip():
                try:
                    import json
                    lora_list = json.loads(loras)
                    if isinstance(lora_list, list):
                        arguments["loras"] = lora_list
                except Exception as e:
                    print(f"Warning: could not parse LoRAs input: {e}")

            # Call FAL API for Qwen Image
            result = ApiHandler.submit_and_get_result("fal-ai/qwen-image", arguments)

            # Convert result into ComfyUI image tensor
            return ResultProcessor.process_image_result(result)

        except Exception as e:
            print(f"Error generating image with Qwen-Image: {str(e)}")
            return (torch.zeros((1, 64, 64, 3)),)  # fallback dummy image
