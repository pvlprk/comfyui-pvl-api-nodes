import os
import torch
import numpy as np
import json
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
    CATEGORY = "PVL_tools_FAL"

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
            
            # Log the full result for debugging (comment out in production)
            print("API Response:", json.dumps(result, indent=2, default=str))
            
            # Check for NSFW indicators in the response
            if self._is_nsfw_response(result):
                print("NSFW content detected in API response - returning empty output")
                return tuple()
            
            # Process the result and get the image tensor
            processed_result = ResultProcessor.process_image_result(result)
            
            # Check if the result is a black image (likely NSFW)
            if processed_result and len(processed_result) > 0:
                img_tensor = processed_result[0]
                # Check if image is entirely black (NSFW content)
                if torch.all(img_tensor == 0):
                    print("NSFW content detected (black image) - returning empty output")
                    return tuple()
                
            return processed_result
            
        except Exception as e:
            # Print error to console
            print(f"Error in FLUX Kontext Max: {str(e)}")
            
            # Return empty tuple to prevent output
            return tuple()
    
    def _is_nsfw_response(self, result):
        """
        Check if the API response indicates NSFW content.
        This method looks for common indicators in the response structure.
        """
        # Check if result is a dictionary
        if not isinstance(result, dict):
            return False
            
        # Check for common NSFW indicator fields
        nsfw_indicators = [
            'nsfw', 'is_nsfw', 'content_flag', 'safety_attributes', 
            'content_filter', 'moderation', 'safety'
        ]
        
        # Check top-level fields
        for indicator in nsfw_indicators:
            if indicator in result:
                # If the field exists and is truthy, consider it NSFW
                if result[indicator]:
                    print(f"NSFW detected via '{indicator}' field: {result[indicator]}")
                    return True
        
        # Check for NSFW flags in images
        if 'images' in result and isinstance(result['images'], list):
            for img in result['images']:
                for indicator in nsfw_indicators:
                    if indicator in img and img[indicator]:
                        print(f"NSFW detected in image via '{indicator}' field: {img[indicator]}")
                        return True
        
        # Check for error messages that might indicate NSFW filtering
        if 'error' in result and isinstance(result['error'], dict):
            error_msg = result['error'].get('message', '').lower()
            if any(term in error_msg for term in ['nsfw', 'safety', 'filter', 'moderation']):
                print(f"NSFW detected via error message: {error_msg}")
                return True
                
        return False
