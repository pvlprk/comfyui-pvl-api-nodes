import os
import numpy as np
from PIL import Image
import folder_paths

class PVL_SaveOrNot:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "image"}),
                "format": (["png", "jpeg", "webp"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "condition": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_condition"
    OUTPUT_NODE = True
    CATEGORY = "PVL_tools"	
    
    def save_condition(self, images, condition, filename, format, quality):
        try:
            # Early return if condition is False
            if not condition:
                return {"ui": {"images": []}}
            
            # Check if images is None or empty
            if images is None:
                return {"ui": {"images": []}}
                
            # Check if images tensor is empty
            if hasattr(images, 'nelement') and images.nelement() == 0:
                return {"ui": {"images": []}}
            
            # Convert images to numpy array
            try:
                images = images.cpu().numpy()
                images = (images * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error converting images: {e}")
                return {"ui": {"images": []}}
            
            # Check if we have any images after conversion
            if len(images) == 0:
                return {"ui": {"images": []}}
            
            # Prepare output paths
            results = []
            for i, img_array in enumerate(images):
                try:
                    # Generate filename with index for batch
                    base_name = f"{filename}_{i}" if len(images) > 1 else filename
                    full_filename = f"{base_name}.{format}"
                    output_path = os.path.join(self.output_dir, full_filename)
                    
                    # Create directories if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Convert to PIL Image
                    img = Image.fromarray(img_array)
                    
                    # Save with appropriate format and quality
                    if format == "png":
                        img.save(output_path, format='PNG')
                    elif format == "jpeg":
                        img.save(output_path, format='JPEG', quality=quality)
                    elif format == "webp":
                        img.save(output_path, format='WEBP', quality=quality)
                    
                    # Add to results for UI
                    results.append({
                        "filename": full_filename,
                        "subfolder": "",
                        "type": self.type
                    })
                except Exception as e:
                    print(f"Error saving image {i}: {e}")
                    continue
            
            return {"ui": {"images": results}}
            
        except Exception as e:
            print(f"Unexpected error in save_condition: {e}")
            return {"ui": {"images": []}}

NODE_CLASS_MAPPINGS = {
    "PVL_SaveOrNot": PVL_SaveOrNot
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_SaveOrNot": "PVL Save Or Not"
}