import torch
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import resize
import comfy.utils

class PVL_Image_Translation_Offset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "shifted_image": ("IMAGE",),
                "mask": ("MASK",),
                "upsample_factor": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 200,
                    "step": 10,
                    "display": "number"
                    
                }),
                "overlap_ratio": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "number"
                }),                
                "delimiter": ("STRING", {
                    "default": "[++]"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("x_offsets_str", "y_offsets_str")
    FUNCTION = "calculate_offset"
    CATEGORY = "image/alignment"
    
    def calculate_offset(self, original_image, shifted_image, mask, upsample_factor, overlap_ratio, delimiter):
        # Get batch sizes (1 if single)
        batch_orig = original_image.shape[0] if original_image.dim() == 4 else 1
        batch_shift = shifted_image.shape[0] if shifted_image.dim() == 4 else 1
        batch_mask = mask.shape[0] if mask.dim() == 4 else 1
        num_pairs = max(batch_orig, batch_shift, batch_mask)
        
        x_offsets = []
        y_offsets = []
        
        for i in range(num_pairs):
            # Select indices: direct pair or repeat last from shorter batch
            orig_idx = min(i, batch_orig - 1)
            shift_idx = min(i, batch_shift - 1)
            mask_idx = min(i, batch_mask - 1)
            
            # Extract images for this pair (handle single vs batch)
            if batch_orig > 1:
                orig = original_image[orig_idx].cpu().numpy()
            else:
                orig = original_image.cpu().numpy()
            
            if batch_shift > 1:
                shifted = shifted_image[shift_idx].cpu().numpy()
            else:
                shifted = shifted_image.cpu().numpy()
            
            # Extract mask for this pair (handle single vs batch)
            if batch_mask > 1:
                mask_np = mask[mask_idx].cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()
            
            # Binarize mask at fixed threshold 0.5: pixels > 0.5 are masked (excluded), others valid
            # Invert to get valid_mask (True for unmasked regions outside change area)
            mask_binary = (mask_np > 0.5).astype(np.float32)
            valid_mask = np.logical_not(mask_binary).astype(bool)
            
            # Ensure valid_mask is 2D
            if valid_mask.ndim > 2:
                valid_mask = valid_mask.squeeze()
            
            # Convert to grayscale if multi-channel (average across channels)
            if orig.ndim == 3 and orig.shape[2] > 1:
                orig_gray = np.mean(orig, axis=2)
                shifted_gray = np.mean(shifted, axis=2)
            else:
                orig_gray = orig.squeeze()
                shifted_gray = shifted.squeeze()
            
            # Ensure valid_mask matches current pair's image shape (resize if needed)
            if valid_mask.shape != orig_gray.shape:
                valid_mask_resized = resize(valid_mask.astype(float), orig_gray.shape, anti_aliasing=False) > 0.5
                valid_mask_pair = valid_mask_resized.astype(bool)
            else:
                valid_mask_pair = valid_mask
            
            # Compute shift using masked phase cross-correlation
            shift, error, diffphase = phase_cross_correlation(
                reference_image=orig_gray,
                moving_image=shifted_gray,
                upsample_factor=upsample_factor,
                reference_mask=valid_mask_pair,
                moving_mask=valid_mask_pair,
                overlap_ratio=overlap_ratio,
                normalization='phase'
            )
            
            # Append offsets with * -1 sign convention
            x_offset = float(shift[1]) * -1
            y_offset = float(shift[0]) * -1
            x_offsets.append(str(x_offset))
            y_offsets.append(str(y_offset))
        
        # Join offsets with delimiter for separate x and y strings
        x_offsets_str = delimiter.join(x_offsets)
        y_offsets_str = delimiter.join(y_offsets)
        
        return (x_offsets_str, y_offsets_str)

NODE_CLASS_MAPPINGS = {"PVL_Image_Translation_Offset": PVL_Image_Translation_Offset}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Image_Translation_Offset": "PVL Image Translation Offset Detector"}
