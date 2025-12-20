# pvl_crop_to_mask.py
# ComfyUI custom node: crop image to mask bounding box with % padding, optional alpha masking.

import torch

class PVL_CropToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vertical_padding": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "horizontal_padding": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop_to_mask"
    CATEGORY = "PVL Nodes/Image"

    def crop_to_mask(self, image, mask, vertical_padding, horizontal_padding, apply_mask):
        """
        image: torch float tensor [B,H,W,3] in 0..1
        mask:  torch float tensor [B,H,W] (or [H,W]) in 0..1
        Returns:
            image_out: [B,h,w,3] or [B,h,w,4]
            mask_out:  [B,h,w]   (ComfyUI MASK)
        """

        # Normalize mask shape to [B,H,W]
        if mask.dim() == 2:
            mask_b = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask_b = mask
        else:
            raise ValueError(f"MASK must be [H,W] or [B,H,W], got {tuple(mask.shape)}")

        if image.dim() != 4 or image.shape[-1] < 3:
            raise ValueError(f"IMAGE must be [B,H,W,C>=3], got {tuple(image.shape)}")

        B, H, W, C = image.shape
        if mask_b.shape[-2:] != (H, W):
            raise ValueError("IMAGE and MASK spatial sizes must match")

        # Combine mask across batch for consistent crop
        combined = mask_b.max(dim=0).values  # [H,W]
        active = combined > 0.0
        coords = torch.nonzero(active, as_tuple=False)

        # Empty mask → return original
        if coords.numel() == 0:
            if apply_mask:
                alpha = mask_b.clamp(0.0, 1.0).unsqueeze(-1)
                rgba = torch.cat([image[..., :3], alpha], dim=-1)
                return rgba, mask_b
            return image, mask_b

        y0 = int(coords[:, 0].min())
        y1 = int(coords[:, 0].max())
        x0 = int(coords[:, 1].min())
        x1 = int(coords[:, 1].max())

        crop_h = y1 - y0 + 1
        crop_w = x1 - x0 + 1

        pad_y = int(round(crop_h * float(vertical_padding)))
        pad_x = int(round(crop_w * float(horizontal_padding)))

        y0p = max(0, y0 - pad_y)
        y1p = min(H - 1, y1 + pad_y)
        x0p = max(0, x0 - pad_x)
        x1p = min(W - 1, x1 + pad_x)

        ys = slice(y0p, y1p + 1)
        xs = slice(x0p, x1p + 1)

        img_crop = image[:, ys, xs, :3].clamp(0.0, 1.0)
        mask_crop = mask_b[:, ys, xs].clamp(0.0, 1.0)

        if apply_mask:
            alpha = mask_crop.unsqueeze(-1)
            img_crop = torch.cat([img_crop, alpha], dim=-1)

        return img_crop, mask_crop
