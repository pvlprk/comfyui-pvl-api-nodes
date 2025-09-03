# pvl_crop2AR.py
# Node: "PVL Crop to Aspect Ratio"
#
# Inputs:
#   - image (IMAGE)
#   - aspect_ratio (STRING), e.g. "16:9" (also accepts "4x5", "3/2", or a float like "1.7778")
#
# Behavior:
#   - If input image already matches the target AR, pass through unchanged.
#   - Else, center-crop to the target AR WITHOUT resizing. Batch-safe.
#
# Category: PVL/Image

import re
from typing import Optional, Tuple

import torch


class PVL_Crop2AR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_to_ar"
    CATEGORY = "PVL_tools"

    # ----------------- Helpers -----------------

    @staticmethod
    def _ensure_4d(img: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is float32 [B,H,W,C] in [0,1].
        Accepts [H,W,C] or [B,H,W,C].
        """
        if img is None or not torch.is_tensor(img):
            raise ValueError("image is required and must be a tensor")
        t = img
        if not torch.is_floating_point(t):
            t = t.float()
        if t.ndim == 3:
            t = t.unsqueeze(0)  # [1,H,W,C]
        if t.ndim != 4:
            raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(img.shape)}")
        return t

    @staticmethod
    def _parse_aspect_ratio(s: str) -> Optional[float]:
        """
        Parse aspect ratio from string.
        Supports "W:H", "WxH", "W/H", or a single float like "1.7778".
        Returns positive float ratio (width/height) or None if invalid.
        """
        if not isinstance(s, str):
            return None
        raw = s.strip().lower().replace(" ", "")
        if not raw:
            return None

        # Try split forms first: "16:9", "4x5", "3/2"
        if any(ch in raw for ch in [":", "x", "×", "/"]):
            parts = re.split(r"[:x×/]", raw)
            if len(parts) == 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    if a > 0 and b > 0:
                        return a / b
                except Exception:
                    return None

        # Fallback: single float string, e.g. "1.7778"
        try:
            val = float(raw)
            if val > 0:
                return val
        except Exception:
            pass
        return None

    @staticmethod
    def _already_matches_ar(W: int, H: int, ratio: float) -> bool:
        """
        Decide if (W,H) already matches `ratio` using integer rounding to pixel grid.
        Consider equal if either rounded dimension matches exactly.
        """
        if W <= 0 or H <= 0 or ratio <= 0:
            return False
        # If we can achieve the ratio without cropping due to rounding, treat as matched.
        expected_w = int(round(H * ratio))
        if expected_w == W:
            return True
        expected_h = int(round(W / ratio))
        if expected_h == H:
            return True
        return False

    # ----------------- Main -----------------

    def crop_to_ar(self, image: torch.Tensor, aspect_ratio: str):
        t = self._ensure_4d(image)  # [B,H,W,C]
        B, H, W, C = t.shape

        ratio = self._parse_aspect_ratio(aspect_ratio)
        if ratio is None or ratio <= 0:
            # Invalid ratio -> pass-through unchanged
            return (t,)

        # If already matches, pass-through
        if self._already_matches_ar(W, H, ratio):
            return (t,)

        # Decide crop region (centered)
        current_ratio = W / H
        if current_ratio > ratio:
            # Too wide -> crop width
            new_w = int(round(H * ratio))
            new_w = max(1, min(W, new_w))
            pad_left = (W - new_w) // 2
            pad_right = pad_left + new_w
            cropped = t[:, :, pad_left:pad_right, :]
        else:
            # Too tall -> crop height
            new_h = int(round(W / ratio))
            new_h = max(1, min(H, new_h))
            pad_top = (H - new_h) // 2
            pad_bottom = pad_top + new_h
            cropped = t[:, pad_top:pad_bottom, :, :]

        return (cropped,)