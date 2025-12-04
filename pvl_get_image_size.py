import torch
import math

class PVL_Get_Image_Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "PVL_tools/image"

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "width",
        "height",
        "width_1K",
        "height_1K",
        "rounded_aspect_ratio",
        "standard_aspect_ratio"
    )

    FUNCTION = "get_size"

    DESCRIPTION = """
Reads the input image and outputs:
- width (int)
- height (int)
- scaled width for 1Mpx (1K) area
- scaled height for 1Mpx (1K) area
- rounded integer aspect ratio (e.g., 16:9)
- closest standard aspect ratio from a fixed list
"""

    # Standard aspect ratios provided by the user
    STANDARD_RATIOS = [
        "21:9",
        "16:9",
        "4:3",
        "3:2",
        "5:4",
        "1:1",
        "4:5",
        "2:3",
        "3:4",
        "9:16",
        "9:21",
    ]

    def _parse_ratio(self, ratio_str):
        """Convert '16:9' -> float(16/9)."""
        w, h = ratio_str.split(":")
        return float(w) / float(h)

    def _closest_standard_ratio(self, ar_float):
        """Return closest ratio from STANDARD_RATIOS."""
        best_ratio = None
        best_diff = float("inf")

        for ratio_str in self.STANDARD_RATIOS:
            r = self._parse_ratio(ratio_str)
            diff = abs(r - ar_float)
            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio_str

        return best_ratio

    def get_size(self, image):
        # image shape: (B, H, W, C)
        _, H, W, _ = image.shape

        width = int(W)
        height = int(H)

        # 1K (1 million pixel) scaling factor
        scale = math.sqrt(1_000_000 / (width * height))

        width_1K = int(round(width * scale))
        height_1K = int(round(height * scale))

        # Rounded integer aspect ratio
        # Example: width=4032, height=3024 → 4:3
        gcd = math.gcd(width, height)
        rw = width // gcd
        rh = height // gcd
        rounded_aspect_ratio = f"{rw}:{rh}"

        # Standard aspect ratio
        ar_float = width / height
        standard_aspect_ratio = self._closest_standard_ratio(ar_float)

        return (
            width,
            height,
            width_1K,
            height_1K,
            rounded_aspect_ratio,
            standard_aspect_ratio,
        )
