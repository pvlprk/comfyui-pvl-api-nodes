import torch
import torch.nn.functional as F
from comfy import model_management
from comfy.utils import common_upscale


class PVL_ImageResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (
                    ["stretch", "resize", "pad", "pad_edge", "crop"],
                    {"default": "resize"},
                ),
                "pad_color": (
                    "STRING",
                    {
                        "default": "0, 0, 0",
                        "tooltip": "Color to use for padding (RGB or RGBA). Example: '255, 255, 255, 0' for transparent white.",
                    },
                ),
                "crop_position": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center"},
                ),
                "divisible_by": (
                    "INT",
                    {"default": 2, "min": 0, "max": 512, "step": 1},
                ),
                "downsize_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only resize if target dimensions are smaller than original.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "PVL_tools"
    DESCRIPTION = """
Resizes the image to the specified width and height.
Keep proportions maintains the aspect ratio of the image.
Downsize only prevents upscaling when enabled.
Supports RGB and RGBA images.
"""

    def resize(
        self,
        image=None,
        width=512,
        height=512,
        keep_proportion="resize",
        upscale_method="bicubic",
        divisible_by=2,
        pad_color="0, 0, 0",
        crop_position="center",
        downsize_only=False,
        device="cpu",
    ):
        if image is None:
            return (None, 0, 0)

        B, H, W, C = image.shape
        original_width = W
        original_height = H

        if downsize_only:
            if width > original_width:
                width = original_width
            if height > original_height:
                height = original_height

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        if width == 0:
            width = W
        if height == 0:
            height = H

        # proportional resizing
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
                new_height = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_height = round(H * ratio)
                new_width = width
            elif width != 0 and height != 0:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)
            else:
                new_width = W
                new_height = H
            if keep_proportion.startswith("pad"):
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top
            width = new_width
            height = new_height

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        out_image = image.clone().to(device)

        # cropping logic
        if keep_proportion == "crop":
            old_width = W
            old_height = H
            old_aspect = old_width / old_height
            new_aspect = width / height

            if old_aspect > new_aspect:
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:
                crop_w = old_width
                crop_h = round(old_width / new_aspect)

            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            elif crop_position == "right":
                x = old_width - crop_w
                y = (old_height - crop_h) // 2

            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)

        # upscale
        out_image = common_upscale(
            out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled"
        ).movedim(1, -1)

        # padding if needed
        if keep_proportion.startswith("pad"):
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height
                out_image, _ = self.pad(
                    out_image,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    0,
                    pad_color,
                    "edge" if keep_proportion == "pad_edge" else "color",
                )

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1])

    def pad(self, image, left, right, top, bottom, extra_padding, color, mode):
        B, H, W, C = image.shape

        # Parse pad color (supports RGB or RGBA)
        bg_color = [float(x.strip()) / 255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * C
        elif len(bg_color) == 3 and C == 4:
            # Add alpha = 1.0 if image has alpha but color doesn’t specify it
            bg_color.append(1.0)
        elif len(bg_color) != C:
            # Normalize if mismatch
            if len(bg_color) < C:
                bg_color += [1.0] * (C - len(bg_color))
            else:
                bg_color = bg_color[:C]

        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        padded_width = W + left + right
        padded_height = H + top + bottom
        out_image = torch.zeros(
            (B, padded_height, padded_width, C), dtype=image.dtype, device=image.device
        )

        for b in range(B):
            if mode == "edge":
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H - 1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W - 1, :]

                out_image[b, :top, :, :] = top_edge.mean(dim=0)
                out_image[b, top + H :, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :left, :] = left_edge.mean(dim=0)
                out_image[b, :, left + W :, :] = right_edge.mean(dim=0)
                out_image[b, top : top + H, left : left + W, :] = image[b]
            else:
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, top : top + H, left : left + W, :] = image[b]

        return (
            out_image,
            torch.ones(
                (B, padded_height, padded_width),
                dtype=image.dtype,
                device=image.device,
            ),
        )


NODE_CLASS_MAPPINGS = {"PVL_ImageResize": PVL_ImageResize}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_ImageResize": "PVL Image Resize"}
