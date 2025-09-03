# pvl_stitch2size.py
# Node: "PVL Stitch 2 Size"
#
# - Up to 10 optional IMAGE inputs (image_1 ... image_10)
# - width, height (exact output size)
# - pad_color as "R,G,B" (e.g. "255,255,255")
# - keep_relative_scale (BOOLEAN): preserve relative scale across inputs
# - Safely ignores connected-but-empty/invalid inputs (no crashes)
#
# Category: PVL/Image

import math
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image


class PVL_Stitch2Size:
    """
    Stitches up to 10 images into a single canvas (width x height),
    choosing an optimal grid, scaling proportionally, and padding with a chosen color.

    If keep_relative_scale=True, a single global scale factor is used for all images
    (tightest fit across the grid), preserving their relative sizes.
    Otherwise, each image scales independently to best fit its own cell.
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt_images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 1}),
                "pad_color": ("STRING", {"default": "255,255,255"}),
                "keep_relative_scale": ("BOOLEAN", {"default": False}),
            },
            "optional": opt_images,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "PVL/Image"

    # ----------------- Utils -----------------

    @staticmethod
    def _parse_pad_color(s: str) -> Tuple[int, int, int]:
        """
        Parse "R,G,B" or "R;G;B" into a 0-255 RGB tuple. Fallback = white.
        """
        try:
            parts = [int(p.strip()) for p in s.replace(";", ",").split(",")]
            if len(parts) != 3:
                raise ValueError
            r, g, b = [max(0, min(255, v)) for v in parts]
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    @staticmethod
    def _normalize_image_input(x) -> Optional[torch.Tensor]:
        """
        Return a 3D or 4D float tensor in [0,1], or None if unusable.
        Silently ignores connected-but-empty inputs (e.g., batch size 0).
        """
        try:
            # Handle lists/tuples by picking the first usable item
            if isinstance(x, (list, tuple)):
                for el in x:
                    t = PVL_Stitch2Size._normalize_image_input(el)
                    if t is not None:
                        return t
                return None

            if x is None or not torch.is_tensor(x):
                return None

            t = x
            if not torch.is_floating_point(t):
                t = t.float()

            # Accept [B,H,W,C] or [H,W,C]
            if t.ndim == 4:
                if t.shape[0] == 0:  # empty batch
                    return None
                return t
            elif t.ndim == 3:
                return t
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def _tensor_to_pil(img_t: torch.Tensor, bg_rgb: Tuple[int, int, int]) -> Optional[Image.Image]:
        """
        Convert ComfyUI IMAGE tensor to PIL RGB. If batched, take first.
        Returns None on failure (silently skip).
        """
        try:
            t = img_t
            if t.ndim == 4:
                if t.shape[0] == 0:
                    return None
                t = t[0]
            elif t.ndim != 3:
                return None

            # Clamp and move to CPU numpy
            t = t.clamp(0.0, 1.0).detach().cpu().numpy()

            # Expect HWC
            if t.ndim == 2:
                t = np.expand_dims(t, -1)

            C = t.shape[2]
            if C == 1:
                t = np.repeat(t, 3, axis=2)
            elif C >= 4:
                # RGBA → composite over pad color
                rgb = t[:, :, :3]
                alpha = t[:, :, 3:4]
                bg = np.array(bg_rgb, dtype=np.float32)[None, None, :] / 255.0
                rgb = rgb * alpha + bg * (1.0 - alpha)
                t = rgb
            else:
                # use first 3 channels
                t = t[:, :, :3]

            arr = (t * 255.0).round().astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            return None

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = (np.array(img).astype(np.float32) / 255.0)
        return torch.from_numpy(arr)[None, ...]

    @staticmethod
    def _choose_grid(n: int, W: int, H: int, sizes: List[Tuple[int, int]], keep_relative_scale: bool) -> Tuple[int, int]:
        """
        Choose (cols, rows) with cols*rows >= n.

        If keep_relative_scale:
            - Score by maximizing the global scale s_global = min_i min((W/cols)/wi, (H/rows)/hi).
        Else:
            - Score by maximizing total used area with per-image independent scales.

        Tie-breakers: grid aspect closest to canvas aspect, then fewer cells.
        """
        assert n == len(sizes)
        target_aspect = W / max(1, H)
        best = None  # (score, -aspect_diff, -cells, cols, rows)

        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            cell_w = W / cols
            cell_h = H / rows
            if cell_w <= 0 or cell_h <= 0:
                continue

            if keep_relative_scale:
                s_glob = min(min(cell_w / iw, cell_h / ih) for (iw, ih) in sizes)
                score = s_glob
            else:
                used_area = 0.0
                for (iw, ih) in sizes:
                    s = min(cell_w / iw, cell_h / ih)
                    used_area += (iw * ih) * (s * s)
                score = used_area

            grid_aspect = cols / rows
            aspect_diff = abs((grid_aspect / target_aspect) - 1.0)
            cells = cols * rows
            key = (score, -aspect_diff, -cells)

            if best is None or key > best[0]:
                best = (key, cols, rows)

        if best is None:
            cols = min(n, max(1, int(round(math.sqrt(n)))))
            rows = math.ceil(n / cols)
            return cols, rows

        _, cols, rows = best
        return cols, rows

    # ----------------- Main -----------------

    def stitch(
        self,
        width: int,
        height: int,
        pad_color: str,
        keep_relative_scale: bool,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        image_5: Optional[torch.Tensor] = None,
        image_6: Optional[torch.Tensor] = None,
        image_7: Optional[torch.Tensor] = None,
        image_8: Optional[torch.Tensor] = None,
        image_9: Optional[torch.Tensor] = None,
        image_10: Optional[torch.Tensor] = None,
    ):
        # Normalize and filter inputs (ignore connected-but-empty)
        raw_inputs = [image_1, image_2, image_3, image_4, image_5,
                      image_6, image_7, image_8, image_9, image_10]
        tensors = []
        for x in raw_inputs:
            t = self._normalize_image_input(x)
            if t is not None:
                tensors.append(t)

        bg_rgb = self._parse_pad_color(pad_color)
        W, H = int(width), int(height)

        # No usable images -> solid canvas
        if len(tensors) == 0:
            blank = Image.new("RGB", (W, H), bg_rgb)
            return (self._pil_to_tensor(blank),)

        # Convert to PIL; skip any that fail conversion
        pil_images: List[Image.Image] = []
        sizes: List[Tuple[int, int]] = []
        for t in tensors:
            pil = self._tensor_to_pil(t, bg_rgb)
            if pil is None:
                continue
            iw, ih = pil.size
            if iw <= 0 or ih <= 0:
                continue
            pil_images.append(pil.convert("RGB"))
            sizes.append((iw, ih))

        # If everything failed, return solid canvas
        if len(pil_images) == 0:
            blank = Image.new("RGB", (W, H), bg_rgb)
            return (self._pil_to_tensor(blank),)

        n = len(pil_images)

        # Select grid
        cols, rows = self._choose_grid(n, W, H, sizes, keep_relative_scale)

        # Compute exact cell edges (pixel-accurate; spreads rounding evenly)
        cell_w_f = W / cols
        cell_h_f = H / rows
        x_edges = [round(c * cell_w_f) for c in range(cols + 1)]
        y_edges = [round(r * cell_h_f) for r in range(rows + 1)]

        # For relative-scale mode, use the smallest cell size across the grid (handles rounding)
        s_global = None
        if keep_relative_scale:
            min_cell_w = min(x_edges[i + 1] - x_edges[i] for i in range(cols))
            min_cell_h = min(y_edges[i + 1] - y_edges[i] for i in range(rows))
            s_global = min(min(min_cell_w / iw, min_cell_h / ih) for (iw, ih) in sizes)
            # Guard: if degenerate, fall back to per-image scaling
            if not (s_global and s_global > 0):
                keep_relative_scale = False
                s_global = None

        # Build canvas and place images
        canvas = Image.new("RGB", (W, H), bg_rgb)
        for idx, pil in enumerate(pil_images):
            r = idx // cols
            c = idx % cols
            x0, x1 = x_edges[c], x_edges[c + 1]
            y0, y1 = y_edges[r], y_edges[r + 1]
            cell_w = max(1, x1 - x0)
            cell_h = max(1, y1 - y0)

            iw, ih = pil.size
            if keep_relative_scale and s_global is not None:
                s = s_global
            else:
                s = min(cell_w / iw, cell_h / ih)

            new_w = max(1, int(round(iw * s)))
            new_h = max(1, int(round(ih * s)))
            resized = pil.resize((new_w, new_h), resample=Image.LANCZOS)

            off_x = x0 + (cell_w - new_w) // 2
            off_y = y0 + (cell_h - new_h) // 2
            canvas.paste(resized, (off_x, off_y))

        # Exact size guard (should already match)
        if canvas.size != (W, H):
            canvas = canvas.resize((W, H), resample=Image.LANCZOS)

        return (self._pil_to_tensor(canvas),)