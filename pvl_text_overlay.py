import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from comfy.utils import ProgressBar

# -------------------------------------------------------------------------
# Font loading: system fonts + optional ./fonts folder (no sup dependency)
# -------------------------------------------------------------------------

ROOT_FONTS = Path(__file__).absolute().parent / "fonts"


def _build_font_data():
    """
    Build font data from system fonts (via matplotlib.font_manager) and
    optional local fonts from ./fonts.

    Returns:
      font_families: list[str] - sorted family names for dropdown
      font_entries: dict[str, list[dict]] - family -> list of entries:
          {
              "path": str,
              "style": str,   # e.g. "normal", "italic", "oblique"
              "weight": int,  # e.g. 400, 700
              "source": "system" | "local"
          }
    """
    font_entries = {}

    # 1) System fonts
    try:
        mgr = font_manager.FontManager()
        for f in mgr.ttflist:
            family = f.name  # readable family name
            entry = {
                "path": f.fname,
                "style": getattr(f, "style", "normal") or "normal",
                "weight": int(getattr(f, "weight", 400) or 400),
                "source": "system",
            }
            font_entries.setdefault(family, []).append(entry)

        print(
            f"[PVL_TextOverlayNode] System fonts: {len(mgr.ttflist)} entries, "
            f"{len(font_entries)} unique families."
        )
    except Exception as e:
        print(f"[PVL_TextOverlayNode] Failed to load system fonts: {e}")

    # 2) Local fonts in ./fonts (treated as regular)
    local_count = 0
    if ROOT_FONTS.is_dir():
        for font_path in ROOT_FONTS.glob("*.[to][tf][f]"):  # ttf / otf
            family = font_path.stem
            entry = {
                "path": str(font_path),
                "style": "normal",
                "weight": 400,
                "source": "local",
            }
            font_entries.setdefault(family, []).append(entry)
            local_count += 1
        print(
            f"[PVL_TextOverlayNode] Loaded {local_count} local fonts from {ROOT_FONTS}"
        )
    else:
        print(
            f"[PVL_TextOverlayNode] No local fonts directory found at {ROOT_FONTS}"
        )

    if not font_entries:
        print("[PVL_TextOverlayNode] No fonts found, will use PIL default only.")

    font_families = sorted(font_entries.keys())
    preview = font_families[:10]
    print(f"[PVL_TextOverlayNode] Example font families: {preview}")

    return font_families, font_entries


def _select_font_path_for_variation(entries, variation: str):
    """
    Given a list of font entries for a family and a requested variation,
    pick the best matching entry and return its path.

    variation: "auto" | "regular" | "italic" | "bold" | "bold_italic"
    entries: list of dicts with keys: path, style, weight, source
    """
    if not entries:
        return None

    if variation == "auto":
        # Prefer local fonts first, then system
        local_entries = [e for e in entries if e["source"] == "local"]
        if local_entries:
            return local_entries[0]["path"]
        return entries[0]["path"]

    variation = variation.lower()

    def is_regular(e):
        s = e["style"].lower()
        w = e["weight"]
        return ("italic" not in s and "oblique" not in s and w <= 500)

    def is_italic(e):
        s = e["style"].lower()
        return ("italic" in s) or ("oblique" in s)

    def is_bold(e):
        w = e["weight"]
        return w >= 600

    def is_bold_italic(e):
        return is_bold(e) and is_italic(e)

    if variation == "regular":
        candidates = [e for e in entries if is_regular(e)]
    elif variation == "italic":
        candidates = [e for e in entries if is_italic(e)]
    elif variation == "bold":
        candidates = [e for e in entries if is_bold(e)]
    elif variation == "bold_italic":
        candidates = [e for e in entries if is_bold_italic(e)]
    else:
        candidates = []

    if not candidates:
        # Fallback order: local regular -> any local -> any system
        local_regular = [e for e in entries if e["source"] == "local" and is_regular(e)]
        if local_regular:
            return local_regular[0]["path"]

        local_any = [e for e in entries if e["source"] == "local"]
        if local_any:
            return local_any[0]["path"]

        return entries[0]["path"]

    # Prefer local over system when multiple candidates
    local_candidates = [e for e in candidates if e["source"] == "local"]
    if local_candidates:
        return local_candidates[0]["path"]

    return candidates[0]["path"]


class PVL_Text_Overlay:
    """
    PVL_TextOverlayNode applies a text overlay to an image.
    You can specify the text, its position (as a percentage of image dimensions),
    font size, font color, and choose from available fonts.

    Fonts:
      - System fonts via matplotlib.font_manager.FontManager.
      - Optional local fonts from a 'fonts' folder next to this file.
      - A 'font variation' dropdown controls Regular / Italic / Bold / Bold Italic.
    """

    FONT_FAMILIES, FONT_ENTRIES = _build_font_data()
    FONT_NAMES = FONT_FAMILIES  # keep old naming for compatibility

    if not FONT_NAMES:
        # No real fonts found, keep a sentinel
        FONT_NAMES = ["Default"]

    DESCRIPTION = """
PVL_TextOverlayNode applies a text overlay to an image.
You can specify the text, its position, font family, variation, size, and color.
"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": (
                    "STRING",
                    {
                        "default": "Hello, ComfyUI!",
                        "multiline": True,
                        "description": "Text to render on the image.",
                    },
                ),
                "font": (
                    s.FONT_NAMES,
                    {
                        "default": s.FONT_NAMES[0],
                        "tooltip": "Font family.",
                    },
                ),
                "font_variation": (
                    ["auto", "regular", "italic", "bold", "bold_italic"],
                    {
                        "default": "auto",
                        "description": (
                            "Font variation: auto picks the best available; "
                            "or force Regular / Italic / Bold / Bold Italic."
                        ),
                    },
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 51,
                        "min": 1,
                        "step": 1,
                        "description": "Font size in pixels.",
                    },
                ),
                "font_color_r": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "description": "Red color value",
                    },
                ),
                "font_color_g": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "description": "Green color value",
                    },
                ),
                "font_color_b": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "description": "Blue color value",
                    },
                ),
                "x_percent": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "description": "X position as percentage from left",
                    },
                ),
                "y_percent": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "description": "Y position as percentage from top",
                    },
                ),
                "anchor": (
                    [
                        "left-top",
                        "center-top",
                        "right-top",
                        "left-center",
                        "center-center",
                        "right-center",
                        "left-bottom",
                        "center-bottom",
                        "right-bottom",
                    ],
                    {
                        "default": "center-center",
                        "description": (
                            "Text anchor point relative to X,Y coordinates "
                            "(e.g. center-center means X,Y is the text center)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_text_overlay"
    CATEGORY = "⚙️PVL Nodes/Text"

    # ------------------------------------------------------------------
    # Anchor mapping
    # ------------------------------------------------------------------
    def _get_pil_anchor(self, anchor_str: str) -> str:
        """
        Map 9-point anchor names to Pillow 'anchor' strings.

        Pillow anchor:
          Horizontal: 'l' (left), 'm' (middle), 'r' (right)
          Vertical:   'a' (ascent/top-ish), 'm' (middle), 'b' (bottom)
        """
        mapping = {
            "left-top": "la",
            "center-top": "ma",
            "right-top": "ra",
            "left-center": "lm",
            "center-center": "mm",
            "right-center": "rm",
            "left-bottom": "lb",
            "center-bottom": "mb",
            "right-bottom": "rb",
        }
        return mapping.get(anchor_str, "mm")  # default to middle-center

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def apply_text_overlay(
        self,
        image: torch.Tensor,
        text: str,
        font: str,
        font_variation: str,
        font_size: int,
        font_color_r: int,
        font_color_g: int,
        font_color_b: int,
        x_percent: float,
        y_percent: float,
        anchor: str,
    ):
        batch_size = image.shape[0]
        result_images = []

        font_color_rgb = (font_color_r, font_color_g, font_color_b)

        # Resolve font path based on family + variation
        font_obj = None
        if self.FONT_ENTRIES and font in self.FONT_ENTRIES:
            entries = self.FONT_ENTRIES[font]
            path = _select_font_path_for_variation(entries, font_variation)
            try:
                font_obj = ImageFont.truetype(path, font_size)
                print(
                    f"[PVL_TextOverlayNode] Using font '{font}' variation '{font_variation}' "
                    f"from '{path}' size={font_size}."
                )
            except Exception as e:
                print(
                    f"[PVL_TextOverlayNode] Error loading font '{font}' at '{path}'. "
                    f"Falling back to default. Error: {e}"
                )

        if font_obj is None:
            try:
                font_obj = ImageFont.load_default(size=font_size)  # Pillow >= 10
                print(
                    "[PVL_TextOverlayNode] Using PIL default font with requested size."
                )
            except Exception:
                font_obj = ImageFont.load_default()
                print(
                    "[PVL_TextOverlayNode] Using PIL default font without explicit size."
                )

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_tensor = image[i]

            # Convert tensor [H,W,C], 0-1 -> PIL image
            pil_image = Image.fromarray(
                (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            ).convert("RGB")

            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size

            x_abs = int(img_width * (x_percent / 100.0))
            y_abs = int(img_height * (y_percent / 100.0))

            # Try using Pillow's built-in anchor support
            try:
                pil_anchor_val = self._get_pil_anchor(anchor)
                draw.text(
                    (x_abs, y_abs),
                    text,
                    font=font_obj,
                    fill=font_color_rgb,
                    anchor=pil_anchor_val,
                )
            except TypeError:
                # Older Pillow without 'anchor' support – manual positioning
                print(
                    "[PVL_TextOverlayNode] Pillow version might not support "
                    "'anchor' argument. Falling back to manual positioning."
                )
                try:
                    bbox = draw.textbbox((x_abs, y_abs), text, font=font_obj)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    horiz, vert = anchor.split("-")

                    # Horizontal alignment
                    if horiz == "center":
                        x_abs -= text_width // 2
                    elif horiz == "right":
                        x_abs -= text_width

                    # Vertical alignment
                    if vert == "center":
                        y_abs -= text_height // 2
                    elif vert == "bottom":
                        y_abs -= text_height

                    draw.text(
                        (x_abs, y_abs),
                        text,
                        font=font_obj,
                        fill=font_color_rgb,
                    )

                except Exception as e_bbox:
                    print(
                        "[PVL_TextOverlayNode] Manual anchor adjustment failed: "
                        f"{e_bbox}. Drawing at top-left without anchor."
                    )
                    draw.text(
                        (x_abs, y_abs),
                        text,
                        font=font_obj,
                        fill=font_color_rgb,
                    )

            img_array = np.array(pil_image).astype(np.float32) / 255.0
            result_images.append(torch.from_numpy(img_array).unsqueeze(0))

            pbar.update_absolute(i + 1, batch_size)

        final_tensor = torch.cat(result_images, dim=0)
        return (final_tensor,)
