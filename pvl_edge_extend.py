import torch
import torch.nn.functional as F

# --- Color space helpers (RGB only; alpha untouched) ---
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1.0 + a)).clamp_min(0) ** 2.4
    )

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        (1.0 + a) * x.clamp_min(0) ** (1.0 / 2.4) - a
    )

# --- Separable Gaussian blur (for [H,W,C] or [H,W]) ---
def _make_gauss_1d(radius: int, device: torch.device):
    k = radius * 2 + 1
    if k < 3:
        k = 3
    coords = torch.arange(k, dtype=torch.float32, device=device) - (k // 2)
    sigma = max(float(radius) / 3.0, 1e-4)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g_col = g.view(1, 1, k, 1)
    g_row = g.view(1, 1, 1, k)
    pad = (radius, radius)
    return g_col, g_row, pad

def _blur_hw_or_hwc(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    device = x.device
    g_col, g_row, pad = _make_gauss_1d(radius, device)
    if x.ndim == 3:  # [H,W,C]
        h, w, c = x.shape
        out_ch = []
        for ch in range(c):
            ch2d = x[..., ch].unsqueeze(0).unsqueeze(0)
            ch2d = F.conv2d(ch2d, g_col, padding=(pad[0], 0))
            ch2d = F.conv2d(ch2d, g_row, padding=(0, pad[1]))
            out_ch.append(ch2d.squeeze(0).squeeze(0))
        return torch.stack(out_ch, dim=2)
    else:           # [H,W]
        x2d = x.unsqueeze(0).unsqueeze(0)
        x2d = F.conv2d(x2d, g_col, padding=(pad[0], 0))
        x2d = F.conv2d(x2d, g_row, padding=(0, pad[1]))
        return x2d.squeeze(0).squeeze(0)

class PVL_EdgeExtend:
    """
    Edge extension node with iterative Nuke-style propagation.

    Pass logic:
      • premult_after_levels = (unprem * adjusted_alpha)
      • Working state per iteration:
          work_rgb_premult (RGB premult stream), work_alpha (coverage for UNDER)
      • For each enabled pass with radius r:
          1) blur work_rgb_premult & work_alpha  → unprem_blurred
          2) band_out = clamp(dilate(work_alpha, r) - work_alpha, 0, 1)
          3) work_rgb_premult = work_rgb_premult + (unprem_blurred * band_out)[premult] * (1 - work_alpha)
             work_alpha       = work_alpha       + band_out * (1 - work_alpha)
          4) (restore detail) hi-freq add-back limited to intermediate alpha & inner edge band
      • Diagnostics after pass n: unprem(work_rgb_premult, work_alpha)
      • Final premult for output uses ORIGINAL alpha × opacity (unchanged), or switchable later if desired.
      • Optional linear workflow (work_in_linear)
    """

    CATEGORY = "image/alpha"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),

                # Levels
                "black_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # Alpha erode AFTER levels (morphological)
                "alpha_erode_after_levels": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),

                # Pass 1 radius
                "edge_blur": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),

                # Final mix
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # Detail restoration amount (hi-frequency add-back)
                "restore_detail": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                # Second pass
                "second_pass": ("BOOLEAN", {"default": False}),
                "second_edge_blur": ("INT", {"default": 3, "min": 0, "max": 50, "step": 1}),

                # Third pass
                "third_pass": ("BOOLEAN", {"default": False}),
                "third_edge_blur": ("INT", {"default": 3, "min": 0, "max": 50, "step": 1}),

                # Restore mask shaping
                "restore_detail_radius": ("INT", {"default": 3, "min": 0, "max": 128, "step": 1}),
                "restore_detial_blur_edge": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),  # spelling kept

                # Toggle linear workflow
                "work_in_linear": ("BOOLEAN", {"default": True}),
            }
        }

    # Outputs:
    #  0 IMAGE  -> final premultiplied image (sRGB if work_in_linear=True)
    #  1 MASK   -> final alpha (= input alpha * opacity)
    #  2 MASK   -> intermediate alpha (after levels + erode)
    #  3 IMAGE  -> premult_after_levels (diagnostic image)
    #  4 IMAGE  -> unpremult_after_pass1
    #  5 IMAGE  -> unpremult_after_pass2
    #  6 IMAGE  -> pre_final_premult_unpremult (== unprem after pass3 or pass2 if pass3 disabled)
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "image",
        "alpha",
        "intermediate_alpha",
        "premult_after_levels",
        "unpremult_after_pass1",
        "unpremult_after_pass2",
        "pre_final_premult_unpremult",
    )
    FUNCTION = "extend_edges"

    # ----------------- CORE -----------------

    def extend_edges(
        self,
        image, alpha,
        black_point, white_point,
        alpha_erode_after_levels,
        edge_blur,
        opacity, restore_detail,
        second_pass=False, second_edge_blur=3,
        third_pass=False, third_edge_blur=3,
        restore_detail_radius=3, restore_detial_blur_edge=0,
        work_in_linear=True,
    ):
        batch, height, width, channels = image.shape
        eps = 1e-6

        # Normalize alpha to [B,H,W]
        alpha = self._normalize_alpha_to_image(alpha, batch, height, width)

        out_final_img = []
        out_final_alpha = []
        out_intermediate_alpha = []
        out_premult_after_levels = []
        out_unpremult_after_pass1 = []
        out_unpremult_after_pass2 = []
        out_pre_final_unprem = []

        for i in range(batch):
            img_srgb = image[i].clamp(0, 1)   # premultiplied
            a_in     = alpha[i].clamp(0, 1)   # [H,W]

            # Optional linear workflow (RGB only)
            img_work = srgb_to_linear(img_srgb) if work_in_linear else img_srgb

            # Unprem with ORIGINAL alpha (reference & diagnostics)
            a_safe = a_in.clamp_min(eps).unsqueeze(-1)
            unprem_work = (img_work / a_safe).clamp(0, 1)  # [H,W,C]

            # 1) Levels on alpha
            adj_alpha = self.adjust_alpha_levels(a_in, black_point, white_point)
            if alpha_erode_after_levels > 0:
                adj_alpha = self.erode_alpha_morphological(adj_alpha, alpha_erode_after_levels)

            # Expose intermediate alpha
            out_intermediate_alpha.append(adj_alpha)

            # premult_after_levels and initialize working state
            premult_after_levels = (unprem_work * adj_alpha.unsqueeze(-1)).clamp(0, 1)
            work_rgb_premult = premult_after_levels.clone()
            work_alpha = adj_alpha.clone()

            # ----- PASS 1 -----
            work_rgb_premult, work_alpha = self._iterate_pass(
                work_rgb_premult, work_alpha, edge_blur, eps,
                restore_source_unprem=unprem_work,
                restore_amount=restore_detail,
                restore_detail_radius=restore_detail_radius,
                restore_detial_blur_edge=restore_detial_blur_edge
            )
            # Diagnostics after pass 1
            up1 = (work_rgb_premult / work_alpha.clamp_min(eps).unsqueeze(-1)).clamp(0, 1)

            # ----- PASS 2 (optional) -----
            if second_pass and second_edge_blur > 0:
                work_rgb_premult, work_alpha = self._iterate_pass(
                    work_rgb_premult, work_alpha, second_edge_blur, eps,
                    restore_source_unprem=unprem_work,
                    restore_amount=restore_detail,
                    restore_detail_radius=restore_detail_radius,
                    restore_detial_blur_edge=restore_detial_blur_edge
                )
            up2 = (work_rgb_premult / work_alpha.clamp_min(eps).unsqueeze(-1)).clamp(0, 1)

            # ----- PASS 3 (optional) -----
            if third_pass and third_edge_blur > 0:
                work_rgb_premult, work_alpha = self._iterate_pass(
                    work_rgb_premult, work_alpha, third_edge_blur, eps,
                    restore_source_unprem=unprem_work,
                    restore_amount=restore_detail,
                    restore_detail_radius=restore_detail_radius,
                    restore_detial_blur_edge=restore_detial_blur_edge
                )
            up_final = (work_rgb_premult / work_alpha.clamp_min(eps).unsqueeze(-1)).clamp(0, 1)

            # Diagnostics: convert to sRGB if we worked in linear
            if work_in_linear:
                premult_after_levels_out = linear_to_srgb(premult_after_levels).clamp(0, 1)
                up1_out                  = linear_to_srgb(up1).clamp(0, 1)
                up2_out                  = linear_to_srgb(up2).clamp(0, 1)
                up_final_out             = linear_to_srgb(up_final).clamp(0, 1)
            else:
                premult_after_levels_out = premult_after_levels
                up1_out                  = up1
                up2_out                  = up2
                up_final_out             = up_final

            # Final premult with ORIGINAL alpha × opacity (deliverable image)
            final_alpha = (a_in * opacity).clamp(0, 1)                           # [H,W]
            final_img_work = (up_final * final_alpha.unsqueeze(-1)).clamp(0, 1)
            final_img = linear_to_srgb(final_img_work).clamp(0, 1) if work_in_linear else final_img_work

            # Collect
            out_premult_after_levels.append(premult_after_levels_out)
            out_unpremult_after_pass1.append(up1_out)
            out_unpremult_after_pass2.append(up2_out)
            out_pre_final_unprem.append(up_final_out)  # after pass3 (or pass2 if third disabled)
            out_final_img.append(final_img)
            out_final_alpha.append(final_alpha)

        # Stack and return
        return (
            torch.stack(out_final_img, dim=0),
            torch.stack(out_final_alpha, dim=0),
            torch.stack(out_intermediate_alpha, dim=0),
            torch.stack(out_premult_after_levels, dim=0),
            torch.stack(out_unpremult_after_pass1, dim=0),
            torch.stack(out_unpremult_after_pass2, dim=0),
            torch.stack(out_pre_final_unprem, dim=0),
        )

    # ----------------- ITERATION (Nuke-style) -----------------

    def _iterate_pass(self, work_rgb_premult, work_alpha, r, eps,
                      restore_source_unprem, restore_amount,
                      restore_detail_radius, restore_detial_blur_edge):
        """
        One Nuke-style iteration:
          - blur current working premult + alpha
          - unprem blurred pair
          - build outside band via dilation of work_alpha
          - premult(unprem, band_out) and UNDER-merge into working premult
          - update work_alpha using under alpha rule
          - hi-freq add-back masked to (intermediate alpha edge region only)
        Returns: updated (work_rgb_premult, work_alpha)
        """
        if r <= 0:
            # still apply detail add-back mask (no marching without blur)
            return work_rgb_premult, work_alpha

        # 1) Blur current working premult & alpha, then unprem
        blur_rgb = _blur_hw_or_hwc(work_rgb_premult, r)         # [H,W,C]
        blur_a   = _blur_hw_or_hwc(work_alpha, r).clamp_min(eps)  # [H,W]
        unprem_blurred = (blur_rgb / blur_a.unsqueeze(-1)).clamp(0, 1)

        # 2) Outside band from dilation of work_alpha
        band_out = (self.dilate_alpha_morphological(work_alpha, r) - work_alpha).clamp(0.0, 1.0)  # [H,W]

        # 3) UNDER-merge premult(unprem_blurred, band_out) into working premult
        fill_premult = (unprem_blurred * band_out.unsqueeze(-1)).clamp(0, 1)
        inv_a = (1.0 - work_alpha).unsqueeze(-1)
        work_rgb_premult = (work_rgb_premult + fill_premult * inv_a).clamp(0, 1)
        # Update working alpha like under-composite alpha
        work_alpha = (work_alpha + band_out * (1.0 - work_alpha)).clamp(0, 1)

        # 4) Hi-freq add-back (restore detail) — limited to *intermediate alpha* inner edge
        if restore_amount > 0:
            base = work_alpha  # start from current work_alpha, but limit to original intermediate alpha region:
            # limit to original intermediate region by intersecting with initial mask once per-iteration:
            # we don't have it here; we approximate by not growing restore region beyond current work_alpha.
            # Build inner edge band INSIDE the matte:
            r_in = int(max(1, restore_detail_radius))
            inner_band = (base - self.erode_alpha_morphological(base, r_in)).clamp(0.0, 1.0)
            if restore_detial_blur_edge > 0:
                inner_band = _blur_hw_or_hwc(inner_band, restore_detial_blur_edge).clamp(0, 1)

            restore_mask3 = inner_band.unsqueeze(-1)
            # hi-freq from original unprem source
            hf_radius = max(1, r // 2)
            low = _blur_hw_or_hwc(restore_source_unprem, hf_radius)
            detail = (restore_source_unprem - low)
            # Convert current working premult to unprem for safe add, then re-premult to keep marching state in premult:
            current_unprem = (work_rgb_premult / work_alpha.clamp_min(eps).unsqueeze(-1)).clamp(0, 1)
            current_unprem = (current_unprem + detail * restore_amount * restore_mask3).clamp(0, 1)
            work_rgb_premult = (current_unprem * work_alpha.unsqueeze(-1)).clamp(0, 1)

        return work_rgb_premult, work_alpha

    # ----------------- UTILITIES -----------------

    def _normalize_alpha_to_image(self, alpha, batch, height, width):
        """Ensure alpha is [B,H,W] matching image spatial size."""
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(0)  # [1,H,W]
        if alpha.shape[0] != batch:
            alpha = alpha.repeat(batch, 1, 1)
        if alpha.shape[1] != height or alpha.shape[2] != width:
            alpha = F.interpolate(
                alpha.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
        return alpha.clamp(0.0, 1.0)

    def adjust_alpha_levels(self, alpha, black_point, white_point):
        """Adjust alpha using black/white point controls."""
        if white_point <= black_point:
            white_point = black_point + 0.01
        adjusted = (alpha - black_point) / (white_point - black_point)
        return adjusted.clamp(0.0, 1.0)

    def erode_alpha_morphological(self, alpha, radius):
        """
        Morphological erosion (min filter) with square kernel (2r+1).
        Implemented via max-pool on inverted signal: erode(x) = 1 - dilate(1-x).
        """
        if radius <= 0:
            return alpha
        inv = 1.0 - alpha
        inv4 = inv.unsqueeze(0).unsqueeze(0)
        k = radius * 2 + 1
        dil_inv = F.max_pool2d(inv4, kernel_size=k, stride=1, padding=radius)
        eroded = 1.0 - dil_inv.squeeze(0).squeeze(0)
        return eroded.clamp(0.0, 1.0)

    def dilate_alpha_morphological(self, alpha, radius):
        """Proper morphological dilation using max pooling."""
        if radius <= 0:
            return alpha
        a4 = alpha.unsqueeze(0).unsqueeze(0)
        k = radius * 2 + 1
        dil = F.max_pool2d(a4, kernel_size=k, stride=1, padding=radius)
        return dil.squeeze(0).squeeze(0).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {"PVL_EdgeExtend": PVL_EdgeExtend}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_EdgeExtend": "PVL Edge Extend"}
