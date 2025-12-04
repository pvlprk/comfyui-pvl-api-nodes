import torch
import torch.nn.functional as F

class PVL_Color_Match:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
                    [
                        'mkl',
                        'hm',
                        'reinhard',
                        'mvgd',
                        'hm-mvgd-hm',
                        'hm-mkl-hm',
                    ],
                    {
                        "default": 'mkl'
                    }
                ),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                # Standard ComfyUI mask: black (0) = ignore, white (1) = analysis area
                "attention_mask": ("MASK",),
            }
        }

    CATEGORY = "PVL_tools/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
Color transfer between images with optional attention mask.

If attention_mask is provided (standard ComfyUI MASK, black/white):
- The white region defines the analysis area on BOTH target and reference.
- Outside the white region, both images are flattened to the mean color of
  their own masked region *only for analysis*.
- The resulting color correction is then applied to the entire ORIGINAL target image
  (no masking in the output stage).

Based on:
https://github.com/hahnec/color-matcher/
"""

    def _prepare_mask_batch(self, mask_tensor, batch_size, H, W, name="mask"):
        """
        Normalize mask to shape (batch_size, H, W), values in [0,1],
        using nearest-neighbor resize. Handles shapes:
        - (B,H,W)
        - (B,1,H,W)
        - (1,H,W)
        - (1,1,H,W)
        """
        if mask_tensor is None:
            return None

        mask = mask_tensor.cpu()

        # Normalize dimensions to (B,H,W)
        if mask.dim() == 4:
            # (B,C,H,W) -> take first channel
            if mask.size(1) > 1:
                mask = mask[:, 0:1, :, :]
            mask = mask[:, 0, :, :]  # (B,H,W)
        elif mask.dim() == 3:
            # (B,H,W) already
            pass
        else:
            raise ValueError(f"PVL_Color_Match: Unexpected {name} shape {mask.shape}; expected 3D or 4D.")

        B_mask, Hm, Wm = mask.shape

        # Adjust batch dimension: either 1 or batch_size
        if B_mask == 1 and batch_size > 1:
            mask = mask.expand(batch_size, Hm, Wm)
        elif B_mask == batch_size:
            pass
        elif batch_size == 1 and B_mask == 1:
            pass
        else:
            raise ValueError(
                f"PVL_Color_Match: {name} batch size ({B_mask}) must be 1 or "
                f"match the image batch size ({batch_size})."
            )

        # Resize to (H, W) with nearest neighbor (keeps 0 / 1 intact)
        mask_resized = F.interpolate(
            mask.unsqueeze(1),      # (B,1,Hm,Wm)
            size=(H, W),
            mode="nearest"
        ).squeeze(1)               # (B,H,W)

        return mask_resized

    def colormatch(self, image_ref, image_target, method, strength=1.0, attention_mask=None):
        try:
            from color_matcher import ColorMatcher
        except Exception:
            raise Exception(
                "Can't import color-matcher, did you install requirements.txt? "
                "Manual install: pip install color-matcher"
            )

        cm = ColorMatcher()

        # Move to CPU
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()

        # Shapes: (B,H,W,C)
        B_t, H_t, W_t, C_t = image_target.shape
        B_r, H_r, W_r, C_r = image_ref.shape

        if C_t != 3 or C_r != 3:
            raise ValueError("PVL_Color_Match: Expected 3-channel images.")

        # Reference batch size must be 1 or match target batch size
        if B_r != 1 and B_r != B_t:
            raise ValueError(
                "PVL_Color_Match: Use either a single reference image or a batch of "
                "reference images matching the target batch size."
            )

        image_target_np = image_target.numpy()  # (B_t,H_t,W_t,3)
        image_ref_np = image_ref.numpy()        # (B_r,H_r,W_r,3)

        # Prepare masks for analysis, if provided.
        mask_target = None
        mask_ref = None

        if attention_mask is not None:
            # Mask aligned to target resolution
            mask_target = self._prepare_mask_batch(
                attention_mask, batch_size=B_t, H=H_t, W=W_t, name="attention_mask (target)"
            )  # (B_t,H_t,W_t)

            # Same logical mask definition, resized to reference resolution
            mask_ref = self._prepare_mask_batch(
                attention_mask, batch_size=B_r, H=H_r, W=W_r, name="attention_mask (ref)"
            )  # (B_r,H_r,W_r)

        out = []

        for i in range(B_t):
            # ORIGINAL target (this is what we finally correct)
            target_np = image_target_np[i]  # (H_t,W_t,3)

            # Reference image (single or per-batch)
            if B_r == 1:
                ref_np = image_ref_np[0]
            else:
                ref_np = image_ref_np[i]

            # Analysis copies (these are modified with mask)
            analysis_target = target_np.copy()
            analysis_ref = ref_np.copy()

            # === Apply mask-driven analysis on TARGET ===
            if mask_target is not None:
                mask_t_i = mask_target[0] if B_t == 1 else mask_target[i]  # (H_t,W_t)
                mask_t_bool = mask_t_i > 0.5

                if mask_t_bool.any():
                    masked_target_pixels = target_np[mask_t_bool]  # (N,3)
                    mean_target = masked_target_pixels.mean(axis=0)
                    # Flatten outside-mask region to mean color (analysis only)
                    analysis_target[~mask_t_bool] = mean_target

            # === Apply mask-driven analysis on REFERENCE ===
            if mask_ref is not None:
                if B_r == 1:
                    mask_r_i = mask_ref[0]
                else:
                    mask_r_i = mask_ref[i]

                mask_r_bool = mask_r_i > 0.5

                if mask_r_bool.any():
                    masked_ref_pixels = ref_np[mask_r_bool]  # (N,3)
                    mean_ref = masked_ref_pixels.mean(axis=0)
                    # Flatten outside-mask region to mean color (analysis only)
                    analysis_ref[~mask_r_bool] = mean_ref

            # === Color transfer on analysis images ===
            try:
                analysis_result = cm.transfer(
                    src=analysis_target,
                    ref=analysis_ref,
                    method=method
                )
            except BaseException as e:
                print(f"[PVL_Color_Match] Error during transfer (item {i}): {e}")
                break

            # === Apply correction to the FULL original target ===
            # IMPORTANT: we do NOT use the mask here at all.
            # We treat (analysis_result - analysis_target) as the "correction field"
            # and add it onto the ORIGINAL target image (global effect).
            correction = analysis_result - analysis_target
            image_result = target_np + strength * correction

            out.append(torch.from_numpy(image_result))

        if not out:
            raise RuntimeError("PVL_Color_Match: color transfer failed for all items.")

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)
