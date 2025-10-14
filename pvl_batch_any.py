# pvl_batch_anything.py
# PVL — Batch Anything

from typing import Any

import torch 
import comfy.utils


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True
    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")


class PVL_BatchAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_1": (any_type, {}),
                "any_2": (any_type, {}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch",)
    FUNCTION = "batch"
    CATEGORY = "EasyUse/Logic"

    # ---- internal helper for latent dicts with 'samples' ----
    def latentBatch(self, any_1: dict, any_2: dict):
        if torch is None:
            raise RuntimeError("Torch is required for latent batching.")

        s1 = any_1["samples"]
        s2 = any_2["samples"]

        # Ensure we copy other keys too
        samples_out = dict(any_1)

        # Match spatial dims; upscales s2 to s1 if needed
        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(
                s2, s1.shape[3], s1.shape[2], "bilinear", "center"
            )

        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = any_1.get("batch_index", list(range(s1.shape[0]))) + \
                                     any_2.get("batch_index", list(range(s2.shape[0])))
        return samples_out

    def batch(self, any_1: Any, any_2: Any):
        # Tensor batching (e.g., images NHWC/NCHW)
        if (torch is not None) and (isinstance(any_1, getattr(torch, "Tensor", ())) or isinstance(any_2, getattr(torch, "Tensor", ()))):
            if any_1 is None:
                return (any_2,)
            if any_2 is None:
                return (any_1,)

            # Make shapes align
            if any_1.shape[1:] != any_2.shape[1:]:
                any_2 = comfy.utils.common_upscale(
                    any_2.movedim(-1, 1),
                    any_1.shape[2],
                    any_1.shape[1],
                    "bilinear",
                    "center",
                ).movedim(1, -1)
            return (torch.cat((any_1, any_2), dim=0),)

        # Primitive + list/tuple merging
        if isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return (any_1,)
            if isinstance(any_2, tuple):
                return (any_2 + (any_1,),)
            if isinstance(any_2, list):
                return (any_2 + [any_1],)
            return ([any_1, any_2],)

        if isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return (any_2,)
            if isinstance(any_1, tuple):
                return (any_1 + (any_2,),)
            if isinstance(any_1, list):
                return (any_1 + [any_2],)
            return ([any_2, any_1],)

        # Latent dicts with 'samples'
        if isinstance(any_1, dict) and 'samples' in any_1:
            if any_2 is None:
                return (any_1,)
            if isinstance(any_2, dict) and 'samples' in any_2:
                return (self.latentBatch(any_1, any_2),)

        if isinstance(any_2, dict) and 'samples' in any_2:
            if any_1 is None:
                return (any_2,)
            if isinstance(any_1, dict) and 'samples' in any_1:
                return (self.latentBatch(any_2, any_1),)

        # Fallback: try Python '+' concatenation
        if any_1 is None:
            return (any_2,)
        if any_2 is None:
            return (any_1,)
        return (any_1 + any_2,)


NODE_CLASS_MAPPINGS = {"PVL_BatchAny": PVL_BatchAny}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_BatchAny": "PVL — Batch Any"}
