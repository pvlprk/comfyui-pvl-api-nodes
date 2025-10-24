# ---------------------------------------------------
# FlowMatch Euler (Sigmas) for ComfyUI
# - Wraps Hugging Face Diffusers' FlowMatchEulerDiscreteScheduler
# - Outputs SIGMAS (and TIMESTEPS) for use with KSampler (Advanced)
# - Optional MODEL input to align device/dtype with your loaded model
# ---------------------------------------------------

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# ComfyUI device helpers
try:
    from comfy import model_management as mm
except Exception:
    mm = None  # Fallback if comfy internals move; we'll still work with user-selected device.

class PVL_FlowMatchEulerSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_steps": ("INT", {"default": 28, "min": 1, "max": 4096}),
                "sigma_min": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 1.0}),
                "sigma_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "use_dynamic_shifting": ("BOOLEAN", {"default": True}),
                "use_karras_sigmas": ("BOOLEAN", {"default": False}),
                "use_exponential_sigmas": ("BOOLEAN", {"default": False}),
                "use_beta_sigmas": ("BOOLEAN", {"default": False}),
                "stochastic_sampling": ("BOOLEAN", {"default": False}),
                "align_your_steps": ("BOOLEAN", {"default": True}),
                "fallback_device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                # If provided, we align device (and potentially defaults later) with the model
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("SIGMAS", "TIMESTEPS",)
    RETURN_NAMES = ("sigmas", "timesteps",)
    FUNCTION = "build"
    CATEGORY = "sampling/schedulers"

    def _pick_device(self, model, fallback_device: str) -> str:
        # Prefer the active ComfyUI torch device if available
        if mm is not None:
            try:
                dev = str(mm.get_torch_device())
                if dev:
                    return dev
            except Exception:
                pass

        # If a MODEL is provided, try to read its device
        if model is not None:
            try:
                # Many Comfy model wrappers expose .model (torch.nn.Module)
                mod = getattr(model, "model", None) or model
                if hasattr(mod, "device"):
                    return str(mod.device)
                # Last resort, read a parameter
                for p in getattr(mod, "parameters", lambda: [])():
                    return str(p.device)
            except Exception:
                pass

        # Fallback to user's selection
        return fallback_device

    def build(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        shift: float,
        use_dynamic_shifting: bool,
        use_karras_sigmas: bool,
        use_exponential_sigmas: bool,
        use_beta_sigmas: bool,
        stochastic_sampling: bool,
        align_your_steps: bool,
        fallback_device: str,
        model=None,
    ):
        device = self._pick_device(model, fallback_device)

        # Instantiate Diffusers' FlowMatch Euler scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            stochastic_sampling=stochastic_sampling,
        )

        # Prepare timesteps/sigmas
        # Note: align_your_steps is supported by this scheduler class.
        scheduler.set_timesteps(num_steps, device=device, align_your_steps=align_your_steps)

        # Extract sigmas (preferred) or fall back if API changes
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None or not torch.is_tensor(sigmas):
            # Conservative fallback: linear from max->min
            sigmas = torch.linspace(sigma_max, sigma_min, num_steps, dtype=torch.float32, device=device)
        else:
            sigmas = sigmas.to(device=device, dtype=torch.float32)

        timesteps = scheduler.timesteps
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype=torch.float32, device=device)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.float32)

        return (sigmas, timesteps)


NODE_CLASS_MAPPINGS = {
    "PVL_FlowMatchEulerSigmas": PVL_FlowMatchEulerSigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_FlowMatchEulerSigmas": "PVL FlowMatch Euler Sigmas"
}
