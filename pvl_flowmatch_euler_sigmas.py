# comfyui/custom_nodes/pvl_flowmatch_euler_sigmas.py
# ---------------------------------------------------
# A ComfyUI node that exposes Hugging Face Diffusers' FlowMatchEulerDiscreteScheduler
# and outputs a sigma schedule compatible with KSampler (Advanced).
# ---------------------------------------------------

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

class PVL_FlowMatchEulerSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_steps": ("INT", {"default": 28, "min": 1, "max": 4096}),
                "sigma_min": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0}),
                "sigma_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "use_dynamic_shifting": ("BOOLEAN", {"default": True}),
                "use_karras_sigmas": ("BOOLEAN", {"default": False}),
                "stochastic_sampling": ("BOOLEAN", {"default": False}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "TIMESTEPS",)
    RETURN_NAMES = ("sigmas", "timesteps",)
    FUNCTION = "build"
    CATEGORY = "sampling/schedulers"

    def build(self, num_steps, sigma_min, sigma_max, shift,
              use_dynamic_shifting, use_karras_sigmas, stochastic_sampling, device):

        # 1. Instantiate the FlowMatch Euler scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            use_karras_sigmas=use_karras_sigmas,
            stochastic_sampling=stochastic_sampling,
        )

        # 2. Generate timesteps/sigmas for inference
        scheduler.set_timesteps(num_steps, device=device)

        # 3. Extract sigmas from the scheduler
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None:
            # Fallback if Diffusers changes internal structure
            sigmas = torch.linspace(sigma_max, sigma_min, num_steps, dtype=torch.float32, device=device)

        timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)

        # Return both sigmas and timesteps
        return (sigmas, timesteps)


NODE_CLASS_MAPPINGS = {
    "PVL_FlowMatchEulerSigmas": PVL_FlowMatchEulerSigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_FlowMatchEulerSigmas": "FlowMatch Euler (Sigmas)"
}