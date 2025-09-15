import torch
from .fal_utils import ImageUtils, ApiHandler, ResultProcessor


class PVL_fal_FluxPulid:
    """
    ComfyUI node for fal-ai/flux-pulid endpoint.
    Generates an image using a prompt and optional reference image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "bad quality, worst quality, text, signature, watermark, extra limbs"
                    }
                ),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0}),
                "true_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "id_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "max_sequence_length": (["128", "256", "512"], {"default": "256"}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "request_id",)
    FUNCTION = "generate"
    CATEGORY = "PVL/FAL"

    def generate(self, prompt, reference_image=None,
                 negative_prompt="bad quality, worst quality, text, signature, watermark, extra limbs",
                 num_inference_steps=20, guidance_scale=4.0, seed=0,
                 true_cfg=1.0, id_weight=1.0, enable_safety_checker=False,
                 max_sequence_length="256", width=1024, height=1024, start_step=0):

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "true_cfg": true_cfg,
            "id_weight": id_weight,
            "enable_safety_checker": enable_safety_checker,
            "max_sequence_length": max_sequence_length,
            "image_size": {"width": width, "height": height},
        }

        if seed > 0:
            payload["seed"] = seed
        if start_step > 0:
            payload["start_step"] = start_step

        if reference_image is not None:
            # Use Base64 data URI method (Method 2)
            payload["reference_image_url"] = ImageUtils.image_to_data_uri(reference_image)

        # Call the API
        result = ApiHandler.submit_and_get_result("fal-ai/flux-pulid", payload)

        # Convert output images to tensors
        tensors = ResultProcessor.process_image_result(result)

        return (tensors[0], result.get("requestId", ""))