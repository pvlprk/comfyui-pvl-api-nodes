import os
from .fal_utils import ApiHandler, ImageUtils, ResultProcessor, FalConfig

class PVL_fal_KontextMaxMulti_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "aspect_ratio": (
                    [
                        None,
                        "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"
                    ],
                    {"default": "1:1"},
                ),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def generate_image(
        self,
        prompt,
        image_1,
        image_2,
        image_3=None,
        image_4=None,
        aspect_ratio="1:1",
        guidance_scale=3.5,
        num_images=1,
        safety_tolerance="2",
        output_format="png",
        sync_mode=False,
        seed=0,
        api_key="",
    ):
        if api_key:
            os.environ["FAL_KEY"] = api_key
            FalConfig()._initialize()

        image_urls = []
        for i, img in enumerate([image_1, image_2, image_3, image_4], 1):
            if img is not None:
                url = ImageUtils.image_to_data_uri(img)
                if url:
                    image_urls.append(url)
                else:
                    print(f"Error: Failed to upload image {i}")
                    return ResultProcessor.create_blank_image()

        if len(image_urls) < 2:
            print("Error: At least 2 images required.")
            return ResultProcessor.create_blank_image()

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        if seed > 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/kontext/max/multi", arguments)
            output = ResultProcessor.process_image_result(result)
            return output
        except Exception as e:
            return ApiHandler.handle_image_generation_error("KontextMaxMulti", e)
