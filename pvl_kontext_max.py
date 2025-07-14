import io
import time
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from inspect import cleandoc
from typing import Union, Optional
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api_nodes.apis.bfl_api import (
    BFLStatus,
    BFLFluxKontextProGenerateRequest,
    BFLFluxProGenerateResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    process_image_response,
)
from server import PromptServer

def convert_image_to_base64(image: torch.Tensor):
    scaled_image = downscale_image_tensor(image, total_pixels=2048 * 2048)
    if len(scaled_image.shape) > 3:
        scaled_image = scaled_image[0]
    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode()

def validate_aspect_ratio(ratio_string, minimum_ratio=1/4, maximum_ratio=4/1,
                          minimum_ratio_str="1:4", maximum_ratio_str="4:1"):
    try:
        width, height = map(int, ratio_string.split(":"))
        ratio = width / height
        if not (minimum_ratio <= ratio <= maximum_ratio):
            raise ValueError
        return ratio_string
    except Exception:
        raise ValueError(
            f"Aspect ratio must be in format 'width:height' between {minimum_ratio_str} and {maximum_ratio_str}."
        )

def validate_string(value: str, strip_whitespace=True):
    if not isinstance(value, str) or (strip_whitespace and not value.strip()):
        raise ValueError("Prompt must be a non-empty string.")

def _poll_until_generated(polling_url: str, timeout=360, node_id: Union[str, None] = None, fallback_image=None):
    start_time = time.time()
    retries_404 = 0
    max_retries_404 = 5
    retry_404_seconds = 2
    retry_202_seconds = 2
    retry_pending_seconds = 1
    request = requests.Request(method=HttpMethod.GET, url=polling_url)

    while True:
        if node_id:
            time_elapsed = time.time() - start_time
            PromptServer.instance.send_progress_text(
                f"Generating ({time_elapsed:.0f}s)", node_id
            )

        try:
            response = requests.Session().send(request.prepare())
        except Exception as e:
            print(f"[FluxNode Error] Request failed: {e}")
            return fallback_image

        if response.status_code == 200:
            try:
                result = response.json()
            except Exception as e:
                print(f"[FluxNode Error] Invalid JSON: {e}")
                return fallback_image

            if result["status"] == BFLStatus.ready:
                img_url = result["result"]["sample"]
                if node_id:
                    PromptServer.instance.send_progress_text(
                        f"Result URL: {img_url}", node_id
                    )
                try:
                    img_response = requests.get(img_url)
                    return process_image_response(img_response)
                except Exception as e:
                    print(f"[FluxNode Error] Failed to fetch image: {e}")
                    return fallback_image

            elif result["status"] in [
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
            ]:
                print(f"[FluxNode Warning] Moderated content: {result['status']}")
                return fallback_image

            elif result["status"] == BFLStatus.error:
                print(f"[FluxNode Warning] API returned error: {result}")
                return fallback_image

            elif result["status"] == BFLStatus.pending:
                time.sleep(retry_pending_seconds)
                continue

        elif response.status_code == 404:
            if retries_404 < max_retries_404:
                retries_404 += 1
                time.sleep(retry_404_seconds)
                continue
            print(f"[FluxNode Warning] Task not found after retries.")
            return fallback_image

        elif response.status_code == 202:
            time.sleep(retry_202_seconds)

        elif time.time() - start_time > timeout:
            print(f"[FluxNode Warning] Timeout after {timeout} seconds.")
            return fallback_image

        else:
            print(f"[FluxNode Warning] Unexpected status: {response.status_code}")
            return fallback_image

def handle_bfl_synchronous_operation(
    operation: SynchronousOperation,
    timeout_bfl_calls=360,
    node_id: Union[str, None] = None,
    fallback_image=None,
):
    response_api: BFLFluxProGenerateResponse = operation.execute()
    return _poll_until_generated(
        response_api.polling_url, timeout=timeout_bfl_calls, node_id=node_id, fallback_image=fallback_image
    )

class PvlKontextMax(ComfyNodeABC):
    """Flux.1 Kontext [max] node with optional fallback image output."""

    MINIMUM_RATIO = 1 / 4
    MAXIMUM_RATIO = 4 / 1
    MINIMUM_RATIO_STR = "1:4"
    MAXIMUM_RATIO_STR = "4:1"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation",
                    },
                ),
                "aspect_ratio": (
                    IO.STRING,
                    {
                        "default": "16:9",
                        "tooltip": "Aspect ratio in format 'W:H' (e.g., 16:9).",
                    },
                ),
                "guidance": (
                    IO.FLOAT,
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 99.0,
                        "step": 0.1,
                        "tooltip": "Guidance strength for image generation",
                    },
                ),
                "steps": (
                    IO.INT,
                    {
                        "default": 50,
                        "min": 1,
                        "max": 150,
                        "tooltip": "Number of denoising steps",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 1234,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Random seed for generation",
                    },
                ),
                "prompt_upsampling": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Enable prompt upsampling (creative mode)",
                    },
                ),
            },
            "optional": {
                "input_image": (IO.IMAGE,),
                "fallback_image": (IO.IMAGE,),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/BFL"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(
        self,
        prompt: str,
        aspect_ratio: str,
        guidance: float,
        steps: int,
        input_image: Optional[torch.Tensor] = None,
        seed=0,
        prompt_upsampling=False,
        fallback_image=None,
        unique_id: Union[str, None] = None,
        **kwargs,
    ):
        aspect_ratio = validate_aspect_ratio(
            aspect_ratio,
            minimum_ratio=self.MINIMUM_RATIO,
            maximum_ratio=self.MAXIMUM_RATIO,
            minimum_ratio_str=self.MINIMUM_RATIO_STR,
            maximum_ratio_str=self.MAXIMUM_RATIO_STR,
        )
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-kontext-max/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxKontextProGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxKontextProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                guidance=round(guidance, 1),
                steps=steps,
                seed=seed,
                aspect_ratio=aspect_ratio,
                input_image=(
                    input_image if input_image is None else convert_image_to_base64(input_image)
                ),
            ),
            auth_kwargs=kwargs,
        )
        output_image = handle_bfl_synchronous_operation(
            operation, node_id=unique_id, fallback_image=fallback_image
        )
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "PvlKontextMax": PvlKontextMax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PvlKontextMax": "Flux.1 Kontext [max] (Silent)",
}
