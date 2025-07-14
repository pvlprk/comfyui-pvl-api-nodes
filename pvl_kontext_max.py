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
    BFLFluxProGenerateRequest,
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


class FluxProImageNodeSilent(ComfyNodeABC):
    """Flux.1 Pro Node with optional fallback image output in case of error/moderation."""

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
                "prompt_upsampling": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Creative modifier (nondeterministic)",
                    },
                ),
                "width": (
                    IO.INT,
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 1440,
                        "step": 32,
                    },
                ),
                "height": (
                    IO.INT,
                    {
                        "default": 768,
                        "min": 256,
                        "max": 1440,
                        "step": 32,
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Random seed used for noise.",
                    },
                ),
            },
            "optional": {
                "image_prompt": (IO.IMAGE,),
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
        prompt_upsampling,
        width: int,
        height: int,
        seed=0,
        image_prompt=None,
        fallback_image=None,
        unique_id: Union[str, None] = None,
        **kwargs,
    ):
        if image_prompt is not None:
            image_prompt = convert_image_to_base64(image_prompt)

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/bfl/flux-pro-1.1/generate",
                method=HttpMethod.POST,
                request_model=BFLFluxProGenerateRequest,
                response_model=BFLFluxProGenerateResponse,
            ),
            request=BFLFluxProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                width=width,
                height=height,
                seed=seed,
                image_prompt=image_prompt,
            ),
            auth_kwargs=kwargs,
        )

        output_image = handle_bfl_synchronous_operation(
            operation,
            node_id=unique_id,
            fallback_image=fallback_image
        )
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "FluxProImageNodeSilent": FluxProImageNodeSilent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxProImageNodeSilent": "Flux.1 Pro Image (Silent)",
}
