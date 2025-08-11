import requests
import base64
import time
import json
import torch
import numpy as np
from io import BytesIO
from PIL import Image

class ComfyDeployNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deployment_id": ("STRING", {"multiline": False}),
                "api_key": ("STRING", {"multiline": False}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "nonHumanChar": ("BOOLEAN", {"default": False}),
                "mode": (["text", "person", "both"],),
                "three_view": ("BOOLEAN", {"default": False}),
                "gemini_gpt_switch": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0}),
                "t5_clip_prompt_splitter": ("BOOLEAN", {"default": False}),
                "add_missing_clothes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "human_reference": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "invoke_comfydeploy_workflow"
    CATEGORY = "PVL_tools"

    def invoke_comfydeploy(self, deployment_id, api_key, batch_size,
                           nonHumanChar, mode, three_view, gemini_gpt_switch,
                           seed, t5_clip_prompt_splitter, add_missing_clothes,
                           prompt=None, human_reference=None):

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        inputs = {}

        if prompt:
            inputs["prompt"] = prompt

        if human_reference is not None:
            buffered = BytesIO()
            image = Image.fromarray(human_reference)
            image.save(buffered, format="PNG")
            image_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            inputs["input_image"] = image_b64

        inputs.update({
            "batch_size": batch_size,
            "nonHumanChar": nonHumanChar,
            "mode": mode,
            "3view": three_view,
            "gemini_gpt_switch": gemini_gpt_switch,
            "seed": seed,
            "T5-CLIP_prompt_splitter": t5_clip_prompt_splitter,
            "add_missing_clothes": add_missing_clothes
        })
        
        print(f"[DEBUG] Prompt: '{prompt}'")
        print("[DEBUG] Payload being sent:")
        print(json.dumps({"deployment_id": deployment_id, "inputs": inputs}, indent=2))
        
        response = requests.post(
            "https://api.comfydeploy.com/api/run/deployment/queue",
            json={"deployment_id": deployment_id, "inputs": inputs},
            headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"Failed to queue run: {response.text}")

        run_id = response.json().get("run_id")
        if not run_id:
            raise Exception("ComfyDeploy did not return a valid run ID.")

        max_wait = 150
        interval = 5
        waited = 0

        image_url = None
        text_url = None

        while waited < max_wait:
            time.sleep(interval)
            waited += interval

            poll_resp = requests.get(
                f"https://api.comfydeploy.com/api/run/{run_id}",
                headers=headers
            )

            if poll_resp.status_code != 200:
                continue

            run_data = poll_resp.json()
            outputs = run_data.get("outputs", [])

            for output in outputs:
                data = output.get("data", {})
                all_files = data.get("files", []) + data.get("images", [])

                for file in all_files:
                    url = file.get("url", "")
                    if url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        image_url = url
                    elif url.lower().endswith(".txt"):
                        text_url = url

            if image_url:
                break

        if not image_url:
            raise TimeoutError("Timed out waiting for ComfyDeploy run to complete.")

        # Download image and convert to tensor
        img_tensor = None
        try:
            # Download image
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            # Open image and preserve alpha if it exists
            img = Image.open(BytesIO(img_response.content))
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            
            # Convert to NumPy array and normalize
            np_img = np.array(img).astype(np.float32) / 255.0
            
            # Convert to PyTorch tensor
            # Shape will be [1, H, W, C] with C = 3 or 4 depending on the image
            img_tensor = torch.from_numpy(np_img).unsqueeze(0)

        except Exception as e:
            raise RuntimeError(f"Failed to download or convert image: {e}")
            
        # Download text
        text_content = ""
        if text_url:
            txt_resp = requests.get(text_url)
            txt_resp.raise_for_status()
            text_content = txt_resp.text

        return (img_tensor, text_content)