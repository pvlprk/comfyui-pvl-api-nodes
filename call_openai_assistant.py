import openai
import time
import torch
from PIL import Image
from io import BytesIO
import numpy as np

class CallAssistantNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "assistant_id": ("STRING", {"multiline": False}),
                "input_text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
                "model": (cls.get_model_list(),),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "call_assistant"
    CATEGORY = "OpenAI"

    @staticmethod
    def get_model_list():
        try:
            models = openai.models.list()
            model_ids = [m.id for m in models.data if m.id.startswith("gpt")]
            return tuple(sorted(set(model_ids)))
        except Exception as e:
            print(f"Failed to fetch model list: {e}")
            return ("gpt-4", "gpt-4o", "gpt-3.5-turbo")

    def upload_image(self, api_key, image_tensor):
        openai.api_key = api_key

        # Convert tensor to numpy and normalize to 0–255 range
        image_np = image_tensor.squeeze().clamp(0, 1).mul(255).byte().numpy()

        # Determine image mode
        if image_np.ndim == 2:
            mode = "L"
        elif image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)
            mode = "L"
        elif image_np.shape[-1] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"

        # Convert to PIL
        image_pil = Image.fromarray(image_np, mode=mode)

        # Save to buffer as PNG
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        buffer.seek(0)

        # Pass a tuple: (filename, file buffer)
        file = openai.files.create(
            file=("image.png", buffer),
            purpose="vision"
        )
        return file.id


    def call_assistant(self, api_key, assistant_id, input_text, seed, model, image=None):
        openai.api_key = api_key
        thread = openai.beta.threads.create()

        if not input_text.strip() and image is None:
            return ("Error: Both text and image inputs are empty.",)

        if input_text.strip():
            openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=input_text.strip(),
                metadata={"seed": str(seed), "model": model}
            )

        if image is not None:
            try:
                file_id = self.upload_image(api_key, image)
                openai.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=[{
                        "type": "image_file",
                        "image_file": {"file_id": file_id}
                    }]
                )
            except Exception as e:
                return (f"Image upload failed: {e}",)

        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            model=model
        )

        while True:
            run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                break
            elif run.status == "failed":
                return (f"Assistant run failed: {run.last_error['message']}",)
            time.sleep(1)

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        return (messages.data[0].content[0].text.value,)
