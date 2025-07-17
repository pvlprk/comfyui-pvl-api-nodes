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
                "api_key": ("STRING", {"multiline": False, "tooltip": "Your OpenAI API key"}),
                "assistant_id": ("STRING", {"multiline": False, "tooltip": "ID of the OpenAI Assistant to call"}),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 999999,
                    "step": 1, "display": "number", "tooltip": "Seed for reproducibility"
                }),
                "model": ("STRING", {
                    "default": "gpt-4",
                    "multiline": False,
                    "tooltip": "Model name (e.g., gpt-4, gpt-4o, gpt-3.5-turbo). You can also connect a string input."
                }),
                "reasoning_effort": (["disabled", "low", "medium", "high"], {
                    "default": "disabled",
                    "tooltip": "Optional reasoning effort to influence response depth"
                }),
            },
            "optional": {
                "input_text": ("STRING", {"default": "", "multiline": True, "forceInput": True, "tooltip": "Prompt text to send to the assistant"}),
                "image": ("IMAGE", {"default": None, "forceInput": True, "tooltip": "Optional image to include in the request"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_response",)
    FUNCTION = "call_assistant"
    CATEGORY = "OpenAI"

    @staticmethod
    def undefined_to_none(value):
        return None if value in (None, "undefined") else value

    def upload_image(self, api_key, image_tensor):
        openai.api_key = api_key
        image_np = image_tensor.squeeze().clamp(0, 1).mul(255).byte().numpy()

        if image_np.ndim == 2:
            mode = "L"
        elif image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)
            mode = "L"
        elif image_np.shape[-1] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"

        image_pil = Image.fromarray(image_np, mode=mode)
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        buffer.seek(0)

        file = openai.files.create(file=("image.png", buffer), purpose="vision")
        return file.id

    def call_assistant(self, api_key, assistant_id, seed, model, reasoning_effort="disabled", input_text="", image=None):
        input_text = self.undefined_to_none(input_text) or ""
        image = self.undefined_to_none(image)

        openai.api_key = api_key

        try:
            thread = openai.beta.threads.create()
        except Exception as e:
            return (f"Failed to create thread: {e}",)

        messages_created = False

        if input_text.strip():
            try:
                openai.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=input_text.strip(),
                    metadata={"seed": str(seed), "model": model}
                )
                messages_created = True
            except Exception as e:
                return (f"Failed to send text message: {e}",)

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
                messages_created = True
            except Exception as e:
                return (f"Image upload failed: {e}",)

        if not messages_created:
            return ("Error: No message content was sent (text and image both missing or failed).",)

        try:
            run_args = {
                "thread_id": thread.id,
                "assistant_id": assistant_id,
                "model": model,
            }

            if reasoning_effort != "disabled":
                run_args["reasoning_effort"] = reasoning_effort

            run = openai.beta.threads.runs.create(**run_args)
        except Exception as e:
            return (f"Failed to create assistant run: {e}",)
            run = openai.beta.threads.runs.create(**run_args)
        except Exception as e:
            return (f"Failed to create assistant run: {e}",)

        while True:
            run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                break
            elif run.status == "failed":
                return (f"Assistant run failed: {run.last_error['message']}",)
            time.sleep(1)

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        return (messages.data[0].content[0].text.value,)
