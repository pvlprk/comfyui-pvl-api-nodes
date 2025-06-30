import openai
import time

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
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "call_assistant"
    CATEGORY = "OpenAI"

    @staticmethod
    def get_model_list():
        try:
            # You might want to cache this in real use to avoid calling too often
            models = openai.models.list()
            model_ids = [m.id for m in models.data if m.id.startswith("gpt")]
            return tuple(sorted(set(model_ids)))
        except Exception as e:
            print(f"Failed to fetch model list: {e}")
            return ("gpt-4", "gpt-4o", "gpt-3.5-turbo")

    def call_assistant(self, api_key, assistant_id, input_text, seed, model):
        openai.api_key = api_key

        # Optional: could store seed or model as message metadata
        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input_text,
            metadata={
                "seed": str(seed),
                "model": model
            }
        )

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
