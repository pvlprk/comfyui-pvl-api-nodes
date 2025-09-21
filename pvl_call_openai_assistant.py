import openai
import time
import os
import logging
from PIL import Image
from io import BytesIO
import numpy as np

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Set up logger
logger = logging.getLogger("OpenAIAssistant")
logger.setLevel(logging.INFO)

SUPPORTED_MODELS = ["gpt-4", "gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

def retry_request(max_retries=3, initial_delay=1, backoff_factor=2, debug=False):
    """Decorator factory for retrying API requests with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = initial_delay
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except (openai.APIError,
                        openai.APITimeoutError,
                        openai.BadRequestError,
                        openai.RateLimitError,
                        Exception) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise RuntimeError(f"Max retries exceeded: {str(e)}")
                    if debug:
                        logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


class CallAssistantNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assistant_id": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
                "model": ("STRING", {"default": "gpt-4.1"}),
                "reasoning_effort": (["disabled", "low", "medium", "high"], {"default": "disabled"}),
                "retry_count": ("INT", {"default": 3, "min": 1, "max": 10,
                                        "tooltip": "Number of retry attempts if API fails"}),
                "timeout_seconds": ("INT", {"default": 25, "min": 5, "max": 600,
                                            "tooltip": "Timeout for assistant run in seconds"}),
                "debug": (["off", "on"], {"default": "off", "tooltip": "Enable verbose debug logging"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False}),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_response",)
    FUNCTION = "call_assistant"
    CATEGORY = "PVL_tools"

    @staticmethod
    def undefined_to_none(value):
        return None if value in (None, "undefined") else value

    @staticmethod
    def get_api_key(api_key=None, debug=False):
        if api_key and api_key.strip():
            if debug:
                logger.debug("Using provided API key")
            return api_key.strip()
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key and env_key.strip():
            if debug:
                logger.debug("Using API key from environment variable")
            return env_key.strip()
        raise RuntimeError("Error: API key is required (either as input or in OPENAI_API_KEY env var).")

    def estimate_tokens(self, text, model="gpt-4"):
        if not text:
            return 0
        if tiktoken:
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(text))
            except Exception:
                return len(text.split())
        return len(text.split())

    def enable_debug_logging(self):
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    def call_assistant(self, assistant_id, seed, model, reasoning_effort,
                       retry_count, timeout_seconds, debug,
                       api_key="", input_text="", image=None):
        debug_mode = (debug == "on")
        if debug_mode:
            self.enable_debug_logging()

        api_key = self.get_api_key(api_key, debug_mode)
        openai.api_key = api_key

        if not assistant_id.strip():
            raise RuntimeError("Error: Assistant ID is required.")

        if debug_mode:
            logger.debug(f"Calling assistant {assistant_id} with model {model}")

        # Wrap API calls with retry
        decorator = retry_request(max_retries=retry_count, debug=debug_mode)

        @decorator
        def create_thread():
            return openai.beta.threads.create()

        @decorator
        def create_message(**kwargs):
            return openai.beta.threads.messages.create(**kwargs)

        @decorator
        def create_run(**kwargs):
            return openai.beta.threads.runs.create(**kwargs)

        @decorator
        def retrieve_run(thread_id, run_id):
            return openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        @decorator
        def list_messages(thread_id):
            return openai.beta.threads.messages.list(thread_id=thread_id)

        # Create thread
        thread = create_thread()

        # Track if any messages created
        messages_created = False

        if input_text.strip():
            create_message(
                thread_id=thread.id,
                role="user",
                content=input_text.strip(),
                metadata={"seed": str(seed), "model": model}
            )
            messages_created = True
            if debug_mode:
                logger.debug(f"Input tokens: {self.estimate_tokens(input_text, model)}")

        if image is not None:
            file_id = self.upload_image(api_key, image, retry_count, debug_mode)
            create_message(
                thread_id=thread.id,
                role="user",
                content=[{"type": "image_file", "image_file": {"file_id": file_id}}],
            )
            messages_created = True

        if not messages_created:
            raise RuntimeError("Error: No content provided (text or image).")

        run_args = {"thread_id": thread.id, "assistant_id": assistant_id, "model": model}
        if reasoning_effort != "disabled":
            run_args["reasoning_effort"] = reasoning_effort

        run = create_run(**run_args)

        start_time = time.time()
        while True:
            run = retrieve_run(thread.id, run.id)
            if run.status == "completed":
                break
            elif run.status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Assistant run {run.status}: {getattr(run, 'last_error', '')}")
            if time.time() - start_time > timeout_seconds:
                raise RuntimeError(f"Assistant run timed out after {timeout_seconds} seconds")
            time.sleep(1)

        messages = list_messages(thread.id)
        if not messages.data:
            raise RuntimeError("Error: No messages found in thread.")

        last_message = messages.data[0]
        if not last_message.content:
            raise RuntimeError("Error: Assistant returned empty content.")

        response_text = last_message.content[0].text.value
        if debug_mode:
            logger.debug(f"Output tokens: {self.estimate_tokens(response_text, model)}")
        return (response_text,)

    def upload_image(self, api_key, image_tensor, retry_count=3, debug=False):
        decorator = retry_request(max_retries=retry_count, debug=debug)

        @decorator
        def do_upload():
            openai.api_key = api_key
            image_np = image_tensor.squeeze().clamp(0, 1).mul(255).byte().numpy()
            mode = "RGB"
            if image_np.ndim == 2:
                mode = "L"
            elif image_np.shape[-1] == 1:
                image_np = image_np.squeeze(-1); mode = "L"
            elif image_np.shape[-1] == 4:
                mode = "RGBA"
            image_pil = Image.fromarray(image_np, mode=mode)
            buffer = BytesIO(); image_pil.save(buffer, format="PNG"); buffer.seek(0)
            file = openai.files.create(file=("image.png", buffer), purpose="vision")
            return file.id

        return do_upload()
