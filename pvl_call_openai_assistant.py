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
                except (openai.OpenAIError, Exception) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise
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

    RETURN_TYPES = ("STRING", "BOOLEAN",)
    RETURN_NAMES = ("assistant_response", "error",)
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

        try:
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
            messages_created = False

            input_tokens = 0
            if input_text.strip():
                create_message(
                    thread_id=thread.id,
                    role="user",
                    content=input_text.strip(),
                    metadata={"seed": str(seed), "model": model}
                )
                messages_created = True
                input_tokens = self.estimate_tokens(input_text, model)
                if debug_mode:
                    logger.debug(f"Input tokens estimated: {input_tokens}")

            if image is not None:
                file_id = self.upload_image(api_key, image, retry_count, debug_mode)
                create_message(
                    thread_id=thread.id,
                    role="user",
                    content=[{"type": "image_file", "image_file": {"file_id": file_id}}],
                )
                messages_created = True
                if debug_mode:
                    logger.debug("Image uploaded and attached to thread.")

            if not messages_created:
                raise RuntimeError("Error: No content provided (text or image).")

            run_args = {"thread_id": thread.id, "assistant_id": assistant_id, "model": model}
            if reasoning_effort != "disabled":
                run_args["reasoning_effort"] = reasoning_effort

            @decorator
            def run_and_wait(**kwargs):
                run = create_run(**kwargs)
                start_time = time.time()
                if debug_mode:
                    logger.debug(f"Run {run.id} started, polling for completion...")
                while True:
                    r = retrieve_run(kwargs["thread_id"], run.id)

                    if r.status == "completed":
                        if debug_mode:
                            logger.debug(f"Run {r.id} completed successfully.")
                        return r

                    if r.status in ["failed", "expired", "cancelled", "incomplete"]:
                        raise RuntimeError(f"Run ended with status {r.status}, error: {getattr(r, 'last_error', None)}")

                    if r.status == "requires_action":
                        raise RuntimeError("Run requires action (tool calls) which is not supported")

                    if time.time() - start_time > timeout_seconds:
                        raise RuntimeError(f"Assistant run timed out after {timeout_seconds} seconds")

                    time.sleep(1)

            run = run_and_wait(**run_args)

            messages = list_messages(thread.id)
            if not messages or not getattr(messages, "data", None):
                raise RuntimeError("No messages returned")

            last_message = messages.data[0]
            if not last_message.content:
                raise RuntimeError("Message has no content")

            text_block = getattr(last_message.content[0], "text", None)
            if not text_block or not getattr(text_block, "value", None):
                raise RuntimeError("Message text content missing")

            response_text = text_block.value
            output_tokens = self.estimate_tokens(response_text, model)
            total_tokens = input_tokens + output_tokens

            if debug_mode:
                logger.debug(f"Output tokens estimated: {output_tokens}")
                logger.debug(f"Total tokens estimated: {total_tokens}")

            return (response_text, True)

        except Exception as e:
            if debug_mode:
                logger.error(f"CallAssistantNode error: {e}")
            return ("", False)

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
