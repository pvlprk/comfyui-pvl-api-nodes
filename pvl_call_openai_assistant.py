import openai
import time
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import requests
import os
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenAIAssistant")

# Supported models for validation
SUPPORTED_MODELS = ["gpt-4", "gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

def retry_request(max_retries=3, initial_delay=1, backoff_factor=2):
    """Decorator for retrying API requests with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = initial_delay
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, openai.APIError, openai.APITimeoutError) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    logger.warning(f"Retry {retry_count}/{max_retries} for {func.__name__} after error: {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor
            
            return None  # Shouldn't reach here
        return wrapper
    return decorator

class CallAssistantNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assistant_id": ("STRING", {"multiline": False, "tooltip": "ID of the OpenAI Assistant to call"}),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 999999,
                    "step": 1, "display": "number", "tooltip": "Seed for reproducibility"
                }),
                "model": ("STRING", {
                    "default": "gpt-4.1",
                    "multiline": False,
                    "tooltip": "Model name (e.g., gpt-4, gpt-4o, gpt-3.5-turbo). You can also connect a string input."
                }),
                "reasoning_effort": (["disabled", "low", "medium", "high"], {
                    "default": "disabled",
                    "tooltip": "Optional reasoning effort to influence response depth"
                }),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "tooltip": "Your OpenAI API key (or set OPENAI_API_KEY environment variable)"}),
                "input_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to send to the assistant"}),
                "image": ("IMAGE", {"tooltip": "Optional image to include in the request"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_response",)
    FUNCTION = "call_GPT_assistant"
    CATEGORY = "PVL_tools"

    @staticmethod
    def undefined_to_none(value):
        """Convert undefined values to None."""
        return None if value in (None, "undefined") else value

    @staticmethod
    def get_api_key(api_key=None):
        """Get API key from parameter or environment variable."""
        # First try the provided api_key
        if api_key and api_key.strip():
            logger.info("Using provided API key")
            return api_key.strip()
        
        # Then try environment variable
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key and env_key.strip():
            logger.info("Using API key from OPENAI_API_KEY environment variable")
            return env_key.strip()
        
        # If neither is available, return None
        return None

    @retry_request(max_retries=3, initial_delay=1, backoff_factor=2)
    def upload_image(self, api_key, image_tensor):
        """Upload an image tensor to OpenAI and return the file ID."""
        openai.api_key = api_key
        image_np = image_tensor.squeeze().clamp(0, 1).mul(255).byte().numpy()

        # Determine image mode based on dimensions and channels
        if image_np.ndim == 2:
            mode = "L"
        elif image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)
            mode = "L"
        elif image_np.shape[-1] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"

        # Convert to PIL Image and upload
        image_pil = Image.fromarray(image_np, mode=mode)
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        buffer.seek(0)

        file = openai.files.create(file=("image.png", buffer), purpose="vision")
        return file.id

    def call_assistant(self, assistant_id, seed, model, reasoning_effort="disabled", api_key="", input_text="", image=None):
        """Main function to call the OpenAI Assistant."""
        try:
            # Get API key from parameter or environment
            api_key = self.get_api_key(api_key)
            
            # Validate inputs
            if not api_key:
                return ("Error: API key is required. Please provide it directly or set OPENAI_API_KEY environment variable.",)
            
            if not assistant_id.strip():
                return ("Error: Assistant ID is required.",)
            
            # Validate model
            if model not in SUPPORTED_MODELS:
                logger.warning(f"Potentially unsupported model: {model}. Supported models are: {', '.join(SUPPORTED_MODELS)}")
            
            # Process inputs
            input_text = self.undefined_to_none(input_text) or ""
            image = self.undefined_to_none(image)
            
            # Set API key
            openai.api_key = api_key
            
            # Create a thread
            try:
                thread = openai.beta.threads.create()
                logger.info(f"Created thread with ID: {thread.id}")
            except Exception as e:
                logger.error(f"Failed to create thread: {str(e)}")
                return (f"Failed to create thread: {str(e)}",)

            # Track if any messages were created
            messages_created = False

            # Add text message if provided
            if input_text.strip():
                try:
                    openai.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=input_text.strip(),
                        metadata={"seed": str(seed), "model": model}
                    )
                    messages_created = True
                    logger.info("Added text message to thread")
                except Exception as e:
                    logger.error(f"Failed to send text message: {str(e)}")
                    return (f"Failed to send text message: {str(e)}",)

            # Add image if provided
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
                    logger.info("Added image to thread")
                except Exception as e:
                    logger.error(f"Image upload failed: {str(e)}")
                    return (f"Image upload failed: {str(e)}",)

            # Check if any content was provided
            if not messages_created:
                error_msg = "Error: No message content was sent (text and image both missing or empty)."
                logger.warning(error_msg)
                return (error_msg,)

            # Create and run the assistant
            try:
                run_args = {
                    "thread_id": thread.id,
                    "assistant_id": assistant_id,
                    "model": model,
                }

                if reasoning_effort != "disabled":
                    run_args["reasoning_effort"] = reasoning_effort

                run = openai.beta.threads.runs.create(**run_args)
                logger.info(f"Started assistant run with ID: {run.id}")
            except Exception as e:
                logger.error(f"Failed to create assistant run: {str(e)}")
                return (f"Failed to create assistant run: {str(e)}",)

            # Wait for the run to complete with timeout
            max_wait_time = 120  # 2 minutes
            start_time = time.time()
            
            while True:
                try:
                    run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                    
                    if run.status == "completed":
                        logger.info("Assistant run completed successfully")
                        break
                    elif run.status == "failed":
                        error_msg = f"Assistant run failed: {run.last_error['message']}"
                        logger.error(error_msg)
                        return (error_msg,)
                    elif run.status in ["expired", "cancelled"]:
                        error_msg = f"Assistant run {run.status}"
                        logger.warning(error_msg)
                        return (error_msg,)
                    
                    # Check for timeout
                    if time.time() - start_time > max_wait_time:
                        error_msg = f"Error: Assistant run timed out after {max_wait_time} seconds."
                        logger.error(error_msg)
                        return (error_msg,)
                    
                    # Add a small delay to avoid excessive API calls
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error retrieving run status: {str(e)}")
                    return (f"Error retrieving run status: {str(e)}",)

            # Retrieve and return the messages
            try:
                messages = openai.beta.threads.messages.list(thread_id=thread.id)
                if messages.data and len(messages.data) > 0:
                    # Get the last message (assistant's response)
                    last_message = messages.data[0]
                    if last_message.content and len(last_message.content) > 0:
                        response_text = last_message.content[0].text.value
                        logger.info("Successfully retrieved assistant response")
                        return (response_text,)
                    else:
                        logger.warning("Assistant returned empty content")
                        return ("Error: Assistant returned empty content.",)
                else:
                    logger.warning("No messages found in thread")
                    return ("Error: No messages found in thread.",)
            except Exception as e:
                logger.error(f"Failed to retrieve messages: {str(e)}")
                return (f"Failed to retrieve messages: {str(e)}",)
                
        except Exception as e:
            logger.error(f"Unexpected error in call_assistant: {str(e)}")
            return (f"Unexpected error: {str(e)}",)
