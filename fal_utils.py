import configparser
import io
import os
import tempfile

import numpy as np
import requests
import torch
from fal_client.client import SyncClient
from PIL import Image


class FalConfig:
    _instance = None
    _client = None
    _key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if os.environ.get("FAL_KEY") is not None:
                print("FAL_KEY found in environment variables")
                self._key = os.environ["FAL_KEY"]
            else:
                print("FAL_KEY not found in environment variables")
                self._key = config["API"]["FAL_KEY"]
                print("FAL_KEY found in config.ini")
                os.environ["FAL_KEY"] = self._key
                print("FAL_KEY set in environment variables")

            if self._key == "<your_fal_api_key_here>":
                print("WARNING: You are using the default FAL API key placeholder!")
        except KeyError:
            print("Error: FAL_KEY not found in config.ini or environment variables")

    def get_client(self):
        if self._client is None:
            self._client = SyncClient(key=self._key)
        return self._client

    def get_key(self):
        return self._key


class ImageUtils:
    @staticmethod
    def tensor_to_pil(image):
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.dtype in [np.float32, np.float64]:
                image_np = (image_np * 255).astype(np.uint8)

            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image):
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            client = FalConfig().get_client()
            image_url = client.upload_file(temp_file_path)
            return image_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)

    @staticmethod
    def mask_to_image(mask):
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result


class ResultProcessor:
    @staticmethod
    def process_image_result(result):
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            # Stack the images along a new first dimension (batch)
            stacked_images = np.stack(images, axis=0)  # (B, H, W, C)
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (img_tensor,)
        
        
class ApiHandler:
    @staticmethod
    def submit_and_get_result(endpoint, arguments):
        try:
            client = FalConfig().get_client()
            handler = client.submit(endpoint, arguments=arguments)
            return handler.get()
        except Exception as e:
            print(f"Error submitting to {endpoint}: {str(e)}")
            raise e

    @staticmethod
    def handle_image_generation_error(model_name, error):
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()
