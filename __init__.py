from .call_openai_assistant import CallAssistantNode
from .comfydeploy_node import ComfyDeployNode

NODE_CLASS_MAPPINGS = {
    "Call OpenAI Assistant": CallAssistantNode,
    "ComfyDeploy API Caller": ComfyDeployNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Call OpenAI Assistant": "Call OpenAI Assistant",
    "ComfyDeploy API Caller": "ComfyDeploy API Caller",
}
