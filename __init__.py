from .pvl_call_openai_assistant import CallAssistantNode
from .pvl_comfydeploy_node import ComfyDeployNode
from .pvl_kontexrt_max import PvlKontextMax

NODE_CLASS_MAPPINGS = {
    "PVL Call OpenAI Assistant": CallAssistantNode,
    "PVL ComfyDeploy API Caller": ComfyDeployNode,
    "PVL KONTEXT MAX": PvlKontextMax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL Call OpenAI Assistant": "PVL Call OpenAI Assistant",
    "PVL ComfyDeploy API Caller": "PVL ComfyDeploy API Caller",
    "PVL KONTEXT MAX": "PVL KONTEXT MAX",
}
