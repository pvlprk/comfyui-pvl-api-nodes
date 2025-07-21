from .pvl_call_openai_assistant import CallAssistantNode
from .pvl_comfydeploy_node import ComfyDeployNode
from .pvl_kontext_max import PvlKontextMax
from .pvl_checkIfNone import IsConnected

NODE_CLASS_MAPPINGS = {
    "PVL Call OpenAI Assistant": CallAssistantNode,
    "PVL ComfyDeploy API Caller": ComfyDeployNode,
    "PVL KONTEXT MAX": PvlKontextMax,
    "PVLCheckIfConnected": IsConnected,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL Call OpenAI Assistant": "PVL Call OpenAI Assistant",
    "PVL ComfyDeploy API Caller": "PVL ComfyDeploy API Caller",
    "PVL KONTEXT MAX": "PVL KONTEXT MAX",
    "PVLCheckIfConnected": "PVL Check If Connected",
}
