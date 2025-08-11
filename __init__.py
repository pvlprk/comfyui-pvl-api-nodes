from .pvl_call_openai_assistant import CallAssistantNode
from .pvl_comfydeploy_node import ComfyDeployNode
from .pvl_kontext_max import PvlKontextMax
from .pvl_checkIfNone import IsConnected
from .pvl_fal_kontext_max_multi import PVL_fal_KontextMaxMulti_API
from .pvl_fal_flux_with_lora import PVL_fal_FluxWithLora_API
from .pvl_fal_flux_dev import PVL_fal_FluxDev_API
from .pvl_fal_kontext_max_single import PVL_fal_KontextMaxSingle_API
from .pvl_fal_kontext_dev_inpaint import PVL_fal_KontextDevInpaint_API
from .pvl_fal_flux_general import PVL_fal_FluxGeneral_API
from .pvl_switchTen import PVL_Switch_Huge

NODE_CLASS_MAPPINGS = {
    "PVL Call OpenAI Assistant": CallAssistantNode,
    "PVL ComfyDeploy API Caller": ComfyDeployNode,
    "PVL KONTEXT MAX": PvlKontextMax,
    "PVLCheckIfConnected": IsConnected,
    "PVL_fal_KontextMaxMulti_API": PVL_fal_KontextMaxMulti_API,
    "PVL_fal_FluxWithLora_API": PVL_fal_FluxWithLora_API,
    "PVL_fal_FluxDev_API": PVL_fal_FluxDev_API,
    "PVL_fal_KontextMaxSingle_API": PVL_fal_KontextMaxSingle_API,
    "PVL_fal_KontextDevInpaint_API": PVL_fal_KontextDevInpaint_API, 
    "PVL_fal_FluxGeneral_API": PVL_fal_FluxGeneral_API,
    "PVL_Switch_Huge": PVL_Switch_Huge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL Call OpenAI Assistant": "PVL Call OpenAI Assistant",
    "PVL ComfyDeploy API Caller": "PVL ComfyDeploy API Caller",
    "PVL KONTEXT MAX": "PVL KONTEXT MAX",
    "PVLCheckIfConnected": "PVL Check If Connected",
    "PVL_fal_KontextMaxMulti_API": "PVL KONTEXT MAX MULTI (fal.ai)",
    "PVL_fal_FluxWithLora_API": "PVL FLUX DEV LORA (fal.ai)",
    "PVL_fal_FluxDev_API": "FLUX DEV (fal.ai)",
    "PVL_fal_KontextMaxSingle_API": "FLUX Kontext Max Single (fal.ai)",
    "PVL_fal_KontextDevInpaint_API": "FLUX Kontext Dev Inpaint (fal.ai)",
    "PVL_fal_FluxGeneral_API": "FLUX General (fal.ai)",
    "PVL_Switch_Huge": "PVL Switch Huge"
 }
