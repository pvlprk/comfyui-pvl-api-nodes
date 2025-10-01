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
from .pvl_fal_kontext_dev_lora import PVL_fal_KontextDevLora_API
from .pvl_fal_kontext_pro import PVL_fal_KontextPro_API
from .pvl_fal_flux_pro11_ultra import PVL_fal_FluxPro_v1_1_Ultra_API
from .pvl_fal_flux_pro_fill import PVL_fal_FluxPro_Fill_API
from .pvl_fal_kontext_dev import PVL_fal_Kontext_Dev_API
from .pvl_fal_lumaphoton_flash_reframe import PVL_fal_LumaPhoton_FlashReframe_API
from .pvl_fal_lumaphoton_reframe import PVL_fal_LumaPhoton_Reframe_API
from .pvl_NoneOutputNode import PVL_NoneOutputNode
from .pvl_SaveOrNot import PVL_SaveOrNot
from .pvl_ImageResize import PVL_ImageResize
from .pvl_ImageStitch import PVL_ImageStitch
from .pvl_OpenPoseMatch import PVL_OpenPoseMatch
from .pvl_OpenPoseMatch_Z import PVL_OpenPoseMatch_Z
from .pvl_fal_nano_banana_edit import PVL_fal_NanoBanana_API
from .pvl_stitch2size import PVL_Stitch2Size
from .pvl_crop2AR import PVL_Crop2AR
from .pvl_style_picker import PVL_StylePicker
from .pvl_google_nano_banana import PVL_Google_NanoBanana_API
from .pvl_fal_seedream4_edit import PVL_fal_SeeDream4_API
from .pvl_fal_qwen_txt2img import PVL_fal_QwenImage_API
from .pvl_fal_flux_pulid import PVL_fal_FluxPulid
from .pvl_google_nano_banana_multi_img import PVL_Google_NanoBanana_Multi_API
from .pvl_gemini_api import PVL_Gemini_API
from .pvl_txt import PVL_Txt
from .pvl_boolean_logic import PVL_BooleanLogic
from .pvl_any2string import PVL_Any2String
from .pvl_fal_remove_bg_v2 import PVL_fal_RemoveBackground_API
from .pvl_fal_depth_anything_v2 import PVL_fal_DepthAnythingV2_API
from .pvl_google_nano_banana_mandatory_img import PVL_Google_NanoBanana_API_mandatory_IMG

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
    "PVL_Switch_Huge": PVL_Switch_Huge,
    "PVL_fal_KontextDevLora_API": PVL_fal_KontextDevLora_API,
    "PVL_fal_KontextPro_API": PVL_fal_KontextPro_API,
    "PVL_fal_FluxPro_v1_1_Ultra_API": PVL_fal_FluxPro_v1_1_Ultra_API,
    "PVL_fal_FluxPro_Fill_API": PVL_fal_FluxPro_Fill_API,
    "PVL_fal_Kontext_Dev_API": PVL_fal_Kontext_Dev_API,
    "PVL_fal_LumaPhoton_FlashReframe_API": PVL_fal_LumaPhoton_FlashReframe_API,
    "PVL_fal_LumaPhoton_Reframe_API": PVL_fal_LumaPhoton_Reframe_API,
    "PVL_NoneOutputNode": PVL_NoneOutputNode,
    "PVL_SaveOrNot": PVL_SaveOrNot,
    "PVL_ImageResize": PVL_ImageResize,
    "PVL_ImageStitch": PVL_ImageStitch,
    "PVL_OpenPoseMatch": PVL_OpenPoseMatch,
    "PVL_OpenPoseMatch_Z": PVL_OpenPoseMatch_Z,
    "PVL_fal_NanoBanana_API": PVL_fal_NanoBanana_API,
    "PVL_Stitch2Size": PVL_Stitch2Size,
    "PVL_Crop2AR": PVL_Crop2AR,
    "PVL_StylePicker": PVL_StylePicker,
    "PVL_Google_NanoBanana_API": PVL_Google_NanoBanana_API,
    "PVL_fal_SeeDream4_API": PVL_fal_SeeDream4_API,
    "PVL_fal_QwenImage_API": PVL_fal_QwenImage_API,
    "PVL_fal_FluxPulid": PVL_fal_FluxPulid,
    "PVL_Google_NanoBanana_Multi_API": PVL_Google_NanoBanana_Multi_API,
    "PVL_Gemini_API": PVL_Gemini_API,
    "PVL_Txt": PVL_Txt,
    "PVL_BooleanLogic": PVL_BooleanLogic,
    "PVL_Any2String": PVL_Any2String,
    "PVL_fal_RemoveBackground_API": PVL_fal_RemoveBackground_API,
    "PVL_fal_DepthAnythingV2_API": PVL_fal_DepthAnythingV2_API,
    "PVL_Google_NanoBanana_API_mandatory_IMG": PVL_Google_NanoBanana_API_mandatory_IMG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL Call OpenAI Assistant": "PVL Call OpenAI Assistant",
    "PVL ComfyDeploy API Caller": "PVL ComfyDeploy API Caller",
    "PVL KONTEXT MAX": "PVL KONTEXT MAX (comfyui.org)",
    "PVLCheckIfConnected": "PVL Check If Connected",
    "PVL_fal_KontextMaxMulti_API": "PVL KONTEXT MAX MULTI (fal.ai)",
    "PVL_fal_FluxWithLora_API": "PVL FLUX DEV LORA (fal.ai)",
    "PVL_fal_FluxDev_API": "PVL FLUX DEV (fal.ai)",
    "PVL_fal_KontextMaxSingle_API": "PVL Kontext Max Single (fal.ai)",
    "PVL_fal_KontextDevInpaint_API": "PVL Kontext Dev Inpaint (fal.ai)",
    "PVL_fal_FluxGeneral_API": "PVL FLUX General (fal.ai)",
    "PVL_Switch_Huge": "PVL Switch Huge",
    "PVL_fal_KontextDevLora_API": "PVL Kontext Dev LoRA (fal.ai)",
    "PVL_fal_KontextPro_API": "PVL Kontext Pro (fal.ai)",
    "PVL_fal_FluxPro_v1_1_Ultra_API": "PVL FluxPro1.1 Ultra (fal.ai)",
    "PVL_fal_FluxPro_Fill_API": "PVL FluxPro Fill (fal.ai)",
    "PVL_fal_Kontext_Dev_API": "PVL Kontext Dev (fal.ai)",
    "PVL_fal_LumaPhoton_Reframe_API": "PVL LumaPhoton Reframe (fal.ai)",
    "PVL_NoneOutputNode": "PVL NoneOutputNode",
    "PVL_SaveOrNot": "PVL Save Or Not",
    "PVL_ImageResize": "PVL Image Resize",
    "PVL_ImageStitch": "PVL Image Stitch",
    "PVL_OpenPoseMatch": "PVL OpenPose Match",
    "PVL_OpenPoseMatch_Z": "PVL OpenPose Match_Z",
    "PVL_fal_NanoBanana_API": "PVL FAL Nano-Banana Edit",
    "PVL_Stitch2Size": "PVL Stitch 2 Size",
    "PVL_Crop2AR": "PVL Crop to Aspect Ratio",
    "PVL_StylePicker": "PVL StylePicker",
    "PVL_Google_NanoBanana_API": "PVL Google NanoBanana API",
    "PVL_fal_SeeDream4_API": "FAL SeeDream4 Edit (fal.ai)",
    "PVL_fal_QwenImage_API": "PVL QwenImage txt2img (fal.ai)",
    "PVL_fal_FluxPulid": "PVL Flux PuLID (fal.ai)",
    "PVL_Google_NanoBanana_Multi_API": "PVL Google NanoBanana API Multi",
    "PVL_Gemini_API": "PVL Gemini Api",
    "PVL_Txt": "PVL Txt",
    "PVL_BooleanLogic": "PVL BooleanLogic",
    "PVL_Any2String": "PVL Any To String",
    "PVL_fal_RemoveBackground_API": "PVL Remove Background V2 (fal.ai)",
    "PVL_fal_DepthAnythingV2_API": "PVL Depth Anything V2 (fal.ai)",
    "PVL_Google_NanoBanana_API_mandatory_IMG": "PVL Google Nano-Banana API mandatory IMG",
}