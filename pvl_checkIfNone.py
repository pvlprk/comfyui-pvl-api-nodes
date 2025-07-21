class IsConnected:
    """
    Returns True if the input is connected (not None), otherwise False.
    This version checks IMAGE-type input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value": ("IMAGE", {
                    "default": None,
                    "tooltip": "Connect any image here. Will return True if connected, False if not."
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_connected",)
    FUNCTION = "check"
    CATEGORY = "logic"

    def check(self, value=None):
        return (value is not None,)


# Required exports for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PVLCheckIfConnected": IsConnected,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVLCheckIfConnected": "PVL Check If Connected",
}
