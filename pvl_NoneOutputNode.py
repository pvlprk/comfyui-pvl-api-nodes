class PVL_NoneOutputNode:
    """
    A node that can either pass through its input or output None to skip downstream execution.
    Input is optional - works even when nothing is connected.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bypass": ("BOOLEAN", {"default": False})  # Toggle: True for pass-through, False for None
            },
            "optional": {
                "input": ("*",)  # Optional input that accepts any data type
            }
        }

    RETURN_TYPES = ("*",)  # Output type matches input type
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "utility"

    def execute(self, bypass, input=None):
        if bypass:
            # When bypass is True, pass the input through (could be None if not connected)
            return (input,)
        else:
            # When bypass is False, return None
            return (None,)