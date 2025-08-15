class PVL_NoneOutputNode:
    """
    A node that can either pass through its input or output None to skip downstream execution.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),  # Accept any input type
                "bypass": ("BOOLEAN", {"default": False})  # Toggle: True for pass-through, False for None
            }
        }

    RETURN_TYPES = ("*",)  # Output type matches input type
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "PVL_tools"

    def execute(self, input, bypass):
        if bypass:
            # When bypass is True, pass the input through
            return (input,)
        else:
            # When bypass is False, return None
            return (None,)