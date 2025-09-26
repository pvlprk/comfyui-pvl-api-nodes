import comfy.model_base as model_base

class PVL_BooleanLogic:
    """
    PVL - Boolean Logic
    A simple node that takes two boolean inputs and outputs a boolean value
    based on the selected logic operator (AND / OR).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bool_a": ("BOOLEAN", {"default": False}),
                "bool_b": ("BOOLEAN", {"default": False}),
                "logic_operator": (["AND", "OR"], {"default": "AND"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "apply_logic"
    CATEGORY = "PVL"

    def apply_logic(self, bool_a, bool_b, logic_operator):
        if logic_operator == "AND":
            return (bool_a and bool_b,)
        elif logic_operator == "OR":
            return (bool_a or bool_b,)
        else:
            # default fallback
            return (False,)

# Register node
NODE_CLASS_MAPPINGS = {
    "PVL_BooleanLogic": PVL_BooleanLogic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_BooleanLogic": "PVL - Boolean Logic"
}
