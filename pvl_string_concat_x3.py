# pvl_string_concat_x3.py
# Autonomous ComfyUI node: PVL String Concat x3
# Concatenates three strings exactly as provided (no automatic spaces)

class PVL_StringConcatX3:
    """
    PVL String Concat x3

    Takes three string inputs and returns their direct concatenation.
    Preserves all whitespace exactly as provided (no extra spaces added).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_1": ("STRING", {"multiline": True, "default": "", "placeholder": "String 1"}),
                "string_2": ("STRING", {"multiline": True, "default": "", "placeholder": "String 2"}),
                "string_3": ("STRING", {"multiline": True, "default": "", "placeholder": "String 3"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "🧩 PVL/Strings"

    def concat(self, string_1: str, string_2: str, string_3: str):
        # Direct concatenation; do not insert additional spaces
        return (f"{string_1}{string_2}{string_3}",)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "PVL_StringConcatX3": PVL_StringConcatX3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_StringConcatX3": "PVL String Concat x3",
}
