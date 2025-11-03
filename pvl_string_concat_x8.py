class PVL_StringConcatX8:
    """
    PVL String Concat x8

    Takes eight string inputs and returns their direct concatenation.
    Preserves all whitespace exactly as provided (no extra spaces added).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_1": ("STRING", {"multiline": True, "default": "", "placeholder": "String 1"}),
                "string_2": ("STRING", {"multiline": True, "default": "", "placeholder": "String 2"}),
                "string_3": ("STRING", {"multiline": True, "default": "", "placeholder": "String 3"}),
                "string_4": ("STRING", {"multiline": True, "default": "", "placeholder": "String 4"}),
                "string_5": ("STRING", {"multiline": True, "default": "", "placeholder": "String 5"}),
                "string_6": ("STRING", {"multiline": True, "default": "", "placeholder": "String 6"}),
                "string_7": ("STRING", {"multiline": True, "default": "", "placeholder": "String 7"}),
                "string_8": ("STRING", {"multiline": True, "default": "", "placeholder": "String 8"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "🧩 PVL/Strings"

    def concat(self, string_1, string_2, string_3, string_4, string_5, string_6, string_7, string_8):
        # Direct concatenation; do not insert additional spaces
        return (f"{string_1}{string_2}{string_3}{string_4}{string_5}{string_6}{string_7}{string_8}",)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "PVL_StringConcatX8": PVL_StringConcatX8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_StringConcatX8": "PVL String Concat x8",
}
