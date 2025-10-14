# pvl_split_string.py

class PVL_SplitString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "text"}),
            },
            "optional": {
                "delimiter": ("STRING", {"multiline": False, "default": "[*]"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("string_1", "string_2", "string_3", "string_4", "string_5", "string_6",)
    FUNCTION = "split"
    CATEGORY = "PVL_Utils"

    def split(self, text, delimiter="[*]"):
        # Handle edge cases: empty delimiter -> no split
        if delimiter is None or delimiter == "":
            parts = [text]
        else:
            parts = text.split(delimiter)

        strings = [part.strip() for part in parts[:4]]
        string_1, string_2, string_3, string_4, string_5, string_6 = strings + [""] * (6 - len(strings))

        return (string_1, string_2, string_3, string_4, string_5, string_6,)


# Required node registration
NODE_CLASS_MAPPINGS = {
    "PVL_SplitString": PVL_SplitString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_SplitString": "PVL — Split String",
}
