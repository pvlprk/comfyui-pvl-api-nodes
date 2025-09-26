class PVL_Txt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"

    CATEGORY = "PVL"

    @staticmethod
    def execute(text):
        return text,

NODE_CLASS_MAPPINGS = {
    "PVL_Txt": PVL_Txt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_Txt": "PVL - Txt",
}