class PVL_StringListToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING",),
                "delimiter": ("STRING", {"default": "[++]"}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "PVL/Utils/Text"

    def convert(self, texts, delimiter):
        # texts is a python list because INPUT_IS_LIST = True
        if texts is None or len(texts) == 0:
            return ("",)

        # Make sure everything is a string (ComfyUI can sometimes pass None)
        safe_texts = ["" if t is None else str(t) for t in texts]
        return (delimiter.join(safe_texts),)


class PVL_StringToStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": "[++]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "PVL/Utils/Text"

    def convert(self, text, delimiter):
        if text is None:
            return ([ "" ],)

        s = str(text)
        d = "" if delimiter is None else str(delimiter)

        # If delimiter is empty, avoid splitting into characters
        if d == "":
            return ([s],)

        parts = s.split(d)
        return (parts,)
