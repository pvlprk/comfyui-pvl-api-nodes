# pvl_string_to_number.py
# PVL — String To Number (clone of CR_StringToNumber)
# Drop this file into ComfyUI/custom_nodes/ and restart.

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")


class PVL_StringToNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "text", "forceInput": True}),
                "round_integer": (["round", "round down", "round up"],),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("INT", "FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "PVL_Utils"  
    
    def convert(self, text, round_integer):
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        # Determine if numeric (supports optional leading '-' and one decimal point)
        if text.startswith('-') and text[1:].replace('.', '', 1).isdigit():
            float_out = -float(text[1:])
        else:
            if text.replace('.', '', 1).isdigit():
                float_out = float(text)
            else:
                print("[Error] PVL String To Number. Not a number.")
                return {}

        if round_integer == "round up":
            if text.startswith('-'):
                int_out = int(float_out)  # toward zero for negatives
            else:
                int_out = int(float_out) + 1  # ceil for positives
        elif round_integer == "round down":
            if text.startswith('-'):
                int_out = int(float_out) - 1  # floor for negatives
            else:
                int_out = int(float_out)      # floor for positives
        else:
            int_out = round(float_out)

        return (int_out, float_out,)


# Registration
NODE_CLASS_MAPPINGS = {"PVL_StringToNumber": PVL_StringToNumber}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_StringToNumber": "PVL — String To Number"}
