# pvl_any_2_string.py

class PVL_Any2String:
    """Convert connected BOOLEAN/INT/FLOAT sockets to strings and join with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # socket-only (no widgets)
                "bool_input": ("BOOLEAN", {"forceInput": True}),
                "int_input": ("INT", {"forceInput": True}),
                "float_input": ("FLOAT", {"forceInput": True}),
            },
            "required": {
                # user-typed delimiter widget
                "delimiter": ("STRING", {"default": ", "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_output",)
    FUNCTION = "do_str"
    CATEGORY = "PVL/Converters"

    def do_str(self, delimiter, bool_input=None, int_input=None, float_input=None):
        parts = []
        if bool_input is not None:
            parts.append("True" if bool_input else "False")
        if int_input is not None:
            parts.append(str(int_input))
        if float_input is not None:
            parts.append(f"{float_input:.4f}")
        return (delimiter.join(parts),)


NODE_CLASS_MAPPINGS = {"PVL_Any2String": PVL_Any2String}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Any2String": "PVL - Any To String"}
