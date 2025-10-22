# PVL_Compare.py

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        # ComfyUI uses "!=" to check type compatibility — always return False
        return False

any_type = AnyType("*")

# Define supported comparison functions
COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a > b": lambda a, b: a > b,
    "a < b": lambda a, b: a < b,
    "a >= b": lambda a, b: a >= b,
    "a <= b": lambda a, b: a <= b,
}

class PVL_Compare:
    @classmethod
    def INPUT_TYPES(cls):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
                "comparison": (compare_functions, {"default": "a == b"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "PVL/Logic"

    def compare(self, a, b, comparison):
        try:
            result = COMPARE_FUNCTIONS[comparison](a, b)
        except Exception as e:
            print(f"[PVL_Compare] Comparison error: {e}")
            result = False
        return (bool(result),)


# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "PVL Compare": PVL_Compare
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL Compare": "PVL Compare"
}
