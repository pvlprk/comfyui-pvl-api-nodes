# PVL_Switch.py

class AlwaysEqualProxy(str):
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False


def MakeSmartType(t):
    if isinstance(t, str):
        return SmartType(t)
    return t


class SmartType(str):
    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        selfset = set(self.split(','))
        otherset = set(other.split(','))
        return not selfset.issubset(otherset)


def VariantSupport():
    def decorator(cls):
        if hasattr(cls, "INPUT_TYPES"):
            old_input_types = getattr(cls, "INPUT_TYPES")
            def new_input_types(*args, **kwargs):
                types = old_input_types(*args, **kwargs)
                for category in ["required", "optional"]:
                    if category not in types:
                        continue
                    for key, value in types[category].items():
                        if isinstance(value, tuple):
                            types[category][key] = (MakeSmartType(value[0]),) + value[1:]
                return types
            setattr(cls, "INPUT_TYPES", new_input_types)

        if hasattr(cls, "RETURN_TYPES"):
            setattr(cls, "RETURN_TYPES", tuple(MakeSmartType(x) for x in cls.RETURN_TYPES))

        if hasattr(cls, "VALIDATE_INPUTS"):
            raise NotImplementedError("VariantSupport does not support VALIDATE_INPUTS yet")
        else:
            def validate_inputs(input_types):
                inputs = cls.INPUT_TYPES()
                for key, value in input_types.items():
                    if isinstance(value, SmartType):
                        continue
                    if "required" in inputs and key in inputs["required"]:
                        expected_type = inputs["required"][key][0]
                    elif "optional" in inputs and key in inputs["optional"]:
                        expected_type = inputs["optional"][key][0]
                    else:
                        expected_type = None
                    if expected_type is not None and MakeSmartType(value) != expected_type:
                        return f"Invalid type of {key}: {value} (expected {expected_type})"
                return True
            setattr(cls, "VALIDATE_INPUTS", validate_inputs)
        return cls
    return decorator


@VariantSupport()
class PVL_Switch:
    """
    PVL_Switch — conditional lazy switch.
    - If switch = True: evaluates and returns on_true; on_false is not evaluated.
    - If switch = False: evaluates and returns on_false; on_true is not evaluated.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": False}),
                # Place "if True" on top
                "on_true": ("*", {"lazy": True}),
                "on_false": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "switch"
    CATEGORY = "🟣PVL/utility"
    DESCRIPTION = "Choose between two inputs based on a boolean switch, evaluating only the selected branch."

    def check_lazy_status(self, switch, on_true=None, on_false=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]
        return []

    def switch(self, switch, on_true=None, on_false=None):
        value = on_true if switch else on_false
        return (value,)


# Optional: make sure ComfyUI can discover the node with nice display name.
NODE_CLASS_MAPPINGS = {
    "PVL_Switch": PVL_Switch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_Switch": "PVL Switch",
}
