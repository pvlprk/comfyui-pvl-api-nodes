import torch
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
            old_return_types = cls.RETURN_TYPES
            setattr(cls, "RETURN_TYPES", tuple(MakeSmartType(x) for x in old_return_types))

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
class PVL_Switch_x10:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        required_cases = {
            "switch_condition": ("STRING", {"default": "", "multiline": False}),
        }

        # Create case_1 ... case_10
        for i in range(1, 11):
            required_cases[f"case_{i}"] = ("STRING", {"default": "", "multiline": False})

        required_cases["input_default"] = ("*", {"lazy": True})

        optional_inputs = {}

        # Create input_1 ... input_10
        for i in range(1, 11):
            optional_inputs[f"input_{i}"] = ("*", {"lazy": True})

        return {
            "required": required_cases,
            "optional": optional_inputs
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_case"
    CATEGORY = "PVL_tools"

    def check_lazy_status(self, switch_condition, **kwargs):
        """
        Automatic lazy input resolver.
        """

        lazy_inputs = []

        # case_1 ... case_10
        case_values = [kwargs.get(f"case_{i}") for i in range(1, 11)]
        input_default = kwargs.get("input_default")
        # input_1 ... input_10
        input_values = [kwargs.get(f"input_{i}") for i in range(1, 11)]

        # Case-specific lazy resolution
        for idx, case_value in enumerate(case_values, start=1):
            if switch_condition == case_value:
                if input_values[idx - 1] is None:
                    lazy_inputs.append(f"input_{idx}")
                return lazy_inputs

        # Default path
        if input_default is None:
            lazy_inputs.append("input_default")
        return lazy_inputs

    def switch_case(self, switch_condition, **kwargs):
        """
        Main switch logic.
        """

        case_values = [kwargs.get(f"case_{i}") for i in range(1, 11)]   # case_1 ... case_10
        input_default = kwargs.get("input_default")
        input_values = [kwargs.get(f"input_{i}") for i in range(1, 11)] # input_1 ... input_10

        output = input_default

        for idx, case_value in enumerate(case_values, start=1):
            if switch_condition == case_value and input_values[idx - 1] is not None:
                output = input_values[idx - 1]
                break

        return (output,)
