@VariantSupport()
class PVL_Switch_Huge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_condition": ("STRING", {"default": "", "multiline": False}),
                "case_1": ("STRING", {"default": "", "multiline": False}),
                "case_2": ("STRING", {"default": "", "multiline": False}),
                "case_3": ("STRING", {"default": "", "multiline": False}),
                "case_4": ("STRING", {"default": "", "multiline": False}),
                "case_5": ("STRING", {"default": "", "multiline": False}),
                "case_6": ("STRING", {"default": "", "multiline": False}),
                "case_7": ("STRING", {"default": "", "multiline": False}),
                "case_8": ("STRING", {"default": "", "multiline": False}),
                "case_9": ("STRING", {"default": "", "multiline": False}),
                "case_10": ("STRING", {"default": "", "multiline": False}),
                "input_default": ("*", {"lazy": True}),
            },
            "optional": {
                "input_1": ("*", {"lazy": True}),
                "input_2": ("*", {"lazy": True}),
                "input_3": ("*", {"lazy": True}),
                "input_4": ("*", {"lazy": True}),
                "input_5": ("*", {"lazy": True}),
                "input_6": ("*", {"lazy": True}),
                "input_7": ("*", {"lazy": True}),
                "input_8": ("*", {"lazy": True}),
                "input_9": ("*", {"lazy": True}),
                "input_10": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_case"
    CATEGORY = "PVL_tools"

    DESCRIPTION = """
    Extended FL_Switch_Big allows you to choose between up to 10 processing paths based on a switch condition.
    """

    def check_lazy_status(self, switch_condition, 
                          case_1, case_2, case_3, case_4, case_5, 
                          case_6, case_7, case_8, case_9, case_10, 
                          input_default=None,
                          input_1=None, input_2=None, input_3=None, 
                          input_4=None, input_5=None, input_6=None, 
                          input_7=None, input_8=None, input_9=None, 
                          input_10=None):
        lazy_inputs = []
        for idx in range(1, 11):
            if switch_condition == locals()[f"case_{idx}"]:
                if locals()[f"input_{idx}"] is None:
                    lazy_inputs.append(f"input_{idx}")
                break
        else:
            if input_default is None:
                lazy_inputs.append("input_default")
        return lazy_inputs

    def switch_case(self, switch_condition, 
                    case_1, case_2, case_3, case_4, case_5, 
                    case_6, case_7, case_8, case_9, case_10,
                    input_default,
                    input_1=None, input_2=None, input_3=None, 
                    input_4=None, input_5=None, input_6=None, 
                    input_7=None, input_8=None, input_9=None, 
                    input_10=None):
        output = input_default
        for idx in range(1, 11):
            if switch_condition == locals()[f"case_{idx}"] and locals()[f"input_{idx}"] is not None:
                output = locals()[f"input_{idx}"]
                break
        return (output,)
