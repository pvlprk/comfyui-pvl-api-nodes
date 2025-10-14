# pvl_simple_math_condition.py
# PVL — Simple Math Condition (with AnyType for wildcard inputs)

import ast
import operator

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        # ComfyUI uses != to check type compatibility; always "compatible"
        return False

any = AnyType("*")  # wildcard type like in many custom nodes


def _to_bool(x):
    try:
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            return len(x) > 0
        if isinstance(x, (int, float)):
            return x != 0
        if isinstance(x, (str, bytes)):
            return bool(x)
        return bool(x)
    except Exception:
        return False


class _SafeEvaluator(ast.NodeVisitor):
    """Safe evaluator supporting numbers, a,b,c, and +,-,*,/,//,%,**, unary +/-, parentheses."""

    _bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, variables):
        self.vars = variables

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self._bin_ops.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported operator")
        return op(left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op = self._unary_ops.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported unary operator")
        return op(operand)

    def visit_Name(self, node):
        if node.id in self.vars:
            return float(self.vars[node.id])
        raise ValueError(f"Unknown variable '{node.id}'")

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants allowed")

    def visit_Num(self, node):  # py3.8 compat
        return float(node.n)

    def generic_visit(self, node):
        raise ValueError("Unsupported expression element")


def _safe_eval(expr: str, a=0.0, b=0.0, c=0.0) -> float:
    expr = (expr or "").strip()
    if not expr:
        return 0.0
    try:
        tree = ast.parse(expr, mode="eval")
        evaluator = _SafeEvaluator({"a": a, "b": b, "c": c})
        val = evaluator.visit(tree)
        return float(val)
    except Exception:
        return 0.0


class PVL_MathCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "a": (any, {"default": 0.0}),
                "b": (any, {"default": 0.0}),
                "c": (any, {"default": 0.0}),
            },
            "required": {
                "evaluate": (any, {"default": 0}),
                "on_true": ("STRING", {"multiline": False, "default": ""}),
                "on_false": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "essentials_mb/utilities"

    def execute(self, evaluate, on_true, on_false, a=0.0, b=0.0, c=0.0):
        expr = on_true if _to_bool(evaluate) else on_false
        result_float = _safe_eval(expr, a, b, c)
        try:
            result_int = int(result_float)
        except Exception:
            result_int = 0
        return (result_int, float(result_float))


NODE_CLASS_MAPPINGS = {"PVL_MathCondition": PVL_MathCondition}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_MathCondition": "PVL — Math Condition"}
