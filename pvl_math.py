# pvl_simple_math.py
# PVL — Simple Math (clone of essentials_mb/utilities SimpleMath)

import ast
import operator as op
import math

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")


class PVL_Math:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "a": (any, {"default": 0.0}),
                "b": (any, {"default": 0.0}),
                "c": (any, {"default": 0.0}),
            },
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "essentials_mb/utilities"

    def execute(self, value, a=0.0, b=0.0, c=0.0, d=0.0):
        if hasattr(a, 'shape'):
            try: a = list(a.shape)
            except Exception: pass
        if hasattr(b, 'shape'):
            try: b = list(b.shape)
            except Exception: pass
        if hasattr(c, 'shape'):
            try: c = list(c.shape)
            except Exception: pass
        if hasattr(d, 'shape'):
            try: d = list(d.shape)
            except Exception: pass

        for name, val in (("a", a), ("b", b), ("c", c), ("d", d)):
            if isinstance(val, str):
                try:
                    parsed = float(val)
                    if name == "a": a = parsed
                    elif name == "b": b = parsed
                    elif name == "c": c = parsed
                    else: d = parsed
                except Exception:
                    pass

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.Mod: op.mod,
            ast.Eq: op.eq,
            ast.NotEq: op.ne,
            ast.Lt: op.lt,
            ast.LtE: op.le,
            ast.Gt: op.gt,
            ast.GtE: op.ge,
            ast.And: lambda x, y: x and y,
            ast.Or: lambda x, y: x or y,
            ast.Not: op.not_,
        }

        op_functions = {
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'len': len,
        }

        def eval_(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id == "a": return a
                if node.id == "b": return b
                if node.id == "c": return c
                if node.id == "d": return d
                return 0
            if isinstance(node, ast.BinOp):
                left = eval_(node.left)
                right = eval_(node.right)
                fn = operators.get(type(node.op))
                return fn(left, right) if fn else 0
            if isinstance(node, ast.UnaryOp):
                operand = eval_(node.operand)
                fn = operators.get(type(node.op))
                return fn(operand) if fn else 0
            if isinstance(node, ast.Compare):
                left = eval_(node.left)
                for oper, comparator in zip(node.ops, node.comparators):
                    fn = operators.get(type(oper))
                    if not fn or not fn(left, eval_(comparator)):
                        return 0
                    left = eval_(comparator)
                return 1
            if isinstance(node, ast.BoolOp):
                vals = [eval_(v) for v in node.values]
                fn = operators.get(type(node.op))
                if not fn: return 0
                out = vals[0]
                for nxt in vals[1:]:
                    out = fn(out, nxt)
                return out
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in op_functions:
                    args = [eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)
                return 0
            if isinstance(node, ast.Subscript):
                value = eval_(node.value)
                # py3.9+: slice is an expr
                try:
                    idx = eval_(node.slice) if not isinstance(node.slice, ast.Index) else eval_(node.slice.value)  # type: ignore
                except Exception:
                    return 0
                try:
                    return value[idx]
                except Exception:
                    return 0
            return 0

        try:
            tree = ast.parse(value, mode='eval')
            result = eval_(tree.body)
        except Exception:
            result = 0.0

        try:
            result_f = float(result)
        except Exception:
            result_f = 0.0

        if math.isnan(result_f) or math.isinf(result_f):
            result_f = 0.0

        return (round(result_f), result_f,)


NODE_CLASS_MAPPINGS = {"PVL_Math": PVL_Math}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Math": "PVL Math"}
