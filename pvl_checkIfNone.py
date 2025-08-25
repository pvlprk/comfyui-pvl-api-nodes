# pvl_checkIfNone.py
# Single-input checker with '*' passthrough and a single 'has_value' BOOLEAN.
# No external deps; includes a minimal SmartType/VariantSupport that avoids recursion.

import copy

# --- SmartType/VariantSupport (permissive '*' / unions, non-recursive) ---

def MakeSmartType(t):
    if isinstance(t, SmartType):
        return t
    if isinstance(t, str):
        return SmartType(t)
    return t

class SmartType(str):
    @staticmethod
    def _to_set(x):
        if isinstance(x, (SmartType, str)):
            return set(str(x).split(","))
        return {str(x)}

    @staticmethod
    def _compatible(a, b):
        if str(a) == "*" or str(b) == "*":
            return True
        aset = SmartType._to_set(a)
        bset = SmartType._to_set(b)
        return bool(aset & bset) or aset.issubset(bset) or bset.issubset(aset)

    def __eq__(self, other):
        return SmartType._compatible(self, other)

    def __ne__(self, other):
        return not SmartType._compatible(self, other)

def VariantSupport():
    def decorator(cls):
        if hasattr(cls, "INPUT_TYPES"):
            old_input_types = getattr(cls, "INPUT_TYPES")
            def new_input_types(*args, **kwargs):
                types = copy.deepcopy(old_input_types(*args, **kwargs))
                for section in ("required", "optional"):
                    if section in types:
                        for key, spec in list(types[section].items()):
                            if isinstance(spec, tuple) and spec:
                                first = MakeSmartType(spec[0])
                                rest = spec[1:] if len(spec) > 1 else ()
                                types[section][key] = (first,) + rest
                            elif isinstance(spec, str):
                                types[section][key] = (MakeSmartType(spec),)
                return types
            setattr(cls, "INPUT_TYPES", new_input_types)

        if hasattr(cls, "RETURN_TYPES"):
            setattr(cls, "RETURN_TYPES",
                    tuple(MakeSmartType(x) for x in getattr(cls, "RETURN_TYPES")))

        if not hasattr(cls, "VALIDATE_INPUTS"):
            @staticmethod
            def VALIDATE_INPUTS(input_types: dict):
                inputs = cls.INPUT_TYPES()
                for section in ("required", "optional"):
                    if section not in inputs:
                        continue
                    for key, spec in inputs[section].items():
                        if key not in input_types:
                            continue  # optional and not connected is fine
                        expected = spec[0] if isinstance(spec, tuple) else spec
                        expected = MakeSmartType(expected)
                        actual = MakeSmartType(input_types[key])
                        if actual != expected:
                            return f"Invalid type for '{key}': {actual} (expected {expected})"
                return True
            setattr(cls, "VALIDATE_INPUTS", VALIDATE_INPUTS)
        return cls
    return decorator

# --- The node ---

_SENTINEL = object()  # detect "not connected" vs "connected None"

@VariantSupport()
class IsConnected:
    """
    One universal '*' input (non-lazy).
    Outputs:
      - value ('*' passthrough)  → same type/value as input (binds wildcard)
      - has_value (BOOLEAN)      → True if connected AND not empty
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # IMPORTANT: no {"lazy": True}; always fetch upstream value
                "value": ("*",),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("value", "has_value")
    FUNCTION = "check"
    CATEGORY = "PVL_tools"

    # ---- Non-emptiness rules ----
    def _has_value(self, v):
        if v is None:
            return False
        # Booleans: presence counts, even if False
        if isinstance(v, bool):
            return True
        # Numbers: presence counts, even if 0
        if isinstance(v, (int, float)):
            return True
        # Strings: require non-whitespace content
        if isinstance(v, str):
            return len(v.strip()) > 0
        # Bytes-like
        if isinstance(v, (bytes, bytearray, memoryview)):
            return len(v) > 0
        # Sequences / mappings
        if isinstance(v, (list, tuple, dict, set)):
            # Conditioning often is list[tuple(...)] — non-empty counts
            if isinstance(v, list) and v and isinstance(v[0], tuple):
                return True
            return len(v) > 0
        # Latent dicts typically have 'samples' tensor
        if isinstance(v, dict) and "samples" in v:
            samples = v.get("samples", None)
            try:
                import torch
                if isinstance(samples, torch.Tensor):
                    return samples.numel() > 0
            except Exception:
                return samples is not None
            return samples is not None
        # Torch tensors: images/masks/others
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return v.numel() > 0
        except Exception:
            pass
        # Fallback: Python truthiness
        try:
            return bool(v)
        except Exception:
            return True  # be permissive if in doubt

    def check(self, value=_SENTINEL):
        v = None if value is _SENTINEL else value
        has_value = self._has_value(v) if value is not _SENTINEL else False
        return (v, has_value)