# pvl_gemini_segmentation.py
import base64
import io
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter
import os  # <--- Added for env var fallback

# -----------------------------
# ComfyUI Node: PVL_GeminiSegmentation
# -----------------------------

API_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PNG_SIG = b"\x89PNG\r\n\x1a\n"

# Utility: Console tag
TAG = "[PVL_GEMINI_SEG]"

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class CallConfig:
    api_key: str
    model: str
    temperature: float
    timeout: int
    seed: int
    combine_masks: bool
    mask_threshold: float
    mask_blur: int
    mask_expand: int  # negative=erode, positive=dilate
    tries: int
    debug: bool
    show_full_json_debug: bool
    debug_max_preview_chars: int


@dataclass
class CallResult:
    idx: int
    ok: bool
    retryable: bool
    mask_tensor: Optional[torch.Tensor]
    labels_json: str
    usage_in: int
    usage_out: int
    elapsed: float
    err_msg: Optional[str]


# -----------------------------
# Core helpers (image, masks)
# -----------------------------
def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    ComfyUI IMAGE is [B,H,W,C] float 0..1. We take first element in batch.
    """
    i = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(i)


def pil_gray_to_mask_tensor(pil_img: Image.Image, threshold: float) -> torch.Tensor:
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    arr = np.array(pil_img, dtype=np.float32)
    bin_arr = (arr > (threshold * 255.0)).astype(np.float32)
    return torch.from_numpy(bin_arr)


def make_blank_mask_tensor(w: int, h: int, tiny: bool = False) -> torch.Tensor:
    """
    If tiny=True -> a 3x3 zero mask. Else -> WxH zero mask.
    """
    if tiny:
        return torch.zeros((3, 3), dtype=torch.float32)
    return torch.zeros((h, w), dtype=torch.float32)


def ensure_png_bytes(b: bytes) -> bool:
    return len(b) >= 8 and b[:8] == PNG_SIG


def safe_b64_to_bytes(b64_str: str) -> Optional[bytes]:
    """
    Robust base64: strip data URLs, whitespace, fix padding.
    """
    s = b64_str.strip()
    # Remove data URL header if present
    if s.startswith("data:"):
        try:
            s = s.split(",", 1)[1]
        except Exception:
            return None
    # Remove whitespace and newlines
    s = "".join(s.split())

    # Some models insert non-base64 chars; try to keep only valid base64 alphabet + '='
    # But do not be too aggressive; first try as-is.
    def try_decode(x: str) -> Optional[bytes]:
        try:
            return base64.b64decode(x, validate=False)
        except Exception:
            # try to pad
            pad = (-len(x)) % 4
            try:
                return base64.b64decode(x + ("=" * pad), validate=False)
            except Exception:
                return None

    raw = try_decode(s)
    return raw


def paste_mask_into_fullsize(
    mask_pil: Image.Image,
    full_size: Tuple[int, int],
    box_2d: List[int],
) -> Image.Image:
    """
    box_2d is [y0, x0, y1, x1] in pixel coords (our prompt enforces pixels).
    """
    y0, x0, y1, x1 = box_2d
    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = min(full_size[1], int(y1))
    x1 = min(full_size[0], int(x1))
    if y1 <= y0 or x1 <= x0:
        # invalid box -> return blank canvas
        return Image.new("L", full_size, 0)

    # Resize mask to bbox
    box_w = x1 - x0
    box_h = y1 - y0
    m = mask_pil.convert("L").resize((box_w, box_h), Image.Resampling.BILINEAR)

    canvas = Image.new("L", full_size, 0)
    canvas.paste(m, (x0, y0))
    return canvas


def morph_and_blur(binary_mask: Image.Image, expand: int, blur_radius: int) -> Image.Image:
    """
    expand > 0 => dilate, expand < 0 => erode, 0 => no morph.
    blur_radius: Gaussian blur radius in pixels (applied after morph).
    """
    m = binary_mask
    if expand != 0:
        k = max(1, 2 * abs(int(expand)) + 1)  # odd kernel
        # Use MinFilter/MaxFilter to emulate erode/dilate on grayscale mask, then rethreshold
        if expand > 0:
            m = m.filter(ImageFilter.MaxFilter(k))
        else:
            m = m.filter(ImageFilter.MinFilter(k))

    if blur_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return m


# -----------------------------
# API call + response handling
# -----------------------------
def build_prompt(object_prompt: str) -> str:
    object_prompt = (object_prompt or "").strip()
    if object_prompt:
        obj_line = f' ONLY for the following object(s): {object_prompt}.'
    else:
        # If no prompt provided, allow general segmentation
        obj_line = " for all salient objects."

    return (
        "You are a vision segmentation model.\n"
        f"Analyze the given image and produce segmentation masks{obj_line}\n"
        "Return your response strictly as a valid JSON array — no markdown, no commentary.\n"
        'Each array entry must include:\n'
        '  "box_2d": [y0, x0, y1, x1] in pixel coordinates,\n'
        '  "mask": base64-encoded PNG image for that region,\n'
        '  "label": short descriptive label.\n'
    )


def make_api_body(prompt: str, pil_img: Image.Image, temperature: float, seed: int) -> Dict[str, Any]:
    buffered = io.BytesIO()
    # Send a moderately sized image; Gemini can accept large but we keep defaults
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": img_b64}},
    ]
    body: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": float(temperature)}}
    if seed and seed > 0:
        # Not all models honor seed, but it's fine to pass.
        body["generationConfig"]["seed"] = int(seed)
    return body


def parse_usage(resp_json: Dict[str, Any]) -> Tuple[int, int]:
    in_tokens = 0
    out_tokens = 0
    um = resp_json.get("usageMetadata") or {}
    in_tokens = int(um.get("promptTokenCount", 0))
    out_tokens = int(um.get("candidatesTokenCount", 0))
    return in_tokens, out_tokens


def extract_json_text(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    Extract the JSON array string from Gemini response. We expect a single text part
    that is already a JSON array. We still handle fence remnants or stray text.
    """
    candidates = resp_json.get("candidates", [])
    if not candidates:
        return None
    # Use first candidate
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str) and t.strip():
            s = t.strip()

            # Strip any leading/trailing code fences or extra text
            if s.startswith("```"):
                # remove first fence line
                s2 = s.split("```", 1)[-1]
                # if it still contains a closing fence
                if "```" in s2:
                    s2 = s2.split("```")[0]
                s = s2.strip()

            # Try to isolate top-level JSON array
            lb = s.find("[")
            rb = s.rfind("]")
            if lb != -1 and rb != -1 and rb > lb:
                return s[lb : rb + 1]
            # fallback: return raw if it starts with [
            if s.startswith("[") and s.endswith("]"):
                return s
            # last fallback: return as is
            return s
    return None


def try_decode_mask_item(
    item: Dict[str, Any], full_size: Tuple[int, int], debug: bool
) -> Optional[Image.Image]:
    """
    Decode a single mask entry into a pasted canvas.
    """
    try:
        box = item.get("box_2d")
        label = item.get("label", "")
        m_b64 = item.get("mask", "")

        if not isinstance(box, list) or len(box) != 4:
            if debug:
                print(f"{TAG} item: invalid box_2d={box!r}")
            return None

        raw = safe_b64_to_bytes(str(m_b64))
        if not raw:
            if debug:
                print(f"{TAG} item: base64 decode failed; len={len(str(m_b64))}")
            return None
        if not ensure_png_bytes(raw):
            if debug:
                print(f"{TAG} item: decoded bytes not PNG; header={raw[:8]!r}")
            return None

        try:
            m_pil = Image.open(io.BytesIO(raw)).convert("L")
        except Exception as e:
            if debug:
                print(f"{TAG} item: PIL open failed: {e}")
            return None

        pasted = paste_mask_into_fullsize(m_pil, full_size, box)
        return pasted
    except Exception as e:
        if debug:
            print(f"{TAG} item decode exception: {e}")
        return None


# -----------------------------
# The Node
# -----------------------------
class PVL_GeminiSegmentation:
    """
    Multi-image segmentation using Gemini 2.5 via REST.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Optional image/prompt pairs — ANY can be left unconnected with no error
        optional_inputs = {}
        for i in range(1, 6):
            optional_inputs[f"image{i}"] = ("IMAGE",)
            optional_inputs[f"prompt{i}"] = ("STRING", {"multiline": True, "default": ""})

        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": (
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    {"default": "gemini-2.5-flash"},
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "tries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 45, "min": 5, "max": 300}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "combine_masks": ("BOOLEAN", {"default": True}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64}),
                "mask_expand": ("INT", {"default": 0, "min": -64, "max": 64}),
                "debug": ("BOOLEAN", {"default": False}),
                "show_full_json_debug": ("BOOLEAN", {"default": False}),
                "debug_max_preview_chars": ("INT", {"default": 1200, "min": 200, "max": 20000}),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "mask1",
        "mask2",
        "mask3",
        "mask4",
        "mask5",
        "labels_json1",
        "labels_json2",
        "labels_json3",
        "labels_json4",
        "labels_json5",
    )
    FUNCTION = "run"
    CATEGORY = "image/segmentation"

    # -------------- internal --------------
    def _single_call(
        self,
        idx: int,
        image_tensor: torch.Tensor,
        prompt_text: str,
        cfg: CallConfig,
    ) -> CallResult:
        """
        Perform one API call with retries handled by caller.
        """
        t0 = time.time()
        usage_in = 0
        usage_out = 0
        err_msg: Optional[str] = None

        try:
            pil_img = tensor_to_pil(image_tensor)

            body = make_api_body(build_prompt(prompt_text), pil_img, cfg.temperature, cfg.seed)
            url = API_URL_TMPL.format(model=cfg.model)
            headers = {"Content-Type": "application/json"}

            if cfg.debug and not cfg.show_full_json_debug:
                print(f'{TAG} PROMPT for image {idx+1}: {body["contents"][0]["parts"][0]["text"]}')

            r = requests.post(
                url,
                params={"key": cfg.api_key},
                headers=headers,
                data=json.dumps(body),
                timeout=cfg.timeout,
            )
            elapsed = time.time() - t0

            if r.status_code != 200:
                err_msg = f"HTTP {r.status_code}"
                if cfg.debug:
                    print(f"{TAG} HTTP ERROR {r.status_code}: {r.text[:cfg.debug_max_preview_chars]}")
                return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, err_msg)

            resp_json = r.json()

            # Token usage (per call)
            in_toks, out_toks = parse_usage(resp_json)
            usage_in += in_toks
            usage_out += out_toks

            if cfg.debug:
                if cfg.show_full_json_debug:
                    raw = json.dumps(resp_json, ensure_ascii=False, indent=2)
                    print(f"{TAG} HTTP 200 | elapsed={elapsed:.2f}s | usage: input={in_toks} output={out_toks} total={in_toks+out_toks}")
                    print(f"{TAG} FULL RAW JSON:\n{raw}")
                else:
                    preview = json.dumps(resp_json, ensure_ascii=False)[: cfg.debug_max_preview_chars]
                    print(f"{TAG} HTTP 200 | elapsed={elapsed:.2f}s | usage: input={in_toks} output={out_toks} total={in_toks+out_toks}")
                    print(f"{TAG} PREVIEW: {preview}")

            json_text = extract_json_text(resp_json)
            if not json_text:
                if cfg.debug:
                    print(f"{TAG} Image {idx+1}: no JSON extracted, will retry.")
                return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, "no_json")

            # Parse the array
            try:
                items = json.loads(json_text)
            except Exception as e:
                if cfg.debug:
                    print(f"{TAG} Image {idx+1}: JSON parse failed: {e}")
                return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, "bad_json")

            if not isinstance(items, list):
                if cfg.debug:
                    print(f"{TAG} Image {idx+1}: JSON is not a list, will retry.")
                return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, "not_array")

            if len(items) == 0:
                # Treat zero masks as failure (retryable)
                if cfg.debug:
                    print(f"{TAG} Image {idx+1}: zero masks in JSON, will retry.")
                return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, "zero_masks")

            # Decode masks
            W, H = pil_img.size
            decoded: List[Image.Image] = []
            for it in items:
                m_canvas = try_decode_mask_item(it, (W, H), cfg.debug)
                if m_canvas is not None:
                    decoded.append(m_canvas)

            if len(decoded) == 0:
                if cfg.debug:
                    print(f"{TAG} Image {idx+1}: no valid decoded masks, will retry.")
                return CallResult(idx, False, True, None, json.dumps(items, ensure_ascii=False), usage_in, usage_out, elapsed, "decode_fail")

            # Combine or pick largest
            if cfg.combine_masks:
                # union
                union = Image.new("L", (W, H), 0)
                for d in decoded:
                    union = Image.fromarray(np.maximum(np.array(union, dtype=np.uint8), np.array(d, dtype=np.uint8)))
                final_mask = union
            else:
                # pick largest area (non-zero)
                areas = [(i, int(np.count_nonzero(np.array(d) > 0))) for i, d in enumerate(decoded)]
                areas.sort(key=lambda x: x[1], reverse=True)
                final_mask = decoded[areas[0][0]]

            # Morph & blur then threshold
            final_mask = morph_and_blur(final_mask, cfg.mask_expand, cfg.mask_blur)
            # Threshold to binary
            mask_tensor = pil_gray_to_mask_tensor(final_mask, cfg.mask_threshold)

            labels_json = json.dumps(
                [
                    {
                        "index": i,
                        "label": str(it.get("label", f"object_{i}")),
                        "box_2d": it.get("box_2d", []),
                    }
                    for i, it in enumerate(items)
                ],
                ensure_ascii=False,
                indent=2,
            )
            return CallResult(idx, True, False, mask_tensor, labels_json, usage_in, usage_out, elapsed, None)

        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            if cfg.debug:
                print(f"{TAG} HTTP TIMEOUT after {cfg.timeout}s")
            return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, "timeout")
        except Exception as e:
            elapsed = time.time() - t0
            if cfg.debug:
                print(f"{TAG} Exception: {e}")
            return CallResult(idx, False, True, None, "[]", usage_in, usage_out, elapsed, str(e))

    # -------------- public --------------
    def run(
        self,
        api_key: str,
        model: str,
        temperature: float,
        tries: int,
        timeout: int,
        seed: int,
        mask_threshold: float,
        combine_masks: bool,
        mask_blur: int,
        mask_expand: int,
        debug: bool,
        show_full_json_debug: bool,
        debug_max_preview_chars: int,
        **kwargs,
    ):
        # 🔹 Added ENV VAR FALLBACK for API KEY
        if not api_key.strip():
            api_key = (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_GENAI_API_KEY")
                or os.getenv("GOOGLE_GENERATIVE_LANGUAGE_API_KEY")
                or ""
            )
            if debug:
                if api_key:
                    print(f"{TAG} Using API key from environment variable.")
                else:
                    print(f"{TAG} Warning: empty API key and no env var found; calls will fail.")

        # Collect optional inputs
        images: List[Optional[torch.Tensor]] = [kwargs.get(f"image{i}") for i in range(1, 6)]
        prompts: List[str] = [(kwargs.get(f"prompt{i}") or "") for i in range(1, 6)]

        # Build jobs for connected images
        jobs: List[Tuple[int, torch.Tensor, str]] = []
        for idx, (img, prm) in enumerate(zip(images, prompts)):
            if img is None:
                continue
            jobs.append((idx, img, prm))

        if debug:
            print(f"{TAG} Attempt 1/{tries}, pending={[j[0] for j in jobs]}")

        cfg = CallConfig(
            api_key=api_key.strip(),
            model=model.strip(),
            temperature=float(temperature),
            timeout=int(timeout),
            seed=int(seed),
            combine_masks=bool(combine_masks),
            mask_threshold=float(mask_threshold),
            mask_blur=int(mask_blur),
            mask_expand=int(mask_expand),
            tries=int(tries),
            debug=bool(debug),
            show_full_json_debug=bool(show_full_json_debug),
            debug_max_preview_chars=int(debug_max_preview_chars),
        )

        # Storage for results
        out_masks: Dict[int, torch.Tensor] = {}
        out_labels: Dict[int, str] = {}

        total_in = 0
        total_out = 0

        start_time = time.time()
        pending = jobs[:]

        for attempt in range(1, cfg.tries + 1):
            if not pending:
                break

            # Execute in parallel
            futures = []
            with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as ex:
                for (idx, img, prm) in pending:
                    futures.append(ex.submit(self._single_call, idx, img, prm, cfg))

                new_pending: List[Tuple[int, torch.Tensor, str]] = []
                for fut in as_completed(futures):
                    res: CallResult = fut.result()
                    total_in += res.usage_in
                    total_out += res.usage_out

                    if res.ok and res.mask_tensor is not None:
                        out_masks[res.idx] = res.mask_tensor
                        out_labels[res.idx] = res.labels_json
                    else:
                        # if failed and retryable, keep for next attempt
                        # otherwise, will be handled after loop
                        # We treat "zero masks" and decode issues as retryable
                        if res.retryable:
                            # find the original tuple
                            for t in jobs:
                                if t[0] == res.idx:
                                    new_pending.append(t)
                                    break

                pending = new_pending

            if pending and attempt < cfg.tries and debug:
                print(f"{TAG} Attempt {attempt+1}/{cfg.tries}, pending={[p[0] for p in pending]}")

        # For any remaining pending or missing results, produce 3x3 blank
        # If an image input was not connected at all, we also output a 3x3 blank.
        for i in range(5):
            if images[i] is None:
                # Missing input -> ignore gracefully and output tiny black mask & empty labels
                out_masks.setdefault(i, make_blank_mask_tensor(3, 3, tiny=True))
                out_labels.setdefault(i, "[]")
                continue

            if i not in out_masks:
                # max retries reached -> tiny 3x3 blank
                if debug:
                    print(f"{TAG} Image {i+1}: max retries reached or no result, using 3×3 blank mask.")
                out_masks[i] = make_blank_mask_tensor(3, 3, tiny=True)
                out_labels[i] = "[]"

        elapsed_all = time.time() - start_time

        if debug:
            print(f"{TAG} TOTAL TOKENS — input={total_in} output={total_out} total={total_in + total_out}")
            succeeded = sum(1 for k in range(5) if k in out_masks and out_masks[k] is not None)
            print(f"{TAG} Completed in {elapsed_all:.2f}s | successful={succeeded}/{len([i for i in images if i is not None])} | seed={seed}")

        # Assemble outputs in fixed order (5 masks + 5 json strings)
        masks_out: List[torch.Tensor] = []
        labels_out: List[str] = []
        for i in range(5):
            m = out_masks.get(i)
            j = out_labels.get(i, "[]")
            if m is None:
                # Fallback safety: tiny mask
                m = make_blank_mask_tensor(3, 3, tiny=True)
            # Ensure mask tensor has shape [H, W] (ComfyUI MASK)
            masks_out.append(m)
            labels_out.append(j)

        return tuple(masks_out + labels_out)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "PVL_GeminiSegmentation": PVL_GeminiSegmentation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_GeminiSegmentation": "PVL Gemini 2.5 Segmentation",
}
