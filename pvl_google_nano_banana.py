
# pvl_google_nano_banana.py
# Node: PVL Google Nano-Banana API
# Author: PVL
# License: MIT

import os, io, json, base64, typing as T, requests, numpy as np, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch

NODE_NAME = "PVL Google Nano-Banana API"
NODE_CATEGORY = "PVL/Google"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash-image-preview"

# Hidden defaults for decoding knobs
_TOP_P = 0.95
_TOP_K = 64
_MAX_TOKENS = 4096

# --- image helpers ---
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1,H,W,C]

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0,1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")

def encode_pil_b64(img: Image.Image, mime: str) -> str:
    buf = io.BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _extract_inline_blob(part: dict) -> T.Optional[bytes]:
    """Return raw image bytes from a response part.
    Accepts both REST (inline_data) and SDK (inlineData) casings and handles bytes/base64."""
    if not isinstance(part, dict):
        return None
    blob = None
    if "inline_data" in part and isinstance(part["inline_data"], dict):
        blob = part["inline_data"].get("data")
    elif "inlineData" in part and isinstance(part["inlineData"], dict):
        blob = part["inlineData"].get("data")
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if isinstance(blob, str):
        try:
            return base64.b64decode(blob, validate=False)
        except Exception:
            return None
    return None

class PVL_Google_NanoBanana_API:
    """
    ComfyUI node for Gemini 2.5 Flash Image Preview ("nano-banana").
    - Prompt + optional input IMAGE batch (as references)
    - Temperature exposed; top_p/top_k/max_tokens fixed internally
    - Optional text capture (also prints to console when enabled)
    - Safety level 0..3 (OFF→HIGH)
    - API key from input or GEMINI_API_KEY
    - num_images → sends multiple requests in parallel and returns a batch
    - Always includes *all* images from each response (no UI switch)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A tiny banana spaceship over a neon city."}),
            },
            "optional": {
                "images": ("IMAGE",),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                # Only affects encoding of input images (output mime is decided by API)
                "output_format": (["png","jpeg"], {"default": "png"}),
                # Print & return any text parts the API includes
                "capture_text_output": ("BOOLEAN", {"default": False}),
                # Safety level: 0=OFF, 1=LOW, 2=MEDIUM, 3=HIGH
                "safety_level": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                # Number of parallel images to generate (sends N parallel requests)
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                # Networking
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "request_id": ("STRING", {"default": ""}),
                # Debug
                "debug_log": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("images","text")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    # ---- helpers ----
    def _gen_config(self, temperature: float, want_text: bool) -> dict:
        # Do NOT set response_mime_type; API only allows text-y types there.
        return {
            "temperature": float(temperature),
            "top_p": float(_TOP_P),
            "top_k": int(_TOP_K),
            "max_output_tokens": int(_MAX_TOKENS),
            "response_modalities": ["IMAGE","TEXT"] if want_text else ["IMAGE"],
        }

    def _encode_images(self, images: T.Optional[torch.Tensor], mime: str) -> list:
        parts = []
        if images is None or not torch.is_tensor(images):
            return parts
        batch = images if images.ndim == 4 else images.unsqueeze(0)
        for i in range(batch.shape[0]):
            pil = tensor_to_pil(batch[i:i+1])
            parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_b64(pil, mime)}})
        return parts

    def _safety(self, level: int) -> T.Optional[list]:
        if level <= 0: return None
        thr = "BLOCK_ONLY_HIGH" if level == 1 else ("BLOCK_MEDIUM_AND_ABOVE" if level == 2 else "BLOCK_LOW_AND_ABOVE")
        cats = ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUAL","HARM_CATEGORY_DANGEROUS_CONTENT"]
        return [{"category": c, "threshold": thr} for c in cats]

    def _coerce_int(self, v, default=0) -> int:
        try:
            if v is None: return default
            if isinstance(v, (int,float)): return int(v)
            s = str(v).strip()
            return int(s) if s != "" else default
        except Exception:
            return default

    def _single_call(self, req: dict, url: str, headers: dict, timeout: int) -> dict:
        r = requests.post(url, headers=headers, data=json.dumps(req), timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text}")
        return r.json()

    # ---- main ----
    def run(self, prompt: str, images: T.Optional[torch.Tensor] = None,
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "",
            temperature: float = 0.6, output_format: str = "png",
            capture_text_output: bool = False,
            safety_level: T.Any = 0, num_images: int = 1,
            timeout_sec: int = 120, request_id: str = "",
            debug_log: bool = False):

        key = (api_key or os.getenv("GEMINI_API_KEY","")).strip()
        if not key:
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")

        input_mime = "image/png" if str(output_format).lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)

        base_req = {
            "contents": [{
                "role": "user",
                "parts": ([{"text": prompt}] if prompt.strip() else []) + self._encode_images(images, input_mime)
            }],
            "generation_config": self._gen_config(temperature, want_text)
        }

        level = self._coerce_int(safety_level, 0)
        s = self._safety(level)
        if s: base_req["safety_settings"] = s

        url = endpoint_override.strip() or DEFAULT_ENDPOINT.format(model=model)
        base_headers = {"Content-Type":"application/json","x-goog-api-key":key}

        # --- parallel requests ---
        N = max(1, int(self._coerce_int(num_images, 1)))
        timeout = int(self._coerce_int(timeout_sec, 120))
        results: list[dict] = [None] * N

        def build_headers(i: int) -> dict:
            h = dict(base_headers)
            rid = (request_id.strip() + f"-{i}") if request_id.strip() else f"pvl-nb-{int(time.time()*1000)}-{i}"
            h["x-goog-request-params"] = f"requestId={rid}"
            return h

        max_workers = min(N, 6)  # be nice to the API
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i in range(N):
                req_i = dict(base_req)  # shallow copy is fine
                hdrs_i = build_headers(i)
                time.sleep(random.uniform(0.01, 0.05))  # tiny jitter reduces burstiness
                fut = ex.submit(self._single_call, req_i, url, hdrs_i, timeout)
                futures[fut] = i
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = {"__error__": str(e)}

        # --- parse all results ---
        out_imgs, out_text = [], []
        for data in results:
            imgs_i, text_i = [], []
            if isinstance(data, dict) and "candidates" in data:
                cands = data.get("candidates") or []
                for cand in cands:
                    parts = (cand.get("content") or {}).get("parts") or []
                    for p in parts:
                        blob = _extract_inline_blob(p)
                        if blob:
                            try:
                                img = Image.open(io.BytesIO(blob)).convert("RGB")
                                imgs_i.append(pil_to_tensor(img))
                            except Exception:
                                pass
                        elif "text" in p and p["text"] is not None:
                            text_i.append(str(p["text"]))
            # Always include ALL images from each response (default ON behavior)
            if imgs_i:
                out_imgs.extend(imgs_i)
            else:
                # no image for this call -> keep placeholder so batch size matches
                out_imgs.append(pil_to_tensor(Image.new("RGB",(1,1),(0,0,0))))
            if text_i:
                out_text.append("\n".join(text_i))

        # --- finalize outputs ---
        images_tensor = torch.cat(out_imgs, dim=0) if len(out_imgs) > 1 else out_imgs[0]
        final_text = ("\n\n---\n\n").join(out_text) if (capture_text_output and out_text) else ""
        if capture_text_output and final_text:
            print(f"[PVL Google NanoBanana Output]:\n{final_text}\n")

        # Optional debug when many requests but no images came back
        if debug_log and all((t.shape[1] == 1 and t.shape[2] == 1) for t in out_imgs):
            print("[PVL Gemini Debug] All calls returned no image data. Check safety, model, or prompt.")

        return (images_tensor, final_text,)

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_API": PVL_Google_NanoBanana_API}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_API": NODE_NAME}
