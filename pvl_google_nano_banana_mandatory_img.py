
# pvl_google_nano_banana.py
# Node: PVL Google Nano-Banana API (SDK + robust FAL fallback)
# Author: PVL
# License: MIT
#
# Requires:
#   pip install google-genai
#
# Changes in this build:
# - FAL fallback now sends ONLY "image_urls" (list) to avoid duplicate outputs from some routes.
# - Enforces num_images: if num_images==1 returns the FIRST image only; otherwise caps to num_images.
# - Keeps sync_mode true and robust response parsing; prints raw FAL JSON when debug is ON.
# - Return order: ("STRING","IMAGE") => (text, images).

import os, io, json, base64, typing as T, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
import torch

NODE_NAME = "PVL Google Nano-Banana API mandatory IMG"
NODE_CATEGORY = "PVL/Google"
DEFAULT_MODEL = "gemini-2.5-flash-image-preview"

_TOP_P = 0.95
_TOP_K = 64
_MAX_TOKENS = 4096

# --- image helpers ---
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0,1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")

def encode_pil_bytes(img: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()

def _extract_image_bytes_from_part(part) -> T.Optional[bytes]:
    try:
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=False)
                except Exception:
                    return None
    except Exception:
        pass
    if isinstance(part, dict):
        if "inline_data" in part and isinstance(part["inline_data"], dict):
            blob = part["inline_data"].get("data")
        elif "inlineData" in part and isinstance(part["inlineData"], dict):
            blob = part["inlineData"].get("data")
        else:
            blob = None
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=False)
            except Exception:
                return None
    return None

def _extract_text_from_part(part) -> T.Optional[str]:
    try:
        txt = getattr(part, "text", None)
        if txt is not None:
            return str(txt)
    except Exception:
        pass
    if isinstance(part, dict) and "text" in part and part["text"] is not None:
        return str(part["text"])
    return None

def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")

class PVL_Google_NanoBanana_API_mandatory_IMG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A tiny banana spaceship over a neon city."}),
                "images": ("IMAGE",),
            },
            "optional": {

                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "endpoint_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "placeholder": "Leave empty to use GEMINI_API_KEY"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "output_format": (["png","jpeg"], {"default": "png"}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "request_id": ("STRING", {"default": ""}),
                "debug_log": ("BOOLEAN", {"default": False}),

                # --- FAL fallback ---
                "use_fal_fallback": ("BOOLEAN", {"default": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False,
                                           "placeholder": "Leave empty to use FAL_KEY"}),
                "fal_route": ("STRING", {"default": "fal-ai/nano-banana/edit"}),
            }
        }

    RETURN_TYPES = ("STRING","IMAGE",)
    RETURN_NAMES = ("text","images")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    # ---- helpers ----
    def _make_client(self, api_key: str, endpoint_override: str):
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError("Google GenAI SDK not installed. Run: pip install google-genai") from e

        http_options = None
        if endpoint_override.strip():
            try:
                http_options = types.HttpOptions(base_url=endpoint_override.strip())
            except Exception:
                http_options = None

        if http_options is not None:
            client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            client = genai.Client(api_key=api_key)
        return client

    def _build_parts(self, prompt: str, images: T.Optional[torch.Tensor], mime: str):
        parts: T.List[dict] = []
        if prompt and prompt.strip():
            parts.append({"text": prompt})
        if images is not None and torch.is_tensor(images):
            batch = images if images.ndim == 4 else images.unsqueeze(0)
            for i in range(batch.shape[0]):
                pil = tensor_to_pil(batch[i:i+1])
                parts.append({"inline_data": {"mime_type": mime, "data": encode_pil_bytes(pil, mime)}})
        return parts

    def _build_config(self, temperature: float, want_text: bool):
        try:
            from google.genai import types
            cfg = types.GenerateContentConfig(
                temperature=float(temperature),
                top_p=float(_TOP_P),
                top_k=int(_TOP_K),
                max_output_tokens=int(_MAX_TOKENS),
                response_modalities=["IMAGE","TEXT"] if want_text else ["IMAGE"],
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            return cfg
        except Exception:
            return {
                "temperature": float(temperature),
                "top_p": float(_TOP_P),
                "top_k": int(_TOP_K),
                "max_output_tokens": int(_MAX_TOKENS),
                "response_modalities": ["IMAGE","TEXT"] if want_text else ["IMAGE"],
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

    def _single_google_call(self, client, model: str, parts: list, cfg, request_id: str, debug: bool):
        kwargs = {}
        try:
            from google.genai import types
            if request_id.strip():
                kwargs["request_options"] = types.RequestOptions(request_id=request_id.strip())
        except Exception:
            pass

        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config=cfg,
            **kwargs
        )

        imgs, texts = [], []
        cands = getattr(resp, "candidates", None) or []
        for cand in cands:
            content = getattr(cand, "content", None)
            finish_reason = getattr(cand, "finish_reason", None)
            if debug:
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        print("[PVL Debug] usage_metadata:", getattr(um, "__dict__", str(um)))
                    print(f"[PVL Debug] finish_reason: {finish_reason}")
                except Exception:
                    pass
            if content is None:
                continue
            parts = getattr(content, "parts", []) or []
            for p in parts:
                blob = _extract_image_bytes_from_part(p)
                if blob:
                    try:
                        pil = Image.open(io.BytesIO(blob)).convert("RGB")
                        imgs.append(pil_to_tensor(pil))
                    except Exception as ex:
                        if debug:
                            print("[PVL Debug] image decode error:", ex)
                else:
                    t = _extract_text_from_part(p)
                    if t:
                        texts.append(t)

        return imgs, texts, resp

    # ---- FAL fallback via Queue API ----
    def _fal_queue_call(self, route: str, prompt: str, image_tensor: T.Optional[torch.Tensor],
                        mime: str, fal_key: str, timeout: int, debug: bool, num_images: int, output_format: str):
        if not fal_key:
            raise RuntimeError("FAL fallback requested but FAL_KEY is missing. Provide fal_api_key or set env FAL_KEY.")
        if image_tensor is None or not torch.is_tensor(image_tensor):
            raise RuntimeError("FAL fallback requires an input image tensor.")

        batch = image_tensor if image_tensor.ndim == 4 else image_tensor.unsqueeze(0)
        # Prepare data URIs (all frames)
        data_urls: T.List[str] = []
        for i in range(batch.shape[0]):
            pil = tensor_to_pil(batch[i:i+1])
            raw = encode_pil_bytes(pil, mime)
            data_urls.append(_data_url(mime, raw))

        base = "https://queue.fal.run"
        submit_url = f"{base}/{route.strip()}"
        headers = {"Authorization": f"Key {fal_key}"}
        payload = {
            "prompt": prompt or "",
            # Only plural form to avoid duplicate outputs
            "image_urls": data_urls,
            "num_images": int(max(1, num_images)),
            "output_format": ("png" if str(output_format).lower()=="png" else "jpeg"),
            "sync_mode": True,
        }
        if debug:
            print(f"[PVL FAL] QUEUE SUBMIT {submit_url} with {len(data_urls)} image(s) and num_images={payload['num_images']}")

        r = requests.post(submit_url, headers=headers, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"FAL queue submit error {r.status_code}: {r.text}")
        try:
            sub = r.json()
        except Exception:
            raise RuntimeError("FAL queue submit returned non-JSON.")

        req_id = sub.get("request_id")
        status_url = sub.get("status_url") or (f"{base}/{route.strip()}/requests/{req_id}/status" if req_id else None)
        resp_url = sub.get("response_url") or (f"{base}/{route.strip()}/requests/{req_id}" if req_id else None)
        if not req_id or not status_url or not resp_url:
            raise RuntimeError("FAL queue submit missing request_id/status_url/response_url.")

        # Poll
        deadline = time.time() + max(5, int(timeout))
        last_status = None
        while time.time() < deadline:
            sr = requests.get(status_url, headers=headers, timeout=10)
            if not sr.ok:
                time.sleep(0.5); continue
            sdata = sr.json()
            last_status = sdata.get("status")
            if last_status == "COMPLETED":
                break
            time.sleep(0.6)
        if last_status != "COMPLETED":
            raise RuntimeError(f"FAL queue did not complete in time (last status={last_status})")

        # Fetch result
        rr = requests.get(resp_url, headers=headers, timeout=15)
        if rr.status_code >= 400:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")
        try:
            rdata = rr.json()
        except Exception:
            raise RuntimeError("FAL result returned non-JSON.")

        if debug:
            try:
                s = json.dumps(rdata)[:1200]
            except Exception:
                s = str(rdata)[:1200]
            print("[PVL FAL] raw response:", s)

        # Normalize containers
        resp = rdata.get("response") if isinstance(rdata, dict) else None
        if not isinstance(resp, dict):
            resp = rdata if isinstance(rdata, dict) else {}

        description = resp.get("description") or rdata.get("description") or resp.get("output_text") or ""

        # Collect potential image items
        buckets: T.List[T.Union[str, dict]] = []
        for key in ("images", "outputs", "artifacts"):
            val = resp.get(key)
            if isinstance(val, list):
                buckets.extend(val)
        for key in ("image", "output", "result"):
            val = resp.get(key)
            if isinstance(val, (str, dict)):
                buckets.append(val)
        for key in ("images", "image", "output", "outputs", "artifacts"):
            val = rdata.get(key) if isinstance(rdata, dict) else None
            if isinstance(val, list):
                buckets.extend(val)
            elif isinstance(val, (str, dict)):
                buckets.append(val)

        # Decode images, but respect num_images cap
        out = []
        def add_image_from_item(item):
            try:
                if isinstance(item, str):
                    url_or_data = item
                elif isinstance(item, dict):
                    url_or_data = item.get("url") or item.get("data") or item.get("image") or item.get("content")
                else:
                    return
                if not isinstance(url_or_data, str):
                    return
                if url_or_data.startswith("data:image/"):
                    b64 = url_or_data.split(",", 1)[1]
                    blob = base64.b64decode(b64)
                else:
                    ir = requests.get(url_or_data, timeout=timeout)
                    if not ir.ok:
                        return
                    blob = ir.content
                pil = Image.open(io.BytesIO(blob)).convert("RGB")
                out.append(pil_to_tensor(pil))
            except Exception as ex:
                if debug:
                    print("[PVL FAL] image decode failed:", ex)

        for item in buckets:
            if len(out) >= int(max(1, num_images)):
                break
            add_image_from_item(item)

        if not out:
            raise RuntimeError("FAL API returned no images.")

        if len(out) == 1:
            images_tensor = out[0]
        else:
            images_tensor = torch.cat(out, dim=0)

        if debug:
            print(f"[PVL FAL] returning {len(out)} image(s) (capped to num_images={int(max(1, num_images))})")

        return images_tensor, description

    # ---- main ----
    def run(self, prompt: str, images: T.Optional[torch.Tensor] = None,
            model: str = DEFAULT_MODEL, endpoint_override: str = "",
            api_key: str = "",
            temperature: float = 0.6, output_format: str = "png",
            capture_text_output: bool = False, num_images: int = 1,
            timeout_sec: int = 120, request_id: str = "",
            debug_log: bool = False,
            use_fal_fallback: bool = False, fal_api_key: str = "", fal_route: str = "fal-ai/nano-banana/edit"):

        key = (api_key or os.getenv("GEMINI_API_KEY","")).strip()
        input_mime = "image/png" if str(output_format).lower() == "png" else "image/jpeg"
        want_text = bool(capture_text_output)

        # If no Google key but fallback is enabled, attempt FAL directly
        if not key:
            if use_fal_fallback:
                fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
                try:
                    img_tensor, fal_text = self._fal_queue_call(fal_route, prompt, images, input_mime, fal_key, int(timeout_sec), debug_log, num_images, output_format)
                    text_out = (fal_text or "") if want_text else ""
                    if text_out:
                        print("[PVL FAL Text]:\n" + text_out)
                    return (text_out, img_tensor)
                except Exception as fe:
                    print(f"[PVL FAL Fallback] FAL call failed without Google key: {fe}")
                    raise RuntimeError(f"Gemini image generation failed and FAL fallback also failed: {fe}")
            raise RuntimeError("Gemini API key missing. Pass api_key or set GEMINI_API_KEY.")

        # Build client & request
        client = self._make_client(key, endpoint_override)
        parts = self._build_parts(prompt, images, input_mime)
        cfg = self._build_config(temperature, want_text)

        if debug_log:
            p_preview = (prompt or "")[:180].replace("\n"," ")
            img_count = (images.shape[0] if (isinstance(images, torch.Tensor) and images.ndim==4) else (1 if isinstance(images, torch.Tensor) else 0))
            print(f"[PVL Debug] prompt chars={len(prompt or '')} preview='{p_preview}...'")
            print(f"[PVL Debug] parts: text={1 if (prompt and prompt.strip()) else 0}, images={img_count}")
            safe_parts = []
            for pr in parts:
                if "text" in pr:
                    safe_parts.append({"text": pr["text"][:120]})
                elif "inline_data" in pr:
                    di = pr["inline_data"]
                    safe_parts.append({"inline_data": {"mime_type": di.get("mime_type","image/*"), "data": "<bytes>"}})
            try:
                temp = getattr(cfg,'temperature',None) if hasattr(cfg,'temperature') else cfg.get('temperature')
                top_p = getattr(cfg,'top_p',None) if hasattr(cfg,'top_p') else cfg.get('top_p')
                top_k = getattr(cfg,'top_k',None) if hasattr(cfg,'top_k') else cfg.get('top_k')
                mot = getattr(cfg,'max_output_tokens',None) if hasattr(cfg,'max_output_tokens') else cfg.get('max_output_tokens')
                mods = getattr(cfg,'response_modalities',None) if hasattr(cfg,'response_modalities') else cfg.get('response_modalities')
            except Exception:
                temp=top_p=top_k=mot=mods=None
            print("[PVL Debug] config:", {"temperature": temp, "top_p": top_p, "top_k": top_k, "max_output_tokens": mot, "modalities": mods})
            print("[PVL Debug] contents:", [{"role":"user","parts": safe_parts}])

        # Parallel Google calls
        N = max(1, int(num_images))
        results = [None] * N
        errors = []

        def call_i(i: int):
            rid = (request_id.strip() + f"-{i}") if request_id.strip() else f"pvl-nb-{int(time.time()*1000)}-{i}"
            return self._single_google_call(client, model, parts, cfg, rid, debug_log)

        max_workers = min(N, 6)
        if N == 1:
            try:
                results[0] = call_i(0)
            except Exception as e:
                errors.append(f"google call 0 failed: {e}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futmap = {ex.submit(call_i, i): i for i in range(N)}
                for fut in as_completed(futmap):
                    i = futmap[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        errors.append(f"google call {i} failed: {e}")

        # Parse all Google results
        out_imgs, out_texts = [], []
        for idx, tup in enumerate(results):
            if tup is None:
                errors.append(f"google call {idx} returned no response")
                continue
            imgs_i, texts_i, resp = tup
            if not imgs_i:
                try:
                    cands = getattr(resp, "candidates", None) or []
                    fin = getattr(cands[0], "finish_reason", None) if cands else None
                except Exception:
                    fin = None
                errors.append(f"google call {idx} returned no images (finish_reason={fin})")
            else:
                out_imgs.extend(imgs_i)
            if texts_i:
                out_texts.append("\n".join(texts_i))

        # If Google failed and fallback enabled -> try FAL
        if errors and use_fal_fallback:
            print("[PVL Fallback] Google call failed; attempting FAL.ai fallback...")
            for e in errors:
                print("[PVL Google Error]", e)
            try:
                fal_key = (fal_api_key or os.getenv("FAL_KEY","")).strip()
                img_tensor, fal_text = self._fal_queue_call(fal_route, prompt, images, input_mime, fal_key, int(timeout_sec), debug_log, num_images, output_format)
                final_text = ""  # Start with empty; then merge Google texts if requested
                if bool(capture_text_output):
                    pieces = []
                    if out_texts:
                        pieces.append("\n\n--- Google ---\n\n" + ("\n".join(out_texts)))
                    if fal_text:
                        pieces.append("\n\n--- FAL ---\n\n" + fal_text)
                    final_text = "".join(pieces)
                if final_text:
                    print("[PVL Fallback Note] Combined text:")
                    print(final_text)
                return (final_text, img_tensor)
            except Exception as fe:
                print("[PVL FAL Error]", fe)
                raise RuntimeError("Both Google and FAL failed. See console for details.")

        # If Google produced errors and fallback not used -> raise
        if errors:
            if out_texts:
                print("[PVL Google Text]:\n" + ("\n\n---\n\n".join(out_texts)))
            raise RuntimeError("Gemini image generation failed: " + " | ".join(errors[:5]))

        if not out_imgs:
            raise RuntimeError("Gemini image generation failed: no images across all Google calls.")

        images_tensor = torch.cat(out_imgs, dim=0) if len(out_imgs) > 1 else out_imgs[0]
        final_text = ("\n\n---\n\n".join(out_texts)) if (bool(capture_text_output) and out_texts) else ""
        if final_text:
            print(f"[PVL Google NanoBanana Output]:\n{final_text}\n")

        return (final_text, images_tensor,)

NODE_CLASS_MAPPINGS = {"PVL_Google_NanoBanana_API_mandatory_IMG": PVL_Google_NanoBanana_API_mandatory_IMG}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_Google_NanoBanana_API_mandatory_IMG": NODE_NAME}
