# pvl_google_nano_banana_pro_multi_api.py
# Node: PVL Google Nano-Banana PRO Multi API (Gemini 3 Pro Image + FAL)
# Author: PVL
# License: MIT
#
# Features:
# - Google Gemini 3 Pro Image (gemini-3-pro-image-preview) text+image generation
# - Multi-prompt via regex delimiter
# - Up to 8 reference images
# - Resolution dropdown: 1K / 2K / 4K  (Gemini 3 Pro Image where supported + Nano Banana Pro via FAL)
# - Optional google_search grounding tools
# - FAL fallback (nano-banana-pro / nano-banana-pro/edit)
# - Parallel calls & simple retry logic for transient errors
#
# Requires:
#   pip install google-genai
#   pip install requests
#
# Environment:
#   GEMINI_API_KEY  - Google Gemini API key (if api_key input left empty)
#   FAL_KEY         - FAL API key (if fal_api_key input left empty)

import os
import re
import io
import time
import base64
import typing as T

import requests
import numpy as np
import torch
from PIL import Image

NODE_NAME = "PVL Google Nano-Banana PRO Multi API"
NODE_CATEGORY = "PVL/API"

DEFAULT_MODEL = "gemini-3-pro-image-preview"


# --------------------------- IMAGE UTILS ------------------------------------


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI tensor [B,H,W,C] (float32 in [0,1]).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W,3]
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI tensor [B,H,W,C] or [H,W,C] to PIL Image.
    """
    if t.ndim == 4:
        t = t[0]  # [H,W,C]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


def encode_pil_bytes(img: Image.Image, mime: str = "image/png") -> bytes:
    buf = io.BytesIO()
    if mime.lower().endswith("jpeg") or mime.lower().endswith("jpg"):
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _data_url(mime: str, raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def stack_images_same_size(tensors: T.List[torch.Tensor], debug: bool = False) -> torch.Tensor:
    """
    Concatenate (B,H,W,C) batches along B. If shapes mismatch, resize to the first image size.
    """
    if not tensors:
        raise RuntimeError("No images to stack.")
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError:
        if debug:
            print("[PVL PRO NODE] Mismatched sizes, resizing to match first image.")
        target_h, target_w = tensors[0].shape[1], tensors[0].shape[2]
        fixed = []
        for t in tensors:
            pil = tensor_to_pil(t)
            rp = pil.resize((target_w, target_h), Image.LANCZOS)
            fixed.append(pil_to_tensor(rp))
        return torch.cat(fixed, dim=0)


def _extract_image_bytes_from_part(part) -> T.Optional[bytes]:
    """
    Extract inline image bytes from a Gemini part. Skips 'thought' parts.
    """
    # Skip thought parts
    try:
        if getattr(part, "thought", False):
            return None
    except Exception:
        pass
    if isinstance(part, dict) and part.get("thought"):
        return None

    # Preferred helper
    try:
        if hasattr(part, "as_image"):
            img = part.as_image()
            if isinstance(img, Image.Image):
                return encode_pil_bytes(img, "image/png")
    except Exception:
        pass

    # Inline data fields
    try:
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline is not None:
            d = getattr(inline, "data", None)
            if isinstance(d, (bytes, bytearray)):
                return bytes(d)
            if isinstance(d, str):
                try:
                    return base64.b64decode(d, validate=False)
                except Exception:
                    return None
    except Exception:
        pass

    # Dict-based fallback
    if isinstance(part, dict):
        blob = None
        if "inline_data" in part and isinstance(part["inline_data"], dict):
            blob = part["inline_data"].get("data")
        elif "inlineData" in part and isinstance(part["inlineData"], dict):
            blob = part["inlineData"].get("data")
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=False)
            except Exception:
                return None
    return None


def _extract_text_from_part(part) -> T.Optional[str]:
    """
    Extract text from a Gemini part, ignoring thought parts.
    """
    try:
        if getattr(part, "thought", False):
            return None
    except Exception:
        pass
    if isinstance(part, dict) and part.get("thought"):
        return None

    try:
        txt = getattr(part, "text", None)
        if txt is not None:
            s = str(txt)
            if s.strip():
                return s
    except Exception:
        pass

    if isinstance(part, dict) and part.get("text") is not None:
        s = str(part["text"])
        if s.strip():
            return s
    return None


# --------------------------- GOOGLE CLIENT CACHE ----------------------------


class GoogleGeminiClientCache:
    _client = None
    _endpoint = None

    @classmethod
    def get_client(cls, api_key: str, endpoint_override: str = ""):
        key = api_key.strip()
        if not key:
            raise RuntimeError("Google Gemini API key is empty.")
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai (google.genai) package not installed. "
                "Install with: pip install google-genai"
            ) from e

        if cls._client is not None and cls._endpoint == endpoint_override:
            return cls._client

        if endpoint_override:
            cls._client = genai.Client(api_key=key, base_url=endpoint_override)
            cls._endpoint = endpoint_override
        else:
            cls._client = genai.Client(api_key=key)
            cls._endpoint = ""
        return cls._client


# --------------------------- NODE IMPLEMENTATION ----------------------------


class PVL_Google_NanoBanana_PRO_Multi_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter prompts separated by regex delimiter",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                # Dropdown for 1K / 2K / 4K
                "resolution": (["1K", "2K", "4K"],),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "retries": ("INT", {"default": 3, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
                "capture_text_output": ("BOOLEAN", {"default": False}),
                "use_google_search": ("BOOLEAN", {"default": False}),
                "use_fal_fallback": ("BOOLEAN", {"default": True}),
                "force_fal": ("BOOLEAN", {"default": False}),
                "delimiter": (
                    "STRING",
                    {
                        "default": "[++]",
                        "multiline": False,
                        "placeholder": "Regex delimiter (e.g. \\n\\n, \\|\\|, etc.)",
                    },
                ),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "aspect_ratio": (
                    "STRING",
                    {
                        "default": "1:1",
                        "multiline": False,
                        "placeholder": "e.g. 1:1, 2:3, 3:2, 4:5, 16:9...",
                    },
                ),
                "endpoint_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional custom endpoint",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "If empty, GEMINI_API_KEY env is used",
                    },
                ),
                "fal_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "If empty, FAL_KEY env is used",
                    },
                ),
                "fal_route_img2img": (
                    "STRING",
                    {"default": "fal-ai/nano-banana-pro/edit", "multiline": False},
                ),
                "fal_route_txt2img": (
                    "STRING",
                    {"default": "fal-ai/nano-banana-pro", "multiline": False},
                ),
                "request_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional request ID for logging",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "images")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    # --------------------------- HELPERS -------------------------------------

    def _make_client(self, api_key: str, endpoint_override: str = ""):
        key = api_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("No Gemini API key. Provide api_key or set GEMINI_API_KEY.")
        return GoogleGeminiClientCache.get_client(key, endpoint_override)

    def _clean_aspect_ratio(self, ar: T.Optional[str]) -> T.Optional[str]:
        if ar is None:
            return None
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", str(ar))
        if not m:
            return None
        return f"{int(m.group(1))}:{int(m.group(2))}"

    def _clean_resolution(self, res: T.Optional[str]) -> T.Optional[str]:
        if not res:
            return None
        s = str(res).strip().upper()
        if s in {"1K", "2K", "4K"}:
            return s
        return None

    def _build_config(
        self,
        want_text: bool,
        aspect_ratio_opt: T.Optional[str],
        resolution_opt: T.Optional[str],
        use_google_search: bool,
        debug: bool,
    ):
        """
        Build GenerateContentConfig for Gemini 3 Pro Image.

        Strategy:
        - Try to build with image_size (1K/2K/4K).
        - If SDK rejects image_size (extra_forbidden), rebuild without it.
        This makes the node work on old and new google-genai versions.
        """
        try:
            from google.genai import types
        except Exception as e:
            # Fallback: dict config — library will still validate and may reject image_size,
            # but at this point we can't do better than surfacing the error.
            if debug:
                print(f"[PVL PRO NODE] google.genai.types import failed, using dict config: {e}")
            cfg: dict = {
                "response_modalities": ["TEXT", "IMAGE"] if want_text else ["IMAGE"],
            }
            img_cfg: dict = {}
            if aspect_ratio_opt:
                img_cfg["aspect_ratio"] = aspect_ratio_opt
            if resolution_opt:
                img_cfg["image_size"] = resolution_opt
            if img_cfg:
                cfg["image_config"] = img_cfg
            if use_google_search:
                cfg["tools"] = [{"google_search": {}}]
            return cfg

        # Try with image_size first
        resp_modalities = ["TEXT", "IMAGE"] if want_text else ["IMAGE"]

        img_cfg_kwargs: dict = {}
        if aspect_ratio_opt:
            img_cfg_kwargs["aspect_ratio"] = aspect_ratio_opt
        if resolution_opt:
            img_cfg_kwargs["image_size"] = resolution_opt

        tools_val = [{"google_search": {}}] if use_google_search else None

        try:
            image_config_obj = types.ImageConfig(**img_cfg_kwargs) if img_cfg_kwargs else None
            cfg_obj = types.GenerateContentConfig(
                response_modalities=resp_modalities,
                image_config=image_config_obj,
                tools=tools_val,
            )
            if debug and resolution_opt:
                print(f"[PVL PRO NODE] Using image_size={resolution_opt} in GenerateContentConfig.")
            return cfg_obj
        except Exception as e:
            # If the installed SDK does not support image_size, error text will complain about it
            msg = str(e)
            if "image_size" in msg or "extra_forbidden" in msg:
                if debug:
                    print(
                        "[PVL PRO NODE] GenerateContentConfig rejected image_size; "
                        "rebuilding config without resolution."
                    )
                # Rebuild without image_size
                img_cfg_kwargs.pop("image_size", None)
                try:
                    image_config_obj = types.ImageConfig(**img_cfg_kwargs) if img_cfg_kwargs else None
                    cfg_obj = types.GenerateContentConfig(
                        response_modalities=resp_modalities,
                        image_config=image_config_obj,
                        tools=tools_val,
                    )
                    return cfg_obj
                except Exception as e2:
                    if debug:
                        print(f"[PVL PRO NODE] Even config without image_size failed: {e2}")
                    raise
            else:
                if debug:
                    print(f"[PVL PRO NODE] GenerateContentConfig failed for another reason: {e}")
                raise

    def _build_call_prompts(self, base_prompts: T.List[str], num_images: int, debug: bool) -> T.List[str]:
        N = max(1, int(num_images))
        if not base_prompts:
            return [""]
        if len(base_prompts) >= N:
            return base_prompts[:N]
        if debug:
            print(
                f"[PVL PRO NODE] Provided {len(base_prompts)} prompts but num_images={N}. "
                "Reusing the last prompt for remaining calls."
            )
        return base_prompts + [base_prompts[-1]] * (N - len(base_prompts))

    # --------------------------- GOOGLE CALLS --------------------------------

    def _google_generate_once(
        self,
        client,
        model: str,
        prompt: str,
        pil_refs: T.List[Image.Image],
        cfg,
        timeout_sec: int,
        debug: bool,
    ) -> T.Tuple[T.List[torch.Tensor], T.List[str]]:
        contents = []
        if prompt:
            contents.append(prompt)
        contents.extend(pil_refs)

        if debug:
            print(
                f"[PVL PRO NODE] Google call model={model}, prompt_len={len(prompt)}, refs={len(pil_refs)}"
            )

        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
                request_options={"timeout": timeout_sec},
            )
        except TypeError:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )

        imgs: T.List[torch.Tensor] = []
        texts: T.List[str] = []

        parts = getattr(resp, "parts", None)
        if parts is None and hasattr(resp, "candidates"):
            parts = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                if content is not None and hasattr(content, "parts"):
                    parts.extend(content.parts)

        if parts is None:
            return imgs, texts

        for part in parts:
            b = _extract_image_bytes_from_part(part)
            if b:
                try:
                    im = Image.open(io.BytesIO(b)).convert("RGB")
                    imgs.append(pil_to_tensor(im))  # [1,H,W,3]
                except Exception as e:
                    if debug:
                        print("[PVL PRO NODE] image decode error:", e)
            else:
                t = _extract_text_from_part(part)
                if t:
                    texts.append(t)

        return imgs, texts

    def _google_generate_with_retries(
        self,
        client,
        model: str,
        prompt: str,
        pil_refs: T.List[Image.Image],
        cfg,
        timeout_sec: int,
        retries: int,
        debug: bool,
    ) -> T.Tuple[T.List[torch.Tensor], T.List[str]]:
        last_exc = None
        attempts = max(0, int(retries)) + 1
        for attempt in range(1, attempts + 1):
            try:
                if debug:
                    print(f"[PVL PRO NODE] Google attempt {attempt}/{attempts} for prompt[:80]={prompt[:80]!r}")
                return self._google_generate_once(
                    client, model, prompt, pil_refs, cfg, timeout_sec, debug
                )
            except Exception as e:
                last_exc = e
                if attempt >= attempts:
                    break
                wait = 1.0 * attempt
                if debug:
                    print(
                        f"[PVL PRO NODE] Google call failed (attempt {attempt}), retrying in {wait:.1f}s: {e}"
                    )
                time.sleep(wait)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Google call failed without exception in retry wrapper.")

    # --------------------------- FAL HELPERS ---------------------------------

    def _fal_submit_only(
        self,
        route: str,
        prompt: str,
        image_tensors: T.List[torch.Tensor],
        mime: str,
        fal_key: str,
        timeout: int,
        debug: bool,
        output_format: str,
        aspect_ratio_opt: T.Optional[str],
        resolution_opt: T.Optional[str],
        sync_mode: bool,
    ):
        if not fal_key:
            raise RuntimeError("FAL requested but FAL_KEY is missing.")

        base = "https://queue.fal.run"
        url = f"{base}/{route.strip()}"

        headers = {
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        }

        data_urls: T.List[str] = []
        for t in image_tensors:
            batch = t if t.ndim == 4 else t.unsqueeze(0)
            for i in range(batch.shape[0]):
                pil = tensor_to_pil(batch[i:i+1])
                raw = encode_pil_bytes(pil, mime)
                data_urls.append(_data_url(mime, raw))

        payload = {
            "prompt": prompt or "",
            "num_images": 1,
            "output_format": "png" if output_format.lower() == "png" else "jpeg",
            "sync_mode": bool(sync_mode),
        }
        if aspect_ratio_opt:
            payload["aspect_ratio"] = aspect_ratio_opt
        if resolution_opt:
            payload["resolution"] = resolution_opt
        if data_urls:
            payload["image_urls"] = data_urls

        if debug:
            print(
                f"[PVL PRO NODE] FAL submit route={route}, prompt[:80]={payload['prompt'][:80]!r}, "
                f"images={len(data_urls)}, aspect_ratio={aspect_ratio_opt}, resolution={resolution_opt}"
            )

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")

        js = r.json() or {}
        req_id = js.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return request_id")

        status_url = js.get("status_url") or f"{base}/{route.strip()}/requests/{req_id}/status"
        resp_url = js.get("response_url") or f"{base}/{route.strip()}/requests/{req_id}"
        return {"request_id": req_id, "status_url": status_url, "response_url": resp_url}

    def _fal_poll_and_fetch(
        self,
        req_id: str,
        status_url: str,
        resp_url: str,
        fal_key: str,
        timeout: int,
        debug: bool,
    ) -> dict:
        headers = {"Authorization": f"Key {fal_key}"}
        if debug:
            print(f"[PVL PRO NODE] FAL poll request_id={req_id}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=min(10, timeout))
                if sr.ok:
                    js = sr.json() or {}
                    st = js.get("status")
                    if st == "COMPLETED":
                        break
                    if st in ("ERROR", "FAILED", "CANCELLED"):
                        msg = js.get("error") or f"FAL status={st}"
                        raise RuntimeError(msg)
                    if debug:
                        print(f"[PVL PRO NODE] FAL status={st}")
                else:
                    if debug:
                        print(f"[PVL PRO NODE] FAL status error {sr.status_code}: {sr.text}")
            except Exception as e:
                if debug:
                    print(f"[PVL PRO NODE] FAL poll exception: {e}")
            time.sleep(1.0)
        else:
            raise TimeoutError("FAL polling timed out.")

        rr = requests.get(resp_url, headers=headers, timeout=timeout)
        if not rr.ok:
            raise RuntimeError(f"FAL result error {rr.status_code}: {rr.text}")
        return rr.json() or {}

    def _fal_extract_images_and_text(self, resp: dict, debug: bool) -> T.Tuple[T.List[torch.Tensor], T.List[str]]:
        imgs: T.List[torch.Tensor] = []
        texts: T.List[str] = []

        images_info = resp.get("images") or []
        if isinstance(images_info, list):
            for im in images_info:
                url = im.get("url")
                if not url:
                    continue
                try:
                    r = requests.get(url, timeout=60)
                    if not r.ok:
                        if debug:
                            print(f"[PVL PRO NODE] FAL image download error {r.status_code}: {url}")
                        continue
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    imgs.append(pil_to_tensor(img))  # [1,H,W,3]
                except Exception as e:
                    if debug:
                        print(f"[PVL PRO NODE] FAL image download exception for {url}: {e}")

        desc = resp.get("description")
        if isinstance(desc, str) and desc.strip():
            texts.append(desc.strip())

        return imgs, texts

    def _parallel_fal_batch(
        self,
        indices: T.List[int],
        prompts: T.List[str],
        fal_key: str,
        route: str,
        image_tensors: T.List[torch.Tensor],
        input_mime: str,
        timeout_sec: int,
        output_format: str,
        aspect_ratio_opt: T.Optional[str],
        resolution_opt: T.Optional[str],
        sync_mode: bool,
        debug: bool,
    ):
        success_map: dict[int, torch.Tensor] = {}
        text_map: dict[int, str] = {}
        error_info: dict[int, str] = {}

        submit_map: dict[int, dict] = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def submit_worker(idx: int, ptxt: str):
            try:
                req = self._fal_submit_only(
                    route,
                    ptxt,
                    image_tensors,
                    input_mime,
                    fal_key,
                    timeout_sec,
                    debug,
                    output_format,
                    aspect_ratio_opt,
                    resolution_opt,
                    sync_mode,
                )
                submit_map[idx] = req
            except Exception as e:
                msg = f"FAL submit failed: {e}"
                error_info[idx] = msg
                print(f"[PVL PRO NODE] {msg}", flush=True)

        with ThreadPoolExecutor(max_workers=min(8, len(indices))) as ex:
            futs = [ex.submit(submit_worker, i, prompts[i]) for i in indices]
            for _ in as_completed(futs):
                pass

        def result_worker(idx: int, req: dict):
            try:
                js = self._fal_poll_and_fetch(
                    req["request_id"],
                    req["status_url"],
                    req["response_url"],
                    fal_key,
                    timeout_sec,
                    debug,
                )
                imgs, txts = self._fal_extract_images_and_text(js, debug)
                if imgs:
                    success_map[idx] = imgs[0]
                if txts:
                    text_map[idx] = "\n".join(txts)
            except Exception as e:
                msg = f"FAL result failed: {e}"
                error_info[idx] = msg
                print(f"[PVL PRO NODE] {msg}", flush=True)

        with ThreadPoolExecutor(max_workers=min(8, len(submit_map))) as ex:
            futs = [ex.submit(result_worker, i, req) for i, req in submit_map.items()]
            for _ in as_completed(futs):
                pass

        return success_map, text_map, error_info

    # --------------------------- RUN MAIN ------------------------------------

    def run(
        self,
        prompt: str,
        delimiter: str = "[++]",
        image_1: T.Optional[torch.Tensor] = None,
        image_2: T.Optional[torch.Tensor] = None,
        image_3: T.Optional[torch.Tensor] = None,
        image_4: T.Optional[torch.Tensor] = None,
        image_5: T.Optional[torch.Tensor] = None,
        image_6: T.Optional[torch.Tensor] = None,
        image_7: T.Optional[torch.Tensor] = None,
        image_8: T.Optional[torch.Tensor] = None,
        aspect_ratio: str = "1:1",
        model: str = DEFAULT_MODEL,
        endpoint_override: str = "",
        api_key: str = "",
        seed: int = 0,  # not used by Gemini 3 image API
        resolution: str = "1K",
        output_format: str = "png",
        capture_text_output: bool = False,
        use_google_search: bool = False,
        num_images: int = 1,
        timeout_sec: int = 120,
        request_id: str = "",
        debug_log: bool = False,
        use_fal_fallback: bool = True,
        force_fal: bool = False,
        sync_mode: bool = False,
        fal_api_key: str = "",
        fal_route_img2img: str = "fal-ai/nano-banana-pro/edit",
        fal_route_txt2img: str = "fal-ai/nano-banana-pro",
        retries: int = 3,
    ):
        t0 = time.time()

        # Aspect ratio
        aspect_ratio_opt = None
        if aspect_ratio and aspect_ratio.strip().lower() != "auto":
            aspect_ratio_opt = self._clean_aspect_ratio(aspect_ratio)
            if not aspect_ratio_opt:
                print(
                    f"[PVL PRO NODE] WARNING: invalid aspect_ratio '{aspect_ratio}', "
                    "omitting from requests."
                )

        # Resolution (docs: must be 1K, 2K, 4K with uppercase K)
        resolution_opt = self._clean_resolution(resolution)
        if resolution and not resolution_opt:
            print(
                f"[PVL PRO NODE] WARNING: invalid resolution '{resolution}'. "
                "Supported: 1K, 2K, 4K. Omitting from requests."
            )

        # Split prompts via delimiter regex
        try:
            base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
        except re.error:
            print(
                f"[PVL PRO NODE] WARNING: invalid regex pattern '{delimiter}', using literal split instead."
            )
            base_prompts = [p.strip() for p in prompt.split(delimiter) if str(p).strip()]

        if not base_prompts:
            base_prompts = [""]

        if debug_log:
            print(f"[PVL PRO NODE] base_prompts={len(base_prompts)}")

        N = max(1, int(num_images))
        want_text = bool(capture_text_output)
        call_prompts = self._build_call_prompts(base_prompts, N, debug_log)

        # Collect reference images
        input_tensors: T.List[torch.Tensor] = []
        pil_refs: T.List[Image.Image] = []
        for img in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]:
            if img is not None and torch.is_tensor(img) and img.numel() > 0:
                input_tensors.append(img)
                pil_refs.append(tensor_to_pil(img))

        is_img2img = len(input_tensors) > 0

        # FAL routes
        route_img2img = (fal_route_img2img or "").strip() or "fal-ai/nano-banana-pro/edit"
        route_txt2img = (fal_route_txt2img or "").strip() or "fal-ai/nano-banana-pro"
        route_to_use = route_img2img if is_img2img else route_txt2img

        # Keys
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        fal_key = (fal_api_key or os.getenv("FAL_KEY", "")).strip()

        # ---------------- FAL ONLY PATH ----------------

        if force_fal:
            if not fal_key:
                raise RuntimeError("force_fal=True but FAL_KEY is missing.")
            if debug_log:
                print("[PVL PRO NODE] force_fal=True, using FAL only.")
            indices = list(range(N))
            fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
                indices,
                call_prompts,
                fal_key,
                route_to_use,
                input_tensors,
                "image/png",
                timeout_sec,
                output_format,
                aspect_ratio_opt,
                resolution_opt,
                sync_mode,
                debug_log,
            )
            imgs_out: T.List[torch.Tensor] = []
            for i in indices:
                if i in fal_succ:
                    imgs_out.append(fal_succ[i])
                else:
                    raise RuntimeError(f"FAL generation failed at index={i}: {fal_errs.get(i, 'Unknown error')}")

            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                parts = []
                for i in indices:
                    if i in fal_texts:
                        parts.append(f"[i={i}] {fal_texts[i]}")
                text_out = "\n".join(parts) if parts else ""
            return text_out, images_tensor

        # ---------------- GOOGLE UNAVAILABLE → FAL ----------------

        if not key:
            if not use_fal_fallback:
                raise RuntimeError("Gemini API key missing. Provide api_key or set GEMINI_API_KEY.")
            if not fal_key:
                raise RuntimeError("Gemini API key missing and FAL_KEY missing.")
            if debug_log:
                print("[PVL PRO NODE] No Gemini key, using FAL only (fallback).")
            indices = list(range(N))
            fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
                indices,
                call_prompts,
                fal_key,
                route_to_use,
                input_tensors,
                "image/png",
                timeout_sec,
                output_format,
                aspect_ratio_opt,
                resolution_opt,
                sync_mode,
                debug_log,
            )

            imgs_out: T.List[torch.Tensor] = []
            for i in indices:
                if i in fal_succ:
                    imgs_out.append(fal_succ[i])
                else:
                    raise RuntimeError(f"FAL generation failed at index={i}: {fal_errs.get(i, 'Unknown error')}")

            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                parts = []
                for i in indices:
                    if i in fal_texts:
                        parts.append(f"[i={i}] {fal_texts[i]}")
                text_out = "\n".join(parts) if parts else ""
            return text_out, images_tensor

        # ---------------- GOOGLE + OPTIONAL FAL FALLBACK ----------------

        client = self._make_client(key, endpoint_override)
        cfg = self._build_config(want_text, aspect_ratio_opt, resolution_opt, use_google_search, debug_log)

        if debug_log:
            print(
                f"[PVL PRO NODE] Google pass — N={N}, model={model}, "
                f"aspect_ratio={aspect_ratio_opt}, resolution={resolution_opt}, "
                f"use_google_search={use_google_search}"
            )

        success_imgs_g: dict[int, torch.Tensor] = {}
        success_txt_g: dict[int, str] = {}
        error_g: dict[int, str] = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def google_worker(idx: int, ptxt: str):
            try:
                imgs, txts = self._google_generate_with_retries(
                    client,
                    model,
                    ptxt,
                    pil_refs,
                    cfg,
                    timeout_sec,
                    retries,
                    debug_log,
                )
                if imgs:
                    success_imgs_g[idx] = imgs[0]
                if txts:
                    success_txt_g[idx] = "\n".join(txts)
            except Exception as e:
                msg = f"Google call failed for index={idx}: {e}"
                error_g[idx] = msg
                print(f"[PVL PRO NODE] {msg}", flush=True)

        indices = list(range(N))
        with ThreadPoolExecutor(max_workers=min(8, N)) as ex:
            futs = [ex.submit(google_worker, i, call_prompts[i]) for i in indices]
            for _ in as_completed(futs):
                pass

        failed_indices = [i for i in indices if i not in success_imgs_g]

        # If no fallback: require full success
        if not use_fal_fallback:
            if failed_indices:
                msg = error_g.get(failed_indices[0], "Google image generation failed.")
                raise RuntimeError(msg)
            imgs_out = [success_imgs_g[i] for i in indices]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                parts = []
                for i in indices:
                    if i in success_txt_g:
                        parts.append(f"[i={i}] {success_txt_g[i]}")
                text_out = "\n".join(parts) if parts else ""
            return text_out, images_tensor

        # Fallback enabled
        if not failed_indices:
            imgs_out = [success_imgs_g[i] for i in indices]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                parts = []
                for i in indices:
                    if i in success_txt_g:
                        parts.append(f"[i={i}] {success_txt_g[i]}")
                text_out = "\n".join(parts) if parts else ""
            return text_out, images_tensor

        if not fal_key:
            if not success_imgs_g:
                raise RuntimeError(
                    "Gemini failed for some items and FAL_KEY is missing. "
                    f"Example error: {error_g.get(failed_indices[0], 'unknown error')}"
                )
            print(
                f"[PVL PRO NODE] WARNING: FAL_KEY missing; returning only "
                f"{len(success_imgs_g)}/{N} successful Gemini outputs."
            )
            imgs_out = [success_imgs_g[i] for i in indices if i in success_imgs_g]
            images_tensor = stack_images_same_size(imgs_out, debug_log)
            text_out = ""
            if want_text:
                parts = []
                for i in indices:
                    if i in success_txt_g:
                        parts.append(f"[i={i}] {success_txt_g[i]}")
                text_out = "\n".join(parts) if parts else ""
            return text_out, images_tensor

        # Send failed ones to FAL
        if debug_log:
            print(
                f"[PVL PRO NODE] Gemini failures={len(failed_indices)}; "
                f"routing those to FAL route={route_to_use}"
            )

        fal_succ, fal_texts, fal_errs = self._parallel_fal_batch(
            failed_indices,
            call_prompts,
            fal_key,
            route_to_use,
            input_tensors,
            "image/png",
            timeout_sec,
            output_format,
            aspect_ratio_opt,
            resolution_opt,
            sync_mode,
            debug_log,
        )

        final_imgs: dict[int, torch.Tensor] = dict(success_imgs_g)
        final_txts: dict[int, str] = dict(success_txt_g)

        for i, img in fal_succ.items():
            final_imgs[i] = img
        for i, txt in fal_texts.items():
            if i not in final_txts:
                final_txts[i] = txt

        if not final_imgs:
            all_errs = list(error_g.values()) + list(fal_errs.values())
            msg = all_errs[0] if all_errs else "Gemini+FAL total failure."
            raise RuntimeError(msg)

        imgs_out: T.List[torch.Tensor] = []
        ordered_indices = [i for i in indices if i in final_imgs]
        for i in ordered_indices:
            imgs_out.append(final_imgs[i])

        images_tensor = stack_images_same_size(imgs_out, debug_log)
        text_out = ""
        if want_text and final_txts:
            parts = []
            for i in ordered_indices:
                if i in final_txts:
                    parts.append(f"[i={i}] {final_txts[i]}")
            text_out = "\n".join(parts) if parts else ""

        if len(ordered_indices) < N:
            print(
                f"[PVL PRO NODE] WARNING: returning only {len(ordered_indices)}/{N} successful items "
                "(Gemini + FAL)."
            )

        print(f"[PVL PRO NODE] Completed in {time.time()-t0:.2f}s")
        return text_out, images_tensor


NODE_CLASS_MAPPINGS = {
    "PVL_Google_NanoBanana_PRO_Multi_API": PVL_Google_NanoBanana_PRO_Multi_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_Google_NanoBanana_PRO_Multi_API": NODE_NAME,
}
