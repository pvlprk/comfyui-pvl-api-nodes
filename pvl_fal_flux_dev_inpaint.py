import re
import time
import io
import base64
import requests
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fal_utils import FalConfig, ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_Flux_Dev_Inpaint_API:
    """
    PVL Flux Dev Inpaint (fal.ai)
    Endpoint: fal-ai/flux-lora/inpainting

    Inputs:
      - image: ComfyUI IMAGE
      - mask:  ComfyUI MASK (float 0..1, where 1.0 = inpaint region)

    Implementation:
      - Uses Base64 PNG data URIs for image_url + mask_url (no fal.storage upload).
      - Queue submit -> poll -> fetch (mirrors PVL Flux With LoRA node pattern).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),

                "image": ("IMAGE",),
                "mask": ("MASK",),

                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048}),

                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),

                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),

                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": (
                    "STRING",
                    {
                        "default": "[++]",
                        "multiline": False,
                        "placeholder": "Delimiter/regex for splitting prompts (e.g. [++], \\n, |)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    # -------- Local fallback error handler --------
    def _handle_error(self, api_name: str, error: Exception):
        print(f"[{api_name} ERROR] {str(error)}")
        blank = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        return (blank,)

    # -------- Detect non-retryable policy errors --------
    def _is_content_policy_violation(self, message_or_json) -> bool:
        try:
            if isinstance(message_or_json, dict):
                et = str(message_or_json.get("type", "")).lower()
                if "content_policy_violation" in et:
                    return True
                err = message_or_json.get("error")
                if isinstance(err, dict):
                    et2 = str(err.get("type", "")).lower()
                    if "content_policy_violation" in et2:
                        return True
                if "content_policy_violation" in str(message_or_json).lower():
                    return True
            elif isinstance(message_or_json, str):
                if "content_policy_violation" in message_or_json.lower():
                    return True
        except Exception:
            pass
        return False

    # -------------------------
    # MASK -> strict B/W (L-mode) -> data URI
    # -------------------------
    def _mask_to_pil_bw(self, mask: torch.Tensor) -> Image.Image:
        """
        ComfyUI MASK:
          - shape: (H,W) or (1,H,W)
          - float in [0,1]
          - 1.0 = inpaint region (white)
          - 0.0 = keep region (black)
        """
        if not isinstance(mask, torch.Tensor):
            raise RuntimeError("Mask must be a torch.Tensor")

        m = mask.detach().cpu().float()

        if m.ndim == 3:
            m = m[0]  # (H,W)

        if m.ndim != 2:
            raise RuntimeError(f"Invalid MASK shape: {tuple(mask.shape)}")

        m = m.clamp(0.0, 1.0)
        m8 = (m * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(m8, mode="L")

    def _pil_to_png_data_uri(self, pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # -------------------------
    # FAL queue submit/poll (matches your example node)
    # -------------------------
    def _direct_fal_submit(self, endpoint: str, arguments: dict, timeout_sec: int, debug: bool):
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        r = requests.post(submit_url, headers=headers, json=arguments, timeout=timeout_sec)
        if not r.ok:
            try:
                js = r.json()
                if self._is_content_policy_violation(js):
                    raise RuntimeError(f"FAL content_policy_violation: {js}")
            except Exception:
                pass
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        status_url = sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}"

        if debug:
            print(f"[FAL SUBMIT OK] request_id={req_id}")

        return {"request_id": req_id, "status_url": status_url, "response_url": resp_url}

    def _submit_request(
        self,
        prompt_text,
        width,
        height,
        steps,
        CFG,
        seed,
        enable_safety_checker,
        output_format,
        sync_mode,
        strength,
        image_data_uri,
        mask_data_uri,
        timeout_sec=120,
        debug=False,
    ):
        arguments = {
            "prompt": prompt_text,
            "num_inference_steps": int(steps),
            "guidance_scale": float(CFG),
            "num_images": 1,
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
            "sync_mode": bool(sync_mode),
            "image_size": {"width": int(width), "height": int(height)},
            "image_url": image_data_uri,
            "mask_url": mask_data_uri,
            "strength": float(strength),
        }
        if seed != -1:
            arguments["seed"] = int(seed)

        if debug:
            safe_args = dict(arguments)
            safe_args["image_url"] = safe_args["image_url"][:48] + "...(data-uri)"
            safe_args["mask_url"] = safe_args["mask_url"][:48] + "...(data-uri)"
            print(f"[FAL SUBMIT] payload: {safe_args}")

        # Prefer ApiHandler if it supports async submit_only (same pattern as your other node)
        if hasattr(ApiHandler, "submit_only"):
            try:
                if "timeout" in ApiHandler.submit_only.__code__.co_varnames:
                    return ApiHandler.submit_only("fal-ai/flux-lora/inpainting", arguments, timeout=timeout_sec, debug=debug)
                return ApiHandler.submit_only("fal-ai/flux-lora/inpainting", arguments)
            except Exception as e:
                raise RuntimeError(f"FAL submit_only failed: {e}")

        return self._direct_fal_submit("fal-ai/flux-lora/inpainting", arguments, timeout_sec, debug)

    def _poll_request(self, request_info, timeout_sec=120, debug=False, item_idx=None, attempt=None, started_at=None):
        # Prefer ApiHandler if it supports async polling (same pattern as your other node)
        if hasattr(ApiHandler, "poll_and_get_result"):
            try:
                if "timeout" in ApiHandler.poll_and_get_result.__code__.co_varnames:
                    return ApiHandler.poll_and_get_result(request_info, timeout=timeout_sec, debug=debug)
                return ApiHandler.poll_and_get_result(request_info)
            except Exception as e:
                raise RuntimeError(f"FAL poll_and_get_result failed: {e}")

        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]
        req_id = request_info.get("request_id", "")[:16]

        deadline = time.time() + timeout_sec
        completed = False
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=min(10, timeout_sec))
                if sr.ok:
                    js = sr.json()
                    st = js.get("status")
                    if debug:
                        elapsed = (time.time() - (started_at or 0.0)) if started_at else 0.0
                        print(f"[FAL POLL] item={item_idx} attempt={attempt} req={req_id} status={st} elapsed={elapsed:.1f}s")
                    if st == "COMPLETED":
                        completed = True
                        break
                    if st == "ERROR":
                        msg = js.get("error") or "Unknown FAL error"
                        payload = js.get("payload")
                        if payload:
                            raise RuntimeError(f"FAL status ERROR: {msg} | details: {payload}")
                        raise RuntimeError(f"FAL status ERROR: {msg}")
            except Exception as e:
                if debug:
                    print(f"[FAL POLL] item={item_idx} attempt={attempt} req={req_id} status_check_error: {e}")
            time.sleep(0.6)

        if not completed:
            raise RuntimeError(f"FAL request {req_id} timed out after {timeout_sec}s")

        rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout_sec))
        if not rr.ok:
            try:
                js = rr.json()
                if self._is_content_policy_violation(js):
                    raise RuntimeError(f"FAL content_policy_violation: {js}")
            except Exception:
                pass
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

        return rr.json().get("response", rr.json())

    # Worker with per-item retries
    def _run_one_with_retries(
        self,
        item_index: int,
        prompt_text: str,
        width: int,
        height: int,
        steps: int,
        CFG: float,
        seed_base: int,
        enable_safety_checker: bool,
        output_format: str,
        sync_mode: bool,
        strength: float,
        image_data_uri: str,
        mask_data_uri: str,
        retries: int,
        timeout_sec: int,
        debug: bool,
    ):
        seed_for_item = seed_base if seed_base == -1 else ((seed_base + item_index) % 4294967296)

        last_err = ""
        for attempt in range(1, int(retries) + 2):
            t0 = time.time()
            try:
                if debug:
                    print(f"[FAL ITEM] item={item_index} attempt={attempt}/{retries+1} seed={seed_for_item}")

                req_info = self._submit_request(
                    prompt_text=prompt_text,
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed=seed_for_item,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    strength=strength,
                    image_data_uri=image_data_uri,
                    mask_data_uri=mask_data_uri,
                    timeout_sec=timeout_sec,
                    debug=debug,
                )

                result = self._poll_request(
                    req_info,
                    timeout_sec=timeout_sec,
                    debug=debug,
                    item_idx=item_index,
                    attempt=attempt,
                    started_at=t0,
                )

                out = ResultProcessor.process_image_result(result)
                img_tensor = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                if debug:
                    print(f"[FAL ITEM OK] item={item_index} attempt={attempt} dt={time.time()-t0:.2f}s")

                return True, img_tensor, ""

            except Exception as e:
                last_err = str(e)
                print(f"[FAL ITEM ERROR] item={item_index} attempt={attempt} -> {last_err}")
                if self._is_content_policy_violation(last_err):
                    if debug:
                        print(f"[FAL ITEM INFO] item={item_index} content_policy_violation detected — stopping retries.")
                    break

        return False, None, last_err

    def _build_call_prompts(self, base_prompts, num_images, debug=False):
        N = max(1, int(num_images))
        if not base_prompts:
            return []
        if len(base_prompts) >= N:
            call_prompts = base_prompts[:N]
        else:
            if debug:
                print(f"[PVL Flux Dev Inpaint] Provided {len(base_prompts)} prompts but num_images={N}. Reusing last prompt.")
            call_prompts = base_prompts + [base_prompts[-1]] * (N - len(base_prompts))
        if debug:
            for i, p in enumerate(call_prompts):
                show = p if len(p) <= 160 else (p[:157] + "...")
                print(f"[PVL Flux Dev Inpaint] Call {i+1} prompt: {show}")
        return call_prompts

    # -------------------------
    # Main ComfyUI entrypoint
    # -------------------------
    def generate_image(
        self,
        prompt,
        image,
        mask,
        width,
        height,
        strength,
        steps,
        CFG,
        seed,
        num_images,
        enable_safety_checker,
        output_format,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        delimiter="[++]",
    ):
        _t0 = time.time()

        try:
            # Split prompts using delimiter with regex support
            try:
                base_prompts = [p.strip() for p in re.split(delimiter, prompt) if str(p).strip()]
            except re.error:
                print(f"[PVL Flux Dev Inpaint WARNING] Invalid regex pattern '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in str(prompt).split(delimiter) if str(p).strip()]

            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")

            call_prompts = self._build_call_prompts(base_prompts, num_images, debug=debug_log)
            N = len(call_prompts)

            print(f"[PVL Flux Dev Inpaint INFO] Processing {N} call(s) | retries={retries} | timeout={timeout_sec}s")

            # Build data URIs ONCE and reuse for all calls (no fal.storage upload)
            image_data_uri = ImageUtils.image_to_data_uri(image)
            mask_pil = self._mask_to_pil_bw(mask)
            mask_data_uri = self._pil_to_png_data_uri(mask_pil)

            if debug_log:
                print("[PVL Flux Dev Inpaint DEBUG] Using image_url + mask_url as data URIs (no storage upload).")

            # Single call path
            if N == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    prompt_text=call_prompts[0],
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    strength=strength,
                    image_data_uri=image_data_uri,
                    mask_data_uri=mask_data_uri,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    _t1 = time.time()
                    print(f"[PVL Flux Dev Inpaint INFO] Successfully generated 1 image in {(_t1 - _t0):.2f}s")
                    return (img_tensor,)
                raise RuntimeError(last_err or "All attempts failed for single request")

            # Multiple calls in parallel
            print(f"[PVL Flux Dev Inpaint INFO] Submitting {N} requests in parallel...")

            results_map = {}
            errors_map = {}
            max_workers = min(N, 6)

            def worker(i):
                return i, *self._run_one_with_retries(
                    item_index=i,
                    prompt_text=call_prompts[i],
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed_base=seed,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    strength=strength,
                    image_data_uri=image_data_uri,
                    mask_data_uri=mask_data_uri,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug=debug_log,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = [executor.submit(worker, i) for i in range(N)]
                for fut in as_completed(futs):
                    i, ok, img_tensor, last_err = fut.result()
                    if ok and torch.is_tensor(img_tensor):
                        results_map[i] = img_tensor
                    else:
                        errors_map[i] = last_err or "Unknown error"

            if not results_map:
                sample_err = next(iter(errors_map.values()), "All FAL requests failed")
                raise RuntimeError(sample_err)

            all_images = [results_map[i] for i in sorted(results_map.keys()) if torch.is_tensor(results_map[i])]
            if not all_images:
                raise RuntimeError("No images were generated from API calls")

            final_tensor = torch.cat(all_images, dim=0)

            _t1 = time.time()
            print(f"[PVL Flux Dev Inpaint INFO] Successfully generated {final_tensor.shape[0]}/{N} images in {(_t1 - _t0):.2f}s")

            failed_idxs = sorted(set(range(N)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(f"[PVL Flux Dev Inpaint ERROR] Item {i+1} failed after {retries+1} attempt(s): {errors_map.get(i,'Unknown error')}")
                print(f"[PVL Flux Dev Inpaint WARNING] Returning only {final_tensor.shape[0]}/{N} successful results.")

            if seed != -1:
                seed_list = [(seed + i) % 4294967296 for i in range(N)]
                print(f"[PVL Flux Dev Inpaint INFO] Seeds used: {seed_list}")

            return (final_tensor,)

        except Exception as e:
            print(f"Error generating image with FLUX Inpaint: {str(e)}")
            return self._handle_error("FLUX_INPAINT", e)
