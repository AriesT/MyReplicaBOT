import asyncio
import json
import logging
import os
import time
import uuid
from io import BytesIO
from typing import Awaitable, Callable, Optional

import aiohttp
from PIL import Image

import config

# ── workflow metadata ─────────────────────────────────────────────────────
WORKFLOW_TYPES  = ("sd15", "sdxl", "flux", "sd3", "hidream")
WORKFLOW_ICONS  = {"sd15": "🎨", "sdxl": "🖼", "flux": "⚡", "sd3": "🔮", "hidream": "🌟"}
WORKFLOW_LABELS = {"sd15": "SD 1.5", "sdxl": "SDXL", "flux": "FLUX", "sd3": "SD 3/3.5", "hidream": "HiDream"}

# FLUX requires these files in models/text_encoders/ and models/vae/
_FLUX_T5_VARIANTS   = {"t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "t5xxl.safetensors"}
_FLUX_CLIP_VARIANTS = {"clip_l.safetensors"}
_FLUX_VAE_VARIANTS  = {"ae.safetensors"}

# HiDream requires 4 text encoders + same VAE as FLUX
_HIDREAM_CLIP_L_VARIANTS = {"clip_l_hidream.safetensors"}
_HIDREAM_CLIP_G_VARIANTS = {"clip_g_hidream.safetensors"}
_HIDREAM_T5_VARIANTS     = {"t5xxl_fp8_e4m3fn_scaled.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors"}
_HIDREAM_LLAMA_VARIANTS  = {"llama_3.1_8b_instruct_fp8_scaled.safetensors", "llama_3.1_8b_instruct_fp8.safetensors"}


def detect_workflow_hint(name: str) -> str:
    """Heuristic: guess workflow type from checkpoint filename."""
    n = name.lower()
    if any(x in n for x in ("hidream", "hi_dream", "hi-dream")):
        return "hidream"
    if any(x in n for x in ("flux", "schnell")):
        return "flux"
    if any(x in n for x in ("xl", "sdxl", "juggernaut", "playground", "pony")):
        return "sdxl"
    if any(x in n for x in ("sd3", "sd35", "sd3.5", "stable-diffusion-3")):
        return "sd3"
    return "sd15"

_CONNECT_TIMEOUT = aiohttp.ClientTimeout(total=10, connect=5)
log = logging.getLogger(__name__)

# ── cleanup of ComfyUI files once the bot has the result ─────────────────
# The bot runs on the same host as ComfyUI, so results (output/) and uploaded
# inputs (input/) are deleted from disk right after they are fetched.

_task_uploads: dict[int, list[str]] = {}


def _register_upload(name: str) -> None:
    task = asyncio.current_task()
    if task is not None:
        _task_uploads.setdefault(id(task), []).append(name)


def _safe_unlink(base: str, *parts: str) -> None:
    if not base:
        return
    root = os.path.realpath(base)
    path = os.path.realpath(os.path.join(root, *[p for p in parts if p]))
    if not path.startswith(root + os.sep):
        return                                  # never touch anything outside the ComfyUI folder
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("cleanup failed for %s: %s", path, e)


async def _cleanup_run(session: aiohttp.ClientSession, prompt_id: Optional[str]) -> None:
    """Delete this task's uploaded inputs and drop the prompt from ComfyUI history."""
    task = asyncio.current_task()
    for name in _task_uploads.pop(id(task), []) if task is not None else []:
        _safe_unlink(config.COMFY_INPUT_DIR, name)
    if prompt_id:
        try:
            await session.post(f"{config.COMFY_URL}/history", json={"delete": [prompt_id]},
                               timeout=_CONNECT_TIMEOUT)
        except Exception:
            pass


def _resize(data: bytes, width: int, height: int) -> bytes:
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


async def upload_image(image_bytes: bytes, filename: str = "input.png") -> str:
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename=filename, content_type="image/png")
        form.add_field("overwrite", "true")
        r = await session.post(f"{config.COMFY_URL}/upload/image", data=form)
        r.raise_for_status()
        name = (await r.json())["name"]
        _register_upload(name)
        return name


def _lora_nodes(s: dict) -> tuple[dict, list, list]:
    """Return (dict of lora nodes, model_src, clip_src).

    Supports multiple LoRAs via s["loras_active"] = [{"name": ..., "strength": ...}, ...]
    Falls back to legacy single-lora fields s["lora"] / s["lora_strength"] for backwards compat.
    """
    active: list[dict] = list(s.get("loras_active") or [])
    # backwards compat: single lora field
    if not active and s.get("lora"):
        active = [{"name": s["lora"], "strength": float(s.get("lora_strength") or 0.8)}]

    if not active:
        return {}, ["4", 0], ["4", 1]

    nodes: dict = {}
    model_src: list = ["4", 0]
    clip_src:  list = ["4", 1]
    for i, item in enumerate(active):
        nid = str(50 + i)   # nodes 50, 51, 52, …
        nodes[nid] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model":          model_src,
                "clip":           clip_src,
                "lora_name":      item["name"],
                "strength_model": float(item.get("strength", 0.8)),
                "strength_clip":  float(item.get("strength", 0.8)),
            },
        }
        model_src = [nid, 0]
        clip_src  = [nid, 1]
    return nodes, model_src, clip_src


def _hires_nodes(s: dict, decoded_image_src: list, model_src: list,
                 width: int, height: int, cfg: float, steps: int,
                 sampler: str, scheduler: str) -> dict:
    """Return extra workflow nodes for HiRes Fix (upscale + second KSampler pass)."""
    hires_denoise = float(s.get("hires_denoise") or 0.45)
    upscale_model = s.get("upscale_model")
    out_w = width * 2
    out_h = height * 2
    nodes: dict = {}
    if upscale_model:
        nodes["30"] = {"class_type": "UpscaleModelLoader",    "inputs": {"model_name": upscale_model}}
        nodes["31"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["30", 0],
                                                                          "image": decoded_image_src}}
        nodes["32"] = {"class_type": "ImageScale",            "inputs": {"image": ["31", 0],
                                                                          "upscale_method": "lanczos",
                                                                          "width": out_w, "height": out_h,
                                                                          "crop": "disabled"}}
        encode_src = ["32", 0]
    else:
        nodes["30"] = {"class_type": "ImageScale", "inputs": {"image": decoded_image_src,
                                                               "upscale_method": "lanczos",
                                                               "width": out_w, "height": out_h,
                                                               "crop": "disabled"}}
        encode_src = ["30", 0]
    nodes["33"] = {"class_type": "VAEEncode", "inputs": {"pixels": encode_src, "vae": ["4", 2]}}
    nodes["34"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed":          uuid.uuid4().int & 0xFFFFFFFF,
            "steps":         steps,
            "cfg":           cfg,
            "sampler_name":  sampler,
            "scheduler":     scheduler,
            "denoise":       hires_denoise,
            "model":         model_src,
            "positive":      ["6", 0],
            "negative":      ["7", 0],
            "latent_image":  ["33", 0],
        },
    }
    nodes["35"] = {"class_type": "VAEDecode", "inputs": {"samples": ["34", 0], "vae": ["4", 2]}}
    return nodes


def _build_workflow(prompt: str, s: dict) -> dict:
    checkpoint    = s.get("checkpoint")      or config.CHECKPOINT
    steps         = s.get("steps")           or config.STEPS
    cfg           = float(s.get("cfg")       or config.CFG_SCALE)
    width         = s.get("width")           or config.IMAGE_WIDTH
    height        = s.get("height")          or config.IMAGE_HEIGHT
    neg           = s.get("negative_prompt") or config.NEGATIVE_PROMPT
    sampler, scheduler = _resolve_sampler(s.get("sampler") or "euler")
    hires_fix     = bool(s.get("hires_fix"))
    lora_nodes, model_src, clip_src = _lora_nodes(s)
    wf = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage",        "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",          "inputs": {"text": prompt, "clip": clip_src}},
        "7": {"class_type": "CLIPTextEncode",          "inputs": {"text": neg,    "clip": clip_src}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": uuid.uuid4().int & 0xFFFFFFFF,
                "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
                "model": model_src, "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }
    wf.update(lora_nodes)
    if hires_fix:
        wf.update(_hires_nodes(s, ["8", 0], model_src, width, height, cfg, steps, sampler, scheduler))
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "tgbot_hires", "images": ["35", 0]}}
    else:
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "tgbot", "images": ["8", 0]}}
    return wf


def _build_workflow_img2img(prompt: str, s: dict, input_filename: str) -> dict:
    checkpoint    = s.get("checkpoint")      or config.CHECKPOINT
    steps         = s.get("steps")           or config.STEPS
    cfg           = float(s.get("cfg")       or config.CFG_SCALE)
    neg           = s.get("negative_prompt") or config.NEGATIVE_PROMPT
    sampler, scheduler = _resolve_sampler(s.get("sampler") or "euler")
    denoise       = float(s.get("denoise")   or 0.75)
    hires_fix     = bool(s.get("hires_fix"))
    width         = s.get("width")           or config.IMAGE_WIDTH
    height        = s.get("height")          or config.IMAGE_HEIGHT
    lora_nodes, model_src, clip_src = _lora_nodes(s)
    wf = {
        "4":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "10": {"class_type": "LoadImage",  "inputs": {"image": input_filename}},
        "11": {"class_type": "VAEEncode",  "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
        "6":  {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_src}},
        "7":  {"class_type": "CLIPTextEncode", "inputs": {"text": neg,    "clip": clip_src}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": uuid.uuid4().int & 0xFFFFFFFF,
                "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler,
                "denoise": denoise,
                "model": model_src, "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["11", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }
    wf.update(lora_nodes)
    if hires_fix:
        wf.update(_hires_nodes(s, ["8", 0], model_src, width, height, cfg, steps, sampler, scheduler))
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "tgbot_hires_i2i", "images": ["35", 0]}}
    else:
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "tgbot_i2i", "images": ["8", 0]}}
    return wf


def _build_workflow_upscale(input_filename: str, width: int, height: int,
                             scale: float, upscale_model: Optional[str]) -> dict:
    out_w = int(width  * scale)
    out_h = int(height * scale)
    if upscale_model:
        return {
            "1": {"class_type": "LoadImage",             "inputs": {"image": input_filename}},
            "2": {"class_type": "UpscaleModelLoader",    "inputs": {"model_name": upscale_model}},
            "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "ImageScale",            "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                                     "width": out_w, "height": out_h, "crop": "disabled"}},
            "5": {"class_type": "SaveImage",             "inputs": {"filename_prefix": "tgbot_upscale", "images": ["4", 0]}},
        }
    else:
        return {
            "1": {"class_type": "LoadImage",  "inputs": {"image": input_filename}},
            "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                                                          "width": out_w, "height": out_h, "crop": "disabled"}},
            "3": {"class_type": "SaveImage",  "inputs": {"filename_prefix": "tgbot_upscale", "images": ["2", 0]}},
        }


def progress_bar(value: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "░" * width
    pct    = min(value / total, 1.0)
    filled = int(width * pct)
    return f"{'▓' * filled}{'░' * (width - filled)}  {int(pct * 100)}%"


async def fetch_checkpoints() -> list[str]:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/CheckpointLoaderSimple")
            if not r.ok:
                return []
            data  = await r.json()
            names = (data.get("CheckpointLoaderSimple", {})
                        .get("input", {})
                        .get("required", {})
                        .get("ckpt_name", [[]])[0])
            return sorted(names) if isinstance(names, list) else []
    except Exception:
        return []


async def fetch_loras() -> list[str]:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/LoraLoader")
            if not r.ok:
                return []
            data  = await r.json()
            names = (data.get("LoraLoader", {})
                        .get("input", {})
                        .get("required", {})
                        .get("lora_name", [[]])[0])
            return sorted(names) if isinstance(names, list) else []
    except Exception:
        return []


async def fetch_upscale_models() -> list[str]:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/UpscaleModelLoader")
            if not r.ok:
                return []
            data     = await r.json()
            raw      = (data.get("UpscaleModelLoader", {})
                           .get("input", {})
                           .get("required", {})
                           .get("model_name", []))
            # Old format: [["a.pth", ...], {...}]  →  raw[0] is the list
            # New format: ["COMBO", {"options": ["a.pth", ...]}]  →  raw[1]["options"]
            if raw and isinstance(raw[0], list):
                names = raw[0]
            elif len(raw) > 1 and isinstance(raw[1], dict):
                names = raw[1].get("options", [])
            else:
                names = []
            return sorted(names) if names else []
    except Exception:
        return []


async def fetch_unet_models() -> list[str]:
    """Fetch models from UNETLoader — covers models/unet/ and models/diffusion_models/ (FLUX, HiDream)."""
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/UNETLoader")
            if not r.ok:
                return []
            data  = await r.json()
            raw   = (data.get("UNETLoader", {})
                        .get("input", {})
                        .get("required", {})
                        .get("unet_name", []))
            names = raw[0] if raw and isinstance(raw[0], list) else (
                raw[1].get("options", []) if len(raw) > 1 and isinstance(raw[1], dict) else [])
            return sorted(names) if names else []
    except Exception:
        return []


async def fetch_all_models() -> list[str]:
    """Fetch all generatable models: checkpoints + unet/diffusion_models (FLUX, HiDream)."""
    ckpts, unets = await asyncio.gather(fetch_checkpoints(), fetch_unet_models())
    seen: set[str] = set()
    result = []
    for name in ckpts + unets:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return sorted(result)


async def fetch_clip_models() -> list[str]:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/DualCLIPLoader")
            if not r.ok:
                return []
            data  = await r.json()
            raw   = (data.get("DualCLIPLoader", {})
                        .get("input", {})
                        .get("required", {})
                        .get("clip_name1", []))
            names = raw[0] if raw and isinstance(raw[0], list) else (
                raw[1].get("options", []) if len(raw) > 1 and isinstance(raw[1], dict) else [])
            return sorted(names) if names else []
    except Exception:
        return []


async def fetch_vae_models() -> list[str]:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/object_info/VAELoader")
            if not r.ok:
                return []
            data  = await r.json()
            raw   = (data.get("VAELoader", {})
                        .get("input", {})
                        .get("required", {})
                        .get("vae_name", []))
            names = raw[0] if raw and isinstance(raw[0], list) else (
                raw[1].get("options", []) if len(raw) > 1 and isinstance(raw[1], dict) else [])
            return sorted(names) if names else []
    except Exception:
        return []


async def check_flux_deps() -> list[str]:
    """Return list of missing dependency descriptions for FLUX generation."""
    clips_raw = await fetch_clip_models()
    vaes_raw  = await fetch_vae_models()
    # normalize to basenames to handle subfolder paths like "t5\t5xxl_fp16.safetensors"
    clips = {n.replace("\\", "/").split("/")[-1] for n in clips_raw}
    vaes  = {n.replace("\\", "/").split("/")[-1] for n in vaes_raw}
    missing = []
    if not (_FLUX_T5_VARIANTS & clips):
        missing.append("t5xxl_fp16.safetensors → ComfyUI/models/text_encoders/t5/")
    if not (_FLUX_CLIP_VARIANTS & clips):
        missing.append("clip_l.safetensors → ComfyUI/models/text_encoders/")
    if not (_FLUX_VAE_VARIANTS & vaes):
        missing.append("ae.safetensors → ComfyUI/models/vae/")
    return missing


async def check_hidream_deps() -> list[str]:
    """Return list of missing dependency descriptions for HiDream generation."""
    clips_raw = await fetch_clip_models()
    vaes_raw  = await fetch_vae_models()
    clips = {n.replace("\\", "/").split("/")[-1] for n in clips_raw}
    vaes  = {n.replace("\\", "/").split("/")[-1] for n in vaes_raw}
    missing = []
    if not (_HIDREAM_CLIP_L_VARIANTS & clips):
        missing.append("clip_l_hidream.safetensors → ComfyUI/models/text_encoders/")
    if not (_HIDREAM_CLIP_G_VARIANTS & clips):
        missing.append("clip_g_hidream.safetensors → ComfyUI/models/text_encoders/")
    if not (_HIDREAM_T5_VARIANTS & clips):
        missing.append("t5xxl_fp8_e4m3fn_scaled.safetensors → ComfyUI/models/text_encoders/")
    if not (_HIDREAM_LLAMA_VARIANTS & clips):
        missing.append("llama_3.1_8b_instruct_fp8_scaled.safetensors → ComfyUI/models/text_encoders/")
    if not (_FLUX_VAE_VARIANTS & vaes):
        missing.append("ae.safetensors → ComfyUI/models/vae/")
    return missing


# Map A1111-style "sampler_karras" names → (comfy_sampler, scheduler)
_SAMPLER_SCHEDULER_MAP: dict[str, tuple[str, str]] = {
    "dpmpp_2m_karras":          ("dpmpp_2m",          "karras"),
    "dpmpp_2s_ancestral_karras":("dpmpp_2s_ancestral", "karras"),
    "dpmpp_sde_karras":         ("dpmpp_sde",          "karras"),
    "dpmpp_3m_sde_karras":      ("dpmpp_3m_sde",       "karras"),
    "euler_karras":             ("euler",              "karras"),
    "heun_karras":              ("heun",               "karras"),
    "lms_karras":               ("lms",               "karras"),
}

def _resolve_sampler(sampler: str) -> tuple[str, str]:
    """Return (sampler_name, scheduler) splitting A1111-style combined names."""
    if sampler in _SAMPLER_SCHEDULER_MAP:
        return _SAMPLER_SCHEDULER_MAP[sampler]
    if sampler.endswith("_karras"):
        return sampler[:-7], "karras"
    return sampler, "normal"


def _best_clip(available: list[str], variants: set[str]) -> str:
    # exact match first
    for name in available:
        if name in variants:
            return name
    # match ignoring subfolder prefix (e.g. "t5\t5xxl_fp16.safetensors")
    for name in available:
        basename = name.replace("\\", "/").split("/")[-1]
        if basename in variants:
            return name
    return next(iter(variants))


def _build_workflow_flux(prompt: str, s: dict,
                         clips: list[str], vaes: list[str]) -> dict:
    checkpoint = s.get("checkpoint") or config.CHECKPOINT
    steps      = int(s.get("steps") or 20)
    cfg        = float(s.get("cfg") or 1.0)
    width      = int(s.get("width") or 1024)
    height     = int(s.get("height") or 1024)
    t5         = _best_clip(clips, _FLUX_T5_VARIANTS)   or "t5xxl_fp16.safetensors"
    clip_l     = _best_clip(clips, _FLUX_CLIP_VARIANTS) or "clip_l.safetensors"
    vae        = _best_clip(vaes,  _FLUX_VAE_VARIANTS)  or "ae.safetensors"
    return {
        "1": {"class_type": "UNETLoader",     "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": t5, "clip_name2": clip_l, "type": "flux"}},
        "3": {"class_type": "VAELoader",      "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentImage","inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed":         uuid.uuid4().int & 0xFFFFFFFF,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": "euler",
                "scheduler":    "simple",
                "denoise":      1.0,
                "model":        ["1", 0],
                "positive":     ["4", 0],
                "negative":     ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {"class_type": "VAEDecode",  "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",  "inputs": {"filename_prefix": "tgbot_flux", "images": ["7", 0]}},
    }


def _build_workflow_hidream(prompt: str, s: dict,
                             clips: list[str], vaes: list[str]) -> dict:
    checkpoint = s.get("checkpoint") or config.CHECKPOINT
    steps      = int(s.get("steps") or 28)
    cfg        = float(s.get("cfg") or 5.0)
    width      = int(s.get("width") or 1024)
    height     = int(s.get("height") or 1024)
    clip_l     = _best_clip(clips, _HIDREAM_CLIP_L_VARIANTS) or "clip_l_hidream.safetensors"
    clip_g     = _best_clip(clips, _HIDREAM_CLIP_G_VARIANTS) or "clip_g_hidream.safetensors"
    t5         = _best_clip(clips, _HIDREAM_T5_VARIANTS)     or "t5xxl_fp8_e4m3fn_scaled.safetensors"
    llama      = _best_clip(clips, _HIDREAM_LLAMA_VARIANTS)  or "llama_3.1_8b_instruct_fp8_scaled.safetensors"
    vae        = _best_clip(vaes,  _FLUX_VAE_VARIANTS)       or "ae.safetensors"
    return {
        "1":  {"class_type": "UNETLoader",     "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
        "2":  {"class_type": "QuadrupleCLIPLoader", "inputs": {
                  "clip_name1": clip_l, "clip_name2": clip_g,
                  "clip_name3": t5,     "clip_name4": llama, "type": "hidream"}},
        "3":  {"class_type": "VAELoader",      "inputs": {"vae_name": vae}},
        "4":  {"class_type": "CLIPTextEncodeHiDream", "inputs": {
                  "clip_l": prompt, "clip_g": prompt, "t5xxl": prompt, "llama": prompt,
                  "clip": ["2", 0]}},
        "5":  {"class_type": "CLIPTextEncodeHiDream", "inputs": {
                  "clip_l": "", "clip_g": "", "t5xxl": "", "llama": "",
                  "clip": ["2", 0]}},
        "6":  {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7":  {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 6.0}},
        "8":  {
            "class_type": "KSampler",
            "inputs": {
                "seed":         uuid.uuid4().int & 0xFFFFFFFF,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": "lcm",
                "scheduler":    "simple",
                "denoise":      1.0,
                "model":        ["7", 0],
                "positive":     ["4", 0],
                "negative":     ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "9":  {"class_type": "VAEDecode",  "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",  "inputs": {"filename_prefix": "tgbot_hidream", "images": ["9", 0]}},
    }


async def ping() -> bool:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            r = await s.get(f"{config.COMFY_URL}/system_stats")
            return r.ok
    except Exception:
        return False


async def get_status() -> dict:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            sr = await s.get(f"{config.COMFY_URL}/system_stats")
            qr = await s.get(f"{config.COMFY_URL}/queue")
            if sr.ok and qr.ok:
                return {"online": True, "stats": await sr.json(), "queue": await qr.json()}
    except Exception:
        pass
    return {"online": False}


async def _run_comfy_workflow(
    workflow: dict,
    on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    steps_hint: int = 20,
    poll_timeout: Optional[float] = None,
) -> bytes:
    """Submit a workflow and return the first output image (legacy SD/FLUX/HiDream/LTX paths).

    Thin adapter over run_workflow_rich(), so legacy jobs also survive interrupts,
    queue deletions and dropped sockets instead of hanging the bot queue.
    """
    last = [0.0]

    async def on_status(st: dict) -> None:
        if on_progress is None or not st.get("step"):
            return
        now = time.monotonic()
        if now - last[0] >= 0.8 or st["step"] == st["total"]:
            last[0] = now
            await on_progress(st["step"], st["total"])

    return await run_workflow_rich(workflow, {}, steps_hint, 0.0, on_status,
                                   poll_timeout or config.POLL_TIMEOUT, "images")


async def generate(
    prompt: str,
    on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    user_settings: Optional[dict] = None,
    input_image: Optional[bytes] = None,
) -> bytes:
    s        = user_settings or {}
    wf_type  = s.get("_workflow_type", "sd15")
    steps    = int(s.get("steps") or config.STEPS)

    if wf_type == "flux":
        if input_image is not None:
            raise RuntimeError("FLUX не підтримує img2img режим. Оберіть text2img або іншу модель.")
        clips    = await fetch_clip_models()
        vaes     = await fetch_vae_models()
        workflow = _build_workflow_flux(prompt, s, clips, vaes)
    elif wf_type == "hidream":
        if input_image is not None:
            raise RuntimeError("HiDream не підтримує img2img режим. Оберіть text2img або іншу модель.")
        clips    = await fetch_clip_models()
        vaes     = await fetch_vae_models()
        workflow = _build_workflow_hidream(prompt, s, clips, vaes)
    elif input_image is not None:
        w       = s.get("width")  or config.IMAGE_WIDTH
        h       = s.get("height") or config.IMAGE_HEIGHT
        resized = _resize(input_image, w, h)
        fname   = await upload_image(resized, f"tgbot_{uuid.uuid4().hex[:8]}.png")
        workflow = _build_workflow_img2img(prompt, s, fname)
    else:
        workflow = _build_workflow(prompt, s)

    return await _run_comfy_workflow(workflow, on_progress, steps)


async def upscale_image(
    image_bytes: bytes,
    width: int,
    height: int,
    scale: float,
    upscale_model: Optional[str] = None,
    on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> bytes:
    fname    = await upload_image(image_bytes, f"tgbot_usc_{uuid.uuid4().hex[:8]}.png")
    workflow = _build_workflow_upscale(fname, width, height, scale, upscale_model)
    return await _run_comfy_workflow(workflow, on_progress, 1)


# ── LTX-Video ─────────────────────────────────────────────────────────────

# LTX-Video requires frames = 8*n + 1  (9, 17, 25, 33, 49, 65, 97 …)
VIDEO_FRAME_PRESETS = [17, 25, 33, 49, 65]
VIDEO_FPS_PRESETS   = [8, 12, 16, 24]
VIDEO_STEPS_PRESETS = [20, 25, 30, 40]
VIDEO_CFG_PRESETS   = [2.0, 3.0, 3.5, 4.0, 5.0, 7.0]
VIDEO_RES_PRESETS   = [
    ("480×288",   480,  288),   # safe for 6 GB VRAM
    ("512×288",   512,  288),
    ("640×352",   640,  352),
    ("768×512",   768,  512),
    ("512×512",   512,  512),
    ("480×480",   480,  480),
    ("1280×480", 1280,  480),   # wide cinematic / banner
]

# T5 variants recognised for LTXV
_LTXV_T5_VARIANTS = {
    "t5xxl_fp8_e4m3fn_scaled.safetensors",
    "t5xxl_fp8_e4m3fn.safetensors",
    "t5\\t5xxl_fp16.safetensors",
    "t5xxl_fp16.safetensors",
}

_VIDEO_NEG_DEFAULT = (
    "low quality, worst quality, deformed, distorted, blurry, "
    "jittery, flickering, artifacts, noisy, ugly, watermark"
)


async def fetch_video_models() -> list[str]:
    """Return LTX-Video checkpoints available in ComfyUI."""
    all_models = await fetch_checkpoints()
    return [m for m in all_models
            if any(k in m.lower() for k in ("ltx", "ltxv", "ltx-video"))]


async def _find_ltxv_t5(clips: list[str]) -> str:
    """Return best available T5 encoder for LTXV (prefer fp8 scaled)."""
    for want in (
        "t5xxl_fp8_e4m3fn_scaled.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors",
        "t5xxl_fp16.safetensors",
    ):
        for c in clips:
            if c.replace("\\", "/").split("/")[-1] == want:
                return c
    # fallback: any T5
    for c in clips:
        if "t5" in c.lower():
            return c
    return "t5xxl_fp8_e4m3fn_scaled.safetensors"


def _ltxv_frames(requested: int) -> int:
    """Snap to nearest valid LTXV frame count (8n+1, min 9)."""
    n = max(1, round((requested - 1) / 8))
    return 8 * n + 1


def _build_workflow_ltxv_t2v(prompt: str, s: dict, t5_name: str) -> dict:
    """LTX-Video text-to-video workflow."""
    model   = s.get("video_model")   or "ltx-video-2b-v0.9.5.safetensors"
    width   = int(s.get("video_width")  or 480)
    height  = int(s.get("video_height") or 288)
    frames  = _ltxv_frames(int(s.get("video_frames") or 25))
    fps     = float(s.get("video_fps")   or 8)
    steps   = int(s.get("video_steps")  or 25)
    cfg     = float(s.get("video_cfg")  or 3.5)
    neg     = s.get("video_negative")   or _VIDEO_NEG_DEFAULT
    seed    = uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF

    return {
        # ── loaders ──────────────────────────────────────────────────────
        "1": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": t5_name, "type": "ltxv"}},
        "2": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model}},
        # ── text encoding ─────────────────────────────────────────────────
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": neg, "clip": ["1", 0]}},
        # ── latent + model patching ───────────────────────────────────────
        "5": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": width, "height": height,
                         "length": frames, "batch_size": 1}},
        "6": {"class_type": "ModelSamplingLTXV",
              "inputs": {"model": ["2", 0], "max_shift": 2.05,
                         "base_shift": 0.95, "latent": ["5", 0]}},
        # ── conditioning + schedule ───────────────────────────────────────
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                         "frame_rate": fps}},
        "8": {"class_type": "LTXVScheduler",
              "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                         "stretch": True, "terminal": 0.1, "latent": ["5", 0]}},
        # ── sampling ──────────────────────────────────────────────────────
        "9":  {"class_type": "KSamplerSelect",  "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise",      "inputs": {"noise_seed": seed}},
        "11": {"class_type": "CFGGuider",
               "inputs": {"model": ["6", 0], "positive": ["7", 0],
                          "negative": ["7", 1], "cfg": cfg}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["11", 0],
                          "sampler": ["9", 0], "sigmas": ["8", 0],
                          "latent_image": ["5", 0]}},
        # ── decode + save ─────────────────────────────────────────────────
        "13": {"class_type": "VAEDecode",
               "inputs": {"samples": ["12", 0], "vae": ["2", 2]}},
        "14": {"class_type": "SaveAnimatedWEBP",
               "inputs": {"images": ["13", 0],
                          "filename_prefix": "tgbot_video",
                          "fps": fps, "lossless": False,
                          "quality": 85, "method": "default"}},
    }


def _build_workflow_ltxv_i2v(prompt: str, s: dict,
                               t5_name: str, input_filename: str) -> dict:
    """
    LTX-Video image-to-video workflow.

    Extra settings read from `s`:
      video_strength       (float 0.5-1.0)  – how much animation deviates from source
      video_loop           (bool)            – add LTXVAddGuide at last frame = seamless loop
      video_img_compression(int  0-100)      – LTXVPreprocess compression level
    """
    model       = s.get("video_model")           or "ltx-video-2b-v0.9.5.safetensors"
    width       = int(s.get("video_width")       or 480)
    height      = int(s.get("video_height")      or 288)
    frames      = _ltxv_frames(int(s.get("video_frames") or 25))
    fps         = float(s.get("video_fps")       or 8)
    steps       = int(s.get("video_steps")       or 25)
    cfg         = float(s.get("video_cfg")       or 3.5)
    neg         = s.get("video_negative")        or _VIDEO_NEG_DEFAULT
    strength    = float(s.get("video_strength")  or 1.0)
    loop        = bool(s.get("video_loop",  False))
    compression = int(s.get("video_img_compression") or 35)
    seed        = uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF

    # ── base workflow nodes ───────────────────────────────────────────────
    wf: dict = {
        # loaders
        "1": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": t5_name, "type": "ltxv"}},
        "2": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model}},
        # text encoding
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": neg, "clip": ["1", 0]}},
        # LTXV conditioning (adds frame_rate metadata)
        "5": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                         "frame_rate": fps}},
        # load + preprocess input image
        "6": {"class_type": "LoadImage",
              "inputs": {"image": input_filename}},
        "7": {"class_type": "LTXVPreprocess",
              "inputs": {"image": ["6", 0], "img_compression": compression}},
        # img2vid  → outputs: [0]=positive COND, [1]=negative COND, [2]=LATENT
        "8": {"class_type": "LTXVImgToVideo",
              "inputs": {"positive": ["5", 0], "negative": ["5", 1],
                         "vae": ["2", 2], "image": ["7", 0],
                         "width": width, "height": height,
                         "length": frames, "batch_size": 1,
                         "strength": strength}},
    }

    # ── seamless loop: pin last frame = first frame via LTXVAddGuide ──────
    # LTXVAddGuide outputs: [0]=positive COND, [1]=negative COND, [2]=LATENT
    if loop:
        wf["17"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive":  ["8", 0],
                "negative":  ["8", 1],
                "vae":       ["2", 2],
                "latent":    ["8", 2],
                "image":     ["7", 0],   # same preprocessed image = last frame = first frame
                "frame_idx": -1,         # -1 means the very last frame
                "strength":  1.0,        # fully guide last frame to input image
            },
        }
        cond_src  = "17"   # downstream uses guide-aware conditioning
        latent_src = "17"
    else:
        cond_src  = "8"
        latent_src = "8"

    # ── model patching + schedule ─────────────────────────────────────────
    wf["9"]  = {"class_type": "ModelSamplingLTXV",
                "inputs": {"model": ["2", 0], "max_shift": 2.05,
                           "base_shift": 0.95, "latent": [latent_src, 2]}}
    wf["10"] = {"class_type": "LTXVScheduler",
                "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                           "stretch": True, "terminal": 0.1,
                           "latent": [latent_src, 2]}}
    # ── sampling ──────────────────────────────────────────────────────────
    wf["11"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["12"] = {"class_type": "RandomNoise",    "inputs": {"noise_seed": seed}}
    wf["13"] = {"class_type": "CFGGuider",
                "inputs": {"model": ["9", 0],
                           "positive": [cond_src, 0],
                           "negative": [cond_src, 1],
                           "cfg": cfg}}
    wf["14"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["12", 0], "guider": ["13", 0],
                           "sampler": ["11", 0], "sigmas": ["10", 0],
                           "latent_image": [latent_src, 2]}}
    # ── decode + save ─────────────────────────────────────────────────────
    wf["15"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["14", 0], "vae": ["2", 2]}}
    wf["16"] = {"class_type": "SaveAnimatedWEBP",
                "inputs": {"images": ["15", 0],
                           "filename_prefix": "tgbot_video_i2v",
                           "fps": fps, "lossless": False,
                           "quality": 85, "method": "default"}}
    return wf


async def generate_video(
    prompt: str,
    on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    user_settings: Optional[dict] = None,
    input_image: Optional[bytes] = None,
) -> bytes:
    """Generate an animated WEBP video using LTX-Video. Returns animated WEBP bytes."""
    s     = user_settings or {}
    steps = int(s.get("video_steps") or 25)

    clips   = await fetch_clip_models()
    t5_name = await _find_ltxv_t5(clips)

    if input_image is not None:
        w, h = int(s.get("video_width") or 480), int(s.get("video_height") or 288)
        resized = _resize(input_image, w, h)
        fname   = await upload_image(resized, f"tgbot_vid_{uuid.uuid4().hex[:8]}.png")
        workflow = _build_workflow_ltxv_i2v(prompt, s, t5_name, fname)
    else:
        workflow = _build_workflow_ltxv_t2v(prompt, s, t5_name)

    # Use extended timeout for video
    return await _run_comfy_workflow(
        workflow, on_progress, steps,
        poll_timeout=config.VIDEO_POLL_TIMEOUT,
    )


# ── Qwen-Image 2.1 ────────────────────────────────────────────────────────
#
# Text-to-image and reference-image editing. The model does not fit into
# 6 GB VRAM, ComfyUI streams the rest from RAM (DynamicVRAM), so progress
# reporting is richer here: stages, s/step, ETA and live latent previews.

QWEN_QUALITY_PRESETS: dict[str, dict] = {
    "turbo":   {"label": "⚡ Турбо",    "steps": 6,  "lora": True,  "hint": "6 кроків, ~25 с"},
    "quality": {"label": "💎 Якість",   "steps": 25, "lora": False, "hint": "25 кроків, ~1.2 хв"},
    "max":     {"label": "👑 Максимум", "steps": 40, "lora": False, "hint": "40 кроків, ~2 хв"},
}
QWEN_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
    "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
}
QWEN_MEGAPIXELS = [0.5, 1.0, 2.0]

_QWEN_TRANSPARENT_TMPL = (
    "This is an RGBA format image with transparency. {p}. "
    "The image has an alpha channel and a transparent background."
)

# measured seconds per sampling step at 1 MP, refined after every run
_qwen_sec_per_step: dict[str, float] = {"turbo": 2.7, "quality": 2.7, "max": 2.7}


def qwen_size(ratio: str, megapixels: float) -> tuple[int, int]:
    a, b = QWEN_RATIOS.get(ratio, (1, 1))
    area = megapixels * 1024 * 1024
    w = (area * a / b) ** 0.5
    h = area / w
    return max(256, int(round(w / 32)) * 32), max(256, int(round(h / 32)) * 32)


def qwen_settings(s: dict) -> dict:
    """Normalise the user's qwen_* settings into a flat dict with defaults."""
    quality = s.get("qwen_quality") if s.get("qwen_quality") in QWEN_QUALITY_PRESETS else "quality"
    ratio   = s.get("qwen_ratio") if s.get("qwen_ratio") in QWEN_RATIOS else "1:1"
    mp      = float(s.get("qwen_mp") or 1.0)
    if mp not in QWEN_MEGAPIXELS:
        mp = 1.0
    w, h = qwen_size(ratio, mp)
    return {
        "quality":     quality,
        "steps":       QWEN_QUALITY_PRESETS[quality]["steps"],
        "lora":        QWEN_QUALITY_PRESETS[quality]["lora"],
        "ratio":       ratio,
        "mp":          mp,
        "width":       w,
        "height":      h,
        "transparent": bool(s.get("qwen_transparent", False)),
        "translate":   bool(s.get("qwen_translate", True)),
        "seed":        s.get("qwen_seed"),            # None → random
        "negative":    s.get("qwen_negative") or "",
        "cfg":         float(s.get("qwen_cfg") or 1.0),
    }


def qwen_estimate(q: dict) -> float:
    """Rough total seconds for a Qwen job (sampling dominates)."""
    per_step = _qwen_sec_per_step.get(q["quality"], 2.7) * (q["width"] * q["height"]) / (1024 * 1024)
    if q["cfg"] > 1.0:
        per_step *= 2
    return q["steps"] * per_step + 10


def _build_workflow_qwen21(prompt: str, q: dict, seed: int,
                           ref_image: Optional[str] = None) -> dict:
    if q["transparent"]:
        prompt = _QWEN_TRANSPARENT_TMPL.format(p=prompt.rstrip(". "))
    model_src: list = ["1", 0]
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": config.QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": config.QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": config.QWEN_VAE}},
    }
    if q["lora"] and config.QWEN_TURBO_LORA:
        wf["20"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model_src, "lora_name": config.QWEN_TURBO_LORA,
                               "strength_model": 1.0}}
        model_src = ["20", 0]
    wf["4"] = {"class_type": "QwenImage21Cache",
               "inputs": {"model": model_src, "device": "auto", "dtype": "default"}}
    enc = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": q["negative"],
           "resolution": 1024, "vae": ["3", 0]}
    if ref_image:
        wf["10"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        enc["images.image_1"] = ["10", 0]
        latent_src = ["5", 2]          # latent sized from the reference image
    else:
        wf["6"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": q["width"], "height": q["height"], "batch_size": 1}}
        latent_src = ["6", 0]
    wf["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": q["steps"], "cfg": q["cfg"],
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                          "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": latent_src}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "tgbot_qwen", "images": ["8", 0]}}
    return wf


# node id → stage shown to the user
QWEN_STAGES: dict[str, str] = {
    "1": "load", "2": "load", "3": "load", "20": "load", "4": "load", "10": "load",
    "5": "encode", "6": "load", "7": "sample", "8": "decode", "9": "save",
}


def _parse_preview(data: bytes) -> Optional[bytes]:
    """Extract JPEG/PNG bytes from a ComfyUI binary WS frame (types 1 and 4)."""
    if len(data) < 8:
        return None
    etype = int.from_bytes(data[:4], "big")
    if etype == 1:                        # PREVIEW_IMAGE: type, img_format, bytes
        return data[8:]
    if etype == 4:                        # PREVIEW_IMAGE_WITH_METADATA: type, meta_len, meta, bytes
        mlen = int.from_bytes(data[4:8], "big")
        return data[8 + mlen:]
    return None


async def _comfy_ahead(session: aiohttp.ClientSession, prompt_id: str) -> int:
    """How many prompts ComfyUI will run before ours (running + pending ahead); -1 if gone."""
    try:
        r = await session.get(f"{config.COMFY_URL}/queue", timeout=_CONNECT_TIMEOUT)
        q = await r.json()
    except Exception:
        return 0
    running = [it[1] for it in q.get("queue_running", [])]
    if prompt_id in running:
        return 0
    pending = sorted(q.get("queue_pending", []), key=lambda it: it[0])
    ids = [it[1] for it in pending]
    if prompt_id not in ids:
        return -1                      # not queued any more: finished, or deleted by ⏹
    return len(running) + ids.index(prompt_id)


async def run_workflow_rich(
    workflow: dict,
    stages: dict[str, str],
    steps_hint: int,
    eta: float,
    on_status: Optional[Callable[[dict], Awaitable[None]]] = None,
    timeout: Optional[float] = None,
    output_key: str = "images",
) -> bytes:
    """Submit a workflow and stream rich status until the first output file arrives.

    stages maps node id → stage name. on_status receives a dict: stage, step, total,
    elapsed, eta, sec_per_step, preview (latest preview JPEG or None), stage_step /
    stage_total (progress of non-sampling nodes, e.g. an LLM), prompt_id and
    comfy_ahead (prompts ComfyUI runs before ours while we wait). It is called on
    every event and on a 2 s heartbeat — throttling is the caller's job.
    """
    client_id = str(uuid.uuid4())
    ws_url    = config.COMFY_URL.replace("http://", "ws://").replace("https://", "wss://")
    t0        = time.monotonic()
    state     = {"stage": "queue", "step": 0, "total": steps_hint, "elapsed": 0.0, "eta": eta,
                 "sec_per_step": None, "preview": None, "stage_step": 0, "stage_total": 0,
                 "prompt_id": None, "comfy_ahead": 0}
    marks     = {"started": None, "sample": None}   # monotonic times of execution / sampling start

    async def emit() -> None:
        if on_status is None:
            return
        state["elapsed"] = time.monotonic() - t0
        try:
            await on_status(dict(state))
        except Exception:
            pass  # never let a UI error break generation

    async def heartbeat(session: aiohttp.ClientSession, prompt_id: str, ws) -> None:
        missing = 0
        while True:
            await asyncio.sleep(2.0)
            if state["stage"] == "queue" and marks["started"] is None:
                ahead = await _comfy_ahead(session, prompt_id)
                missing = missing + 1 if ahead < 0 else 0
                if missing >= 2:       # deleted from the queue before it started
                    marks["gone"] = True
                    await ws.close()
                    return
                state["comfy_ahead"] = max(ahead, 0)
            elif state["stage"] != "sample" and marks["started"] is not None:
                state["eta"] = max(3.0, eta - (time.monotonic() - marks["started"]))
            await emit()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"{ws_url}/ws?clientId={client_id}",
            timeout=aiohttp.ClientTimeout(total=timeout or config.QWEN_POLL_TIMEOUT),
            max_msg_size=0,
        ) as ws:
            resp = await session.post(f"{config.COMFY_URL}/prompt",
                                      json={"prompt": workflow, "client_id": client_id})
            if not resp.ok:
                raise RuntimeError(f"ComfyUI {resp.status}: {await resp.text()}")
            prompt_id: str = (await resp.json())["prompt_id"]
            state["prompt_id"] = prompt_id
            state["comfy_ahead"] = max(await _comfy_ahead(session, prompt_id), 0)
            await emit()
            hb = asyncio.create_task(heartbeat(session, prompt_id, ws))
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        img = _parse_preview(msg.data)
                        if img and state["stage"] == "sample":
                            state["preview"] = img
                            await emit()
                        continue
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    event = json.loads(msg.data)
                    etype = event.get("type")
                    edata = event.get("data", {})
                    if edata.get("prompt_id") != prompt_id and etype != "status":
                        continue

                    if etype in ("execution_start", "execution_cached") and marks["started"] is None:
                        marks["started"] = time.monotonic()
                        if state["stage"] == "queue":
                            state["stage"] = "load"
                        await emit()

                    elif etype == "executing" and edata.get("node"):
                        if marks["started"] is None:
                            marks["started"] = time.monotonic()
                        stage = stages.get(str(edata["node"]), state["stage"]) if stages else state["stage"]
                        if stage == "sample" and marks["sample"] is None:
                            marks["sample"] = time.monotonic()
                        if stage != state["stage"]:
                            state["stage_step"] = state["stage_total"] = 0
                        state["stage"] = stage
                        await emit()

                    elif etype == "progress":
                        step, total = edata.get("value", 0), edata.get("max", steps_hint)
                        if not stages and state["stage"] == "queue":
                            state["stage"] = "sample"          # legacy callers pass no stage map
                        if state["stage"] == "sample":
                            state["step"], state["total"] = step, total
                            if marks["sample"] is not None and step > 0:
                                sps = (time.monotonic() - marks["sample"]) / step
                                state["sec_per_step"] = sps
                                state["eta"] = sps * (total - step) + 5
                        else:
                            state["stage_step"], state["stage_total"] = step, total
                        await emit()

                    elif etype == "executed":
                        texts = edata.get("output", {}).get("text")
                        if texts:
                            state["texts"] = list(texts)
                            if output_key == "text":
                                run_workflow_rich.last_state = dict(state)
                                return "\n".join(texts).encode()
                        for item in edata.get("output", {}).get(output_key, []):
                            if item.get("type") != "output":
                                continue
                            r = await session.get(
                                f"{config.COMFY_URL}/view",
                                params={"filename": item["filename"],
                                        "subfolder": item.get("subfolder", ""), "type": "output"},
                            )
                            r.raise_for_status()
                            state["done_sec_per_step"] = state["sec_per_step"]
                            run_workflow_rich.last_state = dict(state)
                            data = await r.read()
                            _safe_unlink(config.COMFY_OUTPUT_DIR, item.get("subfolder", ""), item["filename"])
                            return data

                    elif etype == "execution_error":
                        raise RuntimeError(f"ComfyUI: {edata.get('exception_message', 'execution error')}")

                    elif etype == "execution_interrupted":
                        raise RuntimeError("ComfyUI: генерацію перервано")
            finally:
                hb.cancel()
                await _cleanup_run(session, prompt_id)
            if marks.get("gone"):
                raise RuntimeError("ComfyUI: генерацію перервано")
            raise RuntimeError("WebSocket closed unexpectedly")

    raise RuntimeError("Workflow completed without output")


async def generate_qwen(
    prompt: str,
    q: dict,
    on_status: Optional[Callable[[dict], Awaitable[None]]] = None,
    input_image: Optional[bytes] = None,
) -> tuple[bytes, int]:
    """Run Qwen-Image 2.1. Returns (png_bytes, seed)."""
    seed = int(q["seed"]) if q.get("seed") is not None else uuid.uuid4().int & 0xFFFFFFFFFFFF
    ref_name = None
    if input_image is not None:
        ref_name = await upload_image(_to_png(input_image), f"tgbot_qref_{uuid.uuid4().hex[:8]}.png")
    workflow = _build_workflow_qwen21(prompt, q, seed, ref_name)
    data = await run_workflow_rich(workflow, QWEN_STAGES, q["steps"], qwen_estimate(q),
                                   on_status, config.QWEN_POLL_TIMEOUT, "images")
    sps = run_workflow_rich.last_state.get("done_sec_per_step")
    if sps:
        # refine the 1 MP estimate for the next ETA
        norm = sps * (1024 * 1024) / (q["width"] * q["height"])
        if q["cfg"] > 1.0:
            norm /= 2
        old = _qwen_sec_per_step.get(q["quality"], norm)
        _qwen_sec_per_step[q["quality"]] = old * 0.5 + norm * 0.5
    return data, seed


def _to_png(data: bytes) -> bytes:
    img = Image.open(BytesIO(data))
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


async def interrupt(prompt_id: Optional[str] = None) -> None:
    """Stop one prompt (running or queued). Without prompt_id — global interrupt."""
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            if prompt_id:
                await s.post(f"{config.COMFY_URL}/queue", json={"delete": [prompt_id]})
                await s.post(f"{config.COMFY_URL}/interrupt", json={"prompt_id": prompt_id})
            else:
                await s.post(f"{config.COMFY_URL}/interrupt")
    except Exception:
        pass


# ── Music & sound: ACE-Step 1.5 (songs) + Stable Audio 3 (sounds) ────────

SONG_DURATIONS = [30, 60, 90, 120, 180, 240]
SONG_LANGUAGES = {"uk": "🇺🇦 Українська", "en": "🇬🇧 English", "pl": "🇵🇱 Polski", "de": "🇩🇪 Deutsch",
                  "es": "🇪🇸 Español", "fr": "🇫🇷 Français", "it": "🇮🇹 Italiano", "ja": "🇯🇵 日本語",
                  "ko": "🇰🇷 한국어", "zh": "🇨🇳 中文"}
SONG_BPMS      = [70, 90, 100, 110, 120, 128, 140, 170]
SONG_KEYS      = ["C major", "G major", "D major", "A major", "E major", "F major",
                  "A minor", "E minor", "D minor", "B minor", "F# minor", "C minor"]
SONG_LMS       = {"1.7b": "qwen_1.7b_ace15.safetensors", "4b": "qwen_4b_ace15.safetensors"}
SONG_INSTRUMENTAL = "[Instrumental]"

SOUND_DURATIONS  = [3, 5, 10, 20, 30, 60, 120]
SOUND_CATEGORIES = {
    "sfx":        ("💥 Звуковий ефект", "Sound effect: {p}"),
    "oneshot":    ("🥁 One-shot семпл",  "One-shot sample, single hit: {p}"),
    "instrument": ("🎻 Інструмент",      "Solo instrument recording: {p}"),
    "ambient":    ("🌲 Атмосфера",       "Ambient soundscape: {p}"),
    "music":      ("🎶 Музика (інструментал)", "{p}"),
}
SOUND_MODELS = {
    "small_sfx": ("⚡ Small SFX — швидка, для ефектів", "stable_audio_3_small_sfx.safetensors"),
    "medium":    ("💎 Medium — універсальна, для музики й атмосфери", "stable_audio_3_medium.safetensors"),
}

SONG_STAGES  = {"1": "load", "2": "load", "3": "load", "4": "load", "7": "load",
                "5": "compose", "6": "compose", "8": "sample", "9": "decode", "10": "save"}
SOUND_STAGES = {"1": "load", "2": "load", "5": "load", "3": "encode", "4": "encode",
                "6": "sample", "7": "decode", "8": "save"}


def song_settings(s: dict) -> dict:
    dur  = int(s.get("song_duration") or 60)
    lang = s.get("song_language") if s.get("song_language") in SONG_LANGUAGES else "uk"
    bpm  = int(s.get("song_bpm") or 120)
    key  = s.get("song_key") if s.get("song_key") in SONG_KEYS else "C major"
    lm   = s.get("song_lm") if s.get("song_lm") in SONG_LMS else "1.7b"
    return {"duration": dur if dur in SONG_DURATIONS else 60, "language": lang, "bpm": bpm,
            "key": key, "timesig": "3" if s.get("song_timesig") == "3" else "4",
            "lm": lm, "codes": bool(s.get("song_codes", True)), "seed": s.get("song_seed")}


def sound_settings(s: dict) -> dict:
    cat = s.get("sound_category") if s.get("sound_category") in SOUND_CATEGORIES else "sfx"
    dur = int(s.get("sound_duration") or 10)
    mdl = s.get("sound_model") if s.get("sound_model") in SOUND_MODELS else "auto"
    if mdl == "auto":
        mdl = "small_sfx" if cat in ("sfx", "oneshot") else "medium"
    return {"category": cat, "duration": dur if dur in SOUND_DURATIONS else 10,
            "model": mdl, "model_setting": s.get("sound_model") or "auto", "seed": s.get("sound_seed")}


def song_estimate(m: dict) -> float:
    lm = 0.08 if m["lm"] == "1.7b" else 1.15          # seconds per audio-code token (5 per sec of audio), RTX 3050
    return 25 + (m["duration"] * 5 * lm if m["codes"] else 0) + m["duration"] * 0.1


def sound_estimate(m: dict) -> float:
    return 12 + m["duration"] * (0.15 if m["model"] == "small_sfx" else 0.4)


def _build_workflow_song(tags: str, lyrics: str, m: dict, seed: int) -> dict:
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": config.ACE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": config.ACE_CLIP, "clip_name2": SONG_LMS[m["lm"]],
                         "type": "ace", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": config.ACE_VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "5": {"class_type": "TextEncodeAceStepAudio1.5",
              "inputs": {"clip": ["2", 0], "tags": tags, "lyrics": lyrics, "seed": seed,
                         "bpm": m["bpm"], "duration": float(m["duration"]),
                         "timesignature": m["timesig"], "language": m["language"],
                         "keyscale": m["key"], "generate_audio_codes": m["codes"],
                         "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyAceStep1.5LatentAudio",
              "inputs": {"seconds": float(m["duration"]), "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0, "model": ["4", 0],
                         "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["9", 0], "filename_prefix": "audio/tgbot_song", "quality": "V0"}},
    }


def _build_workflow_sound(prompt: str, m: dict, seed: int) -> dict:
    text = SOUND_CATEGORIES[m["category"]][1].format(p=prompt)
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": SOUND_MODELS[m["model"]][1]}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": config.SA3_CLIP, "type": "stable_audio", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "5": {"class_type": "EmptyLatentAudio",
              "inputs": {"seconds": float(m["duration"]), "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "lcm",
                         "scheduler": "simple", "denoise": 1.0, "model": ["1", 0],
                         "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudioMP3",
              "inputs": {"audio": ["7", 0], "filename_prefix": "audio/tgbot_sound", "quality": "V0"}},
    }


async def generate_song(tags: str, lyrics: str, m: dict,
                        on_status: Optional[Callable[[dict], Awaitable[None]]] = None
                        ) -> tuple[bytes, int]:
    seed = int(m["seed"]) if m.get("seed") is not None else uuid.uuid4().int & 0xFFFFFFFF
    wf = _build_workflow_song(tags, lyrics or SONG_INSTRUMENTAL, m, seed)
    data = await run_workflow_rich(wf, SONG_STAGES, 8, song_estimate(m), on_status,
                                   config.QWEN_POLL_TIMEOUT, "audio")
    return data, seed


async def generate_sound(prompt: str, m: dict,
                         on_status: Optional[Callable[[dict], Awaitable[None]]] = None
                         ) -> tuple[bytes, int]:
    seed = int(m["seed"]) if m.get("seed") is not None else uuid.uuid4().int & 0xFFFFFFFF
    wf = _build_workflow_sound(prompt, m, seed)
    data = await run_workflow_rich(wf, SOUND_STAGES, 8, sound_estimate(m), on_status,
                                   config.QWEN_POLL_TIMEOUT, "audio")
    return data, seed


# ── Voice: OmniVoice TTS / cloning / speech-to-speech (custom node comfyui_voice) ──

VOICE_LANGUAGES = {"auto": "🌐 Авто", "uk": "🇺🇦 Українська", "en": "🇬🇧 English", "pl": "🇵🇱 Polski",
                   "de": "🇩🇪 Deutsch", "fr": "🇫🇷 Français", "es": "🇪🇸 Español", "it": "🇮🇹 Italiano",
                   "cs": "🇨🇿 Čeština", "ja": "🇯🇵 日本語", "zh": "🇨🇳 中文"}
VOICE_SPEEDS    = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
VOICE_QUALITY   = {"fast": ("⚡ Швидко", 16), "std": ("⚖️ Стандарт", 32), "best": ("💎 Найкраще", 48)}
# built-in designed voices: fixed seed → the same voice every time
VOICE_PRESETS: dict[str, tuple[str, str, int]] = {
    "f_young":  ("👩 Жінка, молода",       "female, young adult, moderate pitch", 1101),
    "f_warm":   ("👩‍🦰 Жінка, низький тембр", "female, middle-aged, low pitch",      1202),
    "m_young":  ("👨 Чоловік, молодий",     "male, young adult, moderate pitch",   2101),
    "m_deep":   ("🧔 Чоловік, глибокий",    "male, middle-aged, very low pitch",   2202),
    "granny":   ("👵 Бабуся",               "female, elderly, moderate pitch",     3101),
    "grandpa":  ("👴 Дідусь",               "male, elderly, low pitch",            3202),
    "kid":      ("🧒 Дитина",               "child, high pitch",                   4101),
    "whisper":  ("🤫 Шепіт",                "female, young adult, whisper",        5101),
}
VOICE_STAGES = {"1": "load", "2": "load", "3": "encode", "4": "sample", "5": "save"}
VC_PITCHES   = [-12, -7, -4, -2, 0, 2, 4, 7, 12]
VC_STAGES    = {"2": "load", "1": "encode", "6": "load", "4": "sample", "5": "save"}
_VC_SAMPLE_TEXT = ("Привіт! Це зразок мого голосу. Я говорю спокійно і рівно, "
                   "щоб тембр було добре чутно, а інтонація звучала природно.")


def voice_settings(s: dict) -> dict:
    lang = s.get("voice_language") if s.get("voice_language") in VOICE_LANGUAGES else "uk"
    spd  = float(s.get("voice_speed") or 1.0)
    ql   = s.get("voice_quality") if s.get("voice_quality") in VOICE_QUALITY else "std"
    pitch = int(s.get("vc_pitch") or 0)
    return {"language": lang, "speed": spd if spd in VOICE_SPEEDS else 1.0, "quality": ql,
            "num_step": VOICE_QUALITY[ql][1], "current": s.get("voice_current") or "preset:f_young",
            "vc_method": "text" if s.get("vc_method") == "text" else "intonation",
            "vc_singing": bool(s.get("vc_singing", False)),
            "vc_pitch": pitch if pitch in VC_PITCHES else 0}


def voice_estimate(chars: int, m: dict, s2s: bool = False, cloned: bool = False) -> float:
    # measured on RTX 3050: ~0.028 s per character at 32 steps (≈35 chars/s) with a built-in
    # voice; a cloned voice prepends its reference to every fragment (≈2.5× slower, measured
    # on a 43k-character lecture: 79 min at 48 steps)
    per_char = 0.028 * m["num_step"] / 32 * (2.5 if cloned else 1.0)
    return 10 + chars * per_char + (8 if s2s else 0)


def _opus_bitrate(chars: int) -> str:
    """Keep long narrations under Telegram's 50 MB bot upload limit (~1 MB per minute at 128k)."""
    return "128k" if chars <= 12000 else "96k" if chars <= 20000 else "64k" if chars <= 45000 else "48k"


def _tts_node(text_src, m: dict, seed: int, ref: Optional[list], ref_text: str, instruct: str) -> dict:
    inputs = {"text": text_src, "language": m["language"], "speed": m["speed"],
              "num_step": m["num_step"], "seed": seed, "ref_text": ref_text or "", "instruct": instruct or ""}
    if ref is not None:
        inputs["ref_audio"] = ref
    return {"class_type": "OmniVoiceTTS", "inputs": inputs}


async def _upload_audio(data: bytes, name: str) -> str:
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("image", data, filename=name, content_type="application/octet-stream")
        form.add_field("overwrite", "true")
        r = await session.post(f"{config.COMFY_URL}/upload/image", data=form)
        r.raise_for_status()
        name = (await r.json())["name"]
        _register_upload(name)
        return name


async def generate_speech(
    text: str, m: dict, voice: dict,
    on_status: Optional[Callable[[dict], Awaitable[None]]] = None,
    source_audio: Optional[bytes] = None,
) -> tuple[bytes, str]:
    """TTS (text) or speech-to-speech (source_audio). voice: {"ref": bytes|None, "ref_text", "instruct", "seed"}.

    Returns (ogg_opus_bytes, spoken_text). For speech-to-speech spoken_text is the Whisper transcript.
    """
    wf: dict = {}
    ref = None
    if voice.get("ref"):
        name = await _upload_audio(voice["ref"], f"tgbot_voice_{uuid.uuid4().hex[:8]}.ogg")
        wf["1"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        ref = ["1", 0]
    if source_audio is not None:
        src = await _upload_audio(source_audio, f"tgbot_s2s_{uuid.uuid4().hex[:8]}.ogg")
        wf["2"] = {"class_type": "LoadAudio", "inputs": {"audio": src}}
        wf["3"] = {"class_type": "OmniVoiceTranscribe", "inputs": {"audio": ["2", 0]}}
        text_src: object = ["3", 0]
    else:
        text_src = text
    seed = int(voice.get("seed") or (uuid.uuid4().int & 0xFFFFFFFF))
    wf["4"] = _tts_node(text_src, m, seed, ref, voice.get("ref_text", ""), voice.get("instruct", ""))
    if voice.get("lora"):
        wf["4"]["inputs"]["lora"] = voice["lora"]
    wf["5"] = {"class_type": "SaveAudioOpus",
               "inputs": {"audio": ["4", 0], "filename_prefix": "audio/tgbot_voice",
                          "quality": _opus_bitrate(len(text or ""))}}
    eta = voice_estimate(len(text) if text else 200, m, source_audio is not None, bool(voice.get("ref")))
    data = await run_workflow_rich(wf, VOICE_STAGES, 1, eta, on_status,
                                   max(config.QWEN_POLL_TIMEOUT, eta * 3), "audio")
    spoken = "\n".join(run_workflow_rich.last_state.get("texts") or []) or text
    return data, spoken


async def transcribe_audio(data: bytes,
                           on_status: Optional[Callable[[dict], Awaitable[None]]] = None) -> str:
    name = await _upload_audio(data, f"tgbot_asr_{uuid.uuid4().hex[:8]}.ogg")
    wf = {"2": {"class_type": "LoadAudio", "inputs": {"audio": name}},
          "3": {"class_type": "OmniVoiceTranscribe", "inputs": {"audio": ["2", 0]}}}
    raw = await run_workflow_rich(wf, VOICE_STAGES, 1, 15, on_status, config.QWEN_POLL_TIMEOUT, "text")
    return raw.decode().strip()


async def convert_voice(
    source_audio: bytes, m: dict, voice: dict,
    on_status: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> bytes:
    """Seed-VC: the same performance (intonation, timing, emotion) in the target voice.

    Target = the cloned voice sample, or — for built-in voices — a sample synthesised by
    OmniVoice in the same workflow. Returns OGG/Opus bytes.
    """
    src = await _upload_audio(source_audio, f"tgbot_vc_{uuid.uuid4().hex[:8]}.ogg")
    wf: dict = {"2": {"class_type": "LoadAudio", "inputs": {"audio": src}}}
    if voice.get("ref"):
        # Seed-VC benefits from a longer (~25 s) reference than OmniVoice (~10 s)
        tgt = await _upload_audio(voice.get("ref_vc") or voice["ref"], f"tgbot_vct_{uuid.uuid4().hex[:8]}.ogg")
        wf["6"] = {"class_type": "LoadAudio", "inputs": {"audio": tgt}}
        target = ["6", 0]
    else:
        wf["1"] = _tts_node(_VC_SAMPLE_TEXT, dict(m, language="uk", speed=1.0), int(voice.get("seed") or 1),
                            None, "", voice.get("instruct", ""))
        target = ["1", 0]
    wf["4"] = {"class_type": "SeedVCConvert",
               "inputs": {"source": ["2", 0], "target": target,
                          "mode": "singing" if m["vc_singing"] else "speech",
                          "diffusion_steps": {"fast": 15, "std": 30, "best": 50}[m["quality"]],
                          "pitch_shift": m["vc_pitch"], "cfg_rate": 0.7}}
    wf["5"] = {"class_type": "SaveAudioOpus",
               "inputs": {"audio": ["4", 0], "filename_prefix": "audio/tgbot_vc", "quality": "128k"}}
    # source loading and sample synthesis run in arbitrary order: show them as one stage for presets
    stages = VC_STAGES if voice.get("ref") else dict(VC_STAGES, **{"2": "encode"})
    return await run_workflow_rich(wf, stages, 1, 45, on_status, config.QWEN_POLL_TIMEOUT, "audio")


async def select_voice_ref(data: bytes, seconds: float) -> bytes:
    """Cut the cleanest `seconds`-long stretch of speech out of a long recording (OGG/Opus)."""
    name = await _upload_audio(data, f"tgbot_ref_{uuid.uuid4().hex[:8]}.ogg")
    wf = {"2": {"class_type": "LoadAudio", "inputs": {"audio": name}},
          "4": {"class_type": "VoiceRefSelect", "inputs": {"audio": ["2", 0], "seconds": float(seconds)}},
          "5": {"class_type": "SaveAudioOpus",
                "inputs": {"audio": ["4", 0], "filename_prefix": "audio/tgbot_ref", "quality": "128k"}}}
    return await run_workflow_rich(wf, VOICE_STAGES, 1, 10, None, 300, "audio")


# ── personal voice model (LoRA finetune of OmniVoice) ─────────────────────
# Training runs inside the ComfyUI container as a separate process (voice_lora.py).

LORA_HOST_ROOT = os.getenv("VOICE_LORA_HOST_ROOT", "/mnt/docker/comfyui/models/omnivoice")
LORA_CT_ROOT   = "/app/models/omnivoice"          # the same folder as seen inside the container
COMFY_CONTAINER = os.getenv("COMFY_CONTAINER", "comfyui")


def lora_steps(seconds: float) -> int:
    return int(min(1500, max(300, seconds * 3)))


def lora_estimate(seconds: float) -> float:
    return 90 + seconds * 0.6 + lora_steps(seconds) * 2.6     # prep + ASR + ~2.6 s/step on RTX 3050


async def free_comfy_vram() -> None:
    try:
        async with aiohttp.ClientSession(timeout=_CONNECT_TIMEOUT) as s:
            await s.post(f"{config.COMFY_URL}/free", json={"unload_models": True, "free_memory": True})
    except Exception:
        pass


async def train_voice_lora(takes: list[bytes], name: str, lang: str,
                           on_progress: Optional[Callable[[dict], Awaitable[None]]] = None) -> dict:
    """Finetune a personal voice adapter. Returns {"lora": <container path>, "seconds", "clips", "steps"}."""
    import shutil
    data_host = os.path.join(LORA_HOST_ROOT, "lora_data", name)
    out_ct    = f"{LORA_CT_ROOT}/lora/{name}"
    shutil.rmtree(data_host, ignore_errors=True)
    os.makedirs(data_host, exist_ok=True)
    paths = []
    for i, blob in enumerate(takes):
        p = os.path.join(data_host, f"take_{i:02d}.ogg")
        with open(p, "wb") as f:
            f.write(blob)
        paths.append(f"{LORA_CT_ROOT}/lora_data/{name}/take_{i:02d}.ogg")
    await free_comfy_vram()
    cmd = ["docker", "exec", COMFY_CONTAINER, "python", "/app/custom_nodes/comfyui_voice/voice_lora.py",
           "--audio", *paths, "--lang", lang if lang and lang != "auto" else "uk", "--out", out_ct]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    result, tail = None, []
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            tail = (tail + [line])[-15:]
            if line.startswith("PROGRESS "):
                ev = json.loads(line[9:])
                if ev.get("stage") == "done":
                    result = ev
                if on_progress:
                    try:
                        await on_progress(ev)
                    except Exception:
                        pass
        rc = await proc.wait()
    except asyncio.CancelledError:
        await stop_voice_lora()
        raise
    finally:
        shutil.rmtree(data_host, ignore_errors=True)
    if rc != 0 or result is None:
        reason = next((l for l in reversed(tail) if l and not l.startswith("PROGRESS")), f"exit {rc}")
        raise RuntimeError(f"навчання не вдалося: {reason[:300]}")
    return {"lora": out_ct, "seconds": result.get("seconds"), "clips": result.get("clips"),
            "steps": result.get("steps")}


async def stop_voice_lora() -> None:
    proc = await asyncio.create_subprocess_exec("docker", "exec", COMFY_CONTAINER, "pkill", "-f", "voice_lora",
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()


def delete_voice_lora(ct_path: str) -> None:
    import shutil
    if not ct_path or not ct_path.startswith(f"{LORA_CT_ROOT}/lora/"):
        return
    host = os.path.join(LORA_HOST_ROOT, os.path.relpath(ct_path, LORA_CT_ROOT))
    if os.path.realpath(host).startswith(os.path.realpath(os.path.join(LORA_HOST_ROOT, "lora")) + os.sep):
        shutil.rmtree(host, ignore_errors=True)
