import asyncio
import json
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
        return (await r.json())["name"]


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
) -> bytes:
    """Submit a workflow to ComfyUI via WebSocket, wait for output, return image bytes."""
    client_id = str(uuid.uuid4())
    ws_url    = config.COMFY_URL.replace("http://", "ws://").replace("https://", "wss://")
    last_upd  = 0.0

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"{ws_url}/ws?clientId={client_id}",
            timeout=aiohttp.ClientTimeout(total=config.POLL_TIMEOUT),
        ) as ws:
            resp = await session.post(
                f"{config.COMFY_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if not resp.ok:
                raise RuntimeError(f"ComfyUI {resp.status}: {await resp.text()}")
            prompt_id: str = (await resp.json())["prompt_id"]

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    etype = event.get("type")
                    edata = event.get("data", {})

                    if etype == "progress" and on_progress:
                        now = time.monotonic()
                        if now - last_upd >= 0.8:
                            last_upd = now
                            try:
                                await on_progress(edata.get("value", 0), edata.get("max", steps_hint))
                            except Exception:
                                pass  # never let a progress callback error break generation

                    elif etype == "executed" and edata.get("prompt_id") == prompt_id:
                        for img in edata.get("output", {}).get("images", []):
                            if img.get("type") == "output":
                                r = await session.get(
                                    f"{config.COMFY_URL}/view",
                                    params={"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": "output"},
                                )
                                r.raise_for_status()
                                return await r.read()

                    elif etype == "execution_error" and edata.get("prompt_id") == prompt_id:
                        raise RuntimeError(f"ComfyUI: {edata.get('exception_message', 'execution error')}")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise RuntimeError("WebSocket closed unexpectedly")

    raise RuntimeError("Workflow completed without output image")


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
