"""
ComfyUI nodes for OmniVoice (k2-fsa): multilingual TTS (600+ languages incl. Ukrainian),
zero-shot voice cloning, voice design, and Whisper transcription.

The model lives in RAM between calls and is moved to the GPU only while a node runs.
Before that, ComfyUI is asked to unload its own models, so a 6 GB card can alternate
between image/music models and speech without running out of VRAM.
"""
import hashlib
import logging
import os
import re
import threading

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

log = logging.getLogger("comfyui_voice")

REPO      = os.environ.get("OMNIVOICE_REPO", "k2-fsa/OmniVoice")
ASR_REPO  = os.environ.get("OMNIVOICE_ASR", "openai/whisper-large-v3-turbo")
CACHE_DIR = os.path.join(folder_paths.models_dir, "omnivoice")

LANGUAGES = ["auto", "uk", "en", "pl", "de", "fr", "es", "it", "pt", "cs", "sk", "bg", "be",
             "ro", "hu", "tr", "ja", "ko", "zh", "ar", "he", "hi", "ka", "kk", "lt", "lv", "et", "ru"]

_lock = threading.Lock()
_model = None
_peft = None                      # PeftModel wrapper once a personal voice (LoRA) has been used
_adapters: set[str] = set()
_prompt_cache: dict[str, object] = {}
LORA_DIR = os.path.join(CACHE_DIR, "lora")


def _adapter_name(path: str) -> str:
    return "v_" + hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:12]


def _activate_lora(model, path: str):
    """Load (once) and activate a personal-voice LoRA on top of the shared base model.
    Returns the OmniVoice module to call generate() on."""
    global _peft
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(LORA_DIR) + os.sep) or \
            not os.path.exists(os.path.join(real, "adapter_config.json")):
        raise ValueError(f"LoRA adapter not found under {LORA_DIR}: {path}")
    from peft import PeftModel
    name = _adapter_name(real)
    if _peft is None:
        _peft = PeftModel.from_pretrained(model, real, adapter_name=name)
        _adapters.add(name)
    elif name not in _adapters:
        _peft.load_adapter(real, adapter_name=name)
        _adapters.add(name)
    _peft.set_adapter(name)
    # adapters are trained/saved in fp32; the shared base runs in fp16
    for n, p in _peft.named_parameters():
        if p.dtype == torch.float32 and ("lora_" in n or "modules_to_save" in n):
            p.data = p.data.to(torch.float16)
    return _peft.base_model.model


def _load():
    global _model
    if _model is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.environ.setdefault("HF_HOME", CACHE_DIR)
        from omnivoice import OmniVoice
        log.info("Loading OmniVoice from %s", REPO)
        _model = OmniVoice.from_pretrained(REPO, device_map="cpu", dtype=torch.float16,
                                           asr_model_name=ASR_REPO, cache_dir=CACHE_DIR)
    return _model


def _move(model, device) -> None:
    model.to(device)
    tok = getattr(model, "audio_tokenizer", None)
    if tok is not None and hasattr(tok, "to"):
        tok.to(device)
    pipe = getattr(model, "_asr_pipe", None)
    if pipe is not None:
        pipe.model.to(device)
        pipe.device = torch.device(device)
    model._asr_device = device


class _OnGPU:
    """Context: free ComfyUI VRAM, move OmniVoice to GPU, move it back to RAM afterwards."""

    def __enter__(self):
        _lock.acquire()
        self.dev = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        self.model = _load()
        _move(self.model, self.dev)
        return self.model

    def __exit__(self, *exc):
        try:
            _move(self.model, "cpu")
        finally:
            mm.soft_empty_cache()
            _lock.release()
        return False


def _to_mono(audio: dict) -> tuple[torch.Tensor, int]:
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.float().cpu(), int(audio["sample_rate"])


def _audio_key(wav: torch.Tensor, sr: int, text: str) -> str:
    h = hashlib.sha1(wav.numpy().tobytes())
    h.update(f"{sr}|{text}".encode())
    return h.hexdigest()


def _chunks(text: str, limit: int = 280) -> list[str]:
    """Split long text on sentence boundaries so progress can be reported per chunk."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?…;:])\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


class OmniVoiceTTS:
    """Text → speech. With ref_audio: voice cloning. With instruct: voice design."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":     ("STRING", {"multiline": True, "default": ""}),
                "language": (LANGUAGES, {"default": "uk"}),
                "speed":    ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "num_step": ("INT", {"default": 32, "min": 4, "max": 64}),
                "seed":     ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_text":  ("STRING", {"multiline": True, "default": ""}),
                "instruct":  ("STRING", {"multiline": False, "default": ""}),
                "lora":      ("STRING", {"multiline": False, "default": "",
                                         "tooltip": "personal voice adapter dir under models/omnivoice/lora"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/omnivoice"

    def run(self, text, language, speed, num_step, seed, ref_audio=None, ref_text="", instruct="", lora=""):
        chunks = _chunks(text)
        if not chunks:
            raise ValueError("Empty text")
        pbar = comfy.utils.ProgressBar(len(chunks) + 1)
        lang = None if language == "auto" else language
        with _OnGPU() as base:
            import contextlib
            if lora:
                model = _activate_lora(base, lora)
                ctx = contextlib.nullcontext()
            else:
                model = base
                ctx = _peft.disable_adapter() if _peft is not None else contextlib.nullcontext()
            with ctx:
                return self._generate(model, chunks, lang, speed, num_step, seed, ref_audio, ref_text,
                                      instruct, pbar, lora)

    def _generate(self, model, chunks, lang, speed, num_step, seed, ref_audio, ref_text, instruct, pbar, lora):
        if True:
            prompt = None
            if ref_audio is not None:
                wav, sr = _to_mono(ref_audio)
                key = _audio_key(wav, sr, (ref_text or "") + "|" + (lora or ""))
                prompt = _prompt_cache.get(key)
                if prompt is None:
                    if not ref_text and model._asr_pipe is None:
                        model.load_asr_model(device=str(mm.get_torch_device()))
                    prompt = model.create_voice_clone_prompt((wav, sr), ref_text=ref_text or None)
                    _prompt_cache[key] = prompt
            pbar.update(1)
            torch.manual_seed(seed)
            pieces = []
            gap = np.zeros(int(model.sampling_rate * 0.25), dtype=np.float32)
            for i, chunk in enumerate(chunks):
                kw = {"text": chunk, "language": lang, "speed": speed, "num_step": num_step}
                if prompt is not None:
                    kw["voice_clone_prompt"] = prompt
                if instruct:
                    kw["instruct"] = instruct
                audio = model.generate(**kw)[0]
                pieces.append(np.asarray(audio, dtype=np.float32))
                if i < len(chunks) - 1:
                    pieces.append(gap)
                pbar.update(1)
            out = np.concatenate(pieces)
        waveform = torch.from_numpy(out).float().view(1, 1, -1)
        return ({"waveform": waveform, "sample_rate": model.sampling_rate},)


class OmniVoiceTranscribe:
    """Speech → text with Whisper (large-v3-turbo). Also shown in the UI/API output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "audio/omnivoice"

    def run(self, audio):
        wav, sr = _to_mono(audio)
        with _OnGPU() as model:
            if model._asr_pipe is None:
                model.load_asr_model(device=str(mm.get_torch_device()))
            text = model.transcribe((wav, sr))
        return {"ui": {"text": [text]}, "result": (text,)}


# ── Seed-VC: voice conversion that keeps the source intonation ───────────
# Seed-VC (GPL-3.0, github.com/Plachtaa/seed-vc) is not vendored: a checkout pinned to a
# reviewed commit is mounted at SEEDVC_DIR. Its checkpoints are cached in /app/checkpoints.

SEEDVC_DIR = os.environ.get("SEEDVC_DIR", "/opt/seed-vc")


def _stub_dac() -> None:
    """Seed-VC imports dac.nn.quantize.VectorQuantize at module level, but the shipped
    configs use vector_quantize: false, so it is never instantiated. A stub avoids pulling
    descript-audio-codec and its heavy dependency tree into the ComfyUI environment."""
    import sys
    import types
    if "dac" in sys.modules:
        return
    try:
        import dac  # noqa: F401  (real package, if someone installed it)
        return
    except ImportError:
        pass

    class VectorQuantize(torch.nn.Module):
        def __init__(self, *a, **kw):
            raise RuntimeError("Seed-VC config needs vector quantization: install descript-audio-codec")

    for name in ("dac", "dac.nn", "dac.nn.quantize"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["dac.nn.quantize"].VectorQuantize = VectorQuantize


def _patch_bigvgan() -> None:
    """huggingface_hub>=1.0 no longer passes proxies/resume_download to _from_pretrained,
    but Seed-VC's bundled BigVGAN declares them as required keyword-only arguments."""
    import importlib
    bv = importlib.import_module("modules.bigvgan.bigvgan")
    cls = bv.BigVGAN
    if getattr(cls, "_hf_compat_patched", False):
        return
    orig = cls.__dict__["_from_pretrained"].__func__

    def _from_pretrained(klass, *args, proxies=None, resume_download=False, **kwargs):
        return orig(klass, *args, proxies=proxies, resume_download=resume_download, **kwargs)

    cls._from_pretrained = classmethod(_from_pretrained)
    cls._hf_compat_patched = True


_vc = None


def _move_vc(wrapper, device) -> None:
    """Move every torch module the Seed-VC wrapper holds (directly, in dicts, or inside
    helper objects such as RMVPE) and update the device attributes it reads."""
    device = torch.device(device)
    seen: set[int] = set()

    def move(obj, depth=0):
        if id(obj) in seen or depth > 2:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
        elif isinstance(obj, dict):
            for v in obj.values():
                move(v, depth + 1)
        elif hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, type)):
            for k, v in list(vars(obj).items()):
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.to(device))
                elif k == "device":
                    setattr(obj, k, device)
                else:
                    move(v, depth + 1)

    move(wrapper)


def _seedvc_wrapper():
    """Seed-VC models are built once, kept in RAM, and moved to the GPU per call."""
    global _vc
    if _vc is None:
        import sys
        _stub_dac()
        if SEEDVC_DIR not in sys.path:
            sys.path.insert(0, SEEDVC_DIR)
        _patch_bigvgan()
        from seed_vc_wrapper import SeedVCWrapper
        _vc = SeedVCWrapper(device=mm.get_torch_device())
    else:
        _move_vc(_vc, mm.get_torch_device())
    return _vc


def _drain(result):
    """convert_voice() is a generator even with stream_output=False: the full audio is its
    return value (StopIteration.value); streamed items are (mp3_bytes, full_audio)."""
    import types
    if not isinstance(result, types.GeneratorType):
        return result
    last = None
    try:
        while True:
            item = next(result)
            if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
                last = item[1]
    except StopIteration as stop:
        return stop.value if stop.value is not None else last


def _write_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    import soundfile as sf
    sf.write(path, wav.squeeze(0).numpy(), sr)


class SeedVCConvert:
    """Speech/singing → the same performance in the target voice (zero-shot, any language)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source":          ("AUDIO",),
                "target":          ("AUDIO",),
                "mode":            (["speech", "singing"], {"default": "speech"}),
                "diffusion_steps": ("INT", {"default": 30, "min": 4, "max": 100}),
                "pitch_shift":     ("INT", {"default": 0, "min": -24, "max": 24}),
                "cfg_rate":        ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/omnivoice"

    def run(self, source, target, mode, diffusion_steps, pitch_shift, cfg_rate):
        import tempfile
        src, src_sr = _to_mono(source)
        tgt, tgt_sr = _to_mono(target)
        tgt = tgt[:, : tgt_sr * 25]            # Seed-VC uses up to ~25 s of reference
        pbar = comfy.utils.ProgressBar(3)
        with _lock, tempfile.TemporaryDirectory() as tmp:
            mm.unload_all_models()
            mm.soft_empty_cache()
            sp, tp = os.path.join(tmp, "src.wav"), os.path.join(tmp, "tgt.wav")
            _write_wav(sp, src, src_sr)
            _write_wav(tp, tgt, tgt_sr)
            pbar.update(1)
            wrapper = _seedvc_wrapper()
            pbar.update(1)
            try:
                singing = mode == "singing"
                out = wrapper.convert_voice(sp, tp, diffusion_steps=diffusion_steps,
                                            inference_cfg_rate=cfg_rate, f0_condition=singing,
                                            auto_f0_adjust=not singing, pitch_shift=pitch_shift,
                                            stream_output=False)
                out = _drain(out)
                sr = 44100 if singing else 22050
            finally:
                _move_vc(wrapper, "cpu")
                del wrapper
                mm.soft_empty_cache()
            pbar.update(1)
        wav = torch.from_numpy(np.asarray(out, dtype=np.float32)).view(1, 1, -1)
        return ({"waveform": wav, "sample_rate": sr},)


# ── picking the best reference clip from a long recording ────────────────

def _best_window(wav: np.ndarray, sr: int, seconds: float) -> tuple[int, int]:
    """Return (start, end) samples of the cleanest `seconds`-long stretch of speech.

    Frames of 20 ms are classified as speech by energy relative to the recording's own
    loud level. A window scores high when it is mostly speech, has few long pauses and
    no clipping. Boundaries are then moved to the nearest quiet frame so words are not cut.
    """
    hop = int(sr * 0.02)
    n = len(wav) // hop
    want = int(seconds * sr)
    if n < 3 or len(wav) <= want:
        return 0, len(wav)
    frames = wav[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    loud = np.percentile(rms, 95)
    speech = rms > max(loud * 0.08, 1e-4)
    clipped = (np.abs(frames) > 0.99).mean(axis=1) > 0.001
    w = int(seconds / 0.02)
    best, best_score = 0, -1e9
    step = max(1, int(0.25 / 0.02))
    for i in range(0, n - w + 1, step):
        sp = speech[i:i + w]
        ratio = sp.mean()
        # longest pause inside the window (seconds)
        gaps, run = 0, 0
        for f in sp:
            run = 0 if f else run + 1
            gaps = max(gaps, run)
        score = ratio - 0.15 * max(0.0, gaps * 0.02 - 0.6) - 2.0 * clipped[i:i + w].mean()
        if score > best_score:
            best, best_score = i, score
    start, end = best, best + w

    def quiet_near(idx: int, direction: int) -> int:
        for k in range(0, int(0.6 / 0.02)):
            j = idx + direction * k
            if 0 <= j < n and not speech[j]:
                return j
        return idx

    start = quiet_near(start, -1)
    end = min(n, quiet_near(end, +1))
    return start * hop, end * hop


class VoiceRefSelect:
    """Long recording → the best short reference clip for cloning (3–10 s for OmniVoice,
    up to ~25 s for Seed-VC). Also trims leading/trailing silence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",),
                             "seconds": ("FLOAT", {"default": 10.0, "min": 3.0, "max": 30.0, "step": 0.5})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/omnivoice"

    def run(self, audio, seconds):
        wav, sr = _to_mono(audio)
        x = wav.squeeze(0).numpy().astype(np.float32)
        peak = float(np.abs(x).max()) or 1.0
        s, e = _best_window(x / peak, sr, seconds)
        clip = x[s:e]
        fade = int(sr * 0.01)
        if len(clip) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        log.info("VoiceRefSelect: %.1fs of %.1fs (from %.1fs)", len(clip) / sr, len(x) / sr, s / sr)
        return ({"waveform": torch.from_numpy(clip.copy()).view(1, 1, -1), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "VoiceRefSelect": VoiceRefSelect,
    "OmniVoiceTTS": OmniVoiceTTS,
    "OmniVoiceTranscribe": OmniVoiceTranscribe,
    "SeedVCConvert": SeedVCConvert,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceRefSelect": "Voice Reference Select (best clip)",
    "OmniVoiceTTS": "OmniVoice TTS / Voice Clone",
    "OmniVoiceTranscribe": "OmniVoice Transcribe (Whisper)",
    "SeedVCConvert": "Seed-VC Voice Conversion (keeps intonation)",
}
