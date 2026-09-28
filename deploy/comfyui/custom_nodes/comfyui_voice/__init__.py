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
_prompt_cache: dict[str, object] = {}


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
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/omnivoice"

    def run(self, text, language, speed, num_step, seed, ref_audio=None, ref_text="", instruct=""):
        chunks = _chunks(text)
        if not chunks:
            raise ValueError("Empty text")
        pbar = comfy.utils.ProgressBar(len(chunks) + 1)
        lang = None if language == "auto" else language
        with _OnGPU() as model:
            prompt = None
            if ref_audio is not None:
                wav, sr = _to_mono(ref_audio)
                key = _audio_key(wav, sr, ref_text or "")
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


NODE_CLASS_MAPPINGS = {
    "OmniVoiceTTS": OmniVoiceTTS,
    "OmniVoiceTranscribe": OmniVoiceTranscribe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceTTS": "OmniVoice TTS / Voice Clone",
    "OmniVoiceTranscribe": "OmniVoice Transcribe (Whisper)",
}
