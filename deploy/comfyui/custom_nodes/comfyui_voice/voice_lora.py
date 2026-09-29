#!/usr/bin/env python3
"""
Personal voice model: LoRA-finetune OmniVoice on a few minutes of one speaker.

Runs as a separate process inside the ComfyUI container (same Python env, same GPU):

    python voice_lora.py --audio take1.ogg [take2.ogg ...] --lang uk --out /app/models/omnivoice/lora/<id>

Pipeline: segment recordings at pauses (3–14 s phrases) → Whisper transcripts →
OmniVoice audio tokens (WebDataset shards) → LoRA training (omnivoice.cli.train,
single GPU, bf16, SDPA) → adapter copied to --out.

Progress is printed as one JSON object per line prefixed with "PROGRESS " so a caller
(the Telegram bot) can show it: {"stage": ..., "step": ..., "total": ..., "loss": ...}.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HUB = os.environ.get("HF_HOME", "/app/models/omnivoice") + "/hub"
SR = 24000


def emit(stage: str, **kw) -> None:
    print("PROGRESS " + json.dumps({"stage": stage, **kw}, ensure_ascii=False), flush=True)


def snapshot(repo: str) -> str:
    base = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    if not snaps:
        raise SystemExit(f"model {repo} is not in the local cache ({base}); run a TTS once first")
    return snaps[-1]


# ── 1. segmentation ───────────────────────────────────────────────────────

def load_mono(path: str) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    peak = float(np.abs(wav).max()) or 1.0
    return (wav / peak * 0.9).astype(np.float32)


def segment(wav: np.ndarray, min_s=3.0, max_s=14.0, min_pause=0.3) -> list[tuple[int, int]]:
    """Cut at pauses; merge neighbouring phrases up to max_s; drop fragments < 1.5 s."""
    hop = int(SR * 0.02)
    n = len(wav) // hop
    rms = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    speech = rms > max(np.percentile(rms, 95) * 0.08, 1e-4)
    # runs of speech separated by pauses >= min_pause
    phrases, start, silent = [], None, 0
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            silent = 0
        elif start is not None:
            silent += 1
            if silent * 0.02 >= min_pause:
                phrases.append((start, i - silent + 1))
                start, silent = None, 0
    if start is not None:
        phrases.append((start, n))
    # merge into training utterances
    out, cur = [], None
    for a, b in phrases:
        if cur is None:
            cur = [a, b]
        elif (b - cur[0]) * 0.02 <= max_s:
            cur[1] = b
        else:
            out.append(tuple(cur))
            cur = [a, b]
        if (cur[1] - cur[0]) * 0.02 >= min_s and (cur[1] - cur[0]) * 0.02 > max_s * 0.7:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    pad = int(0.12 / 0.02)
    res = []
    for a, b in out:
        if (b - a) * 0.02 < 1.5:
            continue
        # split anything still longer than max_s (no pause inside) into equal parts
        parts = int(np.ceil((b - a) * 0.02 / max_s))
        edges = np.linspace(a, b, parts + 1).astype(int)
        for x, y in zip(edges[:-1], edges[1:]):
            res.append((max(0, x - pad) * hop, min(n, y + pad) * hop))
    return res


# ── 2. transcription ──────────────────────────────────────────────────────

def transcribe(clips: list[np.ndarray], lang: str) -> list[str]:
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=snapshot("openai/whisper-large-v3-turbo"),
                   dtype=torch.float16, device="cuda:0")
    texts = []
    for i, c in enumerate(clips):
        import librosa
        a16 = librosa.resample(c, orig_sr=SR, target_sr=16000)
        kw = {"generate_kwargs": {"language": lang, "task": "transcribe"}} if lang and lang != "auto" else {}
        texts.append(asr({"array": a16, "sampling_rate": 16000}, **kw)["text"].strip())
        emit("asr", step=i + 1, total=len(clips))
    del asr
    torch.cuda.empty_cache()
    return texts


# ── 3–4. tokenization + training ──────────────────────────────────────────

def run_logged(cmd: list[str], stage: str, total: int = 0, env=None) -> None:
    """Run a child process, forward tqdm progress (n/total, loss) as PROGRESS lines."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, bufsize=1)
    last = 0.0
    tail: list[str] = []
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if ch == "" and proc.poll() is not None:
            break
        if ch not in ("\r", "\n"):
            buf += ch
            continue
        line, buf = buf.strip(), ""
        if not line:
            continue
        tail = (tail + [line])[-25:]
        m = re.search(r"(\d+)/(\d+) \[", line)
        if m and time.time() - last > 2:
            last = time.time()
            loss = re.search(r"loss=([\d.]+)", line)
            emit(stage, step=int(m.group(1)), total=int(m.group(2)) or total,
                 loss=float(loss.group(1)) if loss else None)
    if proc.returncode != 0:
        print("\n".join(tail), file=sys.stderr)
        raise SystemExit(f"{stage} failed (exit {proc.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--lang", default="uk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = auto from data amount")
    ap.add_argument("--batch_tokens", type=int, default=1536)
    ap.add_argument("--work", default="")
    args = ap.parse_args()

    work = args.work or tempfile.mkdtemp(prefix="voice_lora_")
    os.makedirs(work, exist_ok=True)
    ov = snapshot("k2-fsa/OmniVoice")
    import soundfile as sf

    emit("segment")
    clips = []
    for path in args.audio:
        wav = load_mono(path)
        clips += [wav[a:b] for a, b in segment(wav)]
    total_sec = sum(len(c) for c in clips) / SR
    if total_sec < 30:
        raise SystemExit(f"too little clean speech: {total_sec:.0f}s (need ≥ 30 s, 2–5 min is best)")
    emit("segment", clips=len(clips), seconds=round(total_sec))

    texts = transcribe(clips, args.lang)
    rows = []
    os.makedirs(os.path.join(work, "wavs"), exist_ok=True)
    for i, (c, t) in enumerate(zip(clips, texts)):
        if len(t) < 2:
            continue
        p = os.path.join(work, "wavs", f"utt_{i:04d}.wav")
        sf.write(p, c, SR)
        rows.append({"id": f"utt_{i:04d}", "audio_path": p, "text": t,
                     **({"language_id": args.lang} if args.lang != "auto" else {})})
    with open(os.path.join(work, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    emit("tokenize", clips=len(rows))

    tok_dir = os.path.join(work, "tokens")
    run_logged([sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                "--input_jsonl", os.path.join(work, "train.jsonl"),
                "--tar_output_pattern", f"{tok_dir}/audios/shard-%06d.tar",
                "--jsonl_output_pattern", f"{tok_dir}/txts/shard-%06d.jsonl",
                "--tokenizer_path", os.path.join(ov, "audio_tokenizer"),
                "--nj_per_gpu", "1", "--loader_workers", "2", "--min_num_shards", "1",
                "--shuffle", "True"], "tokenize", len(rows))

    steps = args.steps or int(min(1500, max(300, total_sec * 3)))
    train_cfg = {
        "llm_name_or_path": ov, "init_from_checkpoint": ov,
        "audio_vocab_size": 1025, "audio_mask_id": 1024, "num_audio_codebook": 8,
        "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "drop_cond_ratio": 0.1, "prompt_ratio_range": [0.0, 0.3], "mask_ratio_range": [0.0, 1.0],
        "language_ratio": 0.8, "use_pinyin_ratio": 0.0, "instruct_ratio": 0.0, "only_instruct_ratio": 0.0,
        "use_lora": True, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lora_bias": "none",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_modules_to_save": ["audio_embeddings", "audio_heads"],
        "learning_rate": 1e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "steps": steps, "seed": 42, "warmup_type": "ratio", "warmup_ratio": 0.05, "warmup_steps": 0,
        "attn_implementation": "sdpa", "max_sample_tokens": 2000, "min_sample_tokens": 50,
        "max_batch_size": 16, "batch_tokens": args.batch_tokens, "gradient_accumulation_steps": 1,
        "num_workers": 1, "mixed_precision": "bf16", "allow_tf32": True,
        "logging_steps": 10, "eval_steps": 10 ** 9, "save_steps": steps, "keep_last_n_checkpoints": 1,
    }
    with open(os.path.join(work, "train_config.json"), "w") as f:
        json.dump(train_cfg, f)
    with open(os.path.join(work, "data_config.json"), "w") as f:
        json.dump({"train": [{"manifest_path": [f"{tok_dir}/data.lst"]}]}, f)

    emit("train", step=0, total=steps)
    exp = os.path.join(work, "exp")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_lora_train.py")
    run_logged(["accelerate", "launch", "--num_processes", "1", "--mixed_precision", "bf16",
                launcher,
                "--train_config", os.path.join(work, "train_config.json"),
                "--data_config", os.path.join(work, "data_config.json"),
                "--output_dir", exp], "train", steps, env=env)

    ckpts = sorted(glob.glob(os.path.join(exp, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not ckpts or not os.path.exists(os.path.join(ckpts[-1], "adapter_config.json")):
        raise SystemExit("training finished without a LoRA adapter checkpoint")
    os.makedirs(args.out, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "train_config.json"):
        src = os.path.join(ckpts[-1], name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))
    with open(os.path.join(args.out, "voice_meta.json"), "w") as f:
        json.dump({"seconds": round(total_sec), "clips": len(rows), "steps": steps, "lang": args.lang,
                   "created": int(time.time())}, f)
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    emit("done", out=args.out, seconds=round(total_sec), clips=len(rows), steps=steps)


if __name__ == "__main__":
    main()
