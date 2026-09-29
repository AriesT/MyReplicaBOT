#!/usr/bin/env python3
"""
Low-VRAM launcher for `omnivoice.cli.train` (LoRA only), used by voice_lora.py.

Upstream loads the full model in fp32 without gradient checkpointing — fine on A100s,
too much for a 6 GB card. Without touching OmniVoice's code this wrapper:
  * loads the frozen base model in bf16 (the LoRA "frozen base, trainable adapters" setup),
  * keeps every trainable parameter (LoRA A/B, audio_embeddings, audio_heads) in fp32,
  * enables non-reentrant gradient checkpointing on the Qwen3 backbone,
then runs the regular training entry point with the same CLI arguments.
"""
import logging

import torch

import omnivoice.cli.train as cli
import omnivoice.training.builder as builder
from omnivoice.models.omnivoice import OmniVoice

log = logging.getLogger("voice_lora_train")

_orig_from_pretrained = OmniVoice.from_pretrained.__func__


def _from_pretrained(cls, *args, **kwargs):
    if kwargs.get("train"):
        kwargs["dtype"] = torch.bfloat16
    return _orig_from_pretrained(cls, *args, **kwargs)


OmniVoice.from_pretrained = classmethod(_from_pretrained)

_orig_build = builder.build_model_and_tokenizer


def _find_llm(model):
    for obj in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        llm = getattr(obj, "llm", None) if obj is not None else None
        if llm is not None:
            return llm
    return None


def _build(config):
    if not config.use_lora:
        raise SystemExit("voice_lora_train.py is meant for LoRA runs only")
    model, tokenizer = _orig_build(config)
    trainable = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            trainable += p.numel()
    llm = _find_llm(model)
    if llm is not None and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(llm, "config"):
            llm.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    print(f"[voice_lora_train] base bf16, trainable fp32 params: {trainable:,}, "
          f"gradient checkpointing: {llm is not None}", flush=True)
    return model, tokenizer


builder.build_model_and_tokenizer = _build
cli.build_model_and_tokenizer = _build

if __name__ == "__main__":
    cli.main()
