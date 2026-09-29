# cu130 is required for ComfyUI's optimized int8 kernels (≈6× faster Qwen-Image on RTX 3050)
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git build-essential libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
ARG COMFY_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI /app && cd /app && git checkout $COMFY_REF && pip install -r requirements.txt
# OmniVoice (TTS / voice cloning / LoRA finetuning) for custom_nodes/comfyui_voice.
# Pinned to reviewed GitHub commit 08be0b4c (PyPI 0.2.1 predates LoRA support).
# --no-deps keeps ComfyUI's torch/transformers.
RUN pip install --no-deps "git+https://github.com/k2-fsa/OmniVoice@08be0b4c" \
 && pip install pydub librosa soundfile accelerate munch matplotlib peft webdataset tensorboardX
WORKDIR /app
EXPOSE 8188
CMD ["python3","main.py","--listen","0.0.0.0","--port","8188","--preview-method","latent2rgb"]
