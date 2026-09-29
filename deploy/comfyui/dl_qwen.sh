#!/bin/bash
set -e
cd /mnt/docker/comfyui/models
B=https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main
dl(){ echo "[$(date +%T)] $2"; wget -q -c -O "$2" "$1"; echo "[$(date +%T)] done $2 $(du -h "$2"|cut -f1)"; }
dl $B/vae/qwen_image_2.1_vae_bf16.safetensors vae/qwen_image_2.1_vae_bf16.safetensors
dl https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors loras/qwen21-turbo-6step-r128.safetensors
dl https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs/resolve/main/models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors loras/qwen21-fun-acc-4step.safetensors
dl $B/diffusion_models/qwen_image_2.1_int8_convrot.safetensors diffusion_models/qwen_image_2.1_int8_convrot.safetensors
dl $B/text_encoders/qwen3vl_8b_int8_convrot.safetensors text_encoders/qwen3vl_8b_int8_convrot.safetensors
echo ALLDONE
