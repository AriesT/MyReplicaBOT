#!/bin/bash
cd /mnt/docker/comfyui/models
A=https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files
S=https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main
dl(){ echo "[$(date +%T)] $2"; wget -q -c -O "$2" "$1" && echo "[$(date +%T)] done $2 $(du -h "$2"|cut -f1)" || echo "FAIL $2"; }
dl $A/vae/ace_1.5_vae.safetensors vae/ace_1.5_vae.safetensors
dl $A/text_encoders/qwen_0.6b_ace15.safetensors text_encoders/qwen_0.6b_ace15.safetensors
dl $A/diffusion_models/acestep_v1.5_turbo.safetensors diffusion_models/acestep_v1.5_turbo.safetensors
dl $A/text_encoders/qwen_1.7b_ace15.safetensors text_encoders/qwen_1.7b_ace15.safetensors
dl $A/text_encoders/qwen_4b_ace15.safetensors text_encoders/qwen_4b_ace15.safetensors
dl $S/text_encoders/t5gemma_b_b_ul2.safetensors text_encoders/t5gemma_b_b_ul2.safetensors
dl $S/checkpoints/stable_audio_3_small_sfx.safetensors checkpoints/stable_audio_3_small_sfx.safetensors
dl $S/checkpoints/stable_audio_3_medium.safetensors checkpoints/stable_audio_3_medium.safetensors
echo ALLDONE
