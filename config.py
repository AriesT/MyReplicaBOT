import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN: str = os.environ["BOT_TOKEN"]
COMFY_URL: str = os.getenv("COMFY_URL", "http://192.168.39.39:8188")
CHECKPOINT: str = os.getenv("CHECKPOINT", "v1-5-pruned-emaonly.ckpt")

IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "512"))
IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "512"))
STEPS: int = int(os.getenv("STEPS", "20"))
CFG_SCALE: float = float(os.getenv("CFG_SCALE", "7.0"))
NEGATIVE_PROMPT: str = os.getenv(
    "NEGATIVE_PROMPT",
    "ugly, blurry, low quality, watermark, text, deformed, extra limbs",
)
POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "2.0"))
POLL_TIMEOUT: float = float(os.getenv("POLL_TIMEOUT", "300.0"))
VIDEO_POLL_TIMEOUT: float = float(os.getenv("VIDEO_POLL_TIMEOUT", "1800.0"))  # 30 min for video

# MMORPG Game API integration (optional)
GAME_API_URL: str = os.getenv("GAME_API_URL", "")
GAME_API_KEY: str = os.getenv("GAME_API_KEY", "")

# Qwen-Image 2.1 (files in ComfyUI models/)
QWEN_UNET: str = os.getenv("QWEN_UNET", "qwen_image_2.1_int8_convrot.safetensors")
QWEN_CLIP: str = os.getenv("QWEN_CLIP", "qwen3vl_8b_int8_convrot.safetensors")
QWEN_VAE: str = os.getenv("QWEN_VAE", "qwen_image_2.1_vae_bf16.safetensors")
QWEN_TURBO_LORA: str = os.getenv("QWEN_TURBO_LORA", "qwen21-turbo-6step-r128.safetensors")
QWEN_POLL_TIMEOUT: float = float(os.getenv("QWEN_POLL_TIMEOUT", "1800.0"))
