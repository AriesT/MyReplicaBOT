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
