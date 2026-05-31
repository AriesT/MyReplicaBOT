"""Translate user prompts to English before sending to ComfyUI."""
import asyncio
import logging

log = logging.getLogger(__name__)

_ASCII_THRESHOLD = 0.85  # if > 85% ASCII chars, assume already English


def _looks_english(text: str) -> bool:
    if not text:
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= _ASCII_THRESHOLD


async def to_english(text: str) -> str:
    """Translate text to English. Returns original on error or if already English."""
    text = text.strip()
    if not text or _looks_english(text):
        return text
    try:
        from deep_translator import GoogleTranslator
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: GoogleTranslator(source="auto", target="en").translate(text),
        )
        return result.strip() if result else text
    except Exception as e:
        log.warning("Translation failed, using original prompt: %s", e)
        return text
