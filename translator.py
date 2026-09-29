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
    """Translate text to English. Returns original on error or if already English.

    Google first, MyMemory as a fallback (Google rate-limits bursts), then Google once more.
    """
    text = text.strip()
    if not text or _looks_english(text):
        return text
    try:
        from deep_translator import GoogleTranslator, MyMemoryTranslator
    except ImportError:
        return text
    loop = asyncio.get_event_loop()
    attempts = [
        lambda: GoogleTranslator(source="auto", target="en").translate(text),
        lambda: MyMemoryTranslator(source="uk-UA", target="en-GB").translate(text),
        lambda: GoogleTranslator(source="auto", target="en").translate(text),
    ]
    for i, fn in enumerate(attempts):
        try:
            result = await loop.run_in_executor(None, fn)
            if result and result.strip():
                return result.strip()
        except Exception as e:
            log.warning("Translation attempt %d failed: %s", i + 1, e)
        await asyncio.sleep(0.5 + i)
    log.warning("Translation failed, using original prompt")
    return text
