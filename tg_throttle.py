"""
Rate limiting for progress-message edits.

Telegram answers frequent edits in one chat with "Too Many Requests: retry after N".
Swallowing that and retrying immediately keeps the chat throttled, so progress looks
frozen. All progress updates go through edit(): at most one edit per chat every
MIN_INTERVAL seconds, and after a RetryAfter the chat is left alone for as long as
Telegram asks.
"""
import logging
import time
from typing import Awaitable, Callable

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

log = logging.getLogger(__name__)

MIN_INTERVAL = 3.0
_next_ok: dict[int, float] = {}          # chat id → monotonic time of the next allowed edit


def ready(chat_id: int) -> bool:
    return time.monotonic() >= _next_ok.get(chat_id, 0.0)


async def edit(chat_id: int, action: Callable[[], Awaitable[object]],
               min_interval: float = MIN_INTERVAL, force: bool = False) -> bool:
    """Run one edit if the chat's budget allows it. Returns True if the edit went through.

    force=True ignores our own interval (final/important updates) but still respects a
    Telegram RetryAfter.
    """
    now = time.monotonic()
    wait_until = _next_ok.get(chat_id, 0.0)
    if now < wait_until and (not force or getattr(edit, "_flood", {}).get(chat_id)):
        return False
    try:
        await action()
        _next_ok[chat_id] = now + min_interval
        getattr(edit, "_flood", {}).pop(chat_id, None)
        return True
    except TelegramRetryAfter as e:
        _next_ok[chat_id] = now + float(e.retry_after) + 1.0
        edit._flood[chat_id] = True
        log.warning("Telegram flood control in chat %s: pausing progress edits for %ss", chat_id, e.retry_after)
        return False
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            log.debug("progress edit rejected in chat %s: %s", chat_id, e)
        _next_ok[chat_id] = now + min_interval
        return False


edit._flood = {}
