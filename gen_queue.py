"""
Single-worker generation queue.
Jobs are processed one at a time; waiting users see live position updates.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message

import comfy_client

log = logging.getLogger(__name__)


@dataclass
class GenJob:
    message:       Message
    prompt:        str
    user_settings: dict
    status_msg:    Message
    on_done:       Callable[[Message, bytes, str], Awaitable[None]]
    on_error:      Callable[[Message, str], Awaitable[None]]
    input_image:   Optional[bytes] = None
    batch_index:   int = 1
    batch_total:   int = 1
    cancel_kb:     Optional[InlineKeyboardMarkup] = None
    on_cancel:     Optional[Callable[[Message], Awaitable[None]]] = None


_queue:       list[GenJob]           = []
_lock:        asyncio.Lock           = asyncio.Lock()
_worker_task: Optional[asyncio.Task] = None


# ── public API ────────────────────────────────────────────────────────────

def queue_len() -> int:
    return len(_queue)


async def enqueue(job: GenJob) -> None:
    global _worker_task
    async with _lock:
        _queue.append(job)
        pos = len(_queue)

    if pos > 1:
        await _set_waiting(job, pos - 1)

    async with _lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_worker())


async def cancel_by_msg(msg_id: int) -> list[GenJob]:
    """Remove all waiting (non-running) jobs with given status_msg.message_id."""
    async with _lock:
        to_cancel = [j for j in _queue[1:] if j.status_msg.message_id == msg_id]
        for j in to_cancel:
            _queue.remove(j)
    for j in to_cancel:
        if j.on_cancel:
            try:
                await j.on_cancel(j.message)
            except Exception:
                log.exception("on_cancel callback failed")
    if to_cancel:
        await _broadcast()
    return to_cancel


# ── worker ────────────────────────────────────────────────────────────────

async def _worker() -> None:
    while True:
        async with _lock:
            if not _queue:
                break
            job = _queue[0]

        try:
            await _run(job)
        except Exception:
            log.exception("Unexpected error in queue worker")

        async with _lock:
            if _queue and _queue[0] is job:
                _queue.pop(0)

        await _broadcast()


async def _run(job: GenJob) -> None:
    is_i2i  = job.input_image is not None
    label   = "варіацію" if is_i2i else "зображення"
    counter = f"[{job.batch_index}/{job.batch_total}] " if job.batch_total > 1 else ""

    try:
        # remove cancel button once the job starts executing
        await job.status_msg.edit_text(
            f"⏳ {counter}Підключаюсь до ComfyUI...", reply_markup=None)
    except TelegramBadRequest:
        pass

    async def on_progress(step: int, total: int) -> None:
        bar = comfy_client.progress_bar(step, total)
        try:
            await job.status_msg.edit_text(
                f"⚙️ <b>{counter}Генерую {label}...</b>\n\n"
                f"<code>{bar}</code>\n"
                f"Крок {step} з {total}",
                parse_mode="HTML",
            )
        except TelegramBadRequest:
            pass

    try:
        result = await comfy_client.generate(
            job.prompt,
            on_progress=on_progress,
            user_settings=job.user_settings,
            input_image=job.input_image,
        )
    except Exception as exc:
        log.exception("Generation failed user=%d", job.message.from_user.id)
        try:
            await job.on_error(job.message, _friendly_error(exc))
        except Exception:
            log.exception("on_error callback failed")
        return

    try:
        await job.on_done(job.message, result, job.prompt)
    except Exception:
        log.exception("on_done callback failed")


# ── error message ─────────────────────────────────────────────────────────

def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "could not detect model type" in low:
        name = msg.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        return (
            f"❌ <b>Модель не підтримується</b>\n\n"
            f"<code>{name}</code>\n\n"
            f"ComfyUI не може визначити тип цієї моделі.\n"
            f"Оберіть іншу модель у 🎛 Налаштуваннях генерації."
        )
    if "cuda out of memory" in low or "out of memory" in low:
        return (
            "❌ <b>Недостатньо VRAM</b>\n\n"
            "Спробуйте зменшити розмір зображення або кількість кроків."
        )
    if "websocket" in low or "closed unexpectedly" in low:
        return "❌ <b>З'єднання з ComfyUI перервано</b>\n\nСпробуйте ще раз."
    if "comfyui" in low and ":" in msg:
        detail = msg.split(":", 1)[-1].strip()
        return f"❌ <b>Помилка ComfyUI:</b>\n<code>{detail}</code>"
    return f"❌ <b>Помилка генерації:</b>\n<code>{msg}</code>"


# ── position broadcast ────────────────────────────────────────────────────

async def _broadcast() -> None:
    async with _lock:
        waiting = list(_queue[1:])
    seen: set[int] = set()
    for i, job in enumerate(waiting, start=1):
        mid = job.status_msg.message_id
        if mid not in seen:
            seen.add(mid)
            await _set_waiting(job, i)


async def _set_waiting(job: GenJob, ahead: int) -> None:
    noun = _inflect(ahead)
    try:
        await job.status_msg.edit_text(
            f"🕐 <b>В черзі</b>\n\n"
            f"<code>{'░' * 20}</code>\n"
            f"Попереду: <b>{ahead} {noun}</b>\n"
            f"<i>Повідомлення оновиться, коли дійде ваша черга</i>",
            parse_mode="HTML",
            reply_markup=job.cancel_kb,
        )
    except TelegramBadRequest:
        pass


def _inflect(n: int) -> str:
    if 11 <= n % 100 <= 19:
        return "запитів"
    mod = n % 10
    if mod == 1:
        return "запит"
    if 2 <= mod <= 4:
        return "запити"
    return "запитів"
