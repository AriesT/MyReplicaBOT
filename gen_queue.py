"""
Single-worker generation queue.
Jobs are processed one at a time; waiting users see live position updates.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message

import comfy_client
import tg_throttle

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
    # custom executor (e.g. Qwen-Image) — replaces the default comfy_client.generate() path
    runner:        Optional[Callable[["GenJob"], Awaitable[None]]] = None
    label:         str   = "🎨 Зображення"   # shown to people waiting behind this job
    eta:           float = 60.0             # expected run time, seconds


_TICK = 4.0                      # how often waiting messages are refreshed (only if text changed)
_current: dict = {"job": None, "frac": 0.0, "left": None, "started": 0.0}
_last_text: dict[int, str] = {}  # id(job) → last rendered waiting text
_ticker_task: Optional[asyncio.Task] = None


_queue:       list[GenJob]           = []
_lock:        asyncio.Lock           = asyncio.Lock()
_worker_task: Optional[asyncio.Task] = None


# ── public API ────────────────────────────────────────────────────────────

def queue_len() -> int:
    return len(_queue)


def report(job: GenJob, frac: float, left: Optional[float] = None) -> None:
    """Runners call this with their progress so people in the queue see it."""
    if _current["job"] is job:
        _current["frac"] = max(0.0, min(frac, 1.0))
        _current["left"] = left


def _running_left() -> float:
    job = _current["job"]
    if job is None:
        return 0.0
    if _current["left"] is not None:
        return max(0.0, _current["left"])
    if _current["frac"] > 0.05:
        spent = time.monotonic() - _current["started"]
        return spent * (1 - _current["frac"]) / _current["frac"]
    return max(0.0, job.eta - (time.monotonic() - _current["started"]))


async def enqueue(job: GenJob) -> None:
    global _worker_task
    async with _lock:
        _queue.append(job)
        pos = len(_queue)

    if pos > 1:
        await _set_waiting(job, pos - 1)

    global _ticker_task
    async with _lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_worker())
        if _ticker_task is None or _ticker_task.done():
            _ticker_task = asyncio.create_task(_ticker())


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

        _current.update(job=job, frac=0.0, left=None, started=time.monotonic())
        try:
            await _run(job)
        except Exception:
            log.exception("Unexpected error in queue worker")
        finally:
            _current.update(job=None, frac=0.0, left=None)

        async with _lock:
            if _queue and _queue[0] is job:
                _queue.pop(0)
        _last_text.pop(id(job), None)

        await _broadcast()


async def _run(job: GenJob) -> None:
    if job.runner is not None:
        await job.runner(job)
        return

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
        report(job, step / total if total else 0.0)
        bar = comfy_client.progress_bar(step, total)
        text = (f"⚙️ <b>{counter}Генерую {label}...</b>\n\n"
                f"<code>{bar}</code>\n"
                f"Крок {step} з {total}")
        await tg_throttle.edit(job.status_msg.chat.id,
                               lambda: job.status_msg.edit_text(text, parse_mode="HTML"),
                               force=(step == total))

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

async def _broadcast(force: bool = True) -> None:
    """Refresh every waiting job's message. force=False skips messages whose text is unchanged."""
    async with _lock:
        waiting = list(_queue[1:])
    seen: set[int] = set()
    for i, job in enumerate(waiting, start=1):
        mid = job.status_msg.message_id
        if mid not in seen:
            seen.add(mid)
            await _set_waiting(job, i, force=force)


async def _ticker() -> None:
    """While anyone waits, keep their queue position / current job progress / ETA fresh."""
    while True:
        await asyncio.sleep(_TICK)
        async with _lock:
            if len(_queue) <= 1 and _current["job"] is None:
                if not _queue:
                    break
                continue
        try:
            await _broadcast(force=False)
        except Exception:
            log.exception("queue ticker failed")


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _waiting_text(job: GenJob, ahead: int) -> str:
    running = _current["job"]
    lines = ["🕐 <b>Ви в черзі</b>", "", f"📍 Ваш номер: <b>{ahead + 1}</b>  ·  попереду <b>{ahead} {_inflect(ahead)}</b>"]
    wait = 0.0
    if running is not None:
        pct = int(_current["frac"] * 100)
        filled = int(_current["frac"] * 12)
        lines += ["", f"▶️ Зараз генерується: {running.label}",
                  f"<code>{'▰' * filled}{'▱' * (12 - filled)}  {pct}%</code>"]
        wait += _running_left()
    others = [j for j in _queue[1:] if j is not job][: max(0, ahead - (1 if running is not None else 0))]
    if others:
        names = ", ".join(j.label for j in others[:3]) + (" …" if len(others) > 3 else "")
        lines.append(f"⏭ Далі перед вами: {names}")
    wait += sum(j.eta for j in others)
    lines += ["", f"⏳ Орієнтовно чекати: <b>~{_fmt(wait)}</b>",
              "<i>Оновлюється автоматично · коли дійде черга, тут з'явиться прогрес</i>"]
    return "\n".join(lines)


async def _set_waiting(job: GenJob, ahead: int, force: bool = True) -> None:
    text = _waiting_text(job, ahead)
    key = id(job)
    if not force and _last_text.get(key) == text:
        return
    if await tg_throttle.edit(job.status_msg.chat.id,
                              lambda: job.status_msg.edit_text(text, parse_mode="HTML", reply_markup=job.cancel_kb),
                              force=force):
        _last_text[key] = text


def _inflect(n: int) -> str:
    if 11 <= n % 100 <= 19:
        return "запитів"
    mod = n % 10
    if mod == 1:
        return "запит"
    if 2 <= mod <= 4:
        return "запити"
    return "запитів"
