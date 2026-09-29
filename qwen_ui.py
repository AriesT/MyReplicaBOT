"""
Qwen-Image 2.1 section of the bot.

- 🌀 menu with its own settings (quality preset, aspect ratio, size, transparency, seed…)
- "Qwen mode": while it is on, any plain text message is a Qwen prompt and
  a photo with a caption is a Qwen edit (reference image + instruction)
- live progress card: stages, progress bar, s/step, ETA and latent previews
- result buttons: new variant, re-render in quality, lock seed, original PNG, edit further
"""
import logging
import random
import re
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Optional

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile, CallbackQuery, FSInputFile,
    InlineKeyboardMarkup, InputMediaPhoto, Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from PIL import Image, ImageDraw, ImageFont

import comfy_client as cc
import gen_queue as gq
import history as hist
import translator
import users as db

log    = logging.getLogger(__name__)
router = Router(name="qwen")

TITLE = "🌀 <b>Qwen-Image 2.1</b>"


class QwenCB(CallbackData, prefix="qw"):
    action: str
    value:  Optional[str] = None


class QwenState(StatesGroup):
    waiting_prompt      = State()
    waiting_edit_prompt = State()
    waiting_negative    = State()
    waiting_seed        = State()


# ── access helpers ────────────────────────────────────────────────────────

def _ctx(user) -> tuple[bool, bool]:
    db.sync_id(user.id, user.username or "")
    return db.is_allowed(user.id, user.username or ""), db.is_admin(user.id, user.username or "")


def is_active(tg_id: int) -> bool:
    return bool(db.get_gen_settings(tg_id).get("qwen_active"))


async def _nav(call: CallbackQuery, text: str, **kw) -> None:
    try:
        if call.message.photo or call.message.document:
            await call.message.answer(text, **kw)
        else:
            await call.message.edit_text(text, **kw)
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            await call.message.answer(text, **kw)


def _fmt(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── menu ──────────────────────────────────────────────────────────────────

def _menu_text(tg_id: int) -> str:
    s   = db.get_gen_settings(tg_id)
    q   = cc.qwen_settings(s)
    pre = cc.QWEN_QUALITY_PRESETS[q["quality"]]
    active = bool(s.get("qwen_active"))
    lines = [
        TITLE,
        "<i>7B DiT • текст на зображеннях • прозорий фон • редагування фото</i>",
        "",
        ("🟢 <b>Режим Qwen увімкнено</b> — просто пишіть промпт у чат,\n"
         "а фото з підписом = редагування фото.") if active else
        ("⚪ Режим Qwen вимкнено — увімкніть, щоб кожне повідомлення\n"
         "генерувалось через Qwen."),
        "",
        f"🎚 Якість:   <b>{pre['label']}</b>  <i>({pre['hint']})</i>",
        f"📐 Формат:   <b>{q['ratio']}</b>  →  {q['width']}×{q['height']}",
        f"🔍 Розмір:   <b>{q['mp']:g} МП</b>",
        f"🫥 Прозорий фон: <b>{'так (PNG з альфа-каналом)' if q['transparent'] else 'ні'}</b>",
        f"🌐 Переклад UA→EN: <b>{'так' if q['translate'] else 'ні'}</b>",
        f"🎲 Сід:      <b>{q['seed'] if q['seed'] is not None else 'випадковий'}</b>",
    ]
    if q["negative"]:
        neg = q["negative"][:60] + ("…" if len(q["negative"]) > 60 else "")
        lines.append(f"🚫 Негатив:  <i>{_esc(neg)}</i>  (CFG {q['cfg']:g}, ×2 час)")
    lines += ["", f"⏱ Орієнтовно: <b>~{_fmt(cc.qwen_estimate(q))}</b> на зображення"]
    ahead = gq.queue_len()
    if ahead:
        lines.append(f"🕐 Зараз у черзі: {ahead}")
    return "\n".join(lines)


def kb_menu(tg_id: int) -> InlineKeyboardMarkup:
    s = db.get_gen_settings(tg_id)
    q = cc.qwen_settings(s)
    active = bool(s.get("qwen_active"))
    b = InlineKeyboardBuilder()
    b.button(text="✍️ Згенерувати",            callback_data=QwenCB(action="prompt").pack())
    b.button(text="🎲 Надихни мене",            callback_data=QwenCB(action="inspire").pack())
    b.button(text=("🟢 Режим Qwen: увімк." if active else "⚪ Режим Qwen: вимк."),
             callback_data=QwenCB(action="toggle_active").pack())
    b.button(text=f"🎚 {cc.QWEN_QUALITY_PRESETS[q['quality']]['label']}",
             callback_data=QwenCB(action="pick_quality").pack())
    b.button(text=f"📐 {q['ratio']}",          callback_data=QwenCB(action="pick_ratio").pack())
    b.button(text=f"🔍 {q['mp']:g} МП",        callback_data=QwenCB(action="pick_mp").pack())
    b.button(text=f"🫥 Прозорий: {'так' if q['transparent'] else 'ні'}",
             callback_data=QwenCB(action="toggle_transparent").pack())
    b.button(text=f"🌐 Переклад: {'так' if q['translate'] else 'ні'}",
             callback_data=QwenCB(action="toggle_translate").pack())
    b.button(text=("🔓 Сід: скинути" if q["seed"] is not None else "🎲 Сід: задати"),
             callback_data=QwenCB(action="seed").pack())
    b.button(text=("🚫 Негатив: прибрати" if q["negative"] else "🚫 Негативний промпт"),
             callback_data=QwenCB(action="negative").pack())
    b.button(text="❓ Як це працює",            callback_data=QwenCB(action="help").pack())
    b.button(text="🔙 Головне меню",            callback_data="menu:main")
    b.adjust(2, 1, 3, 2, 2, 1, 1)
    return b.as_markup()


def _kb_picker(items: list[tuple[str, str]], current: str, action: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for value, label in items:
        mark = " ✅" if value == current else ""
        b.button(text=f"{label}{mark}", callback_data=QwenCB(action=action, value=value).pack())
    b.button(text="🔙 Назад", callback_data=QwenCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()


def _kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardBuilder().button(
        text="🔙 Меню Qwen", callback_data=QwenCB(action="menu").pack()).as_markup()


async def show_menu(target: Message, tg_id: int) -> None:
    await target.answer(_menu_text(tg_id), parse_mode="HTML", reply_markup=kb_menu(tg_id))


async def cmd_qwen(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    await state.clear()
    await show_menu(message, message.from_user.id)


@router.callback_query(QwenCB.filter(F.action == "menu"))
async def cb_menu(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.clear()
    await call.answer()
    await _nav(call, _menu_text(call.from_user.id), parse_mode="HTML",
               reply_markup=kb_menu(call.from_user.id))


async def _refresh(call: CallbackQuery, note: str = "") -> None:
    await call.answer(note)
    await _nav(call, _menu_text(call.from_user.id), parse_mode="HTML",
               reply_markup=kb_menu(call.from_user.id))


@router.callback_query(QwenCB.filter(F.action == "toggle_active"))
async def cb_toggle_active(call: CallbackQuery) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    on = not is_active(call.from_user.id)
    db.set_gen_setting(call.from_user.id, "qwen_active", True if on else None)
    if on:   # modes are mutually exclusive
        db.set_gen_setting(call.from_user.id, "voice_active", None)
    await _refresh(call, "🟢 Тепер кожен промпт → Qwen" if on else "⚪ Повернулись до звичайних моделей")


@router.callback_query(QwenCB.filter(F.action.in_({"toggle_transparent", "toggle_translate"})))
async def cb_toggle_flag(call: CallbackQuery, callback_data: QwenCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    q = cc.qwen_settings(db.get_gen_settings(call.from_user.id))
    if callback_data.action == "toggle_transparent":
        db.set_gen_setting(call.from_user.id, "qwen_transparent", not q["transparent"])
    else:
        db.set_gen_setting(call.from_user.id, "qwen_translate", not q["translate"])
    await _refresh(call)


@router.callback_query(QwenCB.filter(F.action == "pick_quality"))
async def cb_pick_quality(call: CallbackQuery) -> None:
    q = cc.qwen_settings(db.get_gen_settings(call.from_user.id))
    items = [(k, f"{v['label']} — {v['hint']}") for k, v in cc.QWEN_QUALITY_PRESETS.items()]
    await call.answer()
    await _nav(call, "🎚 <b>Якість генерації</b>\n\n"
                     "⚡ <b>Турбо</b> — дистильована LoRA, 6 кроків. Швидко і вже дуже гарно.\n"
                     "💎 <b>Якість</b> — оригінальна модель, 25 кроків. Кращі деталі, текстури й текст.\n"
                     "👑 <b>Максимум</b> — 40 кроків, як в офіційному пайплайні.",
               parse_mode="HTML", reply_markup=_kb_picker(items, q["quality"], "set_quality"))


@router.callback_query(QwenCB.filter(F.action == "pick_ratio"))
async def cb_pick_ratio(call: CallbackQuery) -> None:
    s = db.get_gen_settings(call.from_user.id)
    q = cc.qwen_settings(s)
    names = {"1:1": "⬛ 1:1 квадрат", "4:3": "🖥 4:3", "3:4": "📄 3:4", "3:2": "📷 3:2 фото",
             "2:3": "🖼 2:3 портрет", "16:9": "🎬 16:9 кіно", "9:16": "📱 9:16 сторіс"}
    items = []
    for r in cc.QWEN_RATIOS:
        w, h = cc.qwen_size(r, q["mp"])
        items.append((r.replace(":", "x"), f"{names.get(r, r)}  ({w}×{h})"))   # ':' is the callback separator
    await call.answer()
    await _nav(call, "📐 <b>Формат зображення</b>", parse_mode="HTML",
               reply_markup=_kb_picker(items, q["ratio"].replace(":", "x"), "set_ratio"))


@router.callback_query(QwenCB.filter(F.action == "pick_mp"))
async def cb_pick_mp(call: CallbackQuery) -> None:
    q = cc.qwen_settings(db.get_gen_settings(call.from_user.id))
    labels = {0.5: "🐇 0.5 МП — швидше вдвічі", 1.0: "⚖️ 1 МП — оптимально",
              2.0: "🐢 2 МП — нативний 2K, ×2.5 часу"}
    items = [(f"{mp:g}", labels[mp]) for mp in cc.QWEN_MEGAPIXELS]
    await call.answer()
    await _nav(call, "🔍 <b>Розмір (кількість пікселів)</b>", parse_mode="HTML",
               reply_markup=_kb_picker(items, f"{q['mp']:g}", "set_mp"))


@router.callback_query(QwenCB.filter(F.action.in_({"set_quality", "set_ratio", "set_mp"})))
async def cb_set_value(call: CallbackQuery, callback_data: QwenCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    key = {"set_quality": "qwen_quality", "set_ratio": "qwen_ratio", "set_mp": "qwen_mp"}[callback_data.action]
    value = (float(callback_data.value) if key == "qwen_mp"
             else callback_data.value.replace("x", ":") if key == "qwen_ratio" else callback_data.value)
    db.set_gen_setting(call.from_user.id, key, value)
    await _refresh(call, "✅ Збережено")


@router.callback_query(QwenCB.filter(F.action == "seed"))
async def cb_seed(call: CallbackQuery, state: FSMContext) -> None:
    q = cc.qwen_settings(db.get_gen_settings(call.from_user.id))
    if q["seed"] is not None:
        db.set_gen_setting(call.from_user.id, "qwen_seed", None)
        await _refresh(call, "🎲 Сід знову випадковий")
        return
    await state.set_state(QwenState.waiting_seed)
    await call.answer()
    await _nav(call, "🎲 Надішліть число-сід (однаковий сід + промпт = однакова картинка):",
               reply_markup=_kb_back())


@router.message(QwenState.waiting_seed, F.text)
async def handle_seed(message: Message, state: FSMContext) -> None:
    txt = message.text.strip()
    if not txt.isdigit():
        await message.answer("🔢 Потрібне ціле число, напр. <code>42</code>", parse_mode="HTML")
        return
    await state.clear()
    db.set_gen_setting(message.from_user.id, "qwen_seed", int(txt) & 0xFFFFFFFFFFFF)
    await show_menu(message, message.from_user.id)


@router.callback_query(QwenCB.filter(F.action == "negative"))
async def cb_negative(call: CallbackQuery, state: FSMContext) -> None:
    q = cc.qwen_settings(db.get_gen_settings(call.from_user.id))
    if q["negative"]:
        db.set_gen_setting(call.from_user.id, "qwen_negative", None)
        db.set_gen_setting(call.from_user.id, "qwen_cfg", None)
        await _refresh(call, "🚫 Негатив прибрано, CFG = 1")
        return
    await state.set_state(QwenState.waiting_negative)
    await call.answer()
    await _nav(call,
               "🚫 <b>Негативний промпт</b>\n\n"
               "Qwen за замовчуванням працює з CFG = 1, де негатив ігнорується. "
               "Якщо задати негатив — CFG стане 2.5 і генерація триватиме <b>вдвічі довше</b>.\n\n"
               "Надішліть, чого НЕ має бути на зображенні:",
               parse_mode="HTML", reply_markup=_kb_back())


@router.message(QwenState.waiting_negative, F.text)
async def handle_negative(message: Message, state: FSMContext) -> None:
    await state.clear()
    neg = await translator.to_english(message.text.strip())
    db.set_gen_setting(message.from_user.id, "qwen_negative", neg)
    db.set_gen_setting(message.from_user.id, "qwen_cfg", 2.5)
    await show_menu(message, message.from_user.id)


@router.callback_query(QwenCB.filter(F.action == "help"))
async def cb_help(call: CallbackQuery) -> None:
    await call.answer()
    await _nav(call,
               f"{TITLE} — як користуватись\n\n"
               "✍️ <b>Промпт</b> — опишіть картинку словами, можна українською. "
               "Чим детальніше, тим краще: Qwen добре розуміє довгі описи.\n\n"
               "🔤 <b>Текст на картинці</b> — беріть його в лапки: "
               "<code>вивіска з написом \"КАВА\"</code>. Текст у лапках не перекладається.\n\n"
               "🫥 <b>Прозорий фон</b> — для стікерів, іконок, ігрових ассетів. "
               "Результат прийде також PNG-файлом з альфа-каналом.\n\n"
               "🖼 <b>Редагування</b> — у режимі Qwen надішліть фото з підписом-інструкцією: "
               "<code>заміни фон на пляж під час заходу сонця</code>. "
               "Або натисніть ✏️ під готовим результатом.\n\n"
               "⚡ Турбо ≈ 25 с, 💎 Якість ≈ 1.2 хв на RTX 3050. "
               "Генерації йдуть по черзі, у повідомленні видно прогрес і прев'ю.",
               parse_mode="HTML", reply_markup=_kb_back())


# ── prompt entry ──────────────────────────────────────────────────────────

INSPIRATIONS = [
    'A vintage travel poster of Kyiv at golden hour, the Motherland Monument on the hills above the Dnipro, '
    'bold retro headline at the top reads "KYIV", grainy lithograph texture',
    'A hand-lettered chalkboard menu in a cozy coffee shop, lines read "КАВА 45 ₴", "КРУАСАН 60 ₴" and '
    '"ГАРНОГО ДНЯ!", warm morning light, shallow depth of field',
    'Cinematic portrait of an old Carpathian shepherd in a sheepskin coat, dramatic side window light, '
    'deep wrinkles, 85mm photo, misty mountains in the background',
    'An isometric cozy gamer room at night, a ginger cat asleep on a gaming chair, rain on the window, '
    'soft neon purple and teal lighting, highly detailed 3D render',
    'Macro photograph of a dewdrop on a sunflower petal reflecting a blue sky, creamy bokeh',
    'A cyberpunk street food market in heavy rain, holographic signs, steam from noodle stalls, '
    '35mm film look, reflections on wet asphalt',
    'A fantasy RPG inventory icon of a glowing blue crystal sword, centered, clean painterly style',
    'A children\'s book illustration of a hedgehog in a knitted scarf carrying a lantern through '
    'an autumn forest, watercolor and ink',
    'A minimalist tech conference poster, large headline reads "FUTURE IS OPEN", abstract gradient '
    'waves, Swiss typography grid',
    'Food photography of Ukrainian varenyky with sour cream and fried onions on a rustic wooden table, '
    'top-down view, natural light',
]


@router.callback_query(QwenCB.filter(F.action == "prompt"))
async def cb_prompt(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.set_state(QwenState.waiting_prompt)
    await call.answer()
    await _nav(call,
               f"{TITLE}\n\n✍️ Напишіть, що намалювати.\n"
               "<i>Текст для напису на картинці — у лапках.</i>",
               parse_mode="HTML", reply_markup=_kb_back())


@router.callback_query(QwenCB.filter(F.action == "inspire"))
async def cb_inspire(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.clear()
    await call.answer("🎲 Випадковий промпт!")
    await generate(call.message, random.choice(INSPIRATIONS), call.from_user)


@router.message(QwenState.waiting_prompt, F.text)
async def handle_prompt(message: Message, state: FSMContext) -> None:
    await state.clear()
    await generate(message, message.text.strip(), message.from_user)


@router.message(QwenState.waiting_edit_prompt, F.text)
async def handle_edit_prompt(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    img = await _load_ref(message.bot, data)
    if img is None:
        await message.answer("❌ Не вдалося знайти вихідне зображення.", reply_markup=_kb_back())
        return
    await generate(message, message.text.strip(), message.from_user, input_image=img)


@router.message(QwenState.waiting_edit_prompt, F.photo)
async def handle_edit_photo_replace(message: Message, state: FSMContext) -> None:
    await state.update_data(file_id=message.photo[-1].file_id, file_path=None)
    if message.caption:
        await handle_edit_prompt_from_caption(message, state)
    else:
        await message.answer("✅ Фото оновлено. Тепер напишіть, що змінити:")


async def handle_edit_prompt_from_caption(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    img = await _load_ref(message.bot, data)
    await generate(message, message.caption.strip(), message.from_user, input_image=img)


async def _load_ref(bot: Bot, data: dict) -> Optional[bytes]:
    if data.get("file_path") and Path(data["file_path"]).exists():
        return Path(data["file_path"]).read_bytes()
    if data.get("file_id"):
        f   = await bot.get_file(data["file_id"])
        buf = await bot.download_file(f.file_path)
        return buf.read()
    return None


async def handle_photo_in_qwen_mode(message: Message, state: FSMContext) -> None:
    """Called from bot.handle_photo when the user has Qwen mode on."""
    caption = (message.caption or "").strip()
    if caption:
        f   = await message.bot.get_file(message.photo[-1].file_id)
        img = (await message.bot.download_file(f.file_path)).read()
        await generate(message, caption, message.from_user, input_image=img)
        return
    await state.set_state(QwenState.waiting_edit_prompt)
    await state.update_data(file_id=message.photo[-1].file_id, file_path=None)
    await message.answer(
        f"{TITLE} · 🖼 <b>Редагування</b>\n\nФото отримано. Напишіть, що змінити:\n"
        "<i>напр. «зроби зимовий пейзаж», «заміни фон на космос», «додай напис \"SALE\"»</i>",
        parse_mode="HTML", reply_markup=_kb_back())


# ── translation that keeps quoted text intact ─────────────────────────────

_QUOTE_RE = re.compile(r'"([^"]+)"|«([^»]+)»|“([^”]+)”|„([^“”]+)[“”]')


async def _prepare_prompt(prompt: str, translate: bool) -> str:
    if not translate:
        return prompt
    quoted: list[str] = []

    def keep(m: re.Match) -> str:
        quoted.append(next(g for g in m.groups() if g is not None))
        return f" QX{len(quoted) - 1}X "

    masked = _QUOTE_RE.sub(keep, prompt)
    en = await translator.to_english(masked)
    en = re.sub(r"QX\s*(\d+)\s*X", lambda m: f'"{quoted[int(m.group(1))]}"'
                if int(m.group(1)) < len(quoted) else m.group(0), en)
    # if the translator dropped a placeholder, append the lost text so it is not lost
    for t in quoted:
        if f'"{t}"' not in en:
            en += f', with text "{t}"'
    return re.sub(r"\s{2,}", " ", en).strip()


# ── generation ────────────────────────────────────────────────────────────

# recent results for the buttons under the image: entry_id → info
_results: "OrderedDict[str, dict]" = OrderedDict()
_RESULTS_MAX = 300
_running: dict = {}          # info about the job currently on the GPU


def _remember(eid: str, info: dict) -> None:
    _results[eid] = info
    while len(_results) > _RESULTS_MAX:
        _results.popitem(last=False)


async def generate(
    message: Message,
    prompt: str,
    user,
    input_image: Optional[bytes] = None,
    overrides: Optional[dict] = None,
    prepared_prompt: Optional[str] = None,
) -> None:
    allowed, _ = _ctx(user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    if not prompt:
        return
    q = cc.qwen_settings(db.get_gen_settings(user.id))
    if overrides:
        q.update(overrides)
    en_prompt = prepared_prompt or await _prepare_prompt(prompt, q["translate"])
    log.info("Qwen enqueue user=%d edit=%s q=%s prompt=%r", user.id, input_image is not None,
             q["quality"], en_prompt)

    ahead = gq.queue_len()
    eta   = cc.qwen_estimate(q)
    if ahead:
        text = (f"{TITLE}\n\n🕐 <b>В черзі</b> — попереду {ahead} {gq._inflect(ahead)}\n"
                f"<i>Оновлю повідомлення, коли почнеться генерація</i>")
    else:
        text = f"{TITLE}\n\n⏳ Готую генерацію… <i>(~{_fmt(eta)})</i>"
    status = await message.answer(text, parse_mode="HTML")
    cancel_kb = (InlineKeyboardBuilder()
                 .button(text="❌ Відмінити", callback_data=f"cncl:{status.message_id}")
                 .as_markup())

    job_info = {"user": user, "prompt": prompt, "en_prompt": en_prompt, "q": q,
                "input_image": input_image}

    async def runner(job: gq.GenJob) -> None:
        await _run_job(job, job_info)

    async def on_cancel(msg: Message) -> None:
        try:
            await status.edit_text("❌ Генерацію скасовано.", reply_markup=_kb_back())
        except TelegramBadRequest:
            pass

    async def _noop(*_a) -> None:
        return None

    await gq.enqueue(gq.GenJob(
        message=message, prompt=en_prompt, user_settings={}, status_msg=status,
        on_done=_noop, on_error=_noop, input_image=input_image,
        cancel_kb=cancel_kb, on_cancel=on_cancel, runner=runner,
        label=f"🌀 Qwen · {cc.QWEN_QUALITY_PRESETS[q['quality']]['label']}"
              + (" · редагування" if input_image is not None else ""),
        eta=eta,
    ))


# ── progress card ─────────────────────────────────────────────────────────

_STAGES = [("load", "Завантажую модель у відеокарту"), ("encode", "Читаю промпт"),
           ("sample", "Малюю"), ("decode", "Проявляю зображення")]
_STAGE_WEIGHT = {"queue": 0.0, "load": 0.04, "encode": 0.10, "sample": 0.12, "decode": 0.95, "save": 0.99}


def _overall(st: dict) -> float:
    if st["stage"] == "sample" and st["total"]:
        return 0.12 + 0.83 * st["step"] / st["total"]
    return _STAGE_WEIGHT.get(st["stage"], 0.0)


def _bar(frac: float, width: int = 16) -> str:
    filled = int(round(width * max(0.0, min(frac, 1.0))))
    return "▰" * filled + "▱" * (width - filled) + f"  {int(frac * 100)}%"


def _progress_caption(info: dict, st: dict) -> str:
    q     = info["q"]
    pre   = cc.QWEN_QUALITY_PRESETS[q["quality"]]
    mode  = "🖼 редагування" if info["input_image"] is not None else f"{q['width']}×{q['height']}"
    short = info["prompt"][:160] + ("…" if len(info["prompt"]) > 160 else "")
    order = [k for k, _ in _STAGES]
    cur   = st["stage"] if st["stage"] in order else ("decode" if st["stage"] == "save" else None)
    lines = [f"{TITLE} · {pre['label']} · {mode}", f"📝 <i>{_esc(short)}</i>", ""]
    if st["stage"] == "queue":
        ahead = st.get("comfy_ahead") or 0
        lines.append("🕐 <b>Чекаю вільну відеокарту</b>"
                     + (f" — попереду {ahead} {gq._inflect(ahead)}" if ahead else "…"))
    for key, label in _STAGES:
        if cur is None or order.index(key) > order.index(cur):
            icon = "▫️"
        elif key == cur and st["stage"] != "save":
            icon = "🔄"
        else:
            icon = "✅"
        extra = ""
        if key == "sample" and cur == "sample":
            extra = f" — крок <b>{st['step']}/{st['total']}</b>"
        lines.append(f"{icon} {label}{extra}")
    lines += ["", f"<code>{_bar(_overall(st))}</code>"]
    tail = f"⏱ {_fmt(st['elapsed'])}"
    if st.get("eta") is not None and st["stage"] not in ("decode", "save"):
        tail += f" · залишилось ~{_fmt(st['eta'])}"
    if st.get("sec_per_step"):
        tail += f" · {st['sec_per_step']:.1f} с/крок"
    lines.append(tail)
    return "\n".join(lines)


_card_cache: dict[tuple[int, int], bytes] = {}


def _placeholder(w: int, h: int) -> bytes:
    """A dark gradient card shown until the first live preview arrives."""
    scale = 640 / max(w, h)
    size  = (max(64, int(w * scale)), max(64, int(h * scale)))
    if size in _card_cache:
        return _card_cache[size]
    img  = Image.new("RGB", size)
    px   = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            t = (x / size[0] + y / size[1]) / 2
            px[x, y] = (int(40 + 60 * t), int(20 + 30 * (1 - t)), int(90 + 110 * (1 - t)))
    d = ImageDraw.Draw(img)
    try:
        big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size[0] // 12)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size[0] // 26)
    except OSError:
        big = small = ImageFont.load_default()
    cx, cy = size[0] // 2, size[1] // 2
    d.text((cx, cy - size[1] // 18), "Qwen-Image 2.1", font=big, fill=(235, 230, 255), anchor="mm")
    d.text((cx, cy + size[1] // 14), "малюю для вас…", font=small, fill=(190, 180, 230), anchor="mm")
    out = BytesIO()
    img.save(out, format="JPEG", quality=85)
    _card_cache[size] = out.getvalue()
    return _card_cache[size]


def _kb_stop(status_id: int) -> InlineKeyboardMarkup:
    return (InlineKeyboardBuilder()
            .button(text="⏹ Зупинити", callback_data=QwenCB(action="stop", value=str(status_id)).pack())
            .as_markup())


async def _run_job(job: gq.GenJob, info: dict) -> None:
    user    = info["user"]
    q       = info["q"]
    message = job.message
    try:
        await job.status_msg.delete()
    except TelegramBadRequest:
        pass

    st0 = {"stage": "queue", "step": 0, "total": q["steps"], "elapsed": 0.0,
           "eta": cc.qwen_estimate(q), "sec_per_step": None, "preview": None}
    card = await message.answer_photo(
        BufferedInputFile(_placeholder(q["width"], q["height"]), "qwen.jpg"),
        caption=_progress_caption(info, st0), parse_mode="HTML",
    )
    stop_kb = _kb_stop(card.message_id)
    _running.clear()
    _running.update({"card_id": card.message_id, "uid": user.id})

    last_text  = [0.0]
    last_media = [0.0]
    last_prev  = [None]

    async def on_status(st: dict) -> None:
        if st.get("prompt_id"):
            _running["prompt_id"] = st["prompt_id"]
        gq.report(job, _overall(st), st.get("eta"))
        now = time.monotonic()
        new_preview = st["preview"] is not None and st["preview"] is not last_prev[0]
        caption = _progress_caption(info, st)
        try:
            if new_preview and now - last_media[0] >= 4.0:
                last_media[0] = last_text[0] = now
                last_prev[0]  = st["preview"]
                await card.edit_media(
                    InputMediaPhoto(media=BufferedInputFile(st["preview"], "preview.jpg"),
                                    caption=caption, parse_mode="HTML"),
                    reply_markup=stop_kb,
                )
            elif now - last_text[0] >= 2.5:
                last_text[0] = now
                await card.edit_caption(caption=caption, parse_mode="HTML", reply_markup=stop_kb)
        except TelegramBadRequest:
            pass

    t0 = time.monotonic()
    try:
        data, seed = await cc.generate_qwen(info["en_prompt"], q, on_status=on_status,
                                            input_image=info["input_image"])
    except Exception as exc:
        log.exception("Qwen generation failed user=%d", user.id)
        stopped = "перервано" in str(exc)
        text = ("⏹ <b>Генерацію зупинено</b>" if stopped
                else gq._friendly_error(exc))
        try:
            await card.edit_caption(caption=f"{TITLE}\n\n{text}", parse_mode="HTML",
                                    reply_markup=_kb_back())
        except TelegramBadRequest:
            await message.answer(text, parse_mode="HTML", reply_markup=_kb_back())
        return
    finally:
        _running.clear()
    took = time.monotonic() - t0

    db.increment_gen_count(user.id)
    eid, fp = hist.save_image(user.id, data)
    mode    = "qwen-edit" if info["input_image"] is not None else "qwen21"
    with Image.open(BytesIO(data)) as im:
        w, h, has_alpha = im.width, im.height, im.mode in ("RGBA", "LA")
    hist.add_entry(eid, fp, user.id, user.username or "", info["en_prompt"], mode,
                   f"Qwen-Image 2.1 {cc.QWEN_QUALITY_PRESETS[q['quality']]['label']}", w, h)
    _remember(eid, {"prompt": info["prompt"], "en_prompt": info["en_prompt"], "seed": seed,
                    "q": dict(q), "file": fp, "uid": user.id,
                    "edit_src": info["input_image"] is not None})

    try:
        await card.delete()
    except TelegramBadRequest:
        pass

    caption = _result_caption(info, q, seed, took, w, h)
    kb = kb_result(eid, q)
    try:
        await message.answer_photo(FSInputFile(fp), caption=caption, parse_mode="HTML", reply_markup=kb)
    except TelegramBadRequest:
        # photo too big for Telegram's photo limits — send as a file
        await message.answer_document(FSInputFile(fp, filename=f"qwen_{eid}.png"),
                                      caption=caption, parse_mode="HTML", reply_markup=kb)
    if has_alpha or q["transparent"]:
        await message.answer_document(FSInputFile(fp, filename=f"qwen_{eid}.png"),
                                      caption="🫥 PNG з прозорим фоном")


def _result_caption(info: dict, q: dict, seed: int, took: float, w: int, h: int) -> str:
    pre = cc.QWEN_QUALITY_PRESETS[q["quality"]]
    shown = info["prompt"]
    lines = [f"✨ <b>Готово!</b>  {pre['label']} · {w}×{h} · ⏱ {_fmt(took)}"]
    if info["input_image"] is not None:
        lines.append("🖼 Редагування фото")
    lines.append(f"📝 {_esc(shown[:500])}{'…' if len(shown) > 500 else ''}")
    if info["en_prompt"] != info["prompt"]:
        en = info["en_prompt"]
        lines.append(f"🇬🇧 <i>{_esc(en[:300])}{'…' if len(en) > 300 else ''}</i>")
    lines.append(f"🎲 Сід: <code>{seed}</code>")
    text = "\n".join(lines)
    return text[:1020]


def kb_result(eid: str, q: dict) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🔄 Ще варіант",   callback_data=QwenCB(action="again",  value=eid).pack())
    if q["quality"] == "turbo":
        b.button(text="💎 У якості",  callback_data=QwenCB(action="hq",     value=eid).pack())
    b.button(text="✏️ Редагувати",   callback_data=QwenCB(action="edit",   value=eid).pack())
    b.button(text="🔒 Цей сід",      callback_data=QwenCB(action="lock",   value=eid).pack())
    b.button(text="📄 Оригінал PNG", callback_data=QwenCB(action="orig",   value=eid).pack())
    b.button(text="🌀 Меню Qwen",    callback_data=QwenCB(action="menu").pack())
    b.adjust(2, 2, 2)
    return b.as_markup()


def _result_or_alert(eid: Optional[str]) -> Optional[dict]:
    if eid and eid in _results:
        return _results[eid]
    if eid:
        # bot restarted — recover what we can from history
        for e in hist.get_entries():
            if e["id"] == eid:
                return {"prompt": e["prompt"], "en_prompt": e["prompt"], "seed": None,
                        "q": None, "file": e["file_path"], "uid": e["user_id"], "edit_src": False}
    return None


@router.callback_query(QwenCB.filter(F.action.in_({"again", "hq", "lock", "orig", "edit"})))
async def cb_result_action(call: CallbackQuery, callback_data: QwenCB, state: FSMContext) -> None:
    allowed, admin = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    r = _result_or_alert(callback_data.value)
    if r is None:
        await call.answer("⚠️ Цей результат більше недоступний.", show_alert=True)
        return
    if r["uid"] != call.from_user.id and not admin:
        await call.answer("⛔ Це не ваше зображення.", show_alert=True)
        return
    act = callback_data.action

    if act == "orig":
        await call.answer()
        if Path(r["file"]).exists():
            await call.message.answer_document(FSInputFile(r["file"], filename=f"qwen_{callback_data.value}.png"))
        else:
            await call.message.answer("❌ Файл уже видалено з історії.")
        return

    if act == "lock":
        if r["seed"] is None:
            await call.answer("⚠️ Сід цього зображення невідомий (бот перезапускався).", show_alert=True)
            return
        db.set_gen_setting(call.from_user.id, "qwen_seed", r["seed"])
        await call.answer(f"🔒 Сід {r['seed']} зафіксовано. Скинути — у меню Qwen.", show_alert=True)
        return

    if act == "edit":
        await state.set_state(QwenState.waiting_edit_prompt)
        await state.update_data(file_path=r["file"], file_id=None)
        await call.answer()
        await call.message.answer(
            "✏️ <b>Редагування цього зображення</b>\n\nНапишіть, що змінити:\n"
            "<i>напр. «зроби ніч», «додай кота на підвіконня», «зміни напис на \"ВІДЧИНЕНО\"»</i>",
            parse_mode="HTML", reply_markup=_kb_back())
        return

    # again / hq — re-run the same prompt
    overrides: dict = {"seed": None}
    if act == "hq":
        overrides = {"quality": "quality", "steps": cc.QWEN_QUALITY_PRESETS["quality"]["steps"],
                     "lora": False, "seed": r["seed"]}
    if r["q"]:
        for k in ("ratio", "mp", "width", "height", "transparent", "negative", "cfg"):
            overrides.setdefault(k, r["q"][k])
    await call.answer("🔄 Ставлю в чергу…" if act == "again" else "💎 Перемальовую в якості…")
    await generate(call.message, r["prompt"], call.from_user, overrides=overrides,
                   prepared_prompt=r["en_prompt"])


@router.callback_query(QwenCB.filter(F.action == "stop"))
async def cb_stop(call: CallbackQuery, callback_data: QwenCB) -> None:
    _, admin = _ctx(call.from_user)
    if not _running or str(_running.get("card_id")) != callback_data.value:
        await call.answer("⚠️ Ця генерація вже завершилась.", show_alert=True)
        return
    if _running.get("uid") != call.from_user.id and not admin:
        await call.answer("⛔ Це не ваша генерація.", show_alert=True)
        return
    await cc.interrupt(_running.get("prompt_id"))
    await call.answer("⏹ Зупиняю…")
