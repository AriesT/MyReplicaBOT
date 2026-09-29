"""
🎵 Music & sound section.

- 🎤 Songs: ACE-Step 1.5 turbo — style tags + your lyrics (or instrumental),
  vocal language, BPM, key, duration, LLM arrangement
- 🔊 Sounds: Stable Audio 3 — sound effects, one-shots, instruments, ambience
- live progress with stages, LLM token counter and a dancing equalizer
"""
import logging
import random
import re
import time
import uuid
from collections import OrderedDict
from typing import Optional

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

import comfy_client as cc
import gen_queue as gq
import tg_throttle
import translator
import users as db

log    = logging.getLogger(__name__)
router = Router(name="music")

TITLE = "🎵 <b>Музика та звуки</b>"


class MusCB(CallbackData, prefix="mu"):
    action: str
    value:  Optional[str] = None


class MusicState(StatesGroup):
    song_style  = State()
    song_lyrics = State()
    sound_text  = State()


GENRES: dict[str, tuple[str, str]] = {
    "rock":   ("🎸 Рок",        "Modern alternative rock, distorted electric guitars, punchy live drums, driving bass, energetic male vocals, anthemic chorus"),
    "pop":    ("🎹 Поп",        "Catchy modern pop, bright synths, warm female vocals, punchy drums, radio-ready polished production, uplifting chorus"),
    "lofi":   ("🎧 Lo-fi",      "Lo-fi hip hop, dusty vinyl crackle, mellow jazzy piano chords, soft boom-bap drums, warm bass, relaxed late-night mood"),
    "dance":  ("🕺 Денс",       "Euphoric EDM dance track, four-on-the-floor kick, big supersaw synth leads, pumping sidechain bass, festival drop"),
    "cinema": ("🎻 Кіно",       "Epic cinematic orchestral score, soaring strings, powerful brass, taiko drums, choir, heroic and emotional"),
    "hiphop": ("🎤 Хіп-хоп",    "Hard-hitting trap hip hop, heavy 808 bass, crisp hi-hat rolls, dark synths, confident rap vocals"),
    "folk":   ("🪕 Фолк",       "Ukrainian folk-pop, bandura and sopilka, acoustic guitar, warm layered vocals, joyful dance rhythm"),
    "jazz":   ("🎷 Джаз",       "Smooth jazz lounge, saxophone melody, brushed drums, upright bass, Rhodes piano, sultry vocals"),
    "metal":  ("🤘 Метал",      "Heavy metal, down-tuned chugging guitars, double kick drums, powerful screamed and clean vocals, aggressive"),
    "ballad": ("🌙 Балада",     "Emotional piano ballad, intimate vocals, soft strings swelling in the chorus, slow tempo, heartfelt"),
}

SAMPLE_LYRICS = [
    "[Verse 1]\nНад Дніпром світає, місто ще не спить\nКава на Хрещатику, серце стукотить\n\n"
    "[Chorus]\nКиїв, мій Київ, вогні над рікою\nКиїв, мій Київ, завжди ти зі мною",
    "[Verse 1]\nКіт на підвіконні дивиться у сніг\nВін сьогодні знову всіх перехитрив\n\n"
    "[Chorus]\nМяу-мяу, це мій вечір\nМяу-мяу, плед на плечі",
    "[Verse 1]\nМи летимо крізь ніч на старій машині\nРадіо грає, вітер у шибці\n\n"
    "[Chorus]\nЛіто, не йди, залишись хоч на мить\nЛіто, не йди, нам так добре горить",
]


# ── helpers ───────────────────────────────────────────────────────────────

def _ctx(user) -> tuple[bool, bool]:
    db.sync_id(user.id, user.username or "")
    return db.is_allowed(user.id, user.username or ""), db.is_admin(user.id, user.username or "")


def _fmt(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _nav(call: CallbackQuery, text: str, **kw) -> None:
    try:
        if call.message.audio or call.message.photo or call.message.document:
            await call.message.answer(text, **kw)
        else:
            await call.message.edit_text(text, **kw)
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            await call.message.answer(text, **kw)


def _kb_back(to: str = "menu") -> InlineKeyboardMarkup:
    return InlineKeyboardBuilder().button(
        text="🔙 Назад", callback_data=MusCB(action=to).pack()).as_markup()


def _picker(items: list[tuple[str, str]], current: str, action: str, back: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for value, label in items:
        b.button(text=f"{label}{' ✅' if value == current else ''}",
                 callback_data=MusCB(action=action, value=value).pack())
    b.button(text="🔙 Назад", callback_data=MusCB(action=back).pack())
    b.adjust(2)
    return b.as_markup()


# ── main menu ─────────────────────────────────────────────────────────────

def _menu_text(tg_id: int) -> str:
    s  = db.get_gen_settings(tg_id)
    sg = cc.song_settings(s)
    sd = cc.sound_settings(s)
    return (
        f"{TITLE}\n\n"
        "🎤 <b>Пісня</b> — ACE-Step 1.5: жанр + ваш текст → готовий трек з вокалом. "
        "Або інструментал.\n"
        f"   <i>{cc.SONG_LANGUAGES[sg['language']]} · {sg['duration']} с · {sg['bpm']} BPM · "
        f"{sg['key']} · ~{_fmt(cc.song_estimate(sg))}</i>\n\n"
        "🔊 <b>Звук</b> — Stable Audio 3: ефекти, семпли, інструменти, атмосфера.\n"
        f"   <i>{cc.SOUND_CATEGORIES[sd['category']][0]} · {sd['duration']} с · "
        f"~{_fmt(cc.sound_estimate(sd))}</i>"
    )


def kb_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🎤 Створити пісню",        callback_data=MusCB(action="song").pack())
    b.button(text="🔊 Створити звук",         callback_data=MusCB(action="sound").pack())
    b.button(text="🎲 Випадкова пісня",       callback_data=MusCB(action="song_random").pack())
    b.button(text="⚙️ Налаштування пісні",    callback_data=MusCB(action="song_cfg").pack())
    b.button(text="⚙️ Налаштування звуку",    callback_data=MusCB(action="sound_cfg").pack())
    b.button(text="❓ Підказки",              callback_data=MusCB(action="help").pack())
    b.button(text="🔙 Головне меню",          callback_data="menu:main")
    b.adjust(2, 1, 2, 1, 1)
    return b.as_markup()


async def show_menu(message: Message, tg_id: int) -> None:
    await message.answer(_menu_text(tg_id), parse_mode="HTML", reply_markup=kb_menu())


async def cmd_music(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    await state.clear()
    await show_menu(message, message.from_user.id)


@router.callback_query(MusCB.filter(F.action == "menu"))
async def cb_menu(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.clear()
    await call.answer()
    await _nav(call, _menu_text(call.from_user.id), parse_mode="HTML", reply_markup=kb_menu())


@router.callback_query(MusCB.filter(F.action == "help"))
async def cb_help(call: CallbackQuery) -> None:
    await call.answer()
    await _nav(call,
               f"{TITLE} — підказки\n\n"
               "🎤 <b>Стиль пісні</b> — жанр, інструменти, настрій, тип вокалу:\n"
               "<code>сумний інді-рок, жіночий вокал, піаніно, дощ за вікном</code>\n\n"
               "📝 <b>Текст</b> — можна просто рядками, бот сам розставить куплет/приспів. "
               "Або вручну тегами <code>[Verse]</code>, <code>[Chorus]</code>, <code>[Bridge]</code>, "
               "<code>[Outro]</code>.\n\n"
               "🧠 <b>LLM-аранжування</b> — мовна модель спершу «продумує» структуру треку. "
               "Якісніше, але довше. Можна вимкнути в налаштуваннях.\n\n"
               "🔊 <b>Звуки</b> — описуйте що звучить і де: "
               "<code>скрип старих дерев'яних дверей у замку, луна, далекий грім</code>.\n\n"
               "⏱ Пісня 1 хв ≈ 1 хв генерації, звук 10 с ≈ 15 с.",
               parse_mode="HTML", reply_markup=_kb_back())


# ── song settings ─────────────────────────────────────────────────────────

def _song_cfg_text(tg_id: int) -> str:
    m = cc.song_settings(db.get_gen_settings(tg_id))
    return (
        "🎤 <b>Налаштування пісні</b> (ACE-Step 1.5)\n\n"
        f"⏱ Тривалість: <b>{m['duration']} с</b>\n"
        f"🗣 Мова вокалу: <b>{cc.SONG_LANGUAGES[m['language']]}</b>\n"
        f"🥁 Темп: <b>{m['bpm']} BPM</b>\n"
        f"🎼 Тональність: <b>{m['key']}</b>\n"
        f"📏 Розмір: <b>{m['timesig']}/4</b>\n"
        f"🧠 LLM-аранжування: <b>{'так · ' + m['lm'].upper() if m['codes'] else 'ні (швидше)'}</b>\n\n"
        f"⏱ Орієнтовно: ~{_fmt(cc.song_estimate(m))}"
    )


def kb_song_cfg(tg_id: int) -> InlineKeyboardMarkup:
    m = cc.song_settings(db.get_gen_settings(tg_id))
    b = InlineKeyboardBuilder()
    b.button(text=f"⏱ {m['duration']} с",       callback_data=MusCB(action="pk_dur").pack())
    b.button(text=cc.SONG_LANGUAGES[m['language']], callback_data=MusCB(action="pk_lang").pack())
    b.button(text=f"🥁 {m['bpm']} BPM",          callback_data=MusCB(action="pk_bpm").pack())
    b.button(text=f"🎼 {m['key']}",              callback_data=MusCB(action="pk_key").pack())
    b.button(text=f"📏 {m['timesig']}/4",        callback_data=MusCB(action="tg_timesig").pack())
    b.button(text=("🧠 LLM: " + (m['lm'].upper() if m['codes'] else "вимк.")),
             callback_data=MusCB(action="pk_lm").pack())
    b.button(text="🔙 Назад",                     callback_data=MusCB(action="menu").pack())
    b.adjust(2, 2, 2, 1)
    return b.as_markup()


@router.callback_query(MusCB.filter(F.action == "song_cfg"))
async def cb_song_cfg(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await call.answer()
    await _nav(call, _song_cfg_text(call.from_user.id), parse_mode="HTML",
               reply_markup=kb_song_cfg(call.from_user.id))


@router.callback_query(MusCB.filter(F.action.in_({"pk_dur", "pk_lang", "pk_bpm", "pk_key", "pk_lm"})))
async def cb_song_pick(call: CallbackQuery, callback_data: MusCB) -> None:
    m = cc.song_settings(db.get_gen_settings(call.from_user.id))
    a = callback_data.action
    if a == "pk_dur":
        items, cur, title = [(str(d), f"{d} с") for d in cc.SONG_DURATIONS], str(m["duration"]), "⏱ Тривалість"
    elif a == "pk_lang":
        items, cur, title = list(cc.SONG_LANGUAGES.items()), m["language"], "🗣 Мова вокалу"
    elif a == "pk_bpm":
        names = {70: "балада", 90: "хіп-хоп", 100: "поп", 110: "поп-рок", 120: "стандарт",
                 128: "денс", 140: "трап/рок", 170: "drum&bass"}
        items, cur, title = [(str(v), f"{v} · {names[v]}") for v in cc.SONG_BPMS], str(m["bpm"]), "🥁 Темп"
    elif a == "pk_key":
        items, cur, title = [(k, k) for k in cc.SONG_KEYS], m["key"], "🎼 Тональність (major — світло, minor — сумно)"
    else:
        items = [("off", "⚡ Вимк. — найшвидше"), ("1.7b", "🧠 1.7B — баланс"), ("4b", "🧠🧠 4B — глибше, але ~6 хв на хвилину треку")]
        cur, title = (m["lm"] if m["codes"] else "off"), "🧠 LLM-аранжування"
    await call.answer()
    await _nav(call, f"<b>{title}</b>", parse_mode="HTML",
               reply_markup=_picker(items, cur, "set_" + a[3:], "song_cfg"))


@router.callback_query(MusCB.filter(F.action.in_({"set_dur", "set_lang", "set_bpm", "set_key", "set_lm", "tg_timesig"})))
async def cb_song_set(call: CallbackQuery, callback_data: MusCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    uid, a, v = call.from_user.id, callback_data.action, callback_data.value
    if a == "set_dur":
        db.set_gen_setting(uid, "song_duration", int(v))
    elif a == "set_lang":
        db.set_gen_setting(uid, "song_language", v)
    elif a == "set_bpm":
        db.set_gen_setting(uid, "song_bpm", int(v))
    elif a == "set_key":
        db.set_gen_setting(uid, "song_key", v)
    elif a == "set_lm":
        db.set_gen_setting(uid, "song_codes", v != "off")
        if v != "off":
            db.set_gen_setting(uid, "song_lm", v)
    else:
        m = cc.song_settings(db.get_gen_settings(uid))
        db.set_gen_setting(uid, "song_timesig", "4" if m["timesig"] == "3" else "3")
    await call.answer("✅")
    await _nav(call, _song_cfg_text(uid), parse_mode="HTML", reply_markup=kb_song_cfg(uid))


# ── sound settings ────────────────────────────────────────────────────────

def _sound_cfg_text(tg_id: int) -> str:
    m = cc.sound_settings(db.get_gen_settings(tg_id))
    mdl = "авто" if m["model_setting"] == "auto" else cc.SOUND_MODELS[m["model"]][0]
    return (
        "🔊 <b>Налаштування звуку</b> (Stable Audio 3)\n\n"
        f"🏷 Тип: <b>{cc.SOUND_CATEGORIES[m['category']][0]}</b>\n"
        f"⏱ Тривалість: <b>{m['duration']} с</b>\n"
        f"🎛 Модель: <b>{mdl}</b>"
        f"{'  → ' + cc.SOUND_MODELS[m['model']][0].split(' — ')[0] if m['model_setting'] == 'auto' else ''}\n\n"
        f"⏱ Орієнтовно: ~{_fmt(cc.sound_estimate(m))}"
    )


def kb_sound_cfg(tg_id: int) -> InlineKeyboardMarkup:
    m = cc.sound_settings(db.get_gen_settings(tg_id))
    b = InlineKeyboardBuilder()
    b.button(text=cc.SOUND_CATEGORIES[m["category"]][0], callback_data=MusCB(action="sp_cat").pack())
    b.button(text=f"⏱ {m['duration']} с",                callback_data=MusCB(action="sp_dur").pack())
    b.button(text="🎛 Модель",                            callback_data=MusCB(action="sp_mdl").pack())
    b.button(text="🔙 Назад",                             callback_data=MusCB(action="menu").pack())
    b.adjust(2, 1, 1)
    return b.as_markup()


@router.callback_query(MusCB.filter(F.action == "sound_cfg"))
async def cb_sound_cfg(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await call.answer()
    await _nav(call, _sound_cfg_text(call.from_user.id), parse_mode="HTML",
               reply_markup=kb_sound_cfg(call.from_user.id))


@router.callback_query(MusCB.filter(F.action.in_({"sp_cat", "sp_dur", "sp_mdl"})))
async def cb_sound_pick(call: CallbackQuery, callback_data: MusCB) -> None:
    m = cc.sound_settings(db.get_gen_settings(call.from_user.id))
    a = callback_data.action
    if a == "sp_cat":
        items, cur, title = [(k, v[0]) for k, v in cc.SOUND_CATEGORIES.items()], m["category"], "🏷 Тип звуку"
    elif a == "sp_dur":
        items, cur, title = [(str(d), f"{d} с") for d in cc.SOUND_DURATIONS], str(m["duration"]), "⏱ Тривалість"
    else:
        items = [("auto", "🤖 Авто (за типом звуку)")] + [(k, v[0]) for k, v in cc.SOUND_MODELS.items()]
        cur, title = m["model_setting"], "🎛 Модель"
    await call.answer()
    await _nav(call, f"<b>{title}</b>", parse_mode="HTML",
               reply_markup=_picker(items, cur, "ss_" + a[3:], "sound_cfg"))


@router.callback_query(MusCB.filter(F.action.in_({"ss_cat", "ss_dur", "ss_mdl"})))
async def cb_sound_set(call: CallbackQuery, callback_data: MusCB, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    uid, a, v = call.from_user.id, callback_data.action, callback_data.value
    key = {"ss_cat": "sound_category", "ss_dur": "sound_duration", "ss_mdl": "sound_model"}[a]
    db.set_gen_setting(uid, key, int(v) if a == "ss_dur" else v)
    await call.answer("✅")
    if await state.get_state() == MusicState.sound_text.state:
        await _nav(call, _sound_prompt_text(uid), parse_mode="HTML", reply_markup=kb_sound_prompt(uid))
    else:
        await _nav(call, _sound_cfg_text(uid), parse_mode="HTML", reply_markup=kb_sound_cfg(uid))


# ── song flow ─────────────────────────────────────────────────────────────

def kb_genres() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for key, (label, _) in GENRES.items():
        b.button(text=label, callback_data=MusCB(action="genre", value=key).pack())
    b.button(text="🔙 Назад", callback_data=MusCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()


def kb_lyrics() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🎼 Без слів — інструментал", callback_data=MusCB(action="instrumental").pack())
    b.button(text="📝 Взяти приклад тексту",    callback_data=MusCB(action="sample_lyrics").pack())
    b.button(text="🔙 Назад",                   callback_data=MusCB(action="song").pack())
    b.adjust(1)
    return b.as_markup()


@router.callback_query(MusCB.filter(F.action == "song"))
async def cb_song(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.set_state(MusicState.song_style)
    await call.answer()
    await _nav(call,
               "🎤 <b>Нова пісня · крок 1/2 — стиль</b>\n\n"
               "Оберіть жанр кнопкою або опишіть своїми словами:\n"
               "<i>«меланхолійний синтвейв, чоловічий вокал, нічне місто»</i>",
               parse_mode="HTML", reply_markup=kb_genres())


async def _ask_lyrics(target: Message, style_label: str) -> None:
    await target.answer(
        f"🎤 <b>Нова пісня · крок 2/2 — текст</b>\n\n🎨 Стиль: <i>{_esc(style_label[:200])}</i>\n\n"
        "Надішліть текст пісні. Можна просто рядками — бот сам розставить куплети й приспіви.\n"
        "Або оберіть варіант нижче:",
        parse_mode="HTML", reply_markup=kb_lyrics())


@router.callback_query(MusCB.filter(F.action == "genre"))
async def cb_genre(call: CallbackQuery, callback_data: MusCB, state: FSMContext) -> None:
    label, tags = GENRES.get(callback_data.value, GENRES["pop"])
    await state.update_data(style_tags=tags, style_label=label)
    await state.set_state(MusicState.song_lyrics)
    await call.answer()
    await _ask_lyrics(call.message, label)


@router.message(MusicState.song_style, F.text)
async def handle_style(message: Message, state: FSMContext) -> None:
    label = message.text.strip()
    tags  = await translator.to_english(label)
    await state.update_data(style_tags=tags, style_label=label)
    await state.set_state(MusicState.song_lyrics)
    await _ask_lyrics(message, label)


@router.callback_query(MusCB.filter(F.action.in_({"instrumental", "sample_lyrics"})))
async def cb_lyrics_choice(call: CallbackQuery, callback_data: MusCB, state: FSMContext) -> None:
    data = await state.get_data()
    if not data.get("style_tags"):
        await call.answer("⚠️ Почніть спочатку", show_alert=True); return
    await state.clear()
    await call.answer()
    lyrics = "" if callback_data.action == "instrumental" else random.choice(SAMPLE_LYRICS)
    await generate_song(call.message, call.from_user, data["style_tags"], data["style_label"], lyrics)


@router.message(MusicState.song_lyrics, F.text)
async def handle_lyrics(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    await generate_song(message, message.from_user, data.get("style_tags", GENRES["pop"][1]),
                        data.get("style_label", ""), message.text.strip())


@router.callback_query(MusCB.filter(F.action == "song_random"))
async def cb_song_random(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.clear()
    key = random.choice(list(GENRES))
    await call.answer(f"🎲 {GENRES[key][0]}!")
    await generate_song(call.message, call.from_user, GENRES[key][1], GENRES[key][0],
                        random.choice(SAMPLE_LYRICS), overrides={"language": "uk"})


_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*$", re.M)


def _structure_lyrics(text: str) -> str:
    """Add [Verse]/[Chorus] tags when the user sent plain lines."""
    if not text or _TAG_RE.search(text):
        return text
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if len(blocks) == 1:
        lines = blocks[0].splitlines()
        blocks = ["\n".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
    out, verse = [], 0
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            verse += 1
            out.append(f"[Verse {verse}]\n{b}")
        else:
            out.append(f"[Chorus]\n{b}")
    return "\n\n".join(out)


def _detect_language(lyrics: str, current: str) -> str:
    if re.search(r"[іїєґІЇЄҐ]", lyrics):
        return "uk"
    letters = [c for c in lyrics if c.isalpha()]
    if letters and all(ord(c) < 128 for c in letters) and current == "uk":
        return "en"
    return current


# ── sound flow ────────────────────────────────────────────────────────────

def _sound_prompt_text(tg_id: int) -> str:
    m = cc.sound_settings(db.get_gen_settings(tg_id))
    return (
        f"🔊 <b>Новий звук</b> · {cc.SOUND_CATEGORIES[m['category']][0]} · {m['duration']} с\n\n"
        "Опишіть, що має звучати:\n"
        "<i>«гроза в лісі, дощ по листю, далекий грім»\n«лазерний постріл у sci-fi грі»\n"
        "«удар по бочці 808 з довгим хвостом»</i>\n\n"
        "Тип і тривалість можна змінити кнопками 👇"
    )


def kb_sound_prompt(tg_id: int) -> InlineKeyboardMarkup:
    m = cc.sound_settings(db.get_gen_settings(tg_id))
    b = InlineKeyboardBuilder()
    for k, (label, _) in cc.SOUND_CATEGORIES.items():
        b.button(text=f"{label}{' ✅' if k == m['category'] else ''}",
                 callback_data=MusCB(action="ss_cat", value=k).pack())
    for d in (5, 10, 30):
        b.button(text=f"⏱ {d} с{' ✅' if d == m['duration'] else ''}",
                 callback_data=MusCB(action="ss_dur", value=str(d)).pack())
    b.button(text="🔙 Назад", callback_data=MusCB(action="menu").pack())
    b.adjust(2, 2, 1, 3, 1)
    return b.as_markup()


@router.callback_query(MusCB.filter(F.action == "sound"))
async def cb_sound(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.set_state(MusicState.sound_text)
    await call.answer()
    await _nav(call, _sound_prompt_text(call.from_user.id), parse_mode="HTML",
               reply_markup=kb_sound_prompt(call.from_user.id))


@router.message(MusicState.sound_text, F.text)
async def handle_sound(message: Message, state: FSMContext) -> None:
    await state.clear()
    await generate_sound(message, message.from_user, message.text.strip())


# ── queue + progress ──────────────────────────────────────────────────────

_results: "OrderedDict[str, dict]" = OrderedDict()
_running: dict = {}
_EQ = "▁▂▃▄▅▆▇█"

_SONG_STAGES  = [("load", "Завантажую модель"), ("compose", "Продумую аранжування"),
                 ("sample", "Синтезую звук"), ("decode", "Зводжу й кодую MP3")]
_SOUND_STAGES = [("load", "Завантажую модель"), ("encode", "Розбираю опис"),
                 ("sample", "Синтезую звук"), ("decode", "Кодую MP3")]


def _remember(rid: str, info: dict) -> None:
    _results[rid] = info
    while len(_results) > 300:
        _results.popitem(last=False)


def _progress_text(info: dict, st: dict) -> str:
    stages = _SONG_STAGES if info["kind"] == "song" else _SOUND_STAGES
    order  = [k for k, _ in stages]
    cur    = st["stage"] if st["stage"] in order else ("decode" if st["stage"] == "save" else None)
    head   = "🎤 <b>Створюю пісню</b>" if info["kind"] == "song" else "🔊 <b>Створюю звук</b>"
    lines  = [f"{head} · {info['summary']}", f"📝 <i>{_esc(info['label'][:150])}</i>", ""]
    if st["stage"] == "queue":
        ahead = st.get("comfy_ahead") or 0
        lines.append("🕐 <b>Чекаю вільну відеокарту</b>"
                     + (f" — попереду {ahead} {gq._inflect(ahead)}" if ahead else "…"))
    for key, label in stages:
        if cur is None or order.index(key) > order.index(cur):
            icon = "▫️"
        elif key == cur and st["stage"] != "save":
            icon = "🔄"
        else:
            icon = "✅"
        extra = ""
        if key == cur == "compose" and st.get("stage_total"):
            extra = f" — {st['stage_step']}/{st['stage_total']} токенів"
        if key == cur == "sample" and st.get("total"):
            extra = f" — крок {st['step']}/{st['total']}"
        if key == "compose" and not info.get("codes", True):
            label, icon = label + " (вимкнено)", "➖"
        lines.append(f"{icon} {label}{extra}")
    eq = "".join(random.choice(_EQ) for _ in range(18)) if cur else "▁" * 18
    frac = _frac(info, st)
    filled = int(round(16 * frac))
    lines += ["", f"🎚 <code>{eq}</code>",
              f"<code>{'▰' * filled}{'▱' * (16 - filled)}  {int(frac * 100)}%</code>",
              f"⏱ {_fmt(st['elapsed'])} · залишилось ~{_fmt(st['eta'] if st['stage'] != 'queue' else info['eta'])}"]
    return "\n".join(lines)


def _frac(info: dict, st: dict) -> float:
    s = st["stage"]
    if s == "load":
        return 0.05
    if s in ("compose", "encode"):
        if st.get("stage_total"):
            return 0.1 + 0.6 * st["stage_step"] / st["stage_total"]
        return 0.1
    if s == "sample":
        return 0.75 + 0.15 * (st["step"] / st["total"] if st.get("total") else 0)
    if s in ("decode", "save"):
        return 0.95
    return 0.0


def _kb_stop(mid: int) -> InlineKeyboardMarkup:
    return (InlineKeyboardBuilder()
            .button(text="⏹ Зупинити", callback_data=MusCB(action="stop", value=str(mid)).pack())
            .as_markup())


async def _enqueue(message: Message, user, info: dict) -> None:
    ahead = gq.queue_len()
    text = (f"🕐 <b>В черзі</b> — попереду {ahead} {gq._inflect(ahead)}" if ahead
            else f"⏳ Готую… <i>(~{_fmt(info['eta'])})</i>")
    status = await message.answer(text, parse_mode="HTML")
    cancel_kb = (InlineKeyboardBuilder()
                 .button(text="❌ Відмінити", callback_data=f"cncl:{status.message_id}")
                 .as_markup())

    async def runner(job: gq.GenJob) -> None:
        await _run(job, info, user)

    async def on_cancel(msg: Message) -> None:
        try:
            await status.edit_text("❌ Скасовано.", reply_markup=_kb_back())
        except TelegramBadRequest:
            pass

    async def _noop(*_a) -> None:
        return None

    kind = "🎤 Пісня" if info["kind"] == "song" else "🔊 Звук"
    await gq.enqueue(gq.GenJob(message=message, prompt=info["label"], user_settings={},
                               status_msg=status, on_done=_noop, on_error=_noop,
                               cancel_kb=cancel_kb, on_cancel=on_cancel, runner=runner,
                               label=f"{kind} · {info['summary']}", eta=info["eta"]))


async def _run(job: gq.GenJob, info: dict, user) -> None:
    status = job.status_msg
    stop_kb = _kb_stop(status.message_id)
    _running.clear()
    _running.update({"mid": status.message_id, "uid": user.id})
    last = [0.0]

    async def on_status(st: dict) -> None:
        if st.get("prompt_id"):
            _running["prompt_id"] = st["prompt_id"]
        gq.report(job, _frac(info, st), st.get("eta"))
        now = time.monotonic()
        if now - last[0] < 3.0:
            return
        text = _progress_text(info, st)
        if await tg_throttle.edit(status.chat.id, lambda: status.edit_text(
                text, parse_mode="HTML", reply_markup=stop_kb)):
            last[0] = now

    t0 = time.monotonic()
    try:
        if info["kind"] == "song":
            data, seed = await cc.generate_song(info["tags"], info["lyrics"], info["m"], on_status)
        else:
            data, seed = await cc.generate_sound(info["prompt_en"], info["m"], on_status)
    except Exception as exc:
        log.exception("audio generation failed user=%d", user.id)
        text = "⏹ <b>Зупинено</b>" if "перервано" in str(exc) else gq._friendly_error(exc)
        try:
            await status.edit_text(text, parse_mode="HTML", reply_markup=_kb_back())
        except TelegramBadRequest:
            await job.message.answer(text, parse_mode="HTML", reply_markup=_kb_back())
        return
    finally:
        _running.clear()
    took = time.monotonic() - t0
    db.increment_gen_count(user.id)

    rid = uuid.uuid4().hex[:10]
    info = dict(info, seed=seed)
    _remember(rid, info)
    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    m = info["m"]
    if info["kind"] == "song":
        title     = info["label"][:60] or "Пісня"
        performer = "MyReplicaBot · ACE-Step 1.5"
        caption   = (f"🎤 <b>Пісня готова!</b> · {m['duration']} с · {m['bpm']} BPM · {m['key']} · "
                     f"{cc.SONG_LANGUAGES[m['language']]}\n"
                     f"🎨 <i>{_esc(info['label'][:200])}</i>\n"
                     f"{'🎼 Інструментал' if not info['lyrics'] else '📝 З вокалом'} · ⏱ {_fmt(took)} · "
                     f"🎲 <code>{seed}</code>")
    else:
        title     = info["label"][:60] or "Звук"
        performer = "MyReplicaBot · Stable Audio 3"
        caption   = (f"🔊 <b>Звук готовий!</b> · {cc.SOUND_CATEGORIES[m['category']][0]} · "
                     f"{m['duration']} с\n📝 <i>{_esc(info['label'][:200])}</i>\n"
                     f"⏱ {_fmt(took)} · 🎲 <code>{seed}</code>")
    fname = re.sub(r"[^\w\- ]+", "", title, flags=re.U).strip()[:40] or "audio"
    await job.message.answer_audio(
        BufferedInputFile(data, filename=f"{fname}.mp3"),
        title=title, performer=performer, duration=int(m["duration"]),
        caption=caption[:1020], parse_mode="HTML", reply_markup=kb_result(rid, info),
    )


def kb_result(rid: str, info: dict) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🔄 Ще варіант", callback_data=MusCB(action="again", value=rid).pack())
    durations = cc.SONG_DURATIONS if info["kind"] == "song" else cc.SOUND_DURATIONS
    if info["m"]["duration"] < durations[-1]:
        b.button(text="⏩ Довша версія", callback_data=MusCB(action="longer", value=rid).pack())
    if info["kind"] == "song" and info["lyrics"]:
        b.button(text="🎼 Інструментал", callback_data=MusCB(action="instr", value=rid).pack())
    b.button(text="🎵 Меню", callback_data=MusCB(action="menu").pack())
    b.adjust(2, 2)
    return b.as_markup()


@router.callback_query(MusCB.filter(F.action.in_({"again", "longer", "instr"})))
async def cb_result(call: CallbackQuery, callback_data: MusCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    info = _results.get(callback_data.value or "")
    if info is None:
        await call.answer("⚠️ Цей результат більше недоступний (бот перезапускався).", show_alert=True)
        return
    info = dict(info, m=dict(info["m"], seed=None))
    if callback_data.action == "longer":
        durations = cc.SONG_DURATIONS if info["kind"] == "song" else cc.SOUND_DURATIONS
        info["m"]["duration"] = next((d for d in durations if d > info["m"]["duration"]), durations[-1])
        info["m"]["seed"] = info.get("seed")
    if callback_data.action == "instr":
        info["lyrics"] = ""
        info["m"]["seed"] = info.get("seed")
    info["eta"] = cc.song_estimate(info["m"]) if info["kind"] == "song" else cc.sound_estimate(info["m"])
    info["summary"] = _summary(info)
    await call.answer("🔄 У черзі…")
    await _enqueue(call.message, call.from_user, info)


def _summary(info: dict) -> str:
    m = info["m"]
    if info["kind"] == "song":
        return f"{m['duration']} с · {m['bpm']} BPM · {cc.SONG_LANGUAGES[m['language']].split(' ')[0]}"
    return f"{cc.SOUND_CATEGORIES[m['category']][0]} · {m['duration']} с"


async def generate_song(message: Message, user, tags: str, label: str, lyrics: str,
                        overrides: Optional[dict] = None) -> None:
    allowed, _ = _ctx(user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    m = cc.song_settings(db.get_gen_settings(user.id))
    if overrides:
        m.update(overrides)
    lyrics = _structure_lyrics(lyrics.strip())
    if lyrics:
        m["language"] = _detect_language(lyrics, m["language"])
    info = {"kind": "song", "tags": tags, "label": label or tags, "lyrics": lyrics, "m": m,
            "codes": m["codes"], "eta": cc.song_estimate(m)}
    info["summary"] = _summary(info)
    log.info("Song enqueue user=%d tags=%r lyrics=%d chars m=%s", user.id, tags, len(lyrics), m)
    await _enqueue(message, user, info)


async def generate_sound(message: Message, user, prompt: str) -> None:
    allowed, _ = _ctx(user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    if not prompt:
        return
    m  = cc.sound_settings(db.get_gen_settings(user.id))
    en = await translator.to_english(prompt)
    info = {"kind": "sound", "prompt_en": en, "label": prompt, "m": m, "eta": cc.sound_estimate(m)}
    info["summary"] = _summary(info)
    log.info("Sound enqueue user=%d prompt=%r m=%s", user.id, en, m)
    await _enqueue(message, user, info)


@router.callback_query(MusCB.filter(F.action == "stop"))
async def cb_stop(call: CallbackQuery, callback_data: MusCB) -> None:
    _, admin = _ctx(call.from_user)
    if not _running or str(_running.get("mid")) != callback_data.value:
        await call.answer("⚠️ Вже завершено.", show_alert=True)
        return
    if _running.get("uid") != call.from_user.id and not admin:
        await call.answer("⛔ Це не ваша генерація.", show_alert=True)
        return
    await cc.interrupt(_running.get("prompt_id"))
    await call.answer("⏹ Зупиняю…")
