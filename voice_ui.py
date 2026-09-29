"""
🗣 Voice section: OmniVoice (600+ languages incl. Ukrainian).

- 🔊 text → speech with a built-in designed voice or a cloned one
- 🧬 voice cloning from a 5–30 s voice message (consent confirmed first), per-user library
- 🔁 speech → speech: re-voice a voice message in another voice (Whisper + TTS)
- "Voice mode": plain text is spoken, voice messages are re-voiced
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import StateFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

import comfy_client as cc
import gen_queue as gq
import tg_throttle
import users as db

log    = logging.getLogger(__name__)
router = Router(name="voice")

TITLE      = "🗣 <b>Голос і мовлення</b>"
VOICES_DIR = Path(__file__).parent / "voices"
MAX_VOICES = 12
MAX_CLONE_SEC = 300         # accept up to 5 min of reading for cloning
REF_TTS_SEC   = 10.0        # OmniVoice: 3–10 s reference is best (longer = slower and worse)
REF_VC_SEC    = 25.0        # Seed-VC uses up to ~25 s of reference
MAX_TEXT   = 60000          # ≈8 500 words: ≈60 min of speech, ≈28 min of GPU time on RTX 3050
TG_MSG_MAX = 4096           # Telegram's own limit for one text message


class VoiceCB(CallbackData, prefix="vo"):
    action: str
    value:  Optional[str] = None


class VoiceState(StatesGroup):
    tts_text     = State()
    clone_audio  = State()
    clone_name   = State()
    s2s_audio    = State()


# ── helpers ───────────────────────────────────────────────────────────────

def _ctx(user) -> tuple[bool, bool]:
    db.sync_id(user.id, user.username or "")
    return db.is_allowed(user.id, user.username or ""), db.is_admin(user.id, user.username or "")


def _fmt(sec: Optional[float]) -> str:
    sec = max(0, int(sec or 0))
    return f"{sec // 60}:{sec % 60:02d}"


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _nav(call: CallbackQuery, text: str, **kw) -> None:
    try:
        if call.message.voice or call.message.audio or call.message.photo or call.message.document:
            await call.message.answer(text, **kw)
        else:
            await call.message.edit_text(text, **kw)
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            await call.message.answer(text, **kw)


def _kb_back(to: str = "menu") -> InlineKeyboardMarkup:
    return InlineKeyboardBuilder().button(text="🔙 Назад", callback_data=VoiceCB(action=to).pack()).as_markup()


def _cb(value: str) -> str:
    """':' is aiogram's CallbackData separator — voice keys like 'preset:x' travel as 'preset.x'."""
    return value.replace(":", ".", 1)


def _uncb(value: Optional[str]) -> str:
    return (value or "").replace(".", ":", 1)


def is_active(tg_id: int) -> bool:
    return bool(db.get_gen_settings(tg_id).get("voice_active"))


# ── voice library (per user, stored in gen_settings + files) ─────────────

def _my_voices(tg_id: int) -> list[dict]:
    return [v for v in (db.get_gen_settings(tg_id).get("voices") or []) if Path(v.get("file", "")).exists()]


def _save_voices(tg_id: int, voices: list[dict]) -> None:
    db.set_gen_setting(tg_id, "voices", voices or None)


def _resolve_voice(tg_id: int, key: Optional[str] = None) -> tuple[str, dict]:
    """key 'preset:<id>' | 'my:<vid>' → (display name, voice dict for comfy_client)."""
    key = key or cc.voice_settings(db.get_gen_settings(tg_id))["current"]
    if key.startswith("my:"):
        for v in _my_voices(tg_id):
            if v["id"] == key[3:]:
                vc = v.get("file_vc")
                return f"🧬 {v['name']}", {"ref": Path(v["file"]).read_bytes(), "ref_text": v.get("ref_text", ""),
                                           "ref_vc": Path(vc).read_bytes() if vc and Path(vc).exists() else None,
                                           "seed": 7}
    pid = key[7:] if key.startswith("preset:") else "f_young"
    label, instruct, seed = cc.VOICE_PRESETS.get(pid, cc.VOICE_PRESETS["f_young"])
    return label, {"instruct": instruct, "seed": seed}


# ── menu ──────────────────────────────────────────────────────────────────

def _menu_text(tg_id: int) -> str:
    m = cc.voice_settings(db.get_gen_settings(tg_id))
    name, _ = _resolve_voice(tg_id)
    active = is_active(tg_id)
    return (
        f"{TITLE}\n<i>OmniVoice · 600+ мов · клонування голосу за 5–30 с запису</i>\n\n"
        + ("🟢 <b>Режим озвучки увімкнено</b> — текст у чат = мова, голосове = переозвучка.\n\n" if active else "")
        + f"🎙 Голос: <b>{name}</b>\n"
        f"🌐 Мова: <b>{cc.VOICE_LANGUAGES[m['language']]}</b> · ⏩ Швидкість: <b>{m['speed']:g}×</b> · "
        f"{cc.VOICE_QUALITY[m['quality']][0]}\n"
        f"📚 Моїх голосів: <b>{len(_my_voices(tg_id))}</b>"
    )


def kb_menu(tg_id: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🔊 Озвучити текст",          callback_data=VoiceCB(action="tts").pack())
    b.button(text="🔁 Змінити голос у записі",  callback_data=VoiceCB(action="s2s").pack())
    b.button(text="🧬 Клонувати голос",          callback_data=VoiceCB(action="clone").pack())
    b.button(text="🎙 Обрати голос",             callback_data=VoiceCB(action="pick").pack())
    b.button(text=("🟢 Режим озвучки: увімк." if is_active(tg_id) else "⚪ Режим озвучки: вимк."),
             callback_data=VoiceCB(action="toggle").pack())
    b.button(text="⚙️ Налаштування",             callback_data=VoiceCB(action="cfg").pack())
    b.button(text="🔙 Головне меню",             callback_data="menu:main")
    b.adjust(1, 1, 2, 1, 1, 1)
    return b.as_markup()


async def cmd_voice(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    await state.clear()
    await message.answer(_menu_text(message.from_user.id), parse_mode="HTML",
                         reply_markup=kb_menu(message.from_user.id))


@router.callback_query(VoiceCB.filter(F.action == "menu"))
async def cb_menu(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True); return
    await state.clear()
    await call.answer()
    await _nav(call, _menu_text(call.from_user.id), parse_mode="HTML", reply_markup=kb_menu(call.from_user.id))


@router.callback_query(VoiceCB.filter(F.action == "toggle"))
async def cb_toggle(call: CallbackQuery) -> None:
    on = not is_active(call.from_user.id)
    db.set_gen_setting(call.from_user.id, "voice_active", True if on else None)
    if on:   # modes are mutually exclusive
        db.set_gen_setting(call.from_user.id, "qwen_active", None)
    await call.answer("🟢 Тепер текст → мова, голосове → переозвучка" if on else "⚪ Режим озвучки вимкнено")
    await _nav(call, _menu_text(call.from_user.id), parse_mode="HTML", reply_markup=kb_menu(call.from_user.id))


# ── settings ──────────────────────────────────────────────────────────────

def _picker(items: list[tuple[str, str]], cur: str, action: str, back: str = "cfg") -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v, label in items:
        b.button(text=f"{label}{' ✅' if v == cur else ''}", callback_data=VoiceCB(action=action, value=v).pack())
    b.button(text="🔙 Назад", callback_data=VoiceCB(action=back).pack())
    b.adjust(2)
    return b.as_markup()


@router.callback_query(VoiceCB.filter(F.action == "cfg"))
async def cb_cfg(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    m = cc.voice_settings(db.get_gen_settings(call.from_user.id))
    b = InlineKeyboardBuilder()
    b.button(text=f"🌐 {cc.VOICE_LANGUAGES[m['language']]}", callback_data=VoiceCB(action="pk_lang").pack())
    b.button(text=f"⏩ {m['speed']:g}×",                    callback_data=VoiceCB(action="pk_speed").pack())
    b.button(text=cc.VOICE_QUALITY[m["quality"]][0],         callback_data=VoiceCB(action="pk_q").pack())
    b.button(text=("🎭 Зміна голосу: з інтонацією" if m["vc_method"] == "intonation"
                   else "📝 Зміна голосу: через текст"), callback_data=VoiceCB(action="tg_method").pack())
    b.button(text=f"🎤 Спів: {'так' if m['vc_singing'] else 'ні'}", callback_data=VoiceCB(action="tg_sing").pack())
    b.button(text=f"🎚 Тон: {m['vc_pitch']:+d}",               callback_data=VoiceCB(action="pk_pitch").pack())
    b.button(text="🔙 Назад",                                 callback_data=VoiceCB(action="menu").pack())
    b.adjust(3, 1, 2, 1)
    await call.answer()
    await _nav(call, "⚙️ <b>Налаштування мовлення</b>\n\n"
                     "🌐 <b>Мова</b> — якою мовою читати текст («Авто» визначає сама).\n"
                     "⏩ <b>Швидкість</b> — темп мовлення.\n"
                     "💎 <b>Якість</b> — більше кроків = чистіший звук, трохи довше.\n\n"
                     "🔁 <b>Зміна голосу в записі:</b>\n"
                     "🎭 <b>з інтонацією</b> (Seed-VC) — ваші паузи, емоції, темп, але чужий тембр;\n"
                     "📝 <b>через текст</b> (OmniVoice) — слова розпізнаються й начитуються наново.\n"
                     "🎤 <b>Спів</b> — режим для пісень (зберігає мелодію).\n"
                     "🎚 <b>Тон</b> — зсув у півтонах (напр. +12 / −12 для зміни чоловічий ↔ жіночий у співі).",
               parse_mode="HTML", reply_markup=b.as_markup())


@router.callback_query(VoiceCB.filter(F.action.in_({"tg_method", "tg_sing"})))
async def cb_toggle_vc(call: CallbackQuery, callback_data: VoiceCB, state: FSMContext) -> None:
    m = cc.voice_settings(db.get_gen_settings(call.from_user.id))
    if callback_data.action == "tg_method":
        db.set_gen_setting(call.from_user.id, "vc_method", "text" if m["vc_method"] == "intonation" else None)
    else:
        db.set_gen_setting(call.from_user.id, "vc_singing", not m["vc_singing"])
    await call.answer("✅")
    await cb_cfg(call, state)


@router.callback_query(VoiceCB.filter(F.action.in_({"pk_lang", "pk_speed", "pk_q", "pk_pitch"})))
async def cb_pick_cfg(call: CallbackQuery, callback_data: VoiceCB) -> None:
    m = cc.voice_settings(db.get_gen_settings(call.from_user.id))
    a = callback_data.action
    if a == "pk_lang":
        kb = _picker(list(cc.VOICE_LANGUAGES.items()), m["language"], "set_lang")
    elif a == "pk_pitch":
        kb = _picker([(str(p), f"{p:+d}") for p in cc.VC_PITCHES], str(m["vc_pitch"]), "set_pitch")
    elif a == "pk_speed":
        kb = _picker([(f"{s:g}", f"{s:g}×") for s in cc.VOICE_SPEEDS], f"{m['speed']:g}", "set_speed")
    else:
        kb = _picker([(k, f"{v[0]} ({v[1]} кроків)") for k, v in cc.VOICE_QUALITY.items()], m["quality"], "set_q")
    await call.answer()
    await _nav(call, "⚙️ Оберіть значення:", reply_markup=kb)


@router.callback_query(VoiceCB.filter(F.action.in_({"set_lang", "set_speed", "set_q", "set_pitch"})))
async def cb_set_cfg(call: CallbackQuery, callback_data: VoiceCB, state: FSMContext) -> None:
    key = {"set_lang": "voice_language", "set_speed": "voice_speed", "set_q": "voice_quality",
           "set_pitch": "vc_pitch"}[callback_data.action]
    val = (float(callback_data.value) if key == "voice_speed"
           else int(callback_data.value) if key == "vc_pitch" else callback_data.value)
    db.set_gen_setting(call.from_user.id, key, val)
    await call.answer("✅")
    await cb_cfg(call, state)


# ── voice picker / library ────────────────────────────────────────────────

@router.callback_query(VoiceCB.filter(F.action == "pick"))
async def cb_pick(call: CallbackQuery) -> None:
    cur = cc.voice_settings(db.get_gen_settings(call.from_user.id))["current"]
    b = InlineKeyboardBuilder()
    for pid, (label, _, _) in cc.VOICE_PRESETS.items():
        k = f"preset:{pid}"
        b.button(text=f"{label}{' ✅' if k == cur else ''}", callback_data=VoiceCB(action="use", value=_cb(k)).pack())
    mine = _my_voices(call.from_user.id)
    for v in mine:
        k = f"my:{v['id']}"
        b.button(text=f"🧬 {v['name']}{' ✅' if k == cur else ''}", callback_data=VoiceCB(action="use", value=_cb(k)).pack())
    b.button(text="🎧 Прослухати поточний", callback_data=VoiceCB(action="demo").pack())
    if mine:
        b.button(text="🗑 Видалити мій голос", callback_data=VoiceCB(action="del_list").pack())
    b.button(text="🔙 Назад", callback_data=VoiceCB(action="menu").pack())
    b.adjust(2)
    await call.answer()
    await _nav(call, "🎙 <b>Оберіть голос</b>\n\nВбудовані голоси + 🧬 ваші клоновані.",
               parse_mode="HTML", reply_markup=b.as_markup())


@router.callback_query(VoiceCB.filter(F.action == "use"))
async def cb_use(call: CallbackQuery, callback_data: VoiceCB) -> None:
    key = _uncb(callback_data.value)
    db.set_gen_setting(call.from_user.id, "voice_current", key)
    name, _ = _resolve_voice(call.from_user.id, key)
    await call.answer(f"✅ {name}")
    await cb_pick(call)


@router.callback_query(VoiceCB.filter(F.action == "demo"))
async def cb_demo(call: CallbackQuery) -> None:
    await call.answer("🎧 Генерую зразок…")
    m = cc.voice_settings(db.get_gen_settings(call.from_user.id))
    demo = {"uk": "Привіт! Ось так звучить цей голос. Напишіть будь-який текст — і я його озвучу.",
            "en": "Hi! This is how this voice sounds. Send me any text and I will read it aloud."}
    await speak(call.message, call.from_user, demo.get(m["language"], demo["uk"]))


@router.callback_query(VoiceCB.filter(F.action == "del_list"))
async def cb_del_list(call: CallbackQuery) -> None:
    b = InlineKeyboardBuilder()
    for v in _my_voices(call.from_user.id):
        b.button(text=f"🗑 {v['name']}", callback_data=VoiceCB(action="del", value=v["id"]).pack())
    b.button(text="🔙 Назад", callback_data=VoiceCB(action="pick").pack())
    b.adjust(1)
    await call.answer()
    await _nav(call, "🗑 Який голос видалити? (запис-зразок теж буде видалено)", reply_markup=b.as_markup())


@router.callback_query(VoiceCB.filter(F.action == "del"))
async def cb_del(call: CallbackQuery, callback_data: VoiceCB) -> None:
    uid = call.from_user.id
    keep = []
    for v in _my_voices(uid):
        if v["id"] == callback_data.value:
            for field in ("file", "file_vc", "file_full"):
                if v.get(field):
                    Path(v[field]).unlink(missing_ok=True)
        else:
            keep.append(v)
    _save_voices(uid, keep)
    if cc.voice_settings(db.get_gen_settings(uid))["current"] == f"my:{callback_data.value}":
        db.set_gen_setting(uid, "voice_current", None)
    await call.answer("🗑 Видалено")
    await cb_pick(call)


# ── cloning ───────────────────────────────────────────────────────────────

@router.callback_query(VoiceCB.filter(F.action == "clone"))
async def cb_clone(call: CallbackQuery) -> None:
    if len(_my_voices(call.from_user.id)) >= MAX_VOICES:
        await call.answer(f"⚠️ Максимум {MAX_VOICES} голосів. Видаліть непотрібний.", show_alert=True)
        return
    b = InlineKeyboardBuilder()
    b.button(text="✅ Це мій голос / маю дозвіл власника", callback_data=VoiceCB(action="clone_ok").pack())
    b.button(text="🔙 Назад", callback_data=VoiceCB(action="menu").pack())
    b.adjust(1)
    await call.answer()
    await _nav(call,
               "🧬 <b>Клонування голосу</b>\n\n"
               "Бот запам'ятає тембр із запису й зможе говорити ним будь-який текст.\n\n"
               "⚠️ Клонуйте лише <b>власний голос</b> або голос людини, яка <b>дала згоду</b>. "
               "Імітація чужого голосу без дозволу для обману заборонена.\n\n"
               "Натисніть, щоб підтвердити:",
               parse_mode="HTML", reply_markup=b.as_markup())


@router.callback_query(VoiceCB.filter(F.action == "clone_ok"))
async def cb_clone_ok(call: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(VoiceState.clone_audio)
    await call.answer()
    await _nav(call,
               "🎤 Надішліть <b>голосове або аудіофайл від 10 секунд до 5 хвилин</b>.\n\n"
               "Бот сам вибере найчистіший фрагмент: ~10 с для озвучки та ~25 с для зміни голосу "
               "(довший зразок для озвучки лише сповільнює й погіршує клон — так радять автори моделі).\n\n"
               "💡 Для найкращого результату: тиха кімната, без музики, звичайний тон, мікрофон на одній відстані. "
               "Хвилина-дві начитки дає боту з чого вибрати.",
               parse_mode="HTML", reply_markup=_kb_back())


async def _download(message: Message) -> Optional[tuple[bytes, float]]:
    media = message.voice or message.audio or message.document
    if media is None:
        return None
    f = await message.bot.get_file(media.file_id)
    data = (await message.bot.download_file(f.file_path)).read()
    return data, float(getattr(media, "duration", 0) or 0)


@router.message(VoiceState.clone_audio, F.voice | F.audio | F.document)
async def handle_clone_audio(message: Message, state: FSMContext) -> None:
    got = await _download(message)
    if got is None:
        return
    data, dur = got
    if dur and (dur < 3 or dur > MAX_CLONE_SEC):
        await message.answer(f"⏱ Потрібен запис від 10 секунд до {MAX_CLONE_SEC // 60} хвилин. Спробуйте ще раз.")
        return
    status = await message.answer("🧬 Аналізую запис — шукаю найчистіший фрагмент…")
    try:
        ref = await cc.select_voice_ref(data, REF_TTS_SEC) if not dur or dur > REF_TTS_SEC + 2 else data
        ref_vc = await cc.select_voice_ref(data, REF_VC_SEC) if dur and dur > REF_VC_SEC + 4 else None
        await status.edit_text("🧬 Розпізнаю, що сказано у фрагменті…")
        text = await cc.transcribe_audio(ref)
    except Exception as exc:
        log.exception("clone preparation failed")
        await status.edit_text(gq._friendly_error(exc), parse_mode="HTML", reply_markup=_kb_back())
        await state.clear()
        return
    uid = message.from_user.id
    ref_path = _stash(uid, ref)
    vid = Path(ref_path).stem.split("_", 1)[1]
    extra = {}
    if ref_vc:
        p = VOICES_DIR / f"{uid}_{vid}_vc.ogg"
        p.write_bytes(ref_vc)
        extra["clone_vc_path"] = str(p)
    if dur and dur > REF_TTS_SEC + 2:
        p = VOICES_DIR / f"{uid}_{vid}_full.ogg"      # kept for a future personal (LoRA) voice model
        p.write_bytes(data)
        extra["clone_full_path"] = str(p)
    await state.update_data(clone_bytes_path=ref_path, clone_text=text, clone_dur=dur, **extra)
    await state.set_state(VoiceState.clone_name)
    picked = (f"🎯 З {int(dur)} с запису вибрано найкращі ~{int(REF_TTS_SEC)} с для озвучки"
              + (f" і ~{int(REF_VC_SEC)} с для зміни голосу" if ref_vc else "") + ".\n\n") if dur and dur > REF_TTS_SEC + 2 else ""
    await status.edit_text(
        f"✅ {picked}Розпізнано у фрагменті:\n<i>«{_esc(text[:400])}»</i>\n\n"
        "✍️ Як назвати цей голос? (напр. «Мій голос», «Тато»)",
        parse_mode="HTML", reply_markup=_kb_back())


def _stash(uid: int, data: bytes) -> str:
    VOICES_DIR.mkdir(exist_ok=True)
    p = VOICES_DIR / f"{uid}_{uuid.uuid4().hex[:10]}.ogg"
    p.write_bytes(data)
    return str(p)


@router.message(VoiceState.clone_name, F.text)
async def handle_clone_name(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    path = data.get("clone_bytes_path")
    if not path or not Path(path).exists():
        await message.answer("⚠️ Запис загубився, почніть спочатку.", reply_markup=_kb_back())
        return
    uid = message.from_user.id
    vid = Path(path).stem.split("_", 1)[1]
    entry = {"id": vid, "name": message.text.strip()[:40], "file": path,
             "ref_text": data.get("clone_text", ""), "created": int(time.time())}
    for key, field in (("clone_vc_path", "file_vc"), ("clone_full_path", "file_full")):
        if data.get(key):
            entry[field] = data[key]
    voices = _my_voices(uid) + [entry]
    _save_voices(uid, voices)
    db.set_gen_setting(uid, "voice_current", f"my:{vid}")
    await message.answer(f"🧬 Голос <b>{_esc(message.text.strip()[:40])}</b> збережено й обрано!\n"
                         "Зараз озвучу ним тестову фразу 👇", parse_mode="HTML")
    await speak(message, message.from_user,
                "Привіт! Це мій новий цифровий голос. Тепер я можу озвучити будь-який ваш текст.")


# ── TTS & speech-to-speech ────────────────────────────────────────────────

@router.callback_query(VoiceCB.filter(F.action == "tts"))
async def cb_tts(call: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(VoiceState.tts_text)
    name, _ = _resolve_voice(call.from_user.id)
    await call.answer()
    await _nav(call, f"🔊 <b>Озвучення</b> · {name}\n\n"
                     f"Надішліть текст повідомленням (до {TG_MSG_MAX} символів — ліміт Telegram) "
                     f"або <b>файлом .txt / .docx</b> (до {MAX_TEXT:,} символів ≈ 8 500 слів, "
                     f"≈ {MAX_TEXT // 950} хв аудіо).\n"
                     "<i>Порада: розділові знаки = паузи та інтонація.</i>".replace(",", " "),
               parse_mode="HTML", reply_markup=_kb_back())


@router.message(VoiceState.tts_text, F.text)
async def handle_tts_text(message: Message, state: FSMContext) -> None:
    await state.clear()
    await speak(message, message.from_user, message.text.strip())


_TEXT_EXT = (".txt", ".md", ".srt", ".text", ".docx")


def _docx_text(raw: bytes) -> str:
    """Plain text of a .docx (paragraphs → lines) using only the standard library."""
    import io
    import re as _re
    import zipfile
    from xml.etree import ElementTree as ET
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    lines = []
    for p in root.iter(f"{ns}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag in (f"{ns}tab",):
                parts.append(" ")
            elif node.tag in (f"{ns}br", f"{ns}cr"):
                parts.append("\n")
        lines.append("".join(parts))
    return _re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


async def _read_text_file(message: Message) -> Optional[str]:
    doc = message.document
    name = (doc.file_name or "").lower() if doc else ""
    if doc is None or not name.endswith(_TEXT_EXT):
        return None
    if (doc.file_size or 0) > 20_000_000:
        await message.answer("📄 Файл завеликий (макс. 20 МБ).")
        return ""
    f = await message.bot.get_file(doc.file_id)
    raw = (await message.bot.download_file(f.file_path)).read()
    if name.endswith(".docx"):
        try:
            return _docx_text(raw)
        except Exception:
            log.exception("docx parse failed")
            await message.answer("📄 Не вдалося прочитати .docx — збережіть як .txt і надішліть ще раз.")
            return ""
    for enc in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    await message.answer("📄 Не вдалося прочитати файл — збережіть його в UTF-8.")
    return ""


@router.message(VoiceState.tts_text, F.document)
async def handle_tts_file(message: Message, state: FSMContext) -> None:
    text = await _read_text_file(message)
    if text is None:
        await message.answer("📄 Надішліть текст або файл .txt / .docx / .md / .srt")
        return
    await state.clear()
    if text:
        await speak(message, message.from_user, text)


@router.message(StateFilter(None), F.document)
async def handle_text_file_anywhere(message: Message) -> None:
    """In voice mode a .txt file is read aloud."""
    allowed, _ = _ctx(message.from_user)
    if not allowed or not is_active(message.from_user.id):
        return
    text = await _read_text_file(message)
    if text:
        await speak(message, message.from_user, text)


@router.callback_query(VoiceCB.filter(F.action == "s2s"))
async def cb_s2s(call: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(VoiceState.s2s_audio)
    name, _ = _resolve_voice(call.from_user.id)
    await call.answer()
    m = cc.voice_settings(db.get_gen_settings(call.from_user.id))
    how = ("🎭 <b>з інтонацією</b> — ваші емоції й паузи збережуться, зміниться лише тембр"
           + (" · 🎤 режим співу" if m["vc_singing"] else "")) if m["vc_method"] == "intonation" else \
          "📝 <b>через текст</b> — бот розпізнає слова й начитає їх наново"
    await _nav(call, f"🔁 <b>Зміна голосу</b> → {name}\n\n{how}\n\n"
                     "Надішліть голосове повідомлення або аудіо.\n"
                     "<i>Голос — «🎙 Обрати голос», спосіб — «⚙️ Налаштування».</i>",
               parse_mode="HTML", reply_markup=_kb_back())


@router.message(VoiceState.s2s_audio, F.voice | F.audio | F.document)
async def handle_s2s(message: Message, state: FSMContext) -> None:
    await state.clear()
    got = await _download(message)
    if got:
        await speak(message, message.from_user, "", source_audio=got[0])


@router.message(StateFilter(None), F.voice | F.audio)
async def handle_voice_anywhere(message: Message) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        return
    if not is_active(message.from_user.id):
        await message.answer("🗣 Щоб переозвучити голосове, увімкніть <b>режим озвучки</b> у /voice "
                             "або натисніть «🔁 Змінити голос у записі».", parse_mode="HTML")
        return
    got = await _download(message)
    if got:
        await speak(message, message.from_user, "", source_audio=got[0])


# ── queue + progress ──────────────────────────────────────────────────────

_STAGES = [("load", "Готую голос"), ("encode", "Розпізнаю мову"), ("sample", "Синтезую мовлення"),
           ("save", "Кодую аудіо")]
_VC_STAGES = [("load", "Завантажую запис"), ("encode", "Створюю зразок голосу"),
              ("sample", "Перетворюю голос (Seed-VC)"), ("save", "Кодую аудіо")]
_running: dict = {}


def _frac(st: dict) -> float:
    s = st["stage"]
    if s == "load":
        return 0.1
    if s == "encode":
        return 0.25
    if s == "sample":
        tot = st.get("stage_total") or st.get("total") or 1
        done = st.get("stage_step") or st.get("step") or 0
        return 0.3 + 0.6 * done / tot
    if s == "save":
        return 0.95
    return 0.0


def _progress_text(info: dict, st: dict) -> str:
    stages = _VC_STAGES if info.get("vc") else _STAGES
    order = [k for k, _ in stages]
    cur = st["stage"] if st["stage"] in order else None
    head = ("🎭 <b>Змінюю голос зі збереженням інтонації</b>" if info.get("vc")
            else "🔁 <b>Переозвучую запис</b>" if info["s2s"] else "🔊 <b>Озвучую текст</b>")
    lines = [f"{head} · {info['voice_name']}"]
    if info["text"]:
        lines.append(f"📝 <i>{_esc(info['text'][:150])}{'…' if len(info['text']) > 150 else ''}</i>")
    lines.append("")
    if st["stage"] == "queue":
        ahead = st.get("comfy_ahead") or 0
        lines.append("🕐 <b>Чекаю вільну відеокарту</b>" + (f" — попереду {ahead} {gq._inflect(ahead)}" if ahead else "…"))
    for key, label in stages:
        if key == "encode" and (not info["s2s"] or (info.get("vc") and info.get("has_ref"))):
            continue
        if cur is None or order.index(key) > order.index(cur):
            icon = "▫️"
        elif key == cur:
            icon = "🔄"
        else:
            icon = "✅"
        extra = ""
        if key == cur == "sample" and not info.get("vc"):
            tot = st.get("stage_total") or st.get("total") or 0
            done = st.get("stage_step") or st.get("step") or 0
            if tot > 2:                       # 1 = preparing the voice, then one step per text fragment
                extra = f" — фрагмент <b>{max(done - 1, 0)}/{tot - 1}</b>"
        lines.append(f"{icon} {label}{extra}")
    frac = _frac(st)
    filled = int(round(16 * frac))
    lines += ["", f"<code>{'▰' * filled}{'▱' * (16 - filled)}  {int(frac * 100)}%</code>",
              f"⏱ {_fmt(st['elapsed'])} · залишилось ~{_fmt(st.get('eta'))}"]
    return "\n".join(lines)


async def speak(message: Message, user, text: str, source_audio: Optional[bytes] = None) -> None:
    allowed, _ = _ctx(user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    if source_audio is None and not text:
        return
    if len(text) > MAX_TEXT:
        await message.answer(f"✂️ Задовгий текст: {len(text):,} символів ({len(text.split()):,} слів), "
                             f"максимум {MAX_TEXT:,} (≈8 500 слів). Розбийте його на частини.".replace(",", " "))
        return
    m = cc.voice_settings(db.get_gen_settings(user.id))
    if source_audio is not None:
        m = dict(m, language="auto")
    voice_name, voice = _resolve_voice(user.id)
    eta = cc.voice_estimate(len(text) or 200, m, source_audio is not None, bool(voice.get("ref")))
    vc = source_audio is not None and m["vc_method"] == "intonation"
    info = {"text": text, "s2s": source_audio is not None, "voice_name": voice_name,
            "vc": vc, "has_ref": bool(voice.get("ref"))}
    if vc:
        eta = 40 if voice.get("ref") else 55

    if len(text) > TG_MSG_MAX:
        words = len(text.split())
        await message.answer(f"📄 Прийнято {words:,} слів ({len(text):,} символів) ≈ {len(text) // 950} хв аудіо.\n"
                             f"⏱ Генерація триватиме ~{_fmt(eta)} — прогрес за фрагментами буде видно нижче.\n"
                             + ("<i>💎 «Найкраще» для довгого тексту — це ~1.5× довше. "
                                "«⚖️ Стандарт» звучить майже так само, «⚡ Швидко» — ще вдвічі швидше.</i>"
                                if m["quality"] == "best" else
                                "<i>Порада: «⚡ Швидко» в налаштуваннях вдвічі скорочує час.</i>").replace(",", " "),
                             parse_mode="HTML")
    ahead = gq.queue_len()
    status = await message.answer(f"🕐 В черзі — попереду {ahead} {gq._inflect(ahead)}" if ahead
                                  else f"⏳ Готую… <i>(~{_fmt(eta)})</i>", parse_mode="HTML")
    cancel_kb = (InlineKeyboardBuilder()
                 .button(text="❌ Відмінити", callback_data=f"cncl:{status.message_id}").as_markup())
    stop_kb = (InlineKeyboardBuilder()
               .button(text="⏹ Зупинити", callback_data=VoiceCB(action="stop", value=str(status.message_id)).pack())
               .as_markup())

    async def runner(job: gq.GenJob) -> None:
        _running.clear()
        _running.update({"mid": status.message_id, "uid": user.id})
        last = [0.0]

        async def on_status(st: dict) -> None:
            if st.get("prompt_id"):
                _running["prompt_id"] = st["prompt_id"]
            gq.report(job, _frac(st), st.get("eta"))
            now = time.monotonic()
            # long narrations run for many minutes: update less often to stay clear of flood limits
            interval = 3.0 if len(text) < 5000 else 6.0
            if now - last[0] < interval:
                return
            body = _progress_text(info, st)
            if await tg_throttle.edit(status.chat.id, lambda: status.edit_text(
                    body, parse_mode="HTML", reply_markup=stop_kb), min_interval=interval):
                last[0] = now

        t0 = time.monotonic()
        try:
            if vc:
                data = await cc.convert_voice(source_audio, m, voice, on_status)
                spoken = "🎭 ваш запис зі збереженням інтонації" + (" · 🎤 спів" if m["vc_singing"] else "")
            else:
                data, spoken = await cc.generate_speech(text, m, voice, on_status, source_audio)
        except Exception as exc:
            log.exception("speech failed user=%d", user.id)
            err = "⏹ <b>Зупинено</b>" if "перервано" in str(exc) else gq._friendly_error(exc)
            try:
                await status.edit_text(err, parse_mode="HTML", reply_markup=_kb_back())
            except TelegramBadRequest:
                pass
            return
        finally:
            _running.clear()
        db.increment_gen_count(user.id)
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        cap = (f"{'🎭' if vc else '🔁' if info['s2s'] else '🔊'} {voice_name} · ⏱ {_fmt(time.monotonic() - t0)}\n"
               f"📝 <i>{_esc(spoken[:700])}{'…' if len(spoken) > 700 else ''}</i>")
        kb = InlineKeyboardBuilder()
        kb.button(text="🎙 Інший голос", callback_data=VoiceCB(action="pick").pack())
        kb.button(text="🗣 Меню голосу", callback_data=VoiceCB(action="menu").pack())
        await message.answer_voice(BufferedInputFile(data, filename="voice.ogg"), caption=cap[:1020],
                                   parse_mode="HTML", reply_markup=kb.as_markup())

    async def on_cancel(msg: Message) -> None:
        try:
            await status.edit_text("❌ Скасовано.", reply_markup=_kb_back())
        except TelegramBadRequest:
            pass

    async def _noop(*_a) -> None:
        return None

    await gq.enqueue(gq.GenJob(message=message, prompt=text or "speech-to-speech", user_settings={},
                               status_msg=status, on_done=_noop, on_error=_noop, cancel_kb=cancel_kb,
                               on_cancel=on_cancel, runner=runner,
                               label=("🎭 Зміна голосу" if vc else "🔁 Переозвучка" if info["s2s"] else "🔊 Озвучка")
                                     + f" · {voice_name}",
                               eta=eta))


@router.callback_query(VoiceCB.filter(F.action == "stop"))
async def cb_stop(call: CallbackQuery, callback_data: VoiceCB) -> None:
    _, admin = _ctx(call.from_user)
    if not _running or str(_running.get("mid")) != callback_data.value:
        await call.answer("⚠️ Вже завершено.", show_alert=True)
        return
    if _running.get("uid") != call.from_user.id and not admin:
        await call.answer("⛔ Це не ваше завдання.", show_alert=True)
        return
    await cc.interrupt(_running.get("prompt_id"))
    await call.answer("⏹ Зупиняю…")
