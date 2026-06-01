import asyncio
import hashlib
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import BotCommand, BotCommandScopeAllPrivateChats
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery, ErrorEvent, FSInputFile,
    InlineKeyboardMarkup, InputMediaPhoto, Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

import comfy_client
import config
import database
import game_api
import gen_queue as gq
import history as hist
import loras as loras_db
import models as models_db
import upscale_models as upscale_models_db
import translator
import users as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

bot = Bot(token=config.BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())

# ── presets ───────────────────────────────────────────────────────────────

SIZE_PRESETS    = ["512×512", "768×512", "512×768", "768×768", "1024×1024", "1024×768", "768×1024"]
STEPS_PRESETS   = [10, 15, 20, 25, 30, 40]
CFG_PRESETS     = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0]
SAMPLER_PRESETS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_karras", "ddim", "uni_pc"]
DENOISE_PRESETS = [0.3, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0]
BATCH_PRESETS          = [1, 2, 3, 4]
LORA_STRENGTH_PRESETS  = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5]
HIRES_DENOISE_PRESETS  = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
UPSCALE_SCALES         = [("2", "×2  (1024 → 2048)"), ("4", "×4  (512 → 2048)")]

STYLES: dict[str, dict] = {
    "photo":      {"name": "📷 Фото",
                   "suffix":     "photorealistic, hyperrealistic photography, DSLR, 8k, sharp focus, professional photo",
                   "neg_suffix": "painting, illustration, drawing, cartoon, anime, render, cgi, digital art, sketch"},
    "anime":      {"name": "🎌 Аніме",
                   "suffix":     "anime style, manga, detailed anime art, vibrant colors, Studio Ghibli",
                   "neg_suffix": "photorealistic, realistic, 3d render, photograph, live action"},
    "digital":    {"name": "🖥 Цифрове мист.",
                   "suffix":     "digital art, digital painting, concept art, artstation, highly detailed",
                   "neg_suffix": "photograph, photo, realistic, 3d render, low detail"},
    "oil":        {"name": "🖼 Олійний живопис",
                   "suffix":     "oil painting, oil on canvas, classical painting, detailed brushwork, museum quality",
                   "neg_suffix": "digital art, photo, 3d render, anime, flat colors, sharp edges"},
    "watercolor": {"name": "🎨 Акварель",
                   "suffix":     "watercolor painting, soft colors, watercolor illustration, flowing paint",
                   "neg_suffix": "sharp edges, harsh lines, digital art, 3d render, photo, neon colors"},
    "sketch":     {"name": "✏️ Ескіз",
                   "suffix":     "pencil sketch, pencil drawing, hand drawn, detailed graphite sketch",
                   "neg_suffix": "color, colored, photo, realistic, 3d render, painted"},
    "cinematic":  {"name": "🎬 Кіно",
                   "suffix":     "cinematic, movie still, dramatic lighting, film grain, anamorphic lens",
                   "neg_suffix": "flat lighting, overexposed, anime, illustration, cartoon, low quality"},
    "fantasy":    {"name": "🧙 Фентезі",
                   "suffix":     "fantasy art, epic fantasy, magical atmosphere, mystical, detailed illustration",
                   "neg_suffix": "modern, contemporary, realistic photo, mundane, sci-fi, futuristic"},
    "pixel":      {"name": "👾 Піксель-арт",
                   "suffix":     "pixel art, 16-bit, retro game style, pixelated, clean pixels",
                   "neg_suffix": "smooth, anti-aliased, photorealistic, 3d render, blurry, high resolution"},
    "3d":         {"name": "🔮 3D рендер",
                   "suffix":     "3d render, octane render, unreal engine 5, photorealistic 3d, subsurface scattering",
                   "neg_suffix": "flat, 2d, cartoon, anime, sketch, painted, hand drawn"},
    "vintage":    {"name": "📼 Вінтаж",
                   "suffix":     "vintage style, retro aesthetic, old photo, film photography, faded colors, nostalgic",
                   "neg_suffix": "modern, sharp, vivid colors, digital, clean, neon, futuristic"},
    "minimal":    {"name": "⬜ Мінімалізм",
                   "suffix":     "minimalist, clean design, simple composition, elegant, negative space, white background",
                   "neg_suffix": "cluttered, busy, complex, ornate, detailed background, noisy, busy composition"},
}

# ── FSM ───────────────────────────────────────────────────────────────────

class GenState(StatesGroup):
    waiting_prompt = State()

class Img2ImgState(StatesGroup):
    waiting_prompt = State()

class AddUserState(StatesGroup):
    waiting_username = State()
    waiting_role     = State()

class GenSettingsState(StatesGroup):
    waiting_neg_prompt   = State()
    waiting_custom_style = State()

class ModelsState(StatesGroup):
    waiting_add  = State()
    waiting_edit = State()

class LorasState(StatesGroup):
    waiting_edit    = State()
    waiting_trigger = State()

class UpscaleMdlState(StatesGroup):
    waiting_edit = State()

class RegenConfigState(StatesGroup):
    neg_prompt = State()

class StylePickState(StatesGroup):
    waiting = State()

# ── callback data ─────────────────────────────────────────────────────────

class UserCB(CallbackData, prefix="u"):
    action:   str
    username: str           = ""
    role:     Optional[str] = None

class GsCB(CallbackData, prefix="gs"):
    action: str
    value:  Optional[str] = None

class ModelsCB(CallbackData, prefix="mdl"):
    action: str
    value:  Optional[str] = None

# Callback data is limited to 64 bytes. Long model filenames are replaced with
# a short MD5 hash; _resolve_model_id maps it back to the real name.
_NO_IMG2IMG_WF = {"flux", "hidream"}   # workflows that don't support img2img

_model_id_map: dict[str, str] = {}

def _mk_id(name: str) -> str:
    if len(name) <= 40:
        return name
    h = hashlib.md5(name.encode()).hexdigest()[:12]
    _model_id_map[h] = name
    return h

def _resolve_model_id(cb_id: str) -> str:
    if cb_id in _model_id_map:
        return _model_id_map[cb_id]
    for m in models_db.all_models():
        if m == cb_id or hashlib.md5(m.encode()).hexdigest()[:12] == cb_id:
            return m
    return cb_id

_lora_id_map: dict[str, str] = {}

def _mk_lora_id(name: str) -> str:
    if len(name) <= 40:
        return name
    h = hashlib.md5(name.encode()).hexdigest()[:12]
    _lora_id_map[h] = name
    return h

def _resolve_lora_id(lid: str) -> str:
    if lid in _lora_id_map:
        return _lora_id_map[lid]
    for m in loras_db.all_loras():
        if m == lid or hashlib.md5(m.encode()).hexdigest()[:12] == lid:
            return m
    return lid

# ── multi-lora helpers ────────────────────────────────────────────────────

def _get_active_loras(s: dict) -> list[dict]:
    """Return list of active LoRAs: [{"name": ..., "strength": ...}, ...]"""
    return list(s.get("loras_active") or [])

def _loras_active_summary(active: list[dict]) -> str:
    """Short display string for active LoRAs."""
    if not active:
        return "вимкнені"
    lbls = loras_db.labels()
    parts = [f"{_label(l['name'], lbls)} ({l.get('strength', 0.8)})" for l in active]
    return ", ".join(parts)

def _toggle_lora(tg_id: int, name: str, default_strength: float = 0.8) -> bool:
    """Toggle LoRA in user's active list. Returns True if now active, False if removed."""
    s      = db.get_gen_settings(tg_id)
    active = _get_active_loras(s)
    names  = [l["name"] for l in active]
    if name in names:
        active = [l for l in active if l["name"] != name]
        db.set_gen_setting(tg_id, "loras_active", active or None)
        return False
    else:
        active.append({"name": name, "strength": default_strength})
        db.set_gen_setting(tg_id, "loras_active", active)
        return True

def _set_lora_strength(tg_id: int, name: str, strength: float) -> None:
    """Update strength of a specific LoRA in user's active list."""
    s      = db.get_gen_settings(tg_id)
    active = _get_active_loras(s)
    for item in active:
        if item["name"] == name:
            item["strength"] = strength
    db.set_gen_setting(tg_id, "loras_active", active)

class CancelCB(CallbackData, prefix="cncl"):
    msg_id: int

class HistoryCB(CallbackData, prefix="hist"):
    action: str
    idx:    int = 0
    uid:    int = 0   # 0 = self, >0 = specific user, -1 = all (admin)

class RgsCB(CallbackData, prefix="rgs"):
    action: str
    value:  Optional[str] = None

class LorasCB(CallbackData, prefix="lra"):
    action: str
    value:  Optional[str] = None

class StyleCB(CallbackData, prefix="sty"):
    key: str

class UpscaleCB(CallbackData, prefix="usc"):
    action: str
    idx:    int            = 0
    uid:    int            = 0
    scale:  Optional[str]  = None

class UpscaleModelCB(CallbackData, prefix="uscm"):
    idx: int  # -1 = авто/bilinear, 0+ = індекс в списку моделей зі FSM

class RgsUpscaleModelCB(CallbackData, prefix="ruscm"):
    idx: int  # те саме, але для regen-config

class UpscaleMdlCB(CallbackData, prefix="umd"):
    action: str
    value:  Optional[str] = None

class WorkflowCB(CallbackData, prefix="wf"):
    wtype: str  # sd15 | sdxl | flux | sd3

class MultiLoraCB(CallbackData, prefix="ml"):
    action: str                  # tog | str_pick | str_set | lra_off
    lid:    Optional[str] = None # hashed lora id (≤40 chars)
    val:    Optional[str] = None # strength value for str_set

# ── helpers ───────────────────────────────────────────────────────────────

def _short(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name

def _label(name: str, lbls: dict[str, str]) -> str:
    return lbls.get(name) or _short(name)

def _ctx(user) -> tuple[bool, bool]:
    db.sync_id(user.id, user.username or "")
    return db.is_allowed(user.id, user.username or ""), db.is_admin(user.id, user.username or "")

def _users_text() -> str:
    return f"👥 <b>Користувачі</b> ({len(db.all_users())}):"

def _models_text() -> str:
    return f"🤖 <b>Моделі</b> ({len(models_db.all_models())}):"

def _loras_text() -> str:
    return f"🎭 <b>LoRA</b> ({len(loras_db.all_loras())}):"

def _upscale_models_text() -> str:
    return f"🔍 <b>Upscale моделі</b> ({len(upscale_models_db.all_upscale_models())}):"

async def _nav(call: CallbackQuery, text: str, **kwargs) -> None:
    try:
        if call.message.photo or call.message.document or call.message.video:
            await call.message.edit_caption(text, **kwargs)
        else:
            await call.message.edit_text(text, **kwargs)
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e).lower():
            raise

async def _download_photo(file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    buf  = await bot.download_file(file.file_path)
    return buf.read()

def _gen_settings_text(tg_id: int) -> str:
    s       = db.get_gen_settings(tg_id)
    mode    = s.get("mode", "text2img")
    ckpt    = s.get("checkpoint") or config.CHECKPOINT
    w       = s.get("width")      or config.IMAGE_WIDTH
    h       = s.get("height")     or config.IMAGE_HEIGHT
    steps   = s.get("steps")      or config.STEPS
    cfg     = s.get("cfg")        or config.CFG_SCALE
    sampler = s.get("sampler")    or "euler"
    denoise = s.get("denoise")    or 0.75
    batch   = int(s.get("batch_size", 1))
    neg     = s.get("negative_prompt")
    neg_str = f"<i>{neg[:50]}{'…' if len(neg) > 50 else ''}</i>" if neg else "<i>стандартний</i>"
    active_loras_list = _get_active_loras(s)
    custom  = any(s.get(k) for k in ("mode", "steps", "cfg", "width", "height",
                                      "sampler", "checkpoint", "denoise", "negative_prompt",
                                      "batch_size", "loras_active"))
    note    = "\n<i>* змінено відносно стандартних</i>" if custom else ""
    denoise_line = f"🎨 Варіація:  <b>{denoise}</b>{'  *' if s.get('denoise') else ''}\n" if mode == "img2img" else ""
    _mlbls  = models_db.labels()
    _llbls  = loras_db.labels()
    _ulbls  = upscale_models_db.labels()
    if active_loras_list:
        lora_parts = [f"<b>{_label(l['name'], _llbls)}</b> ({l.get('strength', 0.8)})"
                      for l in active_loras_list]
        lora_line = "🎭 LoRA:     " + ", ".join(lora_parts) + "  *\n"
    else:
        lora_line = "🎭 LoRA:     <b>відсутні</b>\n"
    active_styles = s.get("active_styles") or []
    custom_suf    = s.get("custom_style_suffix") or ""
    styles_parts  = [STYLES[k]["name"] for k in active_styles if k in STYLES]
    if custom_suf:
        styles_parts.append(f"✏️ Власний")
    styles_str = "  ".join(styles_parts) if styles_parts else "<i>вимкнені</i>"
    hires_fix     = s.get("hires_fix", False)
    hires_denoise = float(s.get("hires_denoise") or 0.45)
    upscale_model = s.get("upscale_model")
    if hires_fix:
        hires_str = f"<b>ввімкнений ×2</b>  •  denoise {hires_denoise}  *"
    else:
        hires_str = "<b>вимкнений</b>"
    usc_str = f"<b>{_label(upscale_model, _ulbls)}</b>  *" if upscale_model else "<b>авто (bilinear)</b>"
    return (
        "🎛 <b>Налаштування генерації</b>\n\n"
        f"🔄 Режим:    <b>{mode}</b>{'  *' if s.get('mode') else ''}\n"
        f"🤖 Модель:   <b>{_label(ckpt, _mlbls)}</b>{'  *' if s.get('checkpoint') else ''}\n"
        f"📐 Розмір:   <b>{w} × {h}</b>{'  *' if s.get('width') else ''}\n"
        f"🎚 Кроки:    <b>{steps}</b>{'  *' if s.get('steps') else ''}\n"
        f"🎯 CFG:      <b>{cfg}</b>{'  *' if s.get('cfg') else ''}\n"
        f"🎲 Семплер:  <b>{sampler}</b>{'  *' if s.get('sampler') else ''}\n"
        f"{denoise_line}"
        f"🔢 Кількість: <b>{batch}</b>{'  *' if s.get('batch_size') else ''}\n"
        f"{lora_line}"
        f"🎨 Стилі:    {styles_str}\n"
        f"🔍 HiRes:    {hires_str}\n"
        f"🖼 Upscale:  {usc_str}\n"
        f"📝 Негативний: {neg_str}"
        f"{note}"
    )

async def _build_status_text(tg_id: int) -> str:
    s = await comfy_client.get_status()
    if not s["online"]:
        return f"📊 <b>Статус ComfyUI</b>\n\n🔴 <b>Недоступний</b>\n<code>{config.COMFY_URL}</code>"
    stats    = s["stats"]
    queue    = s["queue"]
    sys_info = stats.get("system", {})
    devices  = stats.get("devices", [])
    ram_total = sys_info.get("ram_total", 0) / 1024**3
    ram_free  = sys_info.get("ram_free",  0) / 1024**3
    lines = [
        "📊 <b>Статус ComfyUI</b>", "",
        "🟢 <b>Онлайн</b>",
        f"🖥  RAM: {ram_total - ram_free:.1f} / {ram_total:.1f} GB",
    ]
    for d in devices:
        vt = d.get("total_memory", 0) / 1024**3
        vf = d.get("free_memory",  0) / 1024**3
        lines.append(f"🎮 {d.get('name','GPU')}: {vt-vf:.1f} / {vt:.1f} GB VRAM")
    running = len(queue.get("queue_running", []))
    pending = len(queue.get("queue_pending", []))
    bot_q   = gq.queue_len()
    bot_run = 1 if bot_q > 0 else 0
    bot_wait = max(bot_q - 1, 0)
    lines += ["", "⏳ <b>Черга ComfyUI:</b>",
              f"   В обробці:    {running}",
              f"   В очікуванні: {pending}",
              "", "🤖 <b>Черга бота:</b>",
              f"   Генерується:  {bot_run}",
              f"   Очікують:     {bot_wait}"]

    gs      = db.get_gen_settings(tg_id)
    mode    = gs.get("mode", "text2img")
    ckpt    = gs.get("checkpoint") or config.CHECKPOINT
    w       = gs.get("width")      or config.IMAGE_WIDTH
    h       = gs.get("height")     or config.IMAGE_HEIGHT
    steps   = gs.get("steps")      or config.STEPS
    cfg     = gs.get("cfg")        or config.CFG_SCALE
    sampler = gs.get("sampler")    or "euler"
    denoise = gs.get("denoise")    or 0.75
    neg     = gs.get("negative_prompt")
    neg_str = neg[:40] + ("…" if len(neg) > 40 else "") if neg else "стандартний"
    batch = int(gs.get("batch_size", 1))
    lines += [
        "", "🎛 <b>Ваші налаштування генерації:</b>",
        f"   🔄 Режим:    <b>{mode}</b>",
        f"   🤖 Модель:   <b>{_label(ckpt, models_db.labels())}</b>",
        f"   📐 Розмір:   <b>{w} × {h}</b>",
        f"   🎚 Кроки:    <b>{steps}</b>",
        f"   🎯 CFG:      <b>{cfg}</b>",
        f"   🎲 Семплер:  <b>{sampler}</b>",
    ]
    if mode == "img2img":
        lines.append(f"   🎨 Варіація: <b>{denoise}</b>")
    lines.append(f"   🔢 Кількість: <b>{batch}</b>")
    lines.append(f"   📝 Негатив:   <i>{neg_str}</i>")
    return "\n".join(lines)

# ── keyboards ─────────────────────────────────────────────────────────────

def kb_main(admin: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🎨 Згенерувати зображення", callback_data="gen:start")
    b.button(text="📊 Статус ComfyUI",          callback_data="comfy:status")
    b.button(text="🎛 Налаштування генерації",  callback_data=GsCB(action="menu").pack())
    b.button(text="📈 Моя статистика",           callback_data="stats:my")
    b.button(text="📜 Історія генерацій",        callback_data=HistoryCB(action="show", uid=0).pack())
    if admin:
        b.button(text="⚙️ Налаштування",        callback_data="menu:settings")
    b.adjust(1)
    return b.as_markup()

def kb_settings() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="👥 Управління користувачами", callback_data="menu:users")
    b.button(text="🤖 Управління моделями",      callback_data="menu:models")
    b.button(text="🎭 Управління LoRA",          callback_data="menu:loras")
    b.button(text="🔍 Управління Upscale",       callback_data="menu:upscale_models")
    b.button(text="📊 Статистика генерацій",     callback_data="stats:all")
    b.button(text="📜 Повна історія",            callback_data=HistoryCB(action="show",     uid=-1).pack())
    b.button(text="🗑 Очистити всю історію",      callback_data=HistoryCB(action="clearall", uid=-1).pack())
    if game_api.is_configured():
        icon = "🔴 Стоп MMORPG-генерацію" if _game_gen_running() else "🎮 Генерація предметів MMORPG"
        b.button(text=icon, callback_data="game:gen")
    b.button(text="🔙 Головне меню",             callback_data="menu:main")
    b.adjust(1)
    return b.as_markup()

def kb_users() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for u in db.all_users():
        icon = "👑" if u["role"] == "admin" else "👤"
        b.button(text=f"{icon} @{u['username']}", callback_data=UserCB(action="view", username=u["username"]).pack())
    b.button(text="➕ Додати користувача", callback_data="user:add")
    b.button(text="🔙 Назад",             callback_data="menu:settings")
    b.adjust(1)
    return b.as_markup()

def kb_user_detail(username: str, role: str, tg_id: Optional[int] = None) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    if role == "admin":
        b.button(text="👤 Зробити користувачем",    callback_data=UserCB(action="setrole", username=username, role="user").pack())
    else:
        b.button(text="👑 Зробити адміністратором", callback_data=UserCB(action="setrole", username=username, role="admin").pack())
    b.button(text="🗑 Видалити",  callback_data=UserCB(action="delete", username=username).pack())
    if tg_id:
        b.button(text="📜 Історія", callback_data=HistoryCB(action="show", idx=0, uid=tg_id).pack())
    b.button(text="🔙 До списку", callback_data="menu:users")
    b.adjust(1)
    return b.as_markup()

def kb_confirm_delete_user(username: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Так, видалити", callback_data=UserCB(action="delete_ok", username=username).pack())
    b.button(text="❌ Скасувати",     callback_data=UserCB(action="view",      username=username).pack())
    b.adjust(2)
    return b.as_markup()

def kb_choose_role(username: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="👤 Користувач",    callback_data=f"role:user:{username}")
    b.button(text="👑 Адміністратор", callback_data=f"role:admin:{username}")
    b.button(text="❌ Скасувати",     callback_data="menu:users")
    b.adjust(2, 1)
    return b.as_markup()

def kb_status() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🔄 Оновити", callback_data="comfy:status")
    b.button(text="🔙 Назад",   callback_data="menu:main")
    b.adjust(2)
    return b.as_markup()

def kb_cancel_to_main() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="❌ Скасувати", callback_data="menu:main")
    return b.as_markup()

# ── gen settings keyboards ────────────────────────────────────────────────

def kb_gen_settings(tg_id: int) -> InlineKeyboardMarkup:
    s       = db.get_gen_settings(tg_id)
    mode    = s.get("mode", "text2img")
    ckpt    = s.get("checkpoint") or config.CHECKPOINT
    w       = s.get("width")      or config.IMAGE_WIDTH
    h       = s.get("height")     or config.IMAGE_HEIGHT
    steps   = s.get("steps")      or config.STEPS
    cfg     = s.get("cfg")        or config.CFG_SCALE
    sampler = s.get("sampler")    or "euler"
    denoise = s.get("denoise")    or 0.75
    batch   = int(s.get("batch_size", 1))
    has_neg = bool(s.get("negative_prompt"))
    b = InlineKeyboardBuilder()
    _mlbls = models_db.labels()
    _llbls = loras_db.labels()
    _ulbls = upscale_models_db.labels()
    wf_type = models_db.get_workflow(ckpt)
    if wf_type not in _NO_IMG2IMG_WF:
        b.button(text=f"🔄 Режим: {mode}",                    callback_data=GsCB(action="show", value="mode").pack())
    b.button(text=f"🤖 Модель: {_label(ckpt, _mlbls)}",       callback_data=GsCB(action="show", value="model").pack())
    b.button(text=f"📐 Розмір: {w}×{h}",                      callback_data=GsCB(action="show", value="size").pack())
    b.button(text=f"🎚 Кроки: {steps}",                        callback_data=GsCB(action="show", value="steps").pack())
    b.button(text=f"🎯 CFG: {cfg}",                            callback_data=GsCB(action="show", value="cfg").pack())
    b.button(text=f"🎲 Семплер: {sampler}",                    callback_data=GsCB(action="show", value="sampler").pack())
    active_loras_gs = _get_active_loras(s)
    if mode == "img2img" and wf_type not in _NO_IMG2IMG_WF:
        b.button(text=f"🎨 Сила варіації: {denoise}", callback_data=GsCB(action="show", value="denoise").pack())
    b.button(text=f"🔢 Кількість: {batch}",                    callback_data=GsCB(action="show", value="batch").pack())
    lora_count = len(active_loras_gs)
    lora_label = f"🎭 LoRA: {lora_count} активних" if active_loras_gs else "🎭 LoRA: вимкнені"
    b.button(text=lora_label,                                  callback_data=GsCB(action="show", value="loras").pack())
    b.button(text="📝 Негативний промпт" + (" ✏️" if has_neg else ""),
             callback_data=GsCB(action="show", value="neg").pack())
    active_styles  = s.get("active_styles") or []
    _has_custom    = bool(s.get("custom_style_suffix"))
    _style_count   = len(active_styles) + (1 if _has_custom else 0)
    style_label    = f"🎨 Стилі: {_style_count} активних" if _style_count else "🎨 Стилі: вимкнені"
    hires_fix     = s.get("hires_fix", False)
    hires_denoise = float(s.get("hires_denoise") or 0.45)
    upscale_model = s.get("upscale_model")
    hires_label   = f"🔍 HiRes Fix: ввімкнений ×2 (d={hires_denoise})" if hires_fix else "🔍 HiRes Fix: вимкнений"
    usc_label     = f"🖼 Upscale: {_label(upscale_model, _ulbls)}" if upscale_model else "🖼 Upscale модель: авто"
    b.button(text=style_label,                     callback_data=GsCB(action="show", value="styles").pack())
    b.button(text=hires_label,                     callback_data=GsCB(action="show", value="hires").pack())
    if hires_fix:
        b.button(text=f"🎚 HiRes denoise: {hires_denoise}",
                 callback_data=GsCB(action="show", value="hires_denoise").pack())
    b.button(text=usc_label,                       callback_data=GsCB(action="show", value="upscale_model").pack())
    b.button(text="🔄 Скинути до стандартних",     callback_data=GsCB(action="reset").pack())
    b.button(text="🔙 Назад",                      callback_data="menu:main")
    b.adjust(1)
    return b.as_markup()

def kb_mode_picker(current: str, wf_type: str = "sd15") -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    modes = [("text2img", "🖋 text2img — текст → зображення")]
    if wf_type not in _NO_IMG2IMG_WF:
        modes.append(("img2img", "🖼 img2img  — фото + промпт → варіація"))
    for val, label in modes:
        mark = " ✅" if val == current else ""
        b.button(text=f"{label}{mark}", callback_data=GsCB(action="set_mode", value=val).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_model_picker(current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    lbls = models_db.labels()
    for m in models_db.all_models():
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=GsCB(action="set_model", value=_mk_id(m)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_size_picker(current_w: int, current_h: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for preset in SIZE_PRESETS:
        pw, ph = map(int, preset.split("×"))
        mark = " ✅" if pw == current_w and ph == current_h else ""
        b.button(text=f"{preset}{mark}", callback_data=GsCB(action="set_size", value=preset).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_steps_picker(current: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {10: "швидко", 20: "стандарт", 30: "якісно", 40: "детально"}
    for v in STEPS_PRESETS:
        mark  = " ✅" if v == current else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=GsCB(action="set_steps", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_cfg_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in CFG_PRESETS:
        mark = " ✅" if float(current) == v else ""
        b.button(text=f"{v}{mark}", callback_data=GsCB(action="set_cfg", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(3)
    return b.as_markup()

def kb_sampler_picker(current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in SAMPLER_PRESETS:
        mark = " ✅" if v == current else ""
        b.button(text=f"{v}{mark}", callback_data=GsCB(action="set_sampler", value=v).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_denoise_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {0.3: "слабка", 0.5: "помірна", 0.75: "сильна", 1.0: "повна"}
    for v in DENOISE_PRESETS:
        mark  = " ✅" if float(current) == v else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=GsCB(action="set_denoise", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_batch_picker(current: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {1: "одне", 2: "два", 3: "три", 4: "чотири"}
    for v in BATCH_PRESETS:
        mark  = " ✅" if v == current else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=GsCB(action="set_batch", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_neg_prompt(has_neg: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    if has_neg:
        b.button(text="🗑 Видалити власний промпт", callback_data=GsCB(action="neg_clear").pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_lora_picker(current: Optional[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    mark = " ✅" if not current else ""
    b.button(text=f"❌ Без LoRA{mark}", callback_data=GsCB(action="set_lora", value="__none__").pack())
    lbls = loras_db.labels()
    for m in loras_db.all_loras():
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=GsCB(action="set_lora", value=m).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_lora_strength_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in LORA_STRENGTH_PRESETS:
        mark = " ✅" if float(current) == v else ""
        b.button(text=f"{v}{mark}", callback_data=GsCB(action="set_lora_strength", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(5)
    return b.as_markup()

def kb_style_toggle(tg_id: int) -> InlineKeyboardMarkup:
    gs            = db.get_gen_settings(tg_id)
    active        = set(gs.get("active_styles") or [])
    custom_suffix = gs.get("custom_style_suffix") or ""
    b = InlineKeyboardBuilder()
    for key, style in STYLES.items():
        prefix = "✅ " if key in active else "❌ "
        b.button(text=f"{prefix}{style['name']}",
                 callback_data=GsCB(action="toggle_style", value=key).pack())
    b.adjust(2)
    # Custom suffix row
    if custom_suffix:
        preview = custom_suffix[:28] + ("…" if len(custom_suffix) > 28 else "")
        b.button(text=f"✏️ Власний: {preview}",
                 callback_data=GsCB(action="show", value="set_custom_style").pack())
    else:
        b.button(text="➕ Власний суфікс",
                 callback_data=GsCB(action="show", value="set_custom_style").pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_hires_denoise_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {0.3: "м'яко", 0.45: "стандарт", 0.6: "сильно"}
    for v in HIRES_DENOISE_PRESETS:
        mark  = " ✅" if float(current) == v else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=GsCB(action="set_hires_denoise", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(3)
    return b.as_markup()


def kb_upscale_model_picker(current: Optional[str], available: list[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text=f"🔁 Авто (bilinear){' ✅' if not current else ''}",
             callback_data=UpscaleModelCB(idx=-1).pack())
    lbls = upscale_models_db.labels()
    for i, m in enumerate(available):
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=UpscaleModelCB(idx=i).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()


# ── regen-config keyboards ───────────────────────────────────────────────

def _rgs_text(s: dict) -> str:
    mode    = s.get("mode", "text2img")
    ckpt    = s.get("checkpoint") or config.CHECKPOINT
    w       = s.get("width")      or config.IMAGE_WIDTH
    h       = s.get("height")     or config.IMAGE_HEIGHT
    steps   = s.get("steps")      or config.STEPS
    cfg     = float(s.get("cfg")  or config.CFG_SCALE)
    sampler = s.get("sampler")    or "euler"
    denoise = float(s.get("denoise") or 0.75)
    batch   = int(s.get("batch_size", 1))
    neg     = s.get("negative_prompt")
    neg_str = f"<i>{neg[:50]}{'…' if len(neg) > 50 else ''}</i>" if neg else "<i>стандартний</i>"
    rgs_active_loras = _get_active_loras(s)
    denoise_line = f"🎨 Варіація:   <b>{denoise}</b>\n" if mode == "img2img" else ""
    _mlbls = models_db.labels()
    _llbls = loras_db.labels()
    _ulbls = upscale_models_db.labels()
    if rgs_active_loras:
        lora_parts = [f"<b>{_label(l['name'], _llbls)}</b> ({l.get('strength', 0.8)})"
                      for l in rgs_active_loras]
        lora_line = "🎭 LoRA:      " + ", ".join(lora_parts) + "\n"
    else:
        lora_line = "🎭 LoRA:      <b>відсутні</b>\n"
    hires_fix     = bool(s.get("hires_fix"))
    hires_denoise = float(s.get("hires_denoise") or 0.45)
    upscale_model = s.get("upscale_model")
    hires_str = f"<b>ввімкнений ×2</b>  •  denoise {hires_denoise}" if hires_fix else "<b>вимкнений</b>"
    usc_str   = f"<b>{_label(upscale_model, _ulbls)}</b>" if upscale_model else "<b>авто</b>"
    return (
        "⚙️ <b>Налаштування для цієї генерації</b>\n"
        "<i>Не зберігаються у ваш профіль</i>\n\n"
        f"🔄 Режим:     <b>{mode}</b>\n"
        f"🤖 Модель:    <b>{_label(ckpt, _mlbls)}</b>\n"
        f"📐 Розмір:    <b>{w} × {h}</b>\n"
        f"🎚 Кроки:     <b>{steps}</b>\n"
        f"🎯 CFG:       <b>{cfg}</b>\n"
        f"🎲 Семплер:   <b>{sampler}</b>\n"
        f"{denoise_line}"
        f"🔢 Кількість: <b>{batch}</b>\n"
        f"{lora_line}"
        f"🔍 HiRes:     {hires_str}\n"
        f"🖼 Upscale:   {usc_str}\n"
        f"📝 Негативний: {neg_str}"
    )

def kb_rgs(s: dict) -> InlineKeyboardMarkup:
    mode    = s.get("mode", "text2img")
    ckpt    = s.get("checkpoint") or config.CHECKPOINT
    w       = s.get("width")      or config.IMAGE_WIDTH
    h       = s.get("height")     or config.IMAGE_HEIGHT
    steps   = s.get("steps")      or config.STEPS
    cfg     = float(s.get("cfg")  or config.CFG_SCALE)
    sampler = s.get("sampler")    or "euler"
    denoise = float(s.get("denoise") or 0.75)
    batch   = int(s.get("batch_size", 1))
    has_neg = bool(s.get("negative_prompt"))
    b = InlineKeyboardBuilder()
    _mlbls = models_db.labels()
    _llbls = loras_db.labels()
    _ulbls = upscale_models_db.labels()
    wf_type = models_db.get_workflow(ckpt)
    if wf_type not in _NO_IMG2IMG_WF:
        b.button(text=f"🔄 Режим: {mode}",                  callback_data=RgsCB(action="show", value="mode").pack())
    b.button(text=f"🤖 Модель: {_label(ckpt, _mlbls)}",     callback_data=RgsCB(action="show", value="model").pack())
    b.button(text=f"📐 Розмір: {w}×{h}",                    callback_data=RgsCB(action="show", value="size").pack())
    b.button(text=f"🎚 Кроки: {steps}",                      callback_data=RgsCB(action="show", value="steps").pack())
    b.button(text=f"🎯 CFG: {cfg}",                          callback_data=RgsCB(action="show", value="cfg").pack())
    b.button(text=f"🎲 Семплер: {sampler}",                  callback_data=RgsCB(action="show", value="sampler").pack())
    rgs_kb_active = _get_active_loras(s)
    if mode == "img2img" and wf_type not in _NO_IMG2IMG_WF:
        b.button(text=f"🎨 Варіація: {denoise}", callback_data=RgsCB(action="show", value="denoise").pack())
    b.button(text=f"🔢 Кількість: {batch}",                  callback_data=RgsCB(action="show", value="batch").pack())
    rgs_lora_count = len(rgs_kb_active)
    rgs_lora_label = f"🎭 LoRA: {rgs_lora_count} активних" if rgs_kb_active else "🎭 LoRA: вимкнені"
    b.button(text=rgs_lora_label, callback_data=RgsCB(action="show", value="loras_info").pack())
    b.button(text="📝 Негативний" + (" ✏️" if has_neg else ""),
             callback_data=RgsCB(action="show", value="neg").pack())
    hires_fix     = bool(s.get("hires_fix"))
    hires_denoise = float(s.get("hires_denoise") or 0.45)
    upscale_model = s.get("upscale_model")
    hires_label   = f"🔍 HiRes Fix: ввімкнений ×2 (d={hires_denoise})" if hires_fix else "🔍 HiRes Fix: вимкнений"
    usc_label     = f"🖼 Upscale: {_label(upscale_model, _ulbls)}" if upscale_model else "🖼 Upscale: авто"
    b.button(text=hires_label, callback_data=RgsCB(action="show", value="hires").pack())
    if hires_fix:
        b.button(text=f"🎚 HiRes denoise: {hires_denoise}",
                 callback_data=RgsCB(action="show", value="hires_denoise").pack())
    b.button(text=usc_label,   callback_data=RgsCB(action="show", value="upscale_model").pack())
    b.button(text="✅ Генерувати",               callback_data=RgsCB(action="go").pack())
    b.button(text="🔙 Скасувати",               callback_data=RgsCB(action="cancel").pack())
    b.adjust(1)
    return b.as_markup()

def kb_rgs_mode_picker(current: str, wf_type: str = "sd15") -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    modes = [("text2img", "🖋 text2img — текст → зображення")]
    if wf_type not in _NO_IMG2IMG_WF:
        modes.append(("img2img", "🖼 img2img  — фото + промпт → варіація"))
    for val, label in modes:
        mark = " ✅" if val == current else ""
        b.button(text=f"{label}{mark}", callback_data=RgsCB(action="set_mode", value=val).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_rgs_model_picker(current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    lbls = models_db.labels()
    for m in models_db.all_models():
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=RgsCB(action="set_model", value=_mk_id(m)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_rgs_size_picker(w: int, h: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for preset in SIZE_PRESETS:
        pw, ph = map(int, preset.split("×"))
        mark = " ✅" if pw == w and ph == h else ""
        b.button(text=f"{preset}{mark}", callback_data=RgsCB(action="set_size", value=preset).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_rgs_steps_picker(current: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {10: "швидко", 20: "стандарт", 30: "якісно", 40: "детально"}
    for v in STEPS_PRESETS:
        mark  = " ✅" if v == current else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=RgsCB(action="set_steps", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_rgs_cfg_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in CFG_PRESETS:
        mark = " ✅" if float(current) == v else ""
        b.button(text=f"{v}{mark}", callback_data=RgsCB(action="set_cfg", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(3)
    return b.as_markup()

def kb_rgs_sampler_picker(current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in SAMPLER_PRESETS:
        mark = " ✅" if v == current else ""
        b.button(text=f"{v}{mark}", callback_data=RgsCB(action="set_sampler", value=v).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_rgs_denoise_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {0.3: "слабка", 0.5: "помірна", 0.75: "сильна", 1.0: "повна"}
    for v in DENOISE_PRESETS:
        mark  = " ✅" if float(current) == v else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=RgsCB(action="set_denoise", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_rgs_batch_picker(current: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {1: "одне", 2: "два", 3: "три", 4: "чотири"}
    for v in BATCH_PRESETS:
        mark  = " ✅" if v == current else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=RgsCB(action="set_batch", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(2)
    return b.as_markup()

def kb_rgs_neg(has_neg: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    if has_neg:
        b.button(text="🗑 Видалити власний промпт", callback_data=RgsCB(action="neg_clear").pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_rgs_lora_picker(current: Optional[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    mark = " ✅" if not current else ""
    b.button(text=f"❌ Без LoRA{mark}", callback_data=RgsCB(action="set_lora", value="__none__").pack())
    lbls = loras_db.labels()
    for m in loras_db.all_loras():
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=RgsCB(action="set_lora", value=m).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_rgs_hires_denoise_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    labels = {0.3: "м'яко", 0.45: "стандарт", 0.6: "сильно"}
    for v in HIRES_DENOISE_PRESETS:
        mark  = " ✅" if float(current) == v else ""
        label = f" ({labels[v]})" if v in labels else ""
        b.button(text=f"{v}{label}{mark}", callback_data=RgsCB(action="set_hires_denoise", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(3)
    return b.as_markup()


def kb_multi_lora_picker(tg_id: int) -> InlineKeyboardMarkup:
    """Keyboard to toggle multiple LoRAs on/off."""
    s            = db.get_gen_settings(tg_id)
    active_loras = _get_active_loras(s)
    active_map   = {l["name"]: l.get("strength", 0.8) for l in active_loras}
    lbls         = loras_db.labels()
    all_loras    = loras_db.all_loras()
    b = InlineKeyboardBuilder()
    for name in all_loras:
        lid = _mk_lora_id(name)
        if name in active_map:
            strength = active_map[name]
            b.button(text=f"✅ {_label(name, lbls)} ({strength})",
                     callback_data=MultiLoraCB(action="str_pick", lid=lid).pack())
        else:
            b.button(text=f"➕ {_label(name, lbls)}",
                     callback_data=MultiLoraCB(action="tog", lid=lid).pack())
    b.button(text="🔙 Назад", callback_data=GsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()

def kb_lora_strength_multi(name: str, lid: str, current: float) -> InlineKeyboardMarkup:
    """Keyboard to pick strength for a specific active LoRA."""
    b = InlineKeyboardBuilder()
    for v in LORA_STRENGTH_PRESETS:
        mark = " ✅" if float(current) == v else ""
        b.button(text=f"{v}{mark}",
                 callback_data=MultiLoraCB(action="str_set", lid=lid, val=str(v)).pack())
    b.button(text="🗑 Вимкнути цю LoRA",
             callback_data=MultiLoraCB(action="lra_off", lid=lid).pack())
    b.button(text="🔙 Назад до списку LoRA",
             callback_data=GsCB(action="show", value="loras").pack())
    b.adjust(5)
    return b.as_markup()

def kb_rgs_upscale_model_picker(current: Optional[str], available: list[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text=f"🔁 Авто (bilinear){' ✅' if not current else ''}",
             callback_data=RgsUpscaleModelCB(idx=-1).pack())
    lbls = upscale_models_db.labels()
    for i, m in enumerate(available):
        mark = " ✅" if m == current else ""
        b.button(text=f"{_label(m, lbls)}{mark}", callback_data=RgsUpscaleModelCB(idx=i).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(1)
    return b.as_markup()


def kb_rgs_lora_strength_picker(current: float) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for v in LORA_STRENGTH_PRESETS:
        mark = " ✅" if float(current) == v else ""
        b.button(text=f"{v}{mark}", callback_data=RgsCB(action="set_lora_strength", value=str(v)).pack())
    b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
    b.adjust(5)
    return b.as_markup()

# ── history keyboards / helpers ──────────────────────────────────────────

def kb_hist_nav(idx: int, total: int, uid: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    noop = HistoryCB(action="noop", uid=uid).pack()
    b.button(text="◀",
             callback_data=HistoryCB(action="nav", idx=idx - 1, uid=uid).pack() if idx > 0 else noop)
    b.button(text=f"{idx + 1} / {total}", callback_data=noop)
    b.button(text="▶",
             callback_data=HistoryCB(action="nav", idx=idx + 1, uid=uid).pack() if idx < total - 1 else noop)
    b.button(text="🔄 Перегенерувати", callback_data=HistoryCB(action="regen",   idx=idx, uid=uid).pack())
    b.button(text="🔍 Апскейл",        callback_data=UpscaleCB(action="pick",    idx=idx, uid=uid).pack())
    b.button(text="🗑 Видалити",        callback_data=HistoryCB(action="del",     idx=idx, uid=uid).pack())
    b.button(text="🔙 Назад",           callback_data=HistoryCB(action="back",   uid=uid).pack())
    b.adjust(3, 2, 1, 1)
    return b.as_markup()


def _hist_caption(entry: dict, pos: int, total: int, uid: int, viewer_id: int) -> str:
    from datetime import datetime as _dt
    ts       = _dt.fromisoformat(entry["ts"])
    date_str = ts.strftime("%d.%m %H:%M")
    ckpt     = _label(entry.get("checkpoint", "?"), models_db.labels())
    w, h     = entry.get("width", "?"), entry.get("height", "?")
    mode     = entry.get("mode", "text2img")
    prompt   = entry["prompt"]
    if len(prompt) > 300:
        prompt = prompt[:300] + "…"
    header = f"📜 <b>{pos} / {total}</b>  •  {date_str}"
    if uid != viewer_id:
        header += f"\n👤 @{entry.get('username', '?')}"
    return (
        f"{header}\n\n"
        f"💬 {prompt}\n\n"
        f"🤖 {ckpt}  •  📐 {w}×{h}  •  {mode}"
    )

# ── models admin keyboards ────────────────────────────────────────────────

def kb_models_admin() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    lbls = models_db.labels()
    wfls = models_db.workflows()
    for m in models_db.all_models():
        icon = comfy_client.WORKFLOW_ICONS.get(wfls.get(m, "sd15"), "🎨")
        b.button(text=f"{icon} {_label(m, lbls)}", callback_data=ModelsCB(action="view", value=_mk_id(m)).pack())
    b.button(text="➕ Додати модель", callback_data=ModelsCB(action="add").pack())
    b.button(text="🔙 Назад",         callback_data="menu:settings")
    b.adjust(1)
    return b.as_markup()

def kb_model_detail(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    mid = _mk_id(name)
    b.button(text="✏️ Перейменувати",  callback_data=ModelsCB(action="edit",          value=mid).pack())
    b.button(text="🔄 Тип workflow",   callback_data=ModelsCB(action="workflow_pick",  value=mid).pack())
    b.button(text="🗑 Видалити",       callback_data=ModelsCB(action="delete",         value=mid).pack())
    b.button(text="🔙 До списку",      callback_data="menu:models")
    b.adjust(2, 1, 1)
    return b.as_markup()


def kb_workflow_picker(current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for wt in comfy_client.WORKFLOW_TYPES:
        icon  = comfy_client.WORKFLOW_ICONS[wt]
        label = comfy_client.WORKFLOW_LABELS[wt]
        mark  = " ✅" if wt == current else ""
        b.button(text=f"{icon} {label}{mark}", callback_data=WorkflowCB(wtype=wt).pack())
    b.button(text="🔙 Назад", callback_data="menu:models")
    b.adjust(2, 2, 1)
    return b.as_markup()

def kb_model_confirm_delete(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    mid = _mk_id(name)
    b.button(text="✅ Так, видалити", callback_data=ModelsCB(action="delete_ok", value=mid).pack())
    b.button(text="❌ Скасувати",     callback_data=ModelsCB(action="view",      value=mid).pack())
    b.adjust(2)
    return b.as_markup()

def kb_cancel_to_models() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="❌ Скасувати", callback_data="menu:models")
    return b.as_markup()

def kb_cancel_to_loras() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="❌ Скасувати", callback_data="menu:loras")
    return b.as_markup()

def kb_cancel_to_upscale_models() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="❌ Скасувати", callback_data="menu:upscale_models")
    return b.as_markup()

def kb_loras_admin() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    lbls = loras_db.labels()
    for name in loras_db.all_loras():
        b.button(text=f"🎭 {_label(name, lbls)}", callback_data=LorasCB(action="view", value=name).pack())
    b.button(text="➕ Додати LoRA", callback_data=LorasCB(action="add").pack())
    b.button(text="🔙 Назад",      callback_data="menu:settings")
    b.adjust(1)
    return b.as_markup()

def kb_lora_detail(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✏️ Перейменувати",  callback_data=LorasCB(action="edit",    value=name).pack())
    b.button(text="🔑 Тригер слово",   callback_data=LorasCB(action="trigger", value=name).pack())
    b.button(text="🗑 Видалити",       callback_data=LorasCB(action="delete",  value=name).pack())
    b.button(text="🔙 До списку",      callback_data="menu:loras")
    b.adjust(2, 1, 1)
    return b.as_markup()

def kb_lora_confirm_delete(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Так, видалити", callback_data=LorasCB(action="delete_ok", value=name).pack())
    b.button(text="❌ Скасувати",     callback_data=LorasCB(action="view",      value=name).pack())
    b.adjust(2)
    return b.as_markup()

def kb_comfy_loras(available: list[str], existing: list[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for m in available:
        if m in existing:
            b.button(text=f"✅ {_short(m)}", callback_data=LorasCB(action="add_pick", value=m).pack())
        else:
            b.button(text=f"➕ {_short(m)}", callback_data=LorasCB(action="add_pick", value=m).pack())
    b.button(text="🔙 Назад", callback_data="menu:loras")
    b.adjust(1)
    return b.as_markup()

def kb_upscale_models_admin() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    lbls = upscale_models_db.labels()
    for name in upscale_models_db.all_upscale_models():
        b.button(text=f"🔍 {_label(name, lbls)}", callback_data=UpscaleMdlCB(action="view", value=name).pack())
    b.button(text="➕ Додати Upscale модель", callback_data=UpscaleMdlCB(action="add").pack())
    b.button(text="🔙 Назад",                callback_data="menu:settings")
    b.adjust(1)
    return b.as_markup()

def kb_upscale_model_detail(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✏️ Перейменувати", callback_data=UpscaleMdlCB(action="edit",   value=name).pack())
    b.button(text="🗑 Видалити",      callback_data=UpscaleMdlCB(action="delete", value=name).pack())
    b.button(text="🔙 До списку",     callback_data="menu:upscale_models")
    b.adjust(2, 1)
    return b.as_markup()

def kb_upscale_model_confirm_delete(name: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Так, видалити", callback_data=UpscaleMdlCB(action="delete_ok", value=name).pack())
    b.button(text="❌ Скасувати",     callback_data=UpscaleMdlCB(action="view",      value=name).pack())
    b.adjust(2)
    return b.as_markup()

def kb_comfy_upscale_models(available: list[str], existing: list[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for i, m in enumerate(available):
        prefix = "✅" if m in existing else "➕"
        b.button(text=f"{prefix} {_short(m)}",
                 callback_data=UpscaleMdlCB(action="add_pick", value=str(i)).pack())
    b.button(text="🔙 Назад", callback_data="menu:upscale_models")
    b.adjust(1)
    return b.as_markup()

def kb_comfy_models(available: list[str], existing: list[str]) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for m in available:
        if m in existing:
            b.button(text=f"✅ {_short(m)}", callback_data=ModelsCB(action="add_pick", value=_mk_id(m)).pack())
        else:
            b.button(text=f"➕ {_short(m)}", callback_data=ModelsCB(action="add_pick", value=_mk_id(m)).pack())
    b.button(text="✏️ Ввести вручну", callback_data=ModelsCB(action="add_manual").pack())
    b.button(text="🔙 Назад",         callback_data="menu:models")
    b.adjust(1)
    return b.as_markup()

# ── error handler ─────────────────────────────────────────────────────────

@dp.errors()
async def error_handler(event: ErrorEvent) -> None:
    log.exception("Unhandled error: %s", event.exception)
    cq = event.update.callback_query
    if cq:
        try:
            await cq.answer(f"❌ {type(event.exception).__name__}: {event.exception}", show_alert=True)
        except Exception:
            pass

# ── /start ────────────────────────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    allowed, admin = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    name      = message.from_user.username or message.from_user.first_name
    role_text = "адміністратор" if admin else "користувач"
    await message.answer(
        f"Привіт, <b>{name}</b>! Ви підключені як <i>{role_text}</i>.\n\nОберіть дію:",
        parse_mode="HTML", reply_markup=kb_main(admin),
    )

# ── /help ─────────────────────────────────────────────────────────────────

@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    allowed, admin = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    lines = [
        "🤖 <b>MyReplicaBot — команди</b>\n",
        "/start — головне меню",
        "/gen — швидкий старт генерації",
        "/settings — налаштування генерації",
        "/history — моя історія зображень",
        "/status — статус ComfyUI",
        "/help — ця довідка",
    ]
    if admin:
        lines += [
            "",
            "👑 <b>Адміністраторські:</b>",
            "/users — управління користувачами",
        ]
    await message.answer("\n".join(lines), parse_mode="HTML")


# ── /gen ──────────────────────────────────────────────────────────────────

@dp.message(Command("gen"))
async def cmd_gen(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    gs   = db.get_gen_settings(message.from_user.id)
    mode = gs.get("mode", "text2img")
    if mode == "img2img":
        await message.answer(
            "🖼 <b>Режим img2img</b>\n\nНадішліть фото (можна одразу з підписом як промпт).",
            parse_mode="HTML", reply_markup=kb_cancel_to_main(),
        )
    else:
        await state.set_state(GenState.waiting_prompt)
        await message.answer("✏️ Введіть текстовий промпт для генерації зображення:",
                             reply_markup=kb_cancel_to_main())


# ── /settings ─────────────────────────────────────────────────────────────

@dp.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    await state.clear()
    tg_id = message.from_user.id
    await message.answer(_gen_settings_text(tg_id),
                         parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))


# ── /history ──────────────────────────────────────────────────────────────

@dp.message(Command("history"))
async def cmd_history(message: Message) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    entries = hist.get_user_history(message.from_user.id)
    if not entries:
        await message.answer("📭 Історія порожня — ще немає згенерованих зображень.")
        return
    entry   = entries[0]
    total   = len(entries)
    caption = _hist_caption(entry, 1, total, message.from_user.id, message.from_user.id)
    try:
        await message.answer_photo(
            FSInputFile(entry["file_path"]),
            caption=caption,
            parse_mode="HTML",
            reply_markup=kb_hist_nav(0, total, 0),
        )
    except Exception:
        await message.answer(caption, parse_mode="HTML",
                             reply_markup=kb_hist_nav(0, total, 0))


# ── /status ───────────────────────────────────────────────────────────────

@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    msg = await message.answer("🔄 Перевіряю...")
    await msg.edit_text(await _build_status_text(message.from_user.id),
                        parse_mode="HTML", reply_markup=kb_status())


# ── /users (admin) ────────────────────────────────────────────────────────

@dp.message(Command("users"))
async def cmd_users(message: Message) -> None:
    _, admin = _ctx(message.from_user)
    if not admin:
        await message.answer("⛔ Доступ лише для адміністраторів.")
        return
    await message.answer(_users_text(), parse_mode="HTML", reply_markup=kb_users())


# ── генерація ─────────────────────────────────────────────────────────────

@dp.callback_query(F.data == "gen:start")
async def cb_gen_start(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔ У вас немає доступу.", show_alert=True)
        return
    gs   = db.get_gen_settings(call.from_user.id)
    mode = gs.get("mode", "text2img")
    if mode == "img2img":
        await _nav(call,
                   "🖼 <b>Режим img2img</b>\n\nНадішліть фото (можна одразу з підписом як промпт).",
                   parse_mode="HTML", reply_markup=kb_cancel_to_main())
    else:
        await state.set_state(GenState.waiting_prompt)
        await _nav(call, "✏️ Введіть текстовий промпт для генерації зображення:",
                   reply_markup=kb_cancel_to_main())
    await call.answer()

@dp.message(GenState.waiting_prompt, F.text)
async def handle_gen_prompt(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await state.clear()
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    prompt = message.text.strip()
    if await _maybe_offer_styles(message, state, prompt):
        return
    await state.clear()
    await _do_generate(message, prompt)

@dp.message(GenState.waiting_prompt, F.photo)
async def handle_gen_prompt_photo(message: Message, state: FSMContext) -> None:
    await message.answer("✏️ Потрібен текстовий промпт. Введіть текст або скасуйте.")

# ── img2img — photo handlers ──────────────────────────────────────────────

@dp.message(Img2ImgState.waiting_prompt, F.photo)
async def handle_i2i_replace_photo(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await state.clear()
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    photo   = message.photo[-1]
    caption = (message.caption or "").strip()
    if caption:
        if await _maybe_offer_styles(message, state, caption, img_file_id=photo.file_id):
            return
        await state.clear()
        img = await _download_photo(photo.file_id)
        await _do_generate(message, caption, img)
    else:
        await state.update_data(img2img_file_id=photo.file_id)
        await message.answer("✅ Фото оновлено. Тепер введіть промпт:")

@dp.message(Img2ImgState.waiting_prompt, F.text)
async def handle_i2i_prompt(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await state.clear()
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    data    = await state.get_data()
    file_id = data.get("img2img_file_id")
    prompt  = message.text.strip()
    if await _maybe_offer_styles(message, state, prompt, img_file_id=file_id):
        return
    await state.clear()
    img = await _download_photo(file_id)
    await _do_generate(message, prompt, img)

@dp.message(StateFilter(None), F.photo)
async def handle_photo(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    gs   = db.get_gen_settings(message.from_user.id)
    mode = gs.get("mode", "text2img")
    if mode != "img2img":
        _, admin = _ctx(message.from_user)
        await message.answer(
            "ℹ️ Для генерації варіацій з фото увімкніть режим <b>img2img</b> у 🎛 Налаштуваннях генерації.",
            parse_mode="HTML", reply_markup=kb_main(admin),
        )
        return
    photo   = message.photo[-1]
    caption = (message.caption or "").strip()
    if caption:
        if await _maybe_offer_styles(message, state, caption, img_file_id=photo.file_id):
            return
        img = await _download_photo(photo.file_id)
        await _do_generate(message, caption, img)
    else:
        await state.update_data(img2img_file_id=photo.file_id)
        await state.set_state(Img2ImgState.waiting_prompt)
        await message.answer(
            "✏️ Фото отримано. Введіть промпт для генерації варіації:",
            reply_markup=kb_cancel_to_main(),
        )

@dp.message(StateFilter(None), F.text)
async def handle_text(message: Message, state: FSMContext) -> None:
    allowed, _ = _ctx(message.from_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return
    gs   = db.get_gen_settings(message.from_user.id)
    mode = gs.get("mode", "text2img")
    if mode == "img2img":
        _, admin = _ctx(message.from_user)
        await message.answer(
            "🖼 Ви у режимі <b>img2img</b>. Надішліть фото (з підписом або без).",
            parse_mode="HTML", reply_markup=kb_main(admin),
        )
        return
    prompt = message.text.strip()
    if await _maybe_offer_styles(message, state, prompt):
        return
    await _do_generate(message, prompt)

def _apply_style(prompt: str, style_key: str) -> str:
    style = STYLES.get(style_key)
    if not style:
        return prompt
    return f"{prompt}, {style['suffix']}"


async def _maybe_offer_styles(
    message: Message,
    state: FSMContext,
    prompt: str,
    img_file_id: Optional[str] = None,
) -> bool:
    """Apply style or show picker depending on how many styles are active.

    - 0 styles → return False (caller proceeds without style)
    - 1 style  → auto-apply silently, return False (caller proceeds with modified prompt)
    - 2+ styles → show picker, return True (caller must wait for StyleCB)

    When this returns False the *prompt* variable in the caller is NOT modified here,
    so for the single-style fast-path we call _do_generate directly and return True
    to prevent the caller from calling it again.
    """
    gs            = db.get_gen_settings(message.from_user.id)
    active        = [k for k in (gs.get("active_styles") or []) if k in STYLES]
    custom_suffix = gs.get("custom_style_suffix") or ""

    total = len(active) + (1 if custom_suffix else 0)

    # No styles at all — nothing to do
    if total == 0:
        return False

    # Exactly one style — apply automatically without asking
    if total == 1:
        img_bytes: Optional[bytes] = None
        if img_file_id:
            img_bytes = await _download_photo(img_file_id)
        if active:
            # single standard style
            solo_key  = active[0]
            new_prompt = _apply_style(prompt, solo_key)
            style_neg  = STYLES[solo_key].get("neg_suffix", "")
            gen_settings: Optional[dict] = None
            if style_neg:
                gs2 = dict(gs)
                base_neg = gs2.get("negative_prompt") or config.NEGATIVE_PROMPT
                gs2["negative_prompt"] = f"{base_neg}, {style_neg}"
                gen_settings = gs2
        else:
            # only custom suffix
            new_prompt   = f"{prompt}, {custom_suffix}"
            gen_settings = dict(gs)
        await state.clear()
        await _do_generate(message, new_prompt, input_image=img_bytes,
                           user_settings=gen_settings)
        return True

    # Multiple styles — show picker
    await state.set_state(StylePickState.waiting)
    await state.update_data(style_prompt=prompt, style_img_file_id=img_file_id)
    b = InlineKeyboardBuilder()
    for key in active:
        b.button(text=STYLES[key]["name"], callback_data=StyleCB(key=key).pack())
    if custom_suffix:
        preview = custom_suffix[:20] + ("…" if len(custom_suffix) > 20 else "")
        b.button(text=f"✏️ Власний ({preview})", callback_data=StyleCB(key="__custom__").pack())
    b.button(text="❌ Без стилю", callback_data=StyleCB(key="__none__").pack())
    b.adjust(2)
    await message.answer(
        "🎨 <b>Оберіть стиль зображення:</b>",
        parse_mode="HTML", reply_markup=b.as_markup(),
    )
    return True


async def _do_generate(
    message: Message,
    prompt: str,
    input_image: Optional[bytes] = None,
    user_settings: Optional[dict] = None,
    from_user=None,
) -> None:
    if not prompt:
        return
    # from_user overrides message.from_user when called from callback handlers
    # (call.message.from_user is the bot, not the user)
    actual_user = from_user or message.from_user
    allowed, _ = _ctx(actual_user)
    if not allowed:
        await message.answer("⛔ У вас немає доступу до цього бота.")
        return

    if user_settings is None:
        user_settings = db.get_gen_settings(actual_user.id)

    # inject workflow type so comfy_client picks the right builder
    user_settings = dict(user_settings)
    _ckpt_for_wf = user_settings.get("checkpoint") or config.CHECKPOINT
    user_settings["_workflow_type"] = models_db.get_workflow(_ckpt_for_wf)

    en_prompt = await translator.to_english(prompt)
    if en_prompt != prompt:
        log.info("Translated prompt: %r → %r", prompt, en_prompt)

    # Inject LoRA trigger words (English) after translation
    _active_for_triggers = _get_active_loras(user_settings)
    if _active_for_triggers:
        _trig_map   = loras_db.triggers()
        _trig_words = [_trig_map[l["name"]] for l in _active_for_triggers
                       if _trig_map.get(l["name"])]
        if _trig_words:
            en_prompt = ", ".join(_trig_words) + ", " + en_prompt

    batch_size = int(user_settings.get("batch_size", 1))
    log.info("Enqueue user=%d batch=%d prompt=%r", actual_user.id, batch_size, en_prompt)

    _mode  = user_settings.get("mode", "text2img")
    _ckpt  = user_settings.get("checkpoint") or config.CHECKPOINT
    _w     = user_settings.get("width")      or config.IMAGE_WIDTH
    _h     = user_settings.get("height")     or config.IMAGE_HEIGHT
    if user_settings.get("hires_fix"):
        _w = _w * 2
        _h = _h * 2
    _uname = actual_user.username or ""

    # Build prompt caption once (Telegram photo caption limit: 1024 chars)
    _caption = f"<b>Промпт:</b> {en_prompt}"
    if en_prompt != prompt:
        _orig = prompt if len(prompt) <= 150 else prompt[:150] + "…"
        _caption += f"\n<i>🇺🇦 {_orig}</i>"
    _caption_overflow = len(_caption) > 1024

    remaining    = [batch_size]
    sent_count   = [0]       # how many images have already been sent
    errors:  list[str] = []

    async def _finalize(msg: Message) -> None:
        """Called when all batch jobs are finished. Cleans up status and shows menu."""
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        _, admin = _ctx(actual_user)
        if sent_count[0] == 0 and not errors:
            await msg.answer("❌ Генерацію скасовано.", reply_markup=kb_main(admin))
            return
        # If prompt was too long to fit in photo caption — send it as a text message now
        if _caption_overflow and sent_count[0] > 0:
            await msg.answer(_caption, parse_mode="HTML")
        for err in errors:
            await msg.answer(err, parse_mode="HTML")
        await msg.answer("Оберіть дію:", reply_markup=kb_main(admin))

    async def on_done(msg: Message, image_bytes: bytes, pmt: str) -> None:
        remaining[0] -= 1
        db.increment_gen_count(actual_user.id)

        # Save to disk and history immediately
        eid, fp = hist.save_image(actual_user.id, image_bytes)
        hist.add_entry(eid, fp, actual_user.id, _uname, en_prompt, _mode, _ckpt, _w, _h)

        # Attach caption to the first image only (if it fits)
        photo_caption = None
        if sent_count[0] == 0 and not _caption_overflow:
            photo_caption = _caption
        sent_count[0] += 1

        # Send image right away — don't wait for the rest of the batch
        await msg.answer_photo(
            FSInputFile(fp),
            caption=photo_caption,
            parse_mode="HTML" if photo_caption else None,
        )

        if remaining[0] == 0:
            await _finalize(msg)

    async def on_error(msg: Message, text: str) -> None:
        remaining[0] -= 1
        errors.append(text)
        if remaining[0] == 0:
            await _finalize(msg)

    async def on_cancel(msg: Message) -> None:
        remaining[0] -= 1
        if remaining[0] == 0:
            await _finalize(msg)

    ahead = gq.queue_len()
    if ahead == 0:
        status_text = "⏳ Підключаюсь до ComfyUI..."
    else:
        noun = gq._inflect(ahead)
        status_text = (
            f"🕐 <b>В черзі</b>\n\n"
            f"<code>{'░' * 20}</code>\n"
            f"Попереду: <b>{ahead} {noun}</b>\n"
            f"<i>Повідомлення оновиться, коли дійде ваша черга</i>"
        )
    status = await message.answer(status_text, parse_mode="HTML")

    cancel_kb = (
        InlineKeyboardBuilder()
        .button(text="❌ Відмінити", callback_data=CancelCB(msg_id=status.message_id).pack())
        .as_markup()
    )

    for i in range(batch_size):
        await gq.enqueue(gq.GenJob(
            message=message,
            prompt=en_prompt,
            user_settings=user_settings,
            status_msg=status,
            on_done=on_done,
            on_error=on_error,
            input_image=input_image,
            batch_index=i + 1,
            batch_total=batch_size,
            cancel_kb=cancel_kb,
            on_cancel=on_cancel,
        ))

# ── вибір стилю перед генерацією ─────────────────────────────────────────

@dp.callback_query(StyleCB.filter())
async def cb_style_pick(call: CallbackQuery, callback_data: StyleCB, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    data        = await state.get_data()
    prompt      = data.get("style_prompt", "")
    img_file_id = data.get("style_img_file_id")
    await state.clear()
    await call.answer()

    gen_settings: Optional[dict] = None
    if callback_data.key == "__custom__":
        # Apply custom user suffix
        gs = dict(db.get_gen_settings(call.from_user.id))
        custom_suffix = gs.get("custom_style_suffix") or ""
        if custom_suffix:
            prompt = f"{prompt}, {custom_suffix}"
        gen_settings = gs
    elif callback_data.key != "__none__":
        prompt = _apply_style(prompt, callback_data.key)
        style_neg = STYLES[callback_data.key].get("neg_suffix", "")
        if style_neg:
            gs = dict(db.get_gen_settings(call.from_user.id))
            base_neg = gs.get("negative_prompt") or config.NEGATIVE_PROMPT
            gs["negative_prompt"] = f"{base_neg}, {style_neg}"
            gen_settings = gs

    img_bytes: Optional[bytes] = None
    if img_file_id:
        img_bytes = await _download_photo(img_file_id)

    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass

    await _do_generate(call.message, prompt, input_image=img_bytes,
                       from_user=call.from_user, user_settings=gen_settings)

# ── апскейл зображення ───────────────────────────────────────────────────

@dp.callback_query(UpscaleCB.filter(F.action == "pick"))
async def cb_upscale_pick(call: CallbackQuery, callback_data: UpscaleCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    idx = callback_data.idx
    uid = callback_data.uid
    b   = InlineKeyboardBuilder()
    for val, label in UPSCALE_SCALES:
        b.button(text=f"🔍 {label}",
                 callback_data=UpscaleCB(action="do", idx=idx, uid=uid, scale=val).pack())
    b.button(text="🔙 Назад", callback_data=HistoryCB(action="nav", idx=idx, uid=uid).pack())
    b.adjust(1)
    try:
        await call.message.edit_caption(
            "🔍 <b>Апскейл зображення</b>\n\nОберіть масштаб:",
            parse_mode="HTML", reply_markup=b.as_markup(),
        )
    except TelegramBadRequest:
        pass
    await call.answer()


@dp.callback_query(UpscaleCB.filter(F.action == "do"))
async def cb_upscale_do(call: CallbackQuery, callback_data: UpscaleCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = callback_data.idx
    if not entries or idx >= len(entries):
        await call.answer("Запис не знайдено.", show_alert=True)
        return
    entry         = entries[idx]
    scale         = float(callback_data.scale or "2")
    gs            = db.get_gen_settings(call.from_user.id)
    upscale_model = gs.get("upscale_model")
    # validate that the saved model still exists in DB list
    if upscale_model and upscale_model not in upscale_models_db.all_upscale_models():
        upscale_model = None
    await call.answer()
    try:
        await call.message.edit_caption(
            f"🔍 <b>Апскейл ×{int(scale)}…</b>\n\n<code>{'░' * 20}</code>",
            parse_mode="HTML", reply_markup=None,
        )
    except TelegramBadRequest:
        pass

    status_msg = call.message

    async def on_progress(value: int, total: int) -> None:
        bar = comfy_client.progress_bar(value, total)
        try:
            await status_msg.edit_caption(
                f"🔍 <b>Апскейл ×{int(scale)}…</b>\n\n<code>{bar}</code>",
                parse_mode="HTML",
            )
        except TelegramBadRequest:
            pass

    eff_uid = uid if uid != 0 else call.from_user.id
    try:
        with open(entry["file_path"], "rb") as f:
            image_bytes = f.read()
        w = int(entry.get("width")  or config.IMAGE_WIDTH)
        h = int(entry.get("height") or config.IMAGE_HEIGHT)
        result = await comfy_client.upscale_image(
            image_bytes, w, h, scale, upscale_model, on_progress,
        )
        eid, fp  = hist.save_image(call.from_user.id, result)
        uname    = call.from_user.username or ""
        hist.add_entry(eid, fp, call.from_user.id, uname,
                       f"[upscale ×{int(scale)}] {entry['prompt']}",
                       "upscale", entry.get("checkpoint", ""),
                       int(w * scale), int(h * scale))
        _, admin = _ctx(call.from_user)
        try:
            await call.message.delete()
        except TelegramBadRequest:
            pass
        await call.message.answer_photo(
            FSInputFile(fp),
            caption=(
                f"✅ <b>Апскейл ×{int(scale)} готовий</b>\n\n"
                f"💬 {entry['prompt'][:200]}{'…' if len(entry['prompt']) > 200 else ''}"
            ),
            parse_mode="HTML", reply_markup=kb_main(admin),
        )
    except Exception as exc:
        log.error("Upscale error: %s", exc)
        try:
            await call.message.edit_caption(
                f"❌ Помилка апскейлу: {exc}",
                reply_markup=kb_hist_nav(idx, len(entries), eff_uid),
            )
        except TelegramBadRequest:
            pass


# ── статус ComfyUI ────────────────────────────────────────────────────────

@dp.callback_query(F.data == "comfy:status")
async def cb_comfy_status(call: CallbackQuery) -> None:
    await call.answer()
    await _nav(call, "🔄 Перевіряю...", reply_markup=None)
    await _nav(call, await _build_status_text(call.from_user.id), parse_mode="HTML", reply_markup=kb_status())

# ── налаштування генерації ────────────────────────────────────────────────

@dp.callback_query(GsCB.filter(F.action == "menu"))
async def cb_gs_menu(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))
    await call.answer()

@dp.callback_query(GsCB.filter(F.action == "show"))
async def cb_gs_show(call: CallbackQuery, callback_data: GsCB, state: FSMContext) -> None:
    tg_id = call.from_user.id
    s     = db.get_gen_settings(tg_id)

    if callback_data.value == "mode":
        cur     = s.get("mode", "text2img")
        wf_type = models_db.get_workflow(s.get("checkpoint") or config.CHECKPOINT)
        await _nav(call,
                   "🔄 <b>Оберіть режим генерації:</b>\n\n"
                   "<b>text2img</b> — генерація з текстового промпту\n"
                   "<b>img2img</b>  — варіація вашого фото за промптом",
                   parse_mode="HTML", reply_markup=kb_mode_picker(cur, wf_type))

    elif callback_data.value == "model":
        cur = s.get("checkpoint") or config.CHECKPOINT
        await _nav(call, "🤖 <b>Оберіть модель:</b>",
                   parse_mode="HTML", reply_markup=kb_model_picker(cur))

    elif callback_data.value == "size":
        w = s.get("width")  or config.IMAGE_WIDTH
        h = s.get("height") or config.IMAGE_HEIGHT
        await _nav(call, "📐 <b>Оберіть розмір зображення:</b>\n"
                   "<i>У режимі img2img вхідне фото масштабується до цього розміру.</i>",
                   parse_mode="HTML", reply_markup=kb_size_picker(w, h))

    elif callback_data.value == "steps":
        cur = s.get("steps") or config.STEPS
        await _nav(call,
                   "🎚 <b>Оберіть кількість кроків:</b>\n<i>Більше кроків — краща якість, але повільніше.</i>",
                   parse_mode="HTML", reply_markup=kb_steps_picker(cur))

    elif callback_data.value == "cfg":
        cur = float(s.get("cfg") or config.CFG_SCALE)
        await _nav(call,
                   "🎯 <b>Оберіть CFG Scale:</b>\n<i>Низьке = більше свободи, високе = точніше до промпту.</i>",
                   parse_mode="HTML", reply_markup=kb_cfg_picker(cur))

    elif callback_data.value == "sampler":
        cur = s.get("sampler") or "euler"
        await _nav(call, "🎲 <b>Оберіть семплер:</b>",
                   parse_mode="HTML", reply_markup=kb_sampler_picker(cur))

    elif callback_data.value == "denoise":
        cur = float(s.get("denoise") or 0.75)
        await _nav(call,
                   "🎨 <b>Сила варіації (denoise):</b>\n"
                   "<i>0.3 — слабка (схоже на оригінал), 0.75 — сильна, 1.0 — повна перегенерація.</i>",
                   parse_mode="HTML", reply_markup=kb_denoise_picker(cur))

    elif callback_data.value == "batch":
        cur = int(s.get("batch_size", 1))
        await _nav(call,
                   "🔢 <b>Кількість зображень:</b>\n\n"
                   "<i>Бот згенерує N варіацій одного промпту підряд.\n"
                   "Кожне зображення надсилається окремо по мірі готовності.</i>",
                   parse_mode="HTML", reply_markup=kb_batch_picker(cur))

    elif callback_data.value == "loras":
        active = _get_active_loras(s)
        count  = len(active)
        all_l  = loras_db.all_loras()
        text = (
            "🎭 <b>Вибір LoRA</b>\n\n"
            "Натисніть ➕ щоб увімкнути, або ✅ щоб змінити вагу / вимкнути.\n"
            "Всі активні LoRA накладаються одночасно.\n\n"
            f"<b>Активних: {count} / {len(all_l)}</b>"
        )
        if not all_l:
            text += "\n\n⚠️ Список LoRA порожній. Адмін може додати LoRA у ⚙️ Налаштуваннях."
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_multi_lora_picker(tg_id))

    elif callback_data.value == "neg":
        neg  = s.get("negative_prompt")
        text = "📝 <b>Негативний промпт</b>\n\n"
        text += f"Поточний:\n<code>{neg}</code>\n\n" if neg else "Зараз використовується стандартний.\n\n"
        text += "Надішліть новий негативний промпт:"
        await state.set_state(GenSettingsState.waiting_neg_prompt)
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_neg_prompt(bool(neg)))

    elif callback_data.value == "styles":
        active        = s.get("active_styles") or []
        custom_suffix = s.get("custom_style_suffix") or ""
        count         = len(active)
        text = (
            "🎨 <b>Стилі генерації</b>\n\n"
            "Активовані стилі показуватимуться при введенні промпту.\n"
            "Оберіть стиль — і він додається до промпту перед генерацією.\n"
            "➕ <b>Власний суфікс</b> — ваш особистий текст що додається до кожного промпту.\n\n"
            f"<b>Активовано: {count} / {len(STYLES)}</b>"
        )
        if custom_suffix:
            text += f"\n✏️ Власний: <i>{custom_suffix[:60]}{'…' if len(custom_suffix) > 60 else ''}</i>"
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_style_toggle(tg_id))

    elif callback_data.value == "set_custom_style":
        cur_custom = s.get("custom_style_suffix") or ""
        text = "✏️ <b>Власний суфікс стилю</b>\n\n"
        if cur_custom:
            text += f"Поточний:\n<code>{cur_custom}</code>\n\n"
        else:
            text += "Поки не встановлено.\n\n"
        text += (
            "Введіть текст що автоматично додається до кожного промпту "
            "при виборі стилю <b>✏️ Власний</b>.\n\n"
            "<i>Наприклад: VRay render, game asset, white background</i>"
        )
        b = InlineKeyboardBuilder()
        if cur_custom:
            b.button(text="🗑 Видалити власний суфікс",
                     callback_data=GsCB(action="clear_custom_style").pack())
        b.button(text="🔙 Назад", callback_data=GsCB(action="show", value="styles").pack())
        b.adjust(1)
        await state.set_state(GenSettingsState.waiting_custom_style)
        await _nav(call, text, parse_mode="HTML", reply_markup=b.as_markup())

    elif callback_data.value == "hires":
        hires_fix = bool(s.get("hires_fix"))
        new_val   = not hires_fix
        db.set_gen_setting(tg_id, "hires_fix", new_val if new_val else None)
        state_str = "ввімкнений ×2" if new_val else "вимкнений"
        await call.answer(f"🔍 HiRes Fix: {state_str}")
        await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))
        return

    elif callback_data.value == "hires_denoise":
        cur = float(s.get("hires_denoise") or 0.45)
        await _nav(call,
                   "🎚 <b>HiRes Fix: сила другого проходу (denoise)</b>\n\n"
                   "<i>0.3 — м'яке уточнення деталей\n"
                   "0.45 — стандарт\n"
                   "0.6 — сильна зміна</i>",
                   parse_mode="HTML", reply_markup=kb_hires_denoise_picker(cur))

    elif callback_data.value == "upscale_model":
        cur       = s.get("upscale_model")
        available = upscale_models_db.all_upscale_models()
        await state.update_data(upscale_models_list=available)
        text = "🖼 <b>Upscale модель</b>\n\n"
        if available:
            text += f"Доступно: <b>{len(available)}</b> моделей.\nВикористовується для HiRes Fix і Апскейлу з історії."
        else:
            text += "⚠️ Список порожній. Адмін може додати моделі у ⚙️ Налаштуваннях."
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_upscale_model_picker(cur, available))

    await call.answer()

@dp.callback_query(GsCB.filter(F.action == "set_mode"))
async def cb_gs_set_mode(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "mode", callback_data.value)
    await call.answer(f"✅ Режим: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_model"))
async def cb_gs_set_model(call: CallbackQuery, callback_data: GsCB) -> None:
    name = _resolve_model_id(callback_data.value)
    db.set_gen_setting(call.from_user.id, "checkpoint", name)
    if models_db.get_workflow(name) in _NO_IMG2IMG_WF:
        db.set_gen_setting(call.from_user.id, "mode", "text2img")
    await call.answer(f"✅ Модель: {_label(name, models_db.labels())}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_size"))
async def cb_gs_set_size(call: CallbackQuery, callback_data: GsCB) -> None:
    w, h = map(int, callback_data.value.split("×"))
    db.set_gen_setting(call.from_user.id, "width",  w)
    db.set_gen_setting(call.from_user.id, "height", h)
    await call.answer(f"✅ Розмір {w}×{h}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_steps"))
async def cb_gs_set_steps(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "steps", int(callback_data.value))
    await call.answer(f"✅ Кроки: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_cfg"))
async def cb_gs_set_cfg(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "cfg", float(callback_data.value))
    await call.answer(f"✅ CFG: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_sampler"))
async def cb_gs_set_sampler(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "sampler", callback_data.value)
    await call.answer(f"✅ Семплер: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_denoise"))
async def cb_gs_set_denoise(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "denoise", float(callback_data.value))
    await call.answer(f"✅ Варіація: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_batch"))
async def cb_gs_set_batch(call: CallbackQuery, callback_data: GsCB) -> None:
    db.set_gen_setting(call.from_user.id, "batch_size", int(callback_data.value))
    await call.answer(f"✅ Кількість: {callback_data.value}", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "neg_clear"))
async def cb_gs_neg_clear(call: CallbackQuery) -> None:
    db.set_gen_setting(call.from_user.id, "negative_prompt", None)
    await call.answer("✅ Негативний промпт скинуто", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "clear_custom_style"))
async def cb_gs_clear_custom_style(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    db.set_gen_setting(call.from_user.id, "custom_style_suffix", None)
    await call.answer("✅ Власний суфікс видалено", show_alert=False)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.message(GenSettingsState.waiting_custom_style, F.text)
async def handle_custom_style_input(message: Message, state: FSMContext) -> None:
    await state.clear()
    suffix = message.text.strip()
    db.set_gen_setting(message.from_user.id, "custom_style_suffix", suffix)
    tg_id = message.from_user.id
    await message.answer(
        f"✅ Власний суфікс збережено:\n<code>{suffix}</code>",
        parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_lora"))
async def cb_gs_set_lora(call: CallbackQuery, callback_data: GsCB) -> None:
    tg_id = call.from_user.id
    if callback_data.value == "__none__":
        db.set_gen_setting(tg_id, "lora", None)
        await call.answer("✅ LoRA вимкнено")
    else:
        db.set_gen_setting(tg_id, "lora", callback_data.value)
        await call.answer(f"✅ LoRA: {_label(callback_data.value, loras_db.labels())}", show_alert=False)
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_lora_strength"))
async def cb_gs_set_lora_strength(call: CallbackQuery, callback_data: GsCB) -> None:
    tg_id = call.from_user.id
    db.set_gen_setting(tg_id, "lora_strength", float(callback_data.value))
    await call.answer(f"✅ Сила LoRA: {callback_data.value}", show_alert=False)
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.callback_query(MultiLoraCB.filter(F.action == "tog"))
async def cb_multi_lora_toggle(call: CallbackQuery, callback_data: MultiLoraCB) -> None:
    tg_id  = call.from_user.id
    name   = _resolve_lora_id(callback_data.lid or "")
    now_on = _toggle_lora(tg_id, name)
    lbls   = loras_db.labels()
    if now_on:
        await call.answer(f"✅ {_label(name, lbls)} увімкнено (вага 0.8)")
    else:
        await call.answer(f"❌ {_label(name, lbls)} вимкнено")
    s      = db.get_gen_settings(tg_id)
    active = _get_active_loras(s)
    count  = len(active)
    all_l  = loras_db.all_loras()
    text   = (
        "🎭 <b>Вибір LoRA</b>\n\n"
        "Натисніть ➕ щоб увімкнути, або ✅ щоб змінити вагу / вимкнути.\n\n"
        f"<b>Активних: {count} / {len(all_l)}</b>"
    )
    await _nav(call, text, parse_mode="HTML", reply_markup=kb_multi_lora_picker(tg_id))


@dp.callback_query(MultiLoraCB.filter(F.action == "str_pick"))
async def cb_multi_lora_str_pick(call: CallbackQuery, callback_data: MultiLoraCB) -> None:
    tg_id  = call.from_user.id
    name   = _resolve_lora_id(callback_data.lid or "")
    s      = db.get_gen_settings(tg_id)
    active = _get_active_loras(s)
    cur_s  = next((l.get("strength", 0.8) for l in active if l["name"] == name), 0.8)
    lbls   = loras_db.labels()
    await _nav(call,
               f"🎚 <b>Вага LoRA: {_label(name, lbls)}</b>\n\n"
               f"Поточна: <b>{cur_s}</b>\n"
               "<i>Типові значення: 0.7–0.9. Більше = сильніший вплив.</i>",
               parse_mode="HTML",
               reply_markup=kb_lora_strength_multi(name, callback_data.lid, cur_s))
    await call.answer()


@dp.callback_query(MultiLoraCB.filter(F.action == "str_set"))
async def cb_multi_lora_str_set(call: CallbackQuery, callback_data: MultiLoraCB) -> None:
    tg_id  = call.from_user.id
    name   = _resolve_lora_id(callback_data.lid or "")
    val    = float(callback_data.val or "0.8")
    _set_lora_strength(tg_id, name, val)
    await call.answer(f"✅ Вага: {val}", show_alert=False)
    s      = db.get_gen_settings(tg_id)
    active = _get_active_loras(s)
    cur_s  = next((l.get("strength", 0.8) for l in active if l["name"] == name), val)
    lbls   = loras_db.labels()
    await _nav(call,
               f"🎚 <b>Вага LoRA: {_label(name, lbls)}</b>\n\n"
               f"Поточна: <b>{cur_s}</b>",
               parse_mode="HTML",
               reply_markup=kb_lora_strength_multi(name, callback_data.lid, cur_s))


@dp.callback_query(MultiLoraCB.filter(F.action == "lra_off"))
async def cb_multi_lora_off(call: CallbackQuery, callback_data: MultiLoraCB) -> None:
    tg_id  = call.from_user.id
    name   = _resolve_lora_id(callback_data.lid or "")
    s      = db.get_gen_settings(tg_id)
    active = [l for l in _get_active_loras(s) if l["name"] != name]
    db.set_gen_setting(tg_id, "loras_active", active or None)
    lbls   = loras_db.labels()
    await call.answer(f"❌ {_label(name, lbls)} вимкнено")
    all_l  = loras_db.all_loras()
    text   = (
        "🎭 <b>Вибір LoRA</b>\n\n"
        "Натисніть ➕ щоб увімкнути, або ✅ щоб змінити вагу / вимкнути.\n\n"
        f"<b>Активних: {len(active)} / {len(all_l)}</b>"
    )
    await _nav(call, text, parse_mode="HTML", reply_markup=kb_multi_lora_picker(tg_id))


@dp.callback_query(GsCB.filter(F.action == "toggle_style"))
async def cb_gs_toggle_style(call: CallbackQuery, callback_data: GsCB) -> None:
    tg_id  = call.from_user.id
    gs     = db.get_gen_settings(tg_id)
    active = list(gs.get("active_styles") or [])
    key    = callback_data.value
    if key in active:
        active.remove(key)
        await call.answer(f"❌ {STYLES[key]['name']} вимкнено")
    else:
        active.append(key)
        await call.answer(f"✅ {STYLES[key]['name']} увімкнено")
    db.set_gen_setting(tg_id, "active_styles", active or None)
    active_now = len(active)
    text = (
        "🎨 <b>Стилі генерації</b>\n\n"
        "Активовані стилі показуватимуться при введенні промпту.\n"
        "Оберіть стиль — і він додається до промпту перед генерацією.\n\n"
        f"<b>Активовано: {active_now} / {len(STYLES)}</b>"
    )
    await _nav(call, text, parse_mode="HTML", reply_markup=kb_style_toggle(tg_id))

@dp.callback_query(GsCB.filter(F.action == "set_hires_denoise"))
async def cb_gs_set_hires_denoise(call: CallbackQuery, callback_data: GsCB) -> None:
    tg_id = call.from_user.id
    db.set_gen_setting(tg_id, "hires_denoise", float(callback_data.value))
    await call.answer(f"✅ HiRes denoise: {callback_data.value}", show_alert=False)
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))


@dp.callback_query(UpscaleModelCB.filter())
async def cb_gs_set_upscale_model(call: CallbackQuery, callback_data: UpscaleModelCB,
                                   state: FSMContext) -> None:
    tg_id = call.from_user.id
    if callback_data.idx == -1:
        db.set_gen_setting(tg_id, "upscale_model", None)
        await call.answer("✅ Upscale: авто (bilinear)")
    else:
        data      = await state.get_data()
        available = data.get("upscale_models_list", [])
        if callback_data.idx >= len(available):
            await call.answer("⚠️ Відкрийте список заново", show_alert=True)
            return
        model = available[callback_data.idx]
        db.set_gen_setting(tg_id, "upscale_model", model)
        await call.answer(f"✅ {_label(model, upscale_models_db.labels())}", show_alert=False)
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))


@dp.callback_query(GsCB.filter(F.action == "reset"))
async def cb_gs_reset(call: CallbackQuery) -> None:
    db.reset_gen_settings(call.from_user.id)
    await call.answer("✅ Налаштування скинуто до стандартних", show_alert=True)
    tg_id = call.from_user.id
    await _nav(call, _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

@dp.message(GenSettingsState.waiting_neg_prompt, F.text)
async def handle_neg_prompt_input(message: Message, state: FSMContext) -> None:
    await state.clear()
    db.set_gen_setting(message.from_user.id, "negative_prompt", message.text.strip())
    tg_id = message.from_user.id
    await message.answer(
        _gen_settings_text(tg_id), parse_mode="HTML", reply_markup=kb_gen_settings(tg_id))

# ── управління моделями (адмін) ───────────────────────────────────────────

@dp.callback_query(F.data == "menu:models")
async def cb_menu_models(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await _nav(call, _models_text(), parse_mode="HTML", reply_markup=kb_models_admin())
    await call.answer()

@dp.callback_query(ModelsCB.filter(F.action == "view"))
async def cb_mdl_view(call: CallbackQuery, callback_data: ModelsCB, state: FSMContext) -> None:
    name   = _resolve_model_id(callback_data.value)
    label  = _label(name, models_db.labels())
    wf     = models_db.get_workflow(name)
    wf_str = f"{comfy_client.WORKFLOW_ICONS[wf]} {comfy_client.WORKFLOW_LABELS[wf]}"
    await state.update_data(viewing_model=name)
    deps_line = ""
    if wf == "flux":
        missing = await comfy_client.check_flux_deps()
        if missing:
            deps_line = "\n\n⚠️ <b>Відсутні файли для FLUX:</b>\n" + "\n".join(f"• <code>{m}</code>" for m in missing)
    elif wf == "hidream":
        missing = await comfy_client.check_hidream_deps()
        if missing:
            deps_line = "\n\n⚠️ <b>Відсутні файли для HiDream:</b>\n" + "\n".join(f"• <code>{m}</code>" for m in missing)
    text = f"🤖 <b>{label}</b>\n<code>{name}</code>\n\nWorkflow: {wf_str}{deps_line}"
    await _nav(call, text, parse_mode="HTML", reply_markup=kb_model_detail(name))
    await call.answer()

@dp.callback_query(ModelsCB.filter(F.action == "add"))
async def cb_mdl_add(call: CallbackQuery, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await call.answer()
    await _nav(call, "🔄 Отримую список моделей з ComfyUI...", reply_markup=None)
    available = await comfy_client.fetch_all_models()
    existing  = models_db.all_models()
    if available:
        new_count = sum(1 for m in available if m not in existing)
        text = (
            f"➕ <b>Додати модель</b>\n\n"
            f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
            f"Нових (не в боті): <b>{new_count}</b>\n\n"
            f"✅ — вже додана   ➕ — не додана"
        )
        await call.message.edit_text(text, parse_mode="HTML",
                                     reply_markup=kb_comfy_models(available, existing))
    else:
        await state.set_state(ModelsState.waiting_add)
        await call.message.edit_text(
            "⚠️ ComfyUI недоступний або список порожній.\n\n"
            "➕ <b>Введіть назву файлу вручну:</b>\n<code>dreamshaper_8.safetensors</code>",
            parse_mode="HTML", reply_markup=kb_cancel_to_models())

@dp.callback_query(ModelsCB.filter(F.action == "add_pick"))
async def cb_mdl_add_pick(call: CallbackQuery, callback_data: ModelsCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name = _resolve_model_id(callback_data.value)
    existing = models_db.all_models()
    if name in existing:
        await call.answer(f"Вже є: {_short(name)}", show_alert=False)
        return
    models_db.add_model(name)
    models_db.set_workflow(name, comfy_client.detect_workflow_hint(name))
    await call.answer(f"✅ {_short(name)} додано", show_alert=False)  # new model has no display_name yet
    # refresh the picker with updated list
    available = await comfy_client.fetch_all_models()
    existing  = models_db.all_models()
    new_count = sum(1 for m in available if m not in existing)
    text = (
        f"➕ <b>Додати модель</b>\n\n"
        f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
        f"Нових (не в боті): <b>{new_count}</b>\n\n"
        f"✅ — вже додана   ➕ — не додана"
    )
    await _nav(call, text, parse_mode="HTML",
               reply_markup=kb_comfy_models(available, existing))

@dp.callback_query(ModelsCB.filter(F.action == "add_manual"))
async def cb_mdl_add_manual(call: CallbackQuery, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await state.set_state(ModelsState.waiting_add)
    await _nav(call,
               "✏️ <b>Введіть назву файлу моделі вручну:</b>\n<code>dreamshaper_8.safetensors</code>",
               parse_mode="HTML", reply_markup=kb_cancel_to_models())
    await call.answer()

@dp.message(ModelsState.waiting_add, F.text)
async def handle_model_add(message: Message, state: FSMContext) -> None:
    await state.clear()
    name = message.text.strip()
    if models_db.add_model(name):
        await message.answer(f"✅ Модель <code>{name}</code> додана.",
                             parse_mode="HTML", reply_markup=kb_models_admin())
    else:
        await message.answer(f"⚠️ Модель <code>{name}</code> вже існує.",
                             parse_mode="HTML", reply_markup=kb_models_admin())

@dp.callback_query(ModelsCB.filter(F.action == "edit"))
async def cb_mdl_edit(call: CallbackQuery, callback_data: ModelsCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name  = _resolve_model_id(callback_data.value)
    lbls  = models_db.labels()
    label = _label(name, lbls)
    await state.set_state(ModelsState.waiting_edit)
    await state.update_data(edit_model=name)
    await _nav(call,
               f"✏️ <b>Назва для відображення</b>\n\n"
               f"Файл: <code>{name}</code>\n"
               f"Зараз: <b>{label}</b>\n\n"
               f"Введіть нову назву для відображення (не впливає на генерацію):",
               parse_mode="HTML", reply_markup=kb_cancel_to_models())
    await call.answer()

@dp.message(ModelsState.waiting_edit, F.text)
async def handle_model_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    name = data.get("edit_model", "")
    await state.clear()
    display = message.text.strip()
    if models_db.set_display_name(name, display):
        await message.answer(f"✅ Назва встановлена: <b>{display}</b>\n<code>{name}</code>",
                             parse_mode="HTML", reply_markup=kb_models_admin())
    else:
        await message.answer("⚠️ Не вдалося оновити назву.",
                             parse_mode="HTML", reply_markup=kb_models_admin())

@dp.callback_query(ModelsCB.filter(F.action == "workflow_pick"))
async def cb_mdl_workflow_pick(call: CallbackQuery, callback_data: ModelsCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name = _resolve_model_id(callback_data.value)
    await state.update_data(viewing_model=name)
    label = _label(name, models_db.labels())
    current = models_db.get_workflow(name)
    await _nav(call,
               f"🔄 <b>Тип workflow для моделі</b>\n\n<b>{label}</b>\n<code>{name}</code>",
               parse_mode="HTML", reply_markup=kb_workflow_picker(current))
    await call.answer()


@dp.callback_query(WorkflowCB.filter())
async def cb_mdl_set_workflow(call: CallbackQuery, callback_data: WorkflowCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    data = await state.get_data()
    name = data.get("viewing_model", "")
    if not name:
        await call.answer("⚠️ Відкрийте модель заново", show_alert=True)
        return
    models_db.set_workflow(name, callback_data.wtype)
    icon  = comfy_client.WORKFLOW_ICONS[callback_data.wtype]
    label_wf = comfy_client.WORKFLOW_LABELS[callback_data.wtype]
    await call.answer(f"✅ {icon} {label_wf}", show_alert=False)
    # show updated detail
    lbl    = _label(name, models_db.labels())
    wf_str = f"{icon} {label_wf}"
    deps_line = ""
    if callback_data.wtype == "flux":
        missing = await comfy_client.check_flux_deps()
        if missing:
            deps_line = "\n\n⚠️ <b>Відсутні файли для FLUX:</b>\n" + "\n".join(f"• <code>{m}</code>" for m in missing)
    elif callback_data.wtype == "hidream":
        missing = await comfy_client.check_hidream_deps()
        if missing:
            deps_line = "\n\n⚠️ <b>Відсутні файли для HiDream:</b>\n" + "\n".join(f"• <code>{m}</code>" for m in missing)
    text = f"🤖 <b>{lbl}</b>\n<code>{name}</code>\n\nWorkflow: {wf_str}{deps_line}"
    await _nav(call, text, parse_mode="HTML", reply_markup=kb_model_detail(name))


@dp.callback_query(ModelsCB.filter(F.action == "delete"))
async def cb_mdl_delete(call: CallbackQuery, callback_data: ModelsCB) -> None:
    name  = _resolve_model_id(callback_data.value)
    label = _label(name, models_db.labels())
    await _nav(call, f"❗ Видалити модель <b>{label}</b>?\n<code>{name}</code>",
               parse_mode="HTML", reply_markup=kb_model_confirm_delete(name))
    await call.answer()

@dp.callback_query(ModelsCB.filter(F.action == "delete_ok"))
async def cb_mdl_delete_ok(call: CallbackQuery, callback_data: ModelsCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    if len(models_db.all_models()) <= 1:
        await call.answer("❌ Не можна видалити єдину модель.", show_alert=True)
        return
    name  = _resolve_model_id(callback_data.value)
    label = _label(name, models_db.labels())
    models_db.remove_model(name)
    await call.answer(f"✅ {label} видалено.", show_alert=True)
    await _nav(call, _models_text(), parse_mode="HTML", reply_markup=kb_models_admin())

# ── управління LoRA (адмін) ──────────────────────────────────────────────

@dp.callback_query(F.data == "menu:loras")
async def cb_menu_loras(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await _nav(call, _loras_text(), parse_mode="HTML", reply_markup=kb_loras_admin())
    await call.answer()


@dp.callback_query(LorasCB.filter(F.action == "view"))
async def cb_lora_view(call: CallbackQuery, callback_data: LorasCB) -> None:
    name    = callback_data.value
    lbls    = loras_db.labels()
    label   = _label(name, lbls)
    trigger = loras_db.get_trigger(name)
    trig_str = f"\n🔑 Тригер: <code>{trigger}</code>" if trigger else "\n🔑 Тригер: <i>не встановлено</i>"
    await _nav(call, f"🎭 <b>{label}</b>\n<code>{name}</code>{trig_str}",
               parse_mode="HTML", reply_markup=kb_lora_detail(name))
    await call.answer()


@dp.callback_query(LorasCB.filter(F.action == "add"))
async def cb_lora_add(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await call.answer()
    await _nav(call, "🔄 Отримую список LoRA з ComfyUI...", reply_markup=None)
    available = await comfy_client.fetch_loras()
    existing  = loras_db.all_loras()
    if available:
        new_count = sum(1 for m in available if m not in existing)
        text = (
            f"➕ <b>Додати LoRA</b>\n\n"
            f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
            f"Нових (не в боті): <b>{new_count}</b>\n\n"
            f"✅ — вже додана   ➕ — не додана"
        )
        await call.message.edit_text(text, parse_mode="HTML",
                                     reply_markup=kb_comfy_loras(available, existing))
    else:
        await call.message.edit_text(
            "⚠️ ComfyUI недоступний або LoRA файлів не знайдено.\n\n"
            "Покладіть <code>.safetensors</code> файли у папку <code>ComfyUI/models/loras/</code>",
            parse_mode="HTML", reply_markup=kb_loras_admin())


@dp.callback_query(LorasCB.filter(F.action == "add_pick"))
async def cb_lora_add_pick(call: CallbackQuery, callback_data: LorasCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name     = callback_data.value
    existing = loras_db.all_loras()
    if name in existing:
        loras_db.remove_lora(name)
        await call.answer(f"✅ {_short(name)} видалено зі списку", show_alert=False)
    else:
        loras_db.add_lora(name)
        await call.answer(f"✅ {_short(name)} додано", show_alert=False)
    available = await comfy_client.fetch_loras()
    existing  = loras_db.all_loras()
    new_count = sum(1 for m in available if m not in existing)
    text = (
        f"➕ <b>Додати LoRA</b>\n\n"
        f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
        f"Нових (не в боті): <b>{new_count}</b>\n\n"
        f"✅ — вже додана   ➕ — не додана"
    )
    await _nav(call, text, parse_mode="HTML",
               reply_markup=kb_comfy_loras(available, existing))


@dp.callback_query(LorasCB.filter(F.action == "edit"))
async def cb_lora_edit(call: CallbackQuery, callback_data: LorasCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name  = callback_data.value
    lbls  = loras_db.labels()
    label = _label(name, lbls)
    await state.set_state(LorasState.waiting_edit)
    await state.update_data(edit_lora=name)
    await _nav(call,
               f"✏️ <b>Назва для відображення</b>\n\n"
               f"Файл: <code>{name}</code>\n"
               f"Зараз: <b>{label}</b>\n\n"
               f"Введіть нову назву для відображення (не впливає на генерацію):",
               parse_mode="HTML", reply_markup=kb_cancel_to_loras())
    await call.answer()


@dp.message(LorasState.waiting_edit, F.text)
async def handle_lora_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    name = data.get("edit_lora", "")
    await state.clear()
    display = message.text.strip()
    if loras_db.set_display_name(name, display):
        await message.answer(f"✅ Назва встановлена: <b>{display}</b>\n<code>{name}</code>",
                             parse_mode="HTML", reply_markup=kb_loras_admin())
    else:
        await message.answer("⚠️ Не вдалося оновити назву.",
                             parse_mode="HTML", reply_markup=kb_loras_admin())


@dp.callback_query(LorasCB.filter(F.action == "trigger"))
async def cb_lora_trigger(call: CallbackQuery, callback_data: LorasCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name    = callback_data.value
    lbls    = loras_db.labels()
    label   = _label(name, lbls)
    current = loras_db.get_trigger(name)
    await state.set_state(LorasState.waiting_trigger)
    await state.update_data(trigger_lora=name)
    text = (
        f"🔑 <b>Тригер слово для LoRA</b>\n\n"
        f"LoRA: <b>{label}</b>\n"
        f"Файл: <code>{name}</code>\n\n"
    )
    if current:
        text += f"Поточний тригер: <code>{current}</code>\n\n"
    else:
        text += "Тригер не встановлено.\n\n"
    text += (
        "Введіть тригер-слово або фразу.\n"
        "Воно автоматично додаватиметься до кожного промпту коли ця LoRA активна.\n\n"
        "<i>Наприклад: weic, Hyperrealism style, ohwx man</i>"
    )
    b = InlineKeyboardBuilder()
    if current:
        b.button(text="🗑 Видалити тригер", callback_data=LorasCB(action="trigger_clear", value=name).pack())
    b.button(text="🔙 Скасувати", callback_data=LorasCB(action="view", value=name).pack())
    b.adjust(1)
    await _nav(call, text, parse_mode="HTML", reply_markup=b.as_markup())
    await call.answer()


@dp.callback_query(LorasCB.filter(F.action == "trigger_clear"))
async def cb_lora_trigger_clear(call: CallbackQuery, callback_data: LorasCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await state.clear()
    name = callback_data.value
    loras_db.set_trigger(name, "")
    await call.answer("✅ Тригер видалено", show_alert=False)
    lbls  = loras_db.labels()
    label = _label(name, lbls)
    await _nav(call, f"🎭 <b>{label}</b>\n<code>{name}</code>\n\n🔑 Тригер: <i>не встановлено</i>",
               parse_mode="HTML", reply_markup=kb_lora_detail(name))


@dp.message(LorasState.waiting_trigger, F.text)
async def handle_lora_trigger(message: Message, state: FSMContext) -> None:
    data    = await state.get_data()
    name    = data.get("trigger_lora", "")
    await state.clear()
    trigger = message.text.strip()
    if loras_db.set_trigger(name, trigger):
        lbls  = loras_db.labels()
        label = _label(name, lbls)
        await message.answer(
            f"✅ Тригер збережено:\n"
            f"LoRA: <b>{label}</b>\n"
            f"Тригер: <code>{trigger}</code>",
            parse_mode="HTML", reply_markup=kb_loras_admin())
    else:
        await message.answer("⚠️ Не вдалося зберегти тригер.",
                             parse_mode="HTML", reply_markup=kb_loras_admin())


@dp.callback_query(LorasCB.filter(F.action == "delete"))
async def cb_lora_delete(call: CallbackQuery, callback_data: LorasCB) -> None:
    name = callback_data.value
    await _nav(call, f"❗ Видалити LoRA <b>{_label(name, loras_db.labels())}</b> зі списку бота?\n<code>{name}</code>",
               parse_mode="HTML", reply_markup=kb_lora_confirm_delete(name))
    await call.answer()


@dp.callback_query(LorasCB.filter(F.action == "delete_ok"))
async def cb_lora_delete_ok(call: CallbackQuery, callback_data: LorasCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    label = _label(callback_data.value, loras_db.labels())
    loras_db.remove_lora(callback_data.value)
    await call.answer(f"✅ {label} видалено.", show_alert=True)
    await _nav(call, _loras_text(), parse_mode="HTML", reply_markup=kb_loras_admin())


# ── управління Upscale моделями (адмін) ─────────────────────────────────

@dp.callback_query(F.data == "menu:upscale_models")
async def cb_menu_upscale_models(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await _nav(call, _upscale_models_text(), parse_mode="HTML", reply_markup=kb_upscale_models_admin())
    await call.answer()


@dp.callback_query(UpscaleMdlCB.filter(F.action == "view"))
async def cb_upscale_mdl_view(call: CallbackQuery, callback_data: UpscaleMdlCB) -> None:
    name  = callback_data.value
    lbls  = upscale_models_db.labels()
    label = _label(name, lbls)
    await _nav(call, f"🔍 <b>{label}</b>\n<code>{name}</code>",
               parse_mode="HTML", reply_markup=kb_upscale_model_detail(name))
    await call.answer()


@dp.callback_query(UpscaleMdlCB.filter(F.action == "add"))
async def cb_upscale_mdl_add(call: CallbackQuery, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await call.answer()
    await _nav(call, "🔄 Отримую список Upscale моделей з ComfyUI...", reply_markup=None)
    available = await comfy_client.fetch_upscale_models()
    existing  = upscale_models_db.all_upscale_models()
    if available:
        await state.update_data(comfy_upscale_list=available)
        new_count = sum(1 for m in available if m not in existing)
        text = (
            f"➕ <b>Додати Upscale модель</b>\n\n"
            f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
            f"Нових (не в боті): <b>{new_count}</b>\n\n"
            f"✅ — вже додана   ➕ — не додана"
        )
        await call.message.edit_text(text, parse_mode="HTML",
                                     reply_markup=kb_comfy_upscale_models(available, existing))
    else:
        await call.message.edit_text(
            "⚠️ ComfyUI недоступний або Upscale моделей не знайдено.\n\n"
            "Покладіть <code>.pth</code> файли у папку <code>ComfyUI/models/upscale_models/</code>",
            parse_mode="HTML", reply_markup=kb_upscale_models_admin())


@dp.callback_query(UpscaleMdlCB.filter(F.action == "add_pick"))
async def cb_upscale_mdl_add_pick(call: CallbackQuery, callback_data: UpscaleMdlCB,
                                   state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    data      = await state.get_data()
    available = data.get("comfy_upscale_list", [])
    try:
        idx  = int(callback_data.value)
        name = available[idx]
    except (TypeError, ValueError, IndexError):
        await call.answer("⚠️ Відкрийте список заново", show_alert=True)
        return
    existing = upscale_models_db.all_upscale_models()
    if name in existing:
        upscale_models_db.remove_upscale_model(name)
        await call.answer(f"✅ {_short(name)} видалено зі списку", show_alert=False)
    else:
        upscale_models_db.add_upscale_model(name)
        await call.answer(f"✅ {_short(name)} додано", show_alert=False)
    existing  = upscale_models_db.all_upscale_models()
    new_count = sum(1 for m in available if m not in existing)
    text = (
        f"➕ <b>Додати Upscale модель</b>\n\n"
        f"Знайдено у ComfyUI: <b>{len(available)}</b>\n"
        f"Нових (не в боті): <b>{new_count}</b>\n\n"
        f"✅ — вже додана   ➕ — не додана"
    )
    await _nav(call, text, parse_mode="HTML",
               reply_markup=kb_comfy_upscale_models(available, existing))


@dp.callback_query(UpscaleMdlCB.filter(F.action == "edit"))
async def cb_upscale_mdl_edit(call: CallbackQuery, callback_data: UpscaleMdlCB, state: FSMContext) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    name  = callback_data.value
    lbls  = upscale_models_db.labels()
    label = _label(name, lbls)
    await state.set_state(UpscaleMdlState.waiting_edit)
    await state.update_data(edit_upscale_mdl=name)
    await _nav(call,
               f"✏️ <b>Назва для відображення</b>\n\n"
               f"Файл: <code>{name}</code>\n"
               f"Зараз: <b>{label}</b>\n\n"
               f"Введіть нову назву для відображення (не впливає на генерацію):",
               parse_mode="HTML", reply_markup=kb_cancel_to_upscale_models())
    await call.answer()


@dp.message(UpscaleMdlState.waiting_edit, F.text)
async def handle_upscale_mdl_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    name = data.get("edit_upscale_mdl", "")
    await state.clear()
    display = message.text.strip()
    if upscale_models_db.set_display_name(name, display):
        await message.answer(f"✅ Назва встановлена: <b>{display}</b>\n<code>{name}</code>",
                             parse_mode="HTML", reply_markup=kb_upscale_models_admin())
    else:
        await message.answer("⚠️ Не вдалося оновити назву.",
                             parse_mode="HTML", reply_markup=kb_upscale_models_admin())


@dp.callback_query(UpscaleMdlCB.filter(F.action == "delete"))
async def cb_upscale_mdl_delete(call: CallbackQuery, callback_data: UpscaleMdlCB) -> None:
    name  = callback_data.value
    label = _label(name, upscale_models_db.labels())
    await _nav(call, f"❗ Видалити Upscale модель <b>{label}</b> зі списку?\n<code>{name}</code>",
               parse_mode="HTML", reply_markup=kb_upscale_model_confirm_delete(name))
    await call.answer()


@dp.callback_query(UpscaleMdlCB.filter(F.action == "delete_ok"))
async def cb_upscale_mdl_delete_ok(call: CallbackQuery, callback_data: UpscaleMdlCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    label = _label(callback_data.value, upscale_models_db.labels())
    upscale_models_db.remove_upscale_model(callback_data.value)
    await call.answer(f"✅ {label} видалено.", show_alert=True)
    await _nav(call, _upscale_models_text(), parse_mode="HTML", reply_markup=kb_upscale_models_admin())


# ── історія генерацій ────────────────────────────────────────────────────

def _hist_entries(uid: int, viewer_id: int) -> list[dict]:
    return hist.get_entries(None if uid < 0 else (viewer_id if uid == 0 else uid))

async def _hist_open(target_msg, viewer_id: int, uid: int, idx: int, state=None) -> None:
    """Send (or replace) history photo message."""
    if state:
        await state.clear()
    entries = _hist_entries(uid, viewer_id)
    if not entries:
        label = "Ваша" if uid == 0 or uid == viewer_id else "Повна" if uid < 0 else "Цього користувача"
        b = InlineKeyboardBuilder()
        b.button(text="🔙 Назад",
                 callback_data=HistoryCB(action="back", uid=uid).pack())
        try:
            await target_msg.edit_text(
                f"📜 <b>{label} історія порожня</b>",
                parse_mode="HTML", reply_markup=b.as_markup())
        except TelegramBadRequest:
            await target_msg.answer(
                f"📜 <b>{label} історія порожня</b>",
                parse_mode="HTML", reply_markup=b.as_markup())
        return
    idx = max(0, min(idx, len(entries) - 1))
    entry = entries[idx]
    try:
        await target_msg.delete()
    except TelegramBadRequest:
        pass
    await target_msg.answer_photo(
        FSInputFile(entry["file_path"]),
        caption=_hist_caption(entry, idx + 1, len(entries), uid if uid != 0 else viewer_id, viewer_id),
        parse_mode="HTML",
        reply_markup=kb_hist_nav(idx, len(entries), uid if uid != 0 else viewer_id),
    )

@dp.callback_query(HistoryCB.filter(F.action == "show"))
async def cb_hist_show(call: CallbackQuery, callback_data: HistoryCB, state: FSMContext) -> None:
    allowed, admin = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    uid = callback_data.uid
    if uid > 0 and uid != call.from_user.id and not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    if uid < 0 and not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await call.answer()
    await _hist_open(call.message, call.from_user.id, uid, 0, state)

@dp.callback_query(HistoryCB.filter(F.action == "nav"))
async def cb_hist_nav_handler(call: CallbackQuery, callback_data: HistoryCB) -> None:
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = max(0, min(callback_data.idx, len(entries) - 1))
    if not entries:
        await call.answer()
        return
    entry = entries[idx]
    await call.answer()
    try:
        await call.message.edit_media(
            InputMediaPhoto(
                media=FSInputFile(entry["file_path"]),
                caption=_hist_caption(entry, idx + 1, len(entries), uid if uid != 0 else call.from_user.id, call.from_user.id),
                parse_mode="HTML",
            ),
            reply_markup=kb_hist_nav(idx, len(entries), uid if uid != 0 else call.from_user.id),
        )
    except TelegramBadRequest:
        pass

@dp.callback_query(HistoryCB.filter(F.action == "noop"))
async def cb_hist_noop(call: CallbackQuery) -> None:
    await call.answer()

@dp.callback_query(HistoryCB.filter(F.action == "regen"))
async def cb_hist_regen(call: CallbackQuery, callback_data: HistoryCB) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = callback_data.idx
    if not entries or idx >= len(entries):
        await call.answer("Запис не знайдено.", show_alert=True)
        return
    entry          = entries[idx]
    prompt_preview = entry["prompt"][:200] + ("…" if len(entry["prompt"]) > 200 else "")
    b = InlineKeyboardBuilder()
    b.button(text="⚡ З поточними налаштуваннями",
             callback_data=HistoryCB(action="regen_now", idx=idx, uid=uid).pack())
    b.button(text="⚙️ Змінити налаштування",
             callback_data=HistoryCB(action="regen_cfg", idx=idx, uid=uid).pack())
    b.button(text="🔙 Назад",
             callback_data=HistoryCB(action="nav", idx=idx, uid=uid).pack())
    b.adjust(1)
    try:
        await call.message.edit_caption(
            f"💬 {prompt_preview}\n\n<b>Оберіть спосіб перегенерації:</b>",
            parse_mode="HTML", reply_markup=b.as_markup(),
        )
    except TelegramBadRequest:
        pass
    await call.answer()


@dp.callback_query(HistoryCB.filter(F.action == "regen_now"))
async def cb_hist_regen_now(call: CallbackQuery, callback_data: HistoryCB, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = callback_data.idx
    if not entries or idx >= len(entries):
        await call.answer("Запис не знайдено.", show_alert=True)
        return
    entry = entries[idx]
    await call.answer()
    await state.clear()
    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass
    await _do_generate(call.message, entry["prompt"], from_user=call.from_user)


@dp.callback_query(HistoryCB.filter(F.action == "regen_cfg"))
async def cb_hist_regen_cfg(call: CallbackQuery, callback_data: HistoryCB, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = callback_data.idx
    if not entries or idx >= len(entries):
        await call.answer("Запис не знайдено.", show_alert=True)
        return
    entry   = entries[idx]
    user_gs = db.get_gen_settings(call.from_user.id)
    temp = {
        "mode":            entry.get("mode", "text2img"),
        "checkpoint":      entry.get("checkpoint") or config.CHECKPOINT,
        "width":           entry.get("width")       or config.IMAGE_WIDTH,
        "height":          entry.get("height")      or config.IMAGE_HEIGHT,
        "steps":           user_gs.get("steps")     or config.STEPS,
        "cfg":             user_gs.get("cfg")        or config.CFG_SCALE,
        "sampler":         user_gs.get("sampler")   or "euler",
        "denoise":         user_gs.get("denoise")   or 0.75,
        "batch_size":      int(user_gs.get("batch_size", 1)),
        "negative_prompt": user_gs.get("negative_prompt"),
        "loras_active":    user_gs.get("loras_active"),
        "hires_fix":       user_gs.get("hires_fix"),
        "hires_denoise":   user_gs.get("hires_denoise") or 0.45,
        "upscale_model":   user_gs.get("upscale_model"),
    }
    await state.update_data(regen_prompt=entry["prompt"], regen_settings=temp,
                            regen_idx=idx, regen_uid=uid)
    await call.answer()
    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass
    await call.message.answer(_rgs_text(temp), parse_mode="HTML", reply_markup=kb_rgs(temp))


# ── regen-config callbacks ────────────────────────────────────────────────

async def _rgs_refresh(call: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    s    = data.get("regen_settings", {})
    try:
        await call.message.edit_text(_rgs_text(s), parse_mode="HTML", reply_markup=kb_rgs(s))
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e).lower():
            raise


async def _rgs_set(call: CallbackQuery, state: FSMContext, **updates) -> None:
    data = await state.get_data()
    s    = dict(data.get("regen_settings", {}))
    s.update(updates)
    await state.update_data(regen_settings=s)
    await _rgs_refresh(call, state)


@dp.callback_query(RgsCB.filter(F.action == "menu"))
async def cb_rgs_menu(call: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(None)
    await _rgs_refresh(call, state)
    await call.answer()


@dp.callback_query(RgsCB.filter(F.action == "show"))
async def cb_rgs_show(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    data = await state.get_data()
    s    = data.get("regen_settings", {})
    await call.answer()

    if callback_data.value == "mode":
        wf_type = models_db.get_workflow(s.get("checkpoint") or config.CHECKPOINT)
        await _nav(call, "🔄 <b>Оберіть режим генерації:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_mode_picker(s.get("mode", "text2img"), wf_type))
    elif callback_data.value == "model":
        cur = s.get("checkpoint") or config.CHECKPOINT
        await _nav(call, "🤖 <b>Оберіть модель:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_model_picker(cur))
    elif callback_data.value == "size":
        w = s.get("width")  or config.IMAGE_WIDTH
        h = s.get("height") or config.IMAGE_HEIGHT
        await _nav(call, "📐 <b>Оберіть розмір зображення:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_size_picker(w, h))
    elif callback_data.value == "steps":
        await _nav(call, "🎚 <b>Оберіть кількість кроків:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_steps_picker(s.get("steps") or config.STEPS))
    elif callback_data.value == "cfg":
        await _nav(call, "🎯 <b>Оберіть CFG Scale:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_cfg_picker(float(s.get("cfg") or config.CFG_SCALE)))
    elif callback_data.value == "sampler":
        await _nav(call, "🎲 <b>Оберіть семплер:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_sampler_picker(s.get("sampler") or "euler"))
    elif callback_data.value == "denoise":
        await _nav(call, "🎨 <b>Сила варіації (denoise):</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_denoise_picker(float(s.get("denoise") or 0.75)))
    elif callback_data.value == "batch":
        await _nav(call, "🔢 <b>Кількість зображень:</b>",
                   parse_mode="HTML", reply_markup=kb_rgs_batch_picker(int(s.get("batch_size", 1))))
    elif callback_data.value == "loras_info":
        rgs_active = _get_active_loras(s)
        llbls = loras_db.labels()
        if rgs_active:
            lines = ["🎭 <b>Активні LoRA для цієї генерації:</b>\n"]
            for item in rgs_active:
                trigger = loras_db.get_trigger(item["name"])
                trig_str = f"  🔑 <code>{trigger}</code>" if trigger else ""
                lines.append(f"• <b>{_label(item['name'], llbls)}</b>  вага: {item.get('strength', 0.8)}{trig_str}")
            text = "\n".join(lines)
            text += "\n\n<i>Змінити LoRA можна у 🎛 Налаштуваннях генерації.</i>"
        else:
            text = "🎭 <b>LoRA не активовані</b>\n\n<i>Змінити можна у 🎛 Налаштуваннях генерації.</i>"
        b = InlineKeyboardBuilder()
        b.button(text="🔙 Назад", callback_data=RgsCB(action="menu").pack())
        await _nav(call, text, parse_mode="HTML", reply_markup=b.as_markup())

    elif callback_data.value == "neg":
        neg  = s.get("negative_prompt")
        text = "📝 <b>Негативний промпт</b>\n\n"
        text += f"Поточний:\n<code>{neg}</code>\n\n" if neg else "Зараз використовується стандартний.\n\n"
        text += "Надішліть новий негативний промпт:"
        await state.set_state(RegenConfigState.neg_prompt)
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_rgs_neg(bool(neg)))

    elif callback_data.value == "hires":
        new_val = not bool(s.get("hires_fix"))
        await call.answer(f"🔍 HiRes Fix: {'ввімкнений' if new_val else 'вимкнений'}")
        await _rgs_set(call, state, hires_fix=new_val if new_val else None)
        return

    elif callback_data.value == "hires_denoise":
        cur = float(s.get("hires_denoise") or 0.45)
        await _nav(call,
                   "🎚 <b>HiRes denoise — сила другого проходу</b>\n\n"
                   "<i>0.3 — м'яке уточнення, 0.45 — стандарт, 0.6 — сильна зміна</i>",
                   parse_mode="HTML", reply_markup=kb_rgs_hires_denoise_picker(cur))

    elif callback_data.value == "upscale_model":
        cur       = s.get("upscale_model")
        available = upscale_models_db.all_upscale_models()
        await state.update_data(regen_upscale_models_list=available)
        text = "🖼 <b>Upscale модель</b>\n\n"
        text += f"Доступно: <b>{len(available)}</b> моделей." if available else "⚠️ Список порожній — буде bilinear."
        await _nav(call, text, parse_mode="HTML", reply_markup=kb_rgs_upscale_model_picker(cur, available))


@dp.callback_query(RgsCB.filter(F.action == "set_mode"))
async def cb_rgs_set_mode(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Режим: {callback_data.value}")
    await _rgs_set(call, state, mode=callback_data.value)

@dp.callback_query(RgsCB.filter(F.action == "set_model"))
async def cb_rgs_set_model(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    name = _resolve_model_id(callback_data.value)
    await call.answer(f"✅ Модель: {_label(name, models_db.labels())}")
    kwargs = {"checkpoint": name}
    if models_db.get_workflow(name) in _NO_IMG2IMG_WF:
        kwargs["mode"] = "text2img"
    await _rgs_set(call, state, **kwargs)

@dp.callback_query(RgsCB.filter(F.action == "set_size"))
async def cb_rgs_set_size(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    w, h = map(int, callback_data.value.split("×"))
    await call.answer(f"✅ Розмір {w}×{h}")
    await _rgs_set(call, state, width=w, height=h)

@dp.callback_query(RgsCB.filter(F.action == "set_steps"))
async def cb_rgs_set_steps(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Кроки: {callback_data.value}")
    await _rgs_set(call, state, steps=int(callback_data.value))

@dp.callback_query(RgsCB.filter(F.action == "set_cfg"))
async def cb_rgs_set_cfg(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ CFG: {callback_data.value}")
    await _rgs_set(call, state, cfg=float(callback_data.value))

@dp.callback_query(RgsCB.filter(F.action == "set_sampler"))
async def cb_rgs_set_sampler(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Семплер: {callback_data.value}")
    await _rgs_set(call, state, sampler=callback_data.value)

@dp.callback_query(RgsCB.filter(F.action == "set_denoise"))
async def cb_rgs_set_denoise(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Варіація: {callback_data.value}")
    await _rgs_set(call, state, denoise=float(callback_data.value))

@dp.callback_query(RgsCB.filter(F.action == "set_batch"))
async def cb_rgs_set_batch(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Кількість: {callback_data.value}")
    await _rgs_set(call, state, batch_size=int(callback_data.value))

@dp.callback_query(RgsCB.filter(F.action == "neg_clear"))
async def cb_rgs_neg_clear(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer("✅ Негативний промпт скинуто")
    await _rgs_set(call, state, negative_prompt=None)

@dp.callback_query(RgsCB.filter(F.action == "set_lora"))
async def cb_rgs_set_lora(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    if callback_data.value == "__none__":
        await call.answer("✅ LoRA вимкнено")
        await _rgs_set(call, state, lora=None)
    else:
        await call.answer(f"✅ LoRA: {_label(callback_data.value, loras_db.labels())}", show_alert=False)
        await _rgs_set(call, state, lora=callback_data.value)

@dp.callback_query(RgsCB.filter(F.action == "set_lora_strength"))
async def cb_rgs_set_lora_strength(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ Сила LoRA: {callback_data.value}", show_alert=False)
    await _rgs_set(call, state, lora_strength=float(callback_data.value))

@dp.callback_query(RgsCB.filter(F.action == "set_hires_denoise"))
async def cb_rgs_set_hires_denoise(call: CallbackQuery, callback_data: RgsCB, state: FSMContext) -> None:
    await call.answer(f"✅ HiRes denoise: {callback_data.value}")
    await _rgs_set(call, state, hires_denoise=float(callback_data.value))

@dp.callback_query(RgsUpscaleModelCB.filter())
async def cb_rgs_set_upscale_model(call: CallbackQuery, callback_data: RgsUpscaleModelCB,
                                    state: FSMContext) -> None:
    if callback_data.idx == -1:
        await call.answer("✅ Upscale: авто (bilinear)")
        await _rgs_set(call, state, upscale_model=None)
    else:
        data      = await state.get_data()
        available = data.get("regen_upscale_models_list", [])
        if callback_data.idx >= len(available):
            await call.answer("⚠️ Відкрийте список заново", show_alert=True)
            return
        model = available[callback_data.idx]
        await call.answer(f"✅ {_label(model, upscale_models_db.labels())}", show_alert=False)
        await _rgs_set(call, state, upscale_model=model)

@dp.callback_query(RgsCB.filter(F.action == "go"))
async def cb_rgs_go(call: CallbackQuery, state: FSMContext) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔", show_alert=True)
        return
    data   = await state.get_data()
    prompt = data.get("regen_prompt", "")
    temp   = data.get("regen_settings", {})
    await call.answer()
    await state.clear()
    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass
    await _do_generate(call.message, prompt, user_settings=temp, from_user=call.from_user)

@dp.callback_query(RgsCB.filter(F.action == "cancel"))
async def cb_rgs_cancel(call: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    idx  = data.get("regen_idx", 0)
    uid  = data.get("regen_uid", 0)
    await state.clear()
    await call.answer()
    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass
    await _hist_open(call.message, call.from_user.id, uid, idx)

@dp.message(RegenConfigState.neg_prompt, F.text)
async def handle_regen_neg_prompt(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    s    = dict(data.get("regen_settings", {}))
    s["negative_prompt"] = message.text.strip()
    await state.update_data(regen_settings=s)
    await state.set_state(None)
    await message.answer(_rgs_text(s), parse_mode="HTML", reply_markup=kb_rgs(s))

@dp.callback_query(HistoryCB.filter(F.action == "del"))
async def cb_hist_del(call: CallbackQuery, callback_data: HistoryCB) -> None:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Так",      callback_data=HistoryCB(action="del_ok", idx=callback_data.idx, uid=callback_data.uid).pack())
    b.button(text="❌ Скасувати", callback_data=HistoryCB(action="nav",    idx=callback_data.idx, uid=callback_data.uid).pack())
    b.adjust(2)
    try:
        await call.message.edit_caption(
            "❗ <b>Видалити цей запис з історії?</b>",
            parse_mode="HTML", reply_markup=b.as_markup())
    except TelegramBadRequest:
        pass
    await call.answer()

@dp.callback_query(HistoryCB.filter(F.action == "del_ok"))
async def cb_hist_del_ok(call: CallbackQuery, callback_data: HistoryCB) -> None:
    uid     = callback_data.uid
    entries = _hist_entries(uid, call.from_user.id)
    idx     = callback_data.idx
    if not entries or idx >= len(entries):
        await call.answer("Вже видалено.")
        return
    hist.delete_entry(entries[idx]["id"])
    await call.answer("✅ Видалено")
    entries = _hist_entries(uid, call.from_user.id)
    if not entries:
        _, admin = _ctx(call.from_user)
        try:
            await call.message.delete()
        except TelegramBadRequest:
            pass
        await _hist_send_back(call.message, call.from_user.id, uid, admin)
        return
    new_idx = min(idx, len(entries) - 1)
    entry   = entries[new_idx]
    eff_uid = uid if uid != 0 else call.from_user.id
    try:
        await call.message.edit_media(
            InputMediaPhoto(
                media=FSInputFile(entry["file_path"]),
                caption=_hist_caption(entry, new_idx + 1, len(entries), eff_uid, call.from_user.id),
                parse_mode="HTML",
            ),
            reply_markup=kb_hist_nav(new_idx, len(entries), eff_uid),
        )
    except TelegramBadRequest:
        pass

@dp.callback_query(HistoryCB.filter(F.action == "back"))
async def cb_hist_back(call: CallbackQuery, callback_data: HistoryCB) -> None:
    _, admin = _ctx(call.from_user)
    await call.answer()
    try:
        await call.message.delete()
    except TelegramBadRequest:
        pass
    await _hist_send_back(call.message, call.from_user.id, callback_data.uid, admin)

@dp.callback_query(HistoryCB.filter(F.action == "clearall"))
async def cb_hist_clearall(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    b = InlineKeyboardBuilder()
    b.button(text="✅ Так, видалити все",
             callback_data=HistoryCB(action="clearall_ok", uid=-1).pack())
    b.button(text="❌ Скасувати",
             callback_data="menu:settings")
    b.adjust(2)
    try:
        await call.message.edit_text(
            "❗ <b>Очистити ВСЮ історію генерацій?</b>\n\n"
            "Це видалить усі записи та файли зображень для всіх користувачів.\n"
            "<i>Дія незворотна.</i>",
            parse_mode="HTML", reply_markup=b.as_markup(),
        )
    except TelegramBadRequest:
        pass
    await call.answer()


@dp.callback_query(HistoryCB.filter(F.action == "clearall_ok"))
async def cb_hist_clearall_ok(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    count = hist.clear_all()
    await call.answer(f"✅ Видалено {count} записів.", show_alert=True)
    await _nav(call, "⚙️ <b>Налаштування</b>", parse_mode="HTML", reply_markup=kb_settings())


async def _hist_send_back(msg, viewer_id: int, uid: int, admin: bool) -> None:
    if uid == 0 or uid == viewer_id:
        await msg.answer("Оберіть дію:", reply_markup=kb_main(admin))
    elif uid < 0:
        await msg.answer("⚙️ <b>Налаштування</b>", parse_mode="HTML", reply_markup=kb_settings())
    else:
        u = db.find(telegram_id=uid)
        if u:
            await msg.answer(_user_detail_text(u), parse_mode="HTML",
                             reply_markup=kb_user_detail(u["username"], u["role"], uid))
        else:
            await msg.answer("Оберіть дію:", reply_markup=kb_main(admin))

# ── скасування генерації ─────────────────────────────────────────────────

@dp.callback_query(CancelCB.filter())
async def cb_cancel_job(call: CallbackQuery, callback_data: CancelCB) -> None:
    cancelled = await gq.cancel_by_msg(callback_data.msg_id)
    if not cancelled:
        await call.answer("⚠️ Завдання вже виконується або завершено.", show_alert=True)
        return
    await call.answer("✅ Скасовано", show_alert=False)
    try:
        await call.message.edit_text("❌ <b>Скасовується…</b>", parse_mode="HTML", reply_markup=None)
    except TelegramBadRequest:
        pass

# ── статистика ────────────────────────────────────────────────────────────

@dp.callback_query(F.data == "stats:my")
async def cb_stats_my(call: CallbackQuery) -> None:
    allowed, _ = _ctx(call.from_user)
    if not allowed:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    count = db.find(telegram_id=call.from_user.id)
    count = (count or {}).get("gen_count", 0)
    b = InlineKeyboardBuilder()
    b.button(text="🔙 Назад", callback_data="menu:main")
    await _nav(call,
               f"📈 <b>Ваша статистика</b>\n\n"
               f"🖼 Згенеровано зображень: <b>{count}</b>",
               parse_mode="HTML", reply_markup=b.as_markup())
    await call.answer()

@dp.callback_query(F.data == "stats:all")
async def cb_stats_all(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    users  = db.all_users()
    total  = sum(u.get("gen_count", 0) for u in users)
    sorted_users = sorted(users, key=lambda u: u.get("gen_count", 0), reverse=True)
    lines = [
        "📊 <b>Статистика генерацій</b>", "",
        f"Всього згенеровано: <b>{total}</b> зображень", "",
    ]
    for u in sorted_users:
        icon = "👑" if u["role"] == "admin" else "👤"
        cnt  = u.get("gen_count", 0)
        lines.append(f"{icon} @{u['username']}: <b>{cnt}</b>")
    b = InlineKeyboardBuilder()
    b.button(text="🔙 Назад", callback_data="menu:settings")
    await _nav(call, "\n".join(lines), parse_mode="HTML", reply_markup=b.as_markup())
    await call.answer()

# ── навігація ─────────────────────────────────────────────────────────────

@dp.callback_query(F.data == "menu:main")
async def cb_menu_main(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    await _nav(call, "Оберіть дію:", reply_markup=kb_main(admin))
    await call.answer()

@dp.callback_query(F.data == "menu:settings")
async def cb_menu_settings(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await _nav(call, "⚙️ <b>Налаштування</b>", parse_mode="HTML", reply_markup=kb_settings())
    await call.answer()

@dp.callback_query(F.data == "menu:users")
async def cb_menu_users(call: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    await _nav(call, _users_text(), parse_mode="HTML", reply_markup=kb_users())
    await call.answer()

# ── користувачі ───────────────────────────────────────────────────────────

def _user_detail_text(u: dict) -> str:
    role_label = "👑 Адміністратор" if u["role"] == "admin" else "👤 Користувач"
    id_line    = f"Telegram ID: <code>{u['id']}</code>" if u.get("id") else "Telegram ID: ще не відомий"
    count      = u.get("gen_count", 0)
    return f"<b>@{u['username']}</b>\nРоль: {role_label}\n{id_line}\n🖼 Згенеровано: <b>{count}</b>"

@dp.callback_query(UserCB.filter(F.action == "view"))
async def cb_user_view(call: CallbackQuery, callback_data: UserCB) -> None:
    u = db.find(username=callback_data.username)
    if not u:
        await call.answer("Користувача не знайдено.", show_alert=True)
        return
    await call.message.edit_text(
        _user_detail_text(u),
        parse_mode="HTML", reply_markup=kb_user_detail(u["username"], u["role"], u.get("id")),
    )
    await call.answer()

@dp.callback_query(UserCB.filter(F.action == "setrole"))
async def cb_user_setrole(call: CallbackQuery, callback_data: UserCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    if callback_data.role == "user" and db.admin_count() <= 1:
        await call.answer("❌ Не можна понизити єдиного адміністратора.", show_alert=True)
        return
    db.set_role(callback_data.username, callback_data.role)
    role_text = "адміністратора" if callback_data.role == "admin" else "користувача"
    await call.answer(f"✅ @{callback_data.username} тепер {role_text}.", show_alert=True)
    u = db.find(username=callback_data.username)
    await call.message.edit_text(
        _user_detail_text(u),
        parse_mode="HTML", reply_markup=kb_user_detail(u["username"], u["role"], u.get("id")),
    )

@dp.callback_query(UserCB.filter(F.action == "delete"))
async def cb_user_delete(call: CallbackQuery, callback_data: UserCB) -> None:
    await call.message.edit_text(
        f"❗ Видалити користувача <b>@{callback_data.username}</b>?",
        parse_mode="HTML", reply_markup=kb_confirm_delete_user(callback_data.username),
    )
    await call.answer()

@dp.callback_query(UserCB.filter(F.action == "delete_ok"))
async def cb_user_delete_ok(call: CallbackQuery, callback_data: UserCB) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ заборонено.", show_alert=True)
        return
    if callback_data.username.lower() == (call.from_user.username or "").lower():
        await call.answer("❌ Не можна видалити самого себе.", show_alert=True)
        return
    u = db.find(username=callback_data.username)
    if u and u["role"] == "admin" and db.admin_count() <= 1:
        await call.answer("❌ Не можна видалити єдиного адміністратора.", show_alert=True)
        return
    db.remove(callback_data.username)
    await call.answer(f"✅ @{callback_data.username} видалено.", show_alert=True)
    await call.message.edit_text(_users_text(), parse_mode="HTML", reply_markup=kb_users())

@dp.callback_query(F.data == "user:add")
async def cb_user_add(call: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AddUserState.waiting_username)
    await _nav(call, "➕ Введіть <b>@username</b> нового користувача:",
               parse_mode="HTML", reply_markup=kb_cancel_to_main())
    await call.answer()

@dp.message(AddUserState.waiting_username, F.text)
async def handle_new_username(message: Message, state: FSMContext) -> None:
    username = message.text.strip().lstrip("@")
    if not username or " " in username:
        await message.answer("⚠️ Введіть коректний @username (без пробілів).")
        return
    if db.find(username=username):
        await message.answer(f"⚠️ Користувач @{username} вже існує.", reply_markup=kb_users())
        await state.clear()
        return
    await state.update_data(new_username=username)
    await state.set_state(AddUserState.waiting_role)
    await message.answer(f"Оберіть роль для <b>@{username}</b>:",
                         parse_mode="HTML", reply_markup=kb_choose_role(username))

@dp.message(AddUserState.waiting_role)
async def handle_role_text_instead_of_button(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    username = data.get("new_username", "")
    await message.answer(f"Будь ласка, оберіть роль для <b>@{username}</b> кнопкою:",
                         parse_mode="HTML", reply_markup=kb_choose_role(username))

@dp.callback_query(AddUserState.waiting_role, F.data.startswith("role:"))
async def handle_new_role(call: CallbackQuery, state: FSMContext) -> None:
    _, role, username = call.data.split(":", 2)
    db.add(username, role)
    await state.clear()
    role_text = "адміністратора" if role == "admin" else "користувача"
    await call.answer(f"✅ @{username} додано як {role_text}.", show_alert=True)
    await call.message.edit_text(_users_text(), parse_mode="HTML", reply_markup=kb_users())

# ── MMORPG game item generation ──────────────────────────────────────────

_game_gen_task: Optional[asyncio.Task] = None


def _game_gen_running() -> bool:
    return _game_gen_task is not None and not _game_gen_task.done()


def _kb_game_stop() -> InlineKeyboardMarkup:
    return (InlineKeyboardBuilder()
            .button(text="⛔ Зупинити генерацію", callback_data="game:stop")
            .as_markup())


@dp.callback_query(F.data == "game:gen")
async def cb_game_gen(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔ Доступ лише для адміністраторів.", show_alert=True)
        return
    await call.answer()

    if _game_gen_running():
        await call.answer("⚙️ Генерація вже запущена.", show_alert=True)
        return

    global _game_gen_task
    _game_gen_task = asyncio.create_task(
        _run_game_gen(call.message, call.from_user)
    )


@dp.callback_query(F.data == "game:stop")
async def cb_game_stop(call: CallbackQuery) -> None:
    _, admin = _ctx(call.from_user)
    if not admin:
        await call.answer("⛔", show_alert=True)
        return
    if _game_gen_running():
        _game_gen_task.cancel()
        await call.answer("⛔ Зупиняю…", show_alert=False)
    else:
        await call.answer("ℹ️ Генерація вже завершена.", show_alert=True)
        await call.message.edit_reply_markup(reply_markup=None)


def _to_webp(png_bytes: bytes, quality: int = 85) -> bytes:
    """Convert image bytes to WebP for smaller uploads."""
    from PIL import Image
    from io import BytesIO as _BytesIO
    img = Image.open(_BytesIO(png_bytes)).convert("RGB")
    out = _BytesIO()
    img.save(out, format="WEBP", quality=quality)
    return out.getvalue()


# ── game-gen helpers ──────────────────────────────────────────────────────

async def _gg_edit(status_ref: list, trigger_msg: Message,
                   text: str, **kwargs) -> None:
    """Edit status message; if deleted — recreate it transparently."""
    try:
        await status_ref[0].edit_text(text, **kwargs)
    except TelegramBadRequest as e:
        msg = str(e).lower()
        if "not found" in msg or "can't be edited" in msg:
            try:
                status_ref[0] = await trigger_msg.answer(text, **kwargs)
            except Exception:
                pass
    except Exception:
        pass


async def _gg_wait_for(
    service_name: str,
    check_fn,                  # async () -> bool
    edit_fn,                   # async (text, **kw) -> None  — may be None for silent waits
    interval: int = 30,
) -> None:
    """
    Block until check_fn() returns True.
    Updates the status message (via edit_fn) every `interval` seconds.
    Raises CancelledError transparently if the task is cancelled while waiting.
    """
    import time as _time
    attempt   = 0
    started   = _time.monotonic()

    while not await check_fn():
        attempt += 1
        elapsed  = int(_time.monotonic() - started)
        mins, s  = divmod(elapsed, 60)
        t_str    = f"{mins}хв {s:02d}с" if mins else f"{s}с"
        log.warning("game_gen: %s unavailable, attempt %d, waiting %ds (elapsed %s)",
                    service_name, attempt, interval, t_str)
        if edit_fn:
            await edit_fn(
                f"⏸ <b>{service_name} недоступний</b>\n\n"
                f"⏳ Чекаю відновлення… спроба <b>{attempt}</b>\n"
                f"Пройшло: <b>{t_str}</b>  •  Перевірка через {interval}с",
                parse_mode="HTML", reply_markup=_kb_game_stop(),
            )
        await asyncio.sleep(interval)

    if attempt > 0:
        log.info("game_gen: %s back online after %d attempts", service_name, attempt)
        if edit_fn:
            await edit_fn(
                f"✅ <b>{service_name} відновлено!</b> Продовжую…",
                parse_mode="HTML", reply_markup=_kb_game_stop(),
            )
            await asyncio.sleep(1)


async def _check_game_api() -> bool:
    """Quick availability check for the game API (does NOT count against rate limits)."""
    try:
        await game_api.fetch_items()
        return True
    except asyncio.CancelledError:
        raise
    except Exception:
        return False


async def _gg_generate(
    prompt: str,
    on_progress,
    user_settings: dict,
    edit_fn,
) -> bytes:
    """
    Generate image with automatic recovery:
    - Waits for ComfyUI if offline before starting
    - If ComfyUI drops mid-generation → waits for it, then retries once
    """
    await _gg_wait_for("ComfyUI", comfy_client.ping, edit_fn)

    for attempt in range(2):          # initial try + 1 recovery
        try:
            return await comfy_client.generate(
                prompt, on_progress=on_progress, user_settings=user_settings,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("game_gen: generate error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0 and not await comfy_client.ping():
                # ComfyUI went down mid-generation — wait and retry
                await _gg_wait_for("ComfyUI", comfy_client.ping, edit_fn)
                continue
            raise


async def _gg_upload(
    item_id:    str,
    webp_bytes: bytes,
    filename:   str,
    edit_fn     = None,
) -> tuple[bool, str]:
    """
    Upload with full recovery and status updates:
    - Transient errors (5xx / connection / 429) → wait for Game API (with status msg), retry
    - Permanent errors (4xx except 429)         → return failure immediately
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            success, result = await game_api.upload_image(item_id, webp_bytes, filename)
            if success:
                return True, result
            is_transient = (
                "HTTP 5" in result
                or "HTTP 429" in result
                or "timed out" in result.lower()
                or "connection" in result.lower()
            )
            if not is_transient:
                log.error("game_gen: upload permanent failure %s: %s", item_id, result)
                return False, result
            log.warning("game_gen: upload transient %s (attempt %d): %s",
                        item_id, attempt, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("game_gen: upload exception %s (attempt %d): %s",
                        item_id, attempt, exc)

        # Wait for Game API to come back — with visible status update
        await _gg_wait_for("Game API", _check_game_api,
                           edit_fn=edit_fn, interval=30)
        await asyncio.sleep(min(attempt * 2, 30))


# ── main background task ──────────────────────────────────────────────────

async def _run_game_gen(trigger_msg: Message, admin_user) -> None:
    """Fetch items → generate variants → show each → upload (parallel with retry)."""
    import time as _time
    from aiogram.types import BufferedInputFile

    status_ref: list = [None]   # mutable holder so _gg_edit can replace it

    async def _edit(text: str, **kw) -> None:
        await _gg_edit(status_ref, trigger_msg, text, **kw)

    # ── wait for ComfyUI (initial check) ─────────────────────────────────
    status_ref[0] = await trigger_msg.answer(
        "🔄 Перевіряю ComfyUI…", reply_markup=_kb_game_stop(),
    )
    await _gg_wait_for("ComfyUI", comfy_client.ping, _edit)

    # ── wait for Game API + fetch items ───────────────────────────────────
    await _edit("🔄 Отримую список предметів…",
                parse_mode="HTML", reply_markup=_kb_game_stop())

    items: list[game_api.GameItem] = []
    while not items:
        await _gg_wait_for("Game API", _check_game_api, _edit)
        try:
            items = await game_api.fetch_items()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("game_gen: fetch_items error: %s", exc)
            items = []
            continue
        if not items:
            await _edit("✅ <b>Всі предмети вже мають зображення!</b>",
                        parse_mode="HTML", reply_markup=kb_settings())
            return

    if not items:
        await _edit("✅ <b>Всі предмети вже мають зображення!</b>",
                    parse_mode="HTML", reply_markup=kb_settings())
        return

    total_items = len(items)
    total_slots = sum(i.slots_remaining for i in items)
    await _edit(
        f"🎮 <b>Генерація предметів MMORPG</b>\n\n"
        f"Предметів: <b>{total_items}</b>  •  Зображень: <b>{total_slots}</b>\n"
        f"Модель: <b>{user_settings_ckpt_label(admin_user.id)}</b>\nПочинаю…",
        parse_mode="HTML", reply_markup=_kb_game_stop(),
    )

    # ── gen settings ─────────────────────────────────────────────────────
    user_settings = dict(db.get_gen_settings(admin_user.id))
    _ckpt = user_settings.get("checkpoint") or config.CHECKPOINT
    user_settings["_workflow_type"] = models_db.get_workflow(_ckpt)
    user_settings["mode"]       = "text2img"
    user_settings["batch_size"] = 1

    ok_count   = 0
    fail_count = 0
    fail_names: list[str] = []
    done_slots = 0

    try:
        for item_idx, item in enumerate(items, 1):
            needed = item.slots_remaining

            for variant in range(1, needed + 1):
                done_slots += 1

                # ── progress header ───────────────────────────────────────
                bar_fill    = int(20 * (done_slots - 1) / total_slots)
                slots_bar   = "▓" * bar_fill + "░" * (20 - bar_fill)
                rarity_tag  = f"  <i>[{item.rarity}]</i>" if item.rarity else ""
                variant_tag = f"  <i>варіант {variant}/{needed}</i>" if needed > 1 else ""
                item_header = (
                    f"🎮 <b>Генерація предметів MMORPG</b>\n\n"
                    f"<code>{slots_bar}</code>  {done_slots - 1}/{total_slots}\n"
                    f"📦 {item_idx}/{total_items}  "
                    f"⚙️ <b>{item.name}</b>{rarity_tag}{variant_tag}"
                )
                await _edit(item_header, parse_mode="HTML",
                            reply_markup=_kb_game_stop())

                # ── on_progress ───────────────────────────────────────────
                _last_upd = [0.0]

                async def on_progress(step: int, total_steps: int,
                                      _hdr: str = item_header) -> None:
                    now = _time.monotonic()
                    if now - _last_upd[0] < 1.0:
                        return
                    _last_upd[0] = now
                    step_bar = comfy_client.progress_bar(step, total_steps)
                    await _edit(
                        f"{_hdr}\n\n<code>{step_bar}</code>  крок {step}/{total_steps}",
                        parse_mode="HTML", reply_markup=_kb_game_stop(),
                    )

                # ── generate (ComfyUI-wait + retry) ──────────────────────
                try:
                    png_bytes = await _gg_generate(
                        item.prompt, on_progress, user_settings, _edit,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error("game_gen: generation failed %s v%d: %s",
                              item.id, variant, exc)
                    fail_count += 1
                    fail_names.append(f"{item.name} v{variant} (gen: {str(exc)[:60]})")
                    continue

                # ── convert to WebP ───────────────────────────────────────
                try:
                    webp_bytes = await asyncio.get_running_loop().run_in_executor(
                        None, _to_webp, png_bytes,
                    )
                except Exception as exc:
                    log.warning("game_gen: WebP conversion failed %s: %s", item.id, exc)
                    webp_bytes = png_bytes

                # ── show result ───────────────────────────────────────────
                caption = (
                    f"✅ <b>{item.name}</b>{rarity_tag}"
                    + (f"\nВаріант {variant}/{needed}" if needed > 1 else "")
                    + f"\n<i>Предмет {item_idx}/{total_items}  •  Слот {done_slots}/{total_slots}</i>"
                )
                try:
                    await trigger_msg.answer_photo(
                        BufferedInputFile(png_bytes, filename=f"{item.id}_v{variant}.png"),
                        caption=caption, parse_mode="HTML",
                    )
                except Exception as exc:
                    log.warning("game_gen: send photo failed %s v%d: %s",
                                item.id, variant, exc)

                # ── upload (Game API-wait + retry, inline with status) ────
                await _edit(
                    f"{item_header}\n\n⬆️ Завантажую на сервер…",
                    parse_mode="HTML", reply_markup=_kb_game_stop(),
                )
                success, result = await _gg_upload(
                    item.id, webp_bytes,
                    f"{item.id}_v{variant}.webp",
                    edit_fn=_edit,
                )
                if success:
                    ok_count += 1
                    log.info("game_gen: ✓ %s v%d → %s", item.id, variant, result)
                else:
                    fail_count += 1
                    fail_names.append(f"{item.name} v{variant} (upload: {result[:60]})")
                    log.warning("game_gen: ✗ %s v%d — %s", item.id, variant, result)

    except asyncio.CancelledError:
        log.info("game_gen: cancelled — ok=%d fail=%d done=%d/%d",
                 ok_count, fail_count, done_slots, total_slots)
        await _edit(
            f"⛔ <b>Генерацію зупинено</b>\n\n"
            f"✅ Завантажено: <b>{ok_count}</b>\n"
            f"🖼 Зроблено: <b>{done_slots}</b> з <b>{total_slots}</b>",
            parse_mode="HTML", reply_markup=kb_settings(),
        )
        return

    # ── summary ───────────────────────────────────────────────────────────
    summary_lines = [
        "🎮 <b>Генерація предметів завершена!</b>\n",
        f"✅ Завантажено:  <b>{ok_count}</b>",
        f"❌ Помилок:     <b>{fail_count}</b>",
        f"🖼 Зображень:   <b>{total_slots}</b>  (по {game_api.MAX_CANDIDATES} на предмет)",
        f"📦 Предметів:   <b>{total_items}</b>",
    ]
    if fail_names:
        summary_lines.append("\n<b>Не вдалось:</b>")
        for n in fail_names[:10]:
            summary_lines.append(f"  • {n}")
        if len(fail_names) > 10:
            summary_lines.append(f"  … та ще {len(fail_names) - 10}")

    await _edit("\n".join(summary_lines), parse_mode="HTML", reply_markup=kb_settings())


def user_settings_ckpt_label(tg_id: int) -> str:
    s = db.get_gen_settings(tg_id)
    ckpt = s.get("checkpoint") or config.CHECKPOINT
    return _label(ckpt, models_db.labels())


# ── запуск ────────────────────────────────────────────────────────────────

async def _notify_admins(text: str) -> None:
    for u in db.all_users():
        if u.get("role") == "admin" and u.get("id"):
            try:
                await bot.send_message(u["id"], text, parse_mode="HTML")
            except Exception:
                pass

async def _set_commands() -> None:
    """Register bot commands so they appear in the Telegram command menu."""
    user_commands = [
        BotCommand(command="start",    description="🏠 Головне меню"),
        BotCommand(command="gen",      description="🎨 Згенерувати зображення"),
        BotCommand(command="settings", description="🎛 Налаштування генерації"),
        BotCommand(command="history",  description="📜 Моя історія зображень"),
        BotCommand(command="status",   description="📊 Статус ComfyUI"),
        BotCommand(command="help",     description="❓ Довідка"),
    ]
    await bot.set_my_commands(user_commands, scope=BotCommandScopeAllPrivateChats())
    log.info("Bot commands registered: %s", [c.command for c in user_commands])


async def main() -> None:
    database.init()
    await bot.delete_webhook(drop_pending_updates=True)
    await _set_commands()
    log.info("Checking ComfyUI at %s...", config.COMFY_URL)
    if await comfy_client.ping():
        log.info("ComfyUI is online.")
    else:
        log.warning("ComfyUI is OFFLINE!")
        await _notify_admins(
            f"⚠️ <b>ComfyUI недоступний при старті бота!</b>\n<code>{config.COMFY_URL}</code>"
        )
    log.info("Bot started. Model: %s", config.CHECKPOINT)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
