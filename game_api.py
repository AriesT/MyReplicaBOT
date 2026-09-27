"""
MMORPG ROE — Game API client.

Endpoints (all auth via X-API-Key header):

  GET  {BASE}/api/game-api/items/needs-generation
       → [{id, name, type, rarity, image_prompt, pending_count, …}]
  POST {BASE}/api/game-api/items/{id}/pending-image

  GET  {BASE}/api/game-api/monsters/needs-generation
       → [{id, name, type, image_prompt, lore_en, pending_count}]
  POST {BASE}/api/game-api/monsters/{id}/pending-image

  GET  {BASE}/api/game-api/npcs/needs-generation
       → [{id, name, title, type, image_prompt, lore_en, pending_count}]
  POST {BASE}/api/game-api/npcs/{id}/pending-image

  GET  {BASE}/api/game-api/locations/needs-generation
       → [{id, name, type, is_safe, level_min, level_max, image_prompt, lore_en, pending_count}]
  POST {BASE}/api/game-api/locations/{id}/pending-image

All upload responses: {ok, pending_image: {id, image_url}}

Set GAME_API_URL and GAME_API_KEY in .env to enable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aiohttp

import config

log = logging.getLogger(__name__)

_TIMEOUT       = aiohttp.ClientTimeout(total=60)
MAX_CANDIDATES = 4          # server-side cap per entity


# ── data models ───────────────────────────────────────────────────────────

@dataclass
class GameItem:
    id:            str
    name:          str
    prompt:        str
    item_type:     str = ""
    rarity:        str = ""
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_CANDIDATES - self.pending_count)

    @property
    def tag(self) -> str:
        return self.rarity or self.item_type


@dataclass
class GameMonster:
    id:            str
    name:          str
    prompt:        str
    monster_type:  str = ""
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_CANDIDATES - self.pending_count)

    @property
    def tag(self) -> str:
        return self.monster_type


@dataclass
class GameNPC:
    id:            str
    name:          str
    prompt:        str
    title:         str = ""   # role, e.g. "Цілителька"
    npc_type:      str = ""   # healer / guard / merchant / etc.
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_CANDIDATES - self.pending_count)

    @property
    def tag(self) -> str:
        return self.title or self.npc_type


@dataclass
class GameLocation:
    id:            str
    name:          str
    prompt:        str
    location_type: str  = ""    # city / shop / dungeon / forest / etc.
    is_safe:       bool = False
    level_min:     int  = 0
    level_max:     int  = 0
    pending_count: int  = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_CANDIDATES - self.pending_count)

    @property
    def tag(self) -> str:
        parts = [self.location_type] if self.location_type else []
        if self.level_min:
            lvl = f"lv.{self.level_min}"
            if self.level_max and self.level_max != self.level_min:
                lvl += f"-{self.level_max}"
            parts.append(lvl)
        if self.is_safe:
            parts.append("safe")
        return "  ".join(parts)


# ── public helpers ────────────────────────────────────────────────────────

def is_configured() -> bool:
    return bool(config.GAME_API_URL and config.GAME_API_KEY)


# ── internal shared helpers ───────────────────────────────────────────────

def _require_config() -> None:
    if not is_configured():
        raise RuntimeError(
            "Game API not configured. Set GAME_API_URL and GAME_API_KEY in .env"
        )


async def _fetch_raw(path: str) -> list[dict]:
    """GET an endpoint, raise on HTTP error, always return a list."""
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}{path}"
    headers = {"X-API-Key": config.GAME_API_KEY}
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        r = await session.get(url, headers=headers)
        r.raise_for_status()
        data = await r.json()
    return data if isinstance(data, list) else (data.get("data") or [])


async def _upload(
    path:         str,
    file_bytes:   bytes,
    filename:     str,
    log_label:    str,
    content_type: str = "image/webp",
    url_key:      str = "image_url",
    body_key:     str = "pending_image",
) -> tuple[bool, str]:
    """POST multipart file; return (ok, url_or_error)."""
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}{path}"
    headers = {"X-API-Key": config.GAME_API_KEY}
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        form = aiohttp.FormData()
        form.add_field("file", file_bytes, filename=filename, content_type=content_type)
        r = await session.post(url, headers=headers, data=form)
        if r.ok:
            try:
                body      = await r.json()
                file_url  = (body.get(body_key) or {}).get(url_key, "")
            except Exception:
                file_url  = ""
            log.info("game_api: uploaded %s → %s", log_label, file_url)
            return True, file_url
        body = await r.text()
        log.warning("game_api: upload failed %s  status=%d  body=%s",
                    log_label, r.status, body[:200])
        return False, f"HTTP {r.status}: {body[:120]}"


def _pending(obj: dict) -> int:
    return int(obj.get("pending_count") or 0)


# ── items ─────────────────────────────────────────────────────────────────

async def fetch_items() -> list[GameItem]:
    raw = await _fetch_raw("/api/game-api/items/needs-generation")
    out: list[GameItem] = []
    for obj in raw:
        prompt  = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        pending = _pending(obj)
        if not prompt or pending >= MAX_CANDIDATES:
            continue
        out.append(GameItem(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj["id"]),
            prompt        = prompt,
            item_type     = str(obj.get("type") or ""),
            rarity        = str(obj.get("rarity") or ""),
            pending_count = pending,
        ))
    log.info("game_api: items  — %d need generation (%d slots)",
             len(out), sum(e.slots_remaining for e in out))
    return out


async def upload_item_image(item_id: str, image_bytes: bytes,
                            filename: str = "image.webp") -> tuple[bool, str]:
    return await _upload(f"/api/game-api/items/{item_id}/pending-image",
                         image_bytes, filename, f"item {item_id}")


# ── monsters ──────────────────────────────────────────────────────────────

async def fetch_monsters() -> list[GameMonster]:
    raw = await _fetch_raw("/api/game-api/monsters/needs-generation")
    out: list[GameMonster] = []
    for obj in raw:
        prompt  = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        pending = _pending(obj)
        if not prompt or pending >= MAX_CANDIDATES:
            continue
        out.append(GameMonster(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj["id"]),
            prompt        = prompt,
            monster_type  = str(obj.get("type") or ""),
            pending_count = pending,
        ))
    log.info("game_api: monsters — %d need generation (%d slots)",
             len(out), sum(e.slots_remaining for e in out))
    return out


async def upload_monster_image(monster_id: str, image_bytes: bytes,
                               filename: str = "image.webp") -> tuple[bool, str]:
    return await _upload(f"/api/game-api/monsters/{monster_id}/pending-image",
                         image_bytes, filename, f"monster {monster_id}")


# ── npcs ──────────────────────────────────────────────────────────────────

async def fetch_npcs() -> list[GameNPC]:
    raw = await _fetch_raw("/api/game-api/npcs/needs-generation")
    out: list[GameNPC] = []
    for obj in raw:
        prompt  = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        pending = _pending(obj)
        if not prompt or pending >= MAX_CANDIDATES:
            continue
        out.append(GameNPC(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj["id"]),
            prompt        = prompt,
            title         = str(obj.get("title") or ""),
            npc_type      = str(obj.get("type") or ""),
            pending_count = pending,
        ))
    log.info("game_api: npcs    — %d need generation (%d slots)",
             len(out), sum(e.slots_remaining for e in out))
    return out


async def upload_npc_image(npc_id: str, image_bytes: bytes,
                           filename: str = "image.webp") -> tuple[bool, str]:
    return await _upload(f"/api/game-api/npcs/{npc_id}/pending-image",
                         image_bytes, filename, f"npc {npc_id}")


# ── locations ─────────────────────────────────────────────────────────────

async def fetch_locations() -> list[GameLocation]:
    raw = await _fetch_raw("/api/game-api/locations/needs-generation")
    out: list[GameLocation] = []
    for obj in raw:
        prompt  = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        pending = _pending(obj)
        if not prompt or pending >= MAX_CANDIDATES:
            continue
        lv_min = int(obj.get("level_min") or 0)
        lv_max = int(obj.get("level_max") or 0)
        out.append(GameLocation(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj["id"]),
            prompt        = prompt,
            location_type = str(obj.get("type") or ""),
            is_safe       = bool(obj.get("is_safe")),
            level_min     = lv_min,
            level_max     = lv_max,
            pending_count = pending,
        ))
    log.info("game_api: locations — %d need generation (%d slots)",
             len(out), sum(e.slots_remaining for e in out))
    return out


async def upload_location_image(location_id: str, image_bytes: bytes,
                                filename: str = "image.webp") -> tuple[bool, str]:
    return await _upload(f"/api/game-api/locations/{location_id}/pending-image",
                         image_bytes, filename, f"location {location_id}")


# ── skills ────────────────────────────────────────────────────────────────

@dataclass
class GameSkill:
    id:            str
    name:          str
    prompt:        str
    skill_type:    str = ""
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_CANDIDATES - self.pending_count)

    @property
    def tag(self) -> str:
        return self.skill_type


async def fetch_skills() -> list[GameSkill]:
    raw = await _fetch_raw("/api/game-api/skills/needs-generation")
    out: list[GameSkill] = []
    for obj in raw:
        prompt  = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        pending = _pending(obj)
        if not prompt or pending >= MAX_CANDIDATES:
            continue
        out.append(GameSkill(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj["id"]),
            prompt        = prompt,
            skill_type    = str(obj.get("type") or ""),
            pending_count = pending,
        ))
    log.info("game_api: skills  — %d need generation (%d slots)",
             len(out), sum(e.slots_remaining for e in out))
    return out


async def upload_skill_image(skill_id: str, image_bytes: bytes,
                             filename: str = "image.webp") -> tuple[bool, str]:
    return await _upload(f"/api/game-api/skills/{skill_id}/pending-image",
                         image_bytes, filename, f"skill {skill_id}")


# backward-compat alias
upload_image = upload_item_image


# ── animated (video) generation ───────────────────────────────────────────
#
# Endpoints:
#   GET  {BASE}/api/game-api/{type}/needs-animated-generation
#        → [{id, name, type, animated_image_prompt, pending_animated_count}]
#   POST {BASE}/api/game-api/{type}/{id}/pending-animated
#        Allowed: gif webp mp4 webm apng
#        → {ok, pending_image: {id, image_url}}
#
# Source image for img2video:
#   Items     — GET /api/game-api/items               → image_url
#   Locations — GET /api/game-api/locations/{id}/detail → image_url
#   Monsters  — GET /api/game-api/monsters/{id}         → avatar_url
#   NPCs      — GET /api/game-api/npcs/{id}             → avatar_url
#
# Note: monsters/npcs needs-animated-generation currently returns 404 due to a
# server-side routing conflict ({id} route matches before the static path).
# The code handles this gracefully.

MAX_ANIMATED_CANDIDATES = 4


@dataclass
class GameAnimated:
    """Unified entity for animated (video) generation."""
    id:            str
    name:          str
    prompt:        str    # video generation prompt
    image_url:     str    # relative source image path for img2video
    tag:           str = ""
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        return max(0, MAX_ANIMATED_CANDIDATES - self.pending_count)


def _abs_url(relative: str) -> str:
    """Make a relative /static/... URL absolute using the configured base."""
    if relative.startswith(("http://", "https://")):
        return relative
    return f"{config.GAME_API_URL.rstrip('/')}{relative}"


async def download_image(image_url: str) -> bytes:
    """Download an image from the game server (handles relative URLs)."""
    _require_config()
    url     = _abs_url(image_url)
    headers = {"X-API-Key": config.GAME_API_KEY}
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        r = await session.get(url, headers=headers)
        r.raise_for_status()
        return await r.read()


async def _fetch_animated_raw(path: str) -> list[dict]:
    """GET a needs-animated-generation endpoint; return [] on any error (incl. 404)."""
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}{path}"
    headers = {"X-API-Key": config.GAME_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            r = await session.get(url, headers=headers)
            if not r.ok:
                log.warning("game_api: %s → HTTP %d (skip)", path, r.status)
                return []
            data = await r.json()
        return data if isinstance(data, list) else (data.get("data") or [])
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("game_api: animated fetch error %s: %s", path, exc)
        return []


async def _get_one(path: str) -> dict:
    """GET a single entity detail; return {} on error."""
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}{path}"
    headers = {"X-API-Key": config.GAME_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            r = await session.get(url, headers=headers)
            if not r.ok:
                return {}
            return await r.json()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("game_api: get_one error %s: %s", path, exc)
        return {}


async def _upload_animated(
    path:      str,
    vid_bytes: bytes,
    filename:  str,
    log_label: str,
) -> tuple[bool, str]:
    return await _upload(
        path, vid_bytes, filename, log_label,
        content_type="image/webp",
        url_key="image_url",
        body_key="pending_image",
    )


# ── items — animated ──────────────────────────────────────────────────────

async def fetch_items_animated() -> list[GameAnimated]:
    """
    Items where animated_image_url is None and pending < MAX_ANIMATED_CANDIDATES.
    Source image is fetched from /api/game-api/items (full list has image_url).
    Prompt: animated_image_prompt → else constructed from name/type/rarity.
    """
    raw = await _fetch_animated_raw("/api/game-api/items/needs-animated-generation")
    if not raw:
        return []

    # Build image_url lookup from the full items list (endpoint returns a JSON array)
    _require_config()
    try:
        full_raw: list[dict] = await _fetch_raw("/api/game-api/items")
    except Exception:
        full_raw = []
    image_map: dict[str, str] = {
        str(x["id"]): str(x.get("image_url") or "")
        for x in full_raw if isinstance(x, dict) and x.get("id")
    }

    out: list[GameAnimated] = []
    for obj in raw:
        pending = int(obj.get("pending_animated_count") or 0)
        if pending >= MAX_ANIMATED_CANDIDATES:
            continue
        item_id   = str(obj["id"])
        image_url = image_map.get(item_id, "")
        if not image_url:
            log.debug("game_api: item %s has no image yet, skipping animated", item_id)
            continue
        name      = str(obj.get("name") or item_id)
        item_type = str(obj.get("type") or "")
        prompt    = (obj.get("animated_image_prompt") or "").strip()
        if not prompt:
            prompt = (
                f"cinematic fantasy MMORPG animation, {name}"
                + (f", {item_type} item" if item_type else "")
                + ", smooth motion, magical glow, high quality"
            )
        out.append(GameAnimated(
            id=item_id, name=name, prompt=prompt,
            image_url=image_url, tag=item_type, pending_count=pending,
        ))

    log.info("game_api: items animated — %d need generation", len(out))
    return out


async def upload_item_animated(item_id: str, vid_bytes: bytes,
                               filename: str = "animation.webp") -> tuple[bool, str]:
    return await _upload_animated(
        f"/api/game-api/items/{item_id}/pending-animated",
        vid_bytes, filename, f"item animated {item_id}",
    )


# ── monsters — animated ───────────────────────────────────────────────────

async def fetch_monsters_animated() -> list[GameAnimated]:
    """
    Monsters needing animated image.
    Detail endpoint /api/game-api/monsters/{id} provides avatar_url and image_prompt.
    NOTE: needs-animated-generation may return 404 due to server routing conflict.
    """
    raw = await _fetch_animated_raw("/api/game-api/monsters/needs-animated-generation")
    if not raw:
        return []

    out: list[GameAnimated] = []
    for obj in raw:
        pending = int(obj.get("pending_animated_count") or 0)
        if pending >= MAX_ANIMATED_CANDIDATES:
            continue
        monster_id = str(obj["id"])
        detail     = await _get_one(f"/api/game-api/monsters/{monster_id}")
        image_url  = str(detail.get("avatar_url") or detail.get("image_url") or "")
        if not image_url:
            log.debug("game_api: monster %s has no image yet, skipping animated", monster_id)
            continue
        name         = str(obj.get("name") or monster_id)
        monster_type = str(obj.get("type") or "")
        prompt       = (obj.get("animated_image_prompt") or "").strip()
        if not prompt:
            prompt = (
                detail.get("image_prompt")
                or f"cinematic fantasy MMORPG animation, {name}"
                   + (f", {monster_type} monster" if monster_type else "")
                   + ", dynamic movement, dramatic lighting, high quality"
            )
        out.append(GameAnimated(
            id=monster_id, name=name, prompt=prompt,
            image_url=image_url, tag=monster_type, pending_count=pending,
        ))

    log.info("game_api: monsters animated — %d need generation", len(out))
    return out


async def upload_monster_animated(monster_id: str, vid_bytes: bytes,
                                  filename: str = "animation.webp") -> tuple[bool, str]:
    return await _upload_animated(
        f"/api/game-api/monsters/{monster_id}/pending-animated",
        vid_bytes, filename, f"monster animated {monster_id}",
    )


# ── npcs — animated ───────────────────────────────────────────────────────

async def fetch_npcs_animated() -> list[GameAnimated]:
    """
    NPCs needing animated image.
    Detail endpoint /api/game-api/npcs/{id} provides avatar_url and image_prompt.
    NOTE: needs-animated-generation may return 404 due to server routing conflict.
    """
    raw = await _fetch_animated_raw("/api/game-api/npcs/needs-animated-generation")
    if not raw:
        return []

    out: list[GameAnimated] = []
    for obj in raw:
        pending = int(obj.get("pending_animated_count") or 0)
        if pending >= MAX_ANIMATED_CANDIDATES:
            continue
        npc_id    = str(obj["id"])
        detail    = await _get_one(f"/api/game-api/npcs/{npc_id}")
        image_url = str(detail.get("avatar_url") or detail.get("image_url") or "")
        if not image_url:
            log.debug("game_api: npc %s has no image yet, skipping animated", npc_id)
            continue
        name     = str(obj.get("name") or npc_id)
        npc_type = str(obj.get("type") or "")
        prompt   = (obj.get("animated_image_prompt") or "").strip()
        if not prompt:
            prompt = (
                detail.get("image_prompt")
                or f"cinematic fantasy MMORPG animation, {name}"
                   + (f", {npc_type} NPC" if npc_type else "")
                   + ", smooth motion, ambient lighting, high quality"
            )
        out.append(GameAnimated(
            id=npc_id, name=name, prompt=prompt,
            image_url=image_url, tag=npc_type, pending_count=pending,
        ))

    log.info("game_api: npcs animated — %d need generation", len(out))
    return out


async def upload_npc_animated(npc_id: str, vid_bytes: bytes,
                              filename: str = "animation.webp") -> tuple[bool, str]:
    return await _upload_animated(
        f"/api/game-api/npcs/{npc_id}/pending-animated",
        vid_bytes, filename, f"npc animated {npc_id}",
    )


# ── locations — animated ──────────────────────────────────────────────────

async def fetch_locations_animated() -> list[GameAnimated]:
    """
    Locations needing animated image.
    Detail endpoint /api/game-api/locations/{id}/detail provides image_url and image_prompt.
    """
    raw = await _fetch_animated_raw("/api/game-api/locations/needs-animated-generation")
    if not raw:
        return []

    out: list[GameAnimated] = []
    for obj in raw:
        pending = int(obj.get("pending_animated_count") or 0)
        if pending >= MAX_ANIMATED_CANDIDATES:
            continue
        loc_id    = str(obj["id"])
        detail    = await _get_one(f"/api/game-api/locations/{loc_id}/detail")
        image_url = str(detail.get("image_url") or "")
        if not image_url:
            log.debug("game_api: location %s has no image yet, skipping animated", loc_id)
            continue
        name     = str(obj.get("name") or loc_id)
        loc_type = str(obj.get("type") or "")
        prompt   = (obj.get("animated_image_prompt") or "").strip()
        if not prompt:
            prompt = (
                detail.get("image_prompt")
                or f"cinematic fantasy MMORPG environment animation, {name}"
                   + (f", {loc_type}" if loc_type else "")
                   + ", atmospheric motion, wind, ambient life, high quality"
            )
        out.append(GameAnimated(
            id=loc_id, name=name, prompt=prompt,
            image_url=image_url, tag=loc_type, pending_count=pending,
        ))

    log.info("game_api: locations animated — %d need generation", len(out))
    return out


async def upload_location_animated(location_id: str, vid_bytes: bytes,
                                   filename: str = "animation.webp") -> tuple[bool, str]:
    return await _upload_animated(
        f"/api/game-api/locations/{location_id}/pending-animated",
        vid_bytes, filename, f"location animated {location_id}",
    )


# ── skills — animated ─────────────────────────────────────────────────────

async def fetch_skills_animated() -> list[GameAnimated]:
    """
    Skills needing animated image.
    Source image fetched from /api/game-api/skills (full list has image_url).
    Prompt: animated_image_prompt → else constructed from name/type.
    """
    raw = await _fetch_animated_raw("/api/game-api/skills/needs-animated-generation")
    if not raw:
        return []

    # Build image_url lookup from the full skills list
    _require_config()
    try:
        full_raw: list[dict] = await _fetch_raw("/api/game-api/skills")
    except Exception:
        full_raw = []
    image_map: dict[str, str] = {
        str(x["id"]): str(x.get("image_url") or x.get("avatar_url") or "")
        for x in full_raw if isinstance(x, dict) and x.get("id")
    }

    out: list[GameAnimated] = []
    for obj in raw:
        pending = int(obj.get("pending_animated_count") or 0)
        if pending >= MAX_ANIMATED_CANDIDATES:
            continue
        skill_id  = str(obj["id"])
        image_url = image_map.get(skill_id, "")
        if not image_url:
            log.debug("game_api: skill %s has no image yet, skipping animated", skill_id)
            continue
        name       = str(obj.get("name") or skill_id)
        skill_type = str(obj.get("type") or "")
        prompt     = (obj.get("animated_image_prompt") or "").strip()
        if not prompt:
            prompt = (
                f"cinematic fantasy MMORPG skill animation, {name}"
                + (f", {skill_type} ability" if skill_type else "")
                + ", magical effect, glowing particles, smooth motion, high quality"
            )
        out.append(GameAnimated(
            id=skill_id, name=name, prompt=prompt,
            image_url=image_url, tag=skill_type, pending_count=pending,
        ))

    log.info("game_api: skills animated — %d need generation", len(out))
    return out


async def upload_skill_animated(skill_id: str, vid_bytes: bytes,
                                filename: str = "animation.webp") -> tuple[bool, str]:
    return await _upload_animated(
        f"/api/game-api/skills/{skill_id}/pending-animated",
        vid_bytes, filename, f"skill animated {skill_id}",
    )
