"""
MMORPG Game API client.

Endpoints:
  GET  {GAME_API_URL}/api/game-api/items/needs-generation
       → items where image_url is empty AND pending_count < 4
       → fields: id, name, type, rarity, image_prompt, pending_count

  POST {GAME_API_URL}/api/game-api/items/{id}/pending-image  (multipart)
       → adds to candidates list (max 4 per item)
       → returns {id, image_url}

Auth: X-API-Key header.
Set GAME_API_URL and GAME_API_KEY in .env to enable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aiohttp

import config

log = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=60)
MAX_CANDIDATES = 4


# ── data model ────────────────────────────────────────────────────────────

@dataclass
class GameItem:
    id:            str
    name:          str
    prompt:        str
    type:          str = ""
    rarity:        str = ""
    pending_count: int = 0

    @property
    def slots_remaining(self) -> int:
        """How many more images can be uploaded for this item."""
        return max(0, MAX_CANDIDATES - self.pending_count)


# ── public API ────────────────────────────────────────────────────────────

def is_configured() -> bool:
    return bool(config.GAME_API_URL and config.GAME_API_KEY)


async def fetch_items() -> list[GameItem]:
    """Return items that still need generation candidates."""
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}/api/game-api/items/needs-generation"
    headers = {"X-API-Key": config.GAME_API_KEY}

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        r = await session.get(url, headers=headers)
        r.raise_for_status()
        data = await r.json()

    raw = data if isinstance(data, list) else data.get("items", data.get("data", []))
    items: list[GameItem] = []
    for obj in raw:
        prompt = (obj.get("image_prompt") or obj.get("prompt") or "").strip()
        if not prompt:
            continue
        pending = int(obj.get("pending_count") or 0)
        if pending >= MAX_CANDIDATES:
            continue  # already full — shouldn't happen (server filters), but be safe
        items.append(GameItem(
            id            = str(obj["id"]),
            name          = str(obj.get("name") or obj.get("title") or obj["id"]),
            prompt        = prompt,
            type          = str(obj.get("type") or ""),
            rarity        = str(obj.get("rarity") or ""),
            pending_count = pending,
        ))
    log.info(
        "game_api: %d items need generation, %d total slots",
        len(items), sum(i.slots_remaining for i in items),
    )
    return items


async def upload_image(
    item_id:     str,
    image_bytes: bytes,
    filename:    str = "image.webp",
) -> tuple[bool, str]:
    """
    Upload one image candidate for an item.
    Returns (success, image_url_or_error_message).
    """
    _require_config()
    url     = f"{config.GAME_API_URL.rstrip('/')}/api/game-api/items/{item_id}/pending-image"
    headers = {"X-API-Key": config.GAME_API_KEY}

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        form = aiohttp.FormData()
        form.add_field("file", image_bytes, filename=filename, content_type="image/webp")
        r = await session.post(url, headers=headers, data=form)
        if r.ok:
            try:
                body = await r.json()
                image_url = body.get("image_url", "")
            except Exception:
                image_url = ""
            log.info("game_api: uploaded candidate for item %s → %s", item_id, image_url)
            return True, image_url
        body = await r.text()
        log.warning("game_api: upload failed item=%s status=%d body=%s",
                    item_id, r.status, body[:200])
        return False, f"HTTP {r.status}: {body[:120]}"


# ── internal ──────────────────────────────────────────────────────────────

def _require_config() -> None:
    if not is_configured():
        raise RuntimeError(
            "Game API not configured. Set GAME_API_URL and GAME_API_KEY in .env"
        )
