import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import database as _db

IMAGES_DIR   = Path(__file__).parent / "images"
MAX_PER_USER = 50


def _ensure_dir() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)


def save_image(user_id: int, data: bytes) -> tuple[str, str]:
    """Save PNG bytes to disk. Returns (entry_id, file_path)."""
    _ensure_dir()
    entry_id  = uuid.uuid4().hex[:12]
    file_path = str(IMAGES_DIR / f"{user_id}_{entry_id}.png")
    Path(file_path).write_bytes(data)
    return entry_id, file_path


def add_entry(
    entry_id:   str,
    file_path:  str,
    user_id:    int,
    username:   str,
    prompt:     str,
    mode:       str,
    checkpoint: str,
    width:      int,
    height:     int,
) -> None:
    c  = _db.get()
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    c.execute(
        """INSERT INTO history(id, user_id, username, prompt, file_path, ts, mode, checkpoint, width, height)
           VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (entry_id, user_id, username, prompt, file_path, ts, mode, checkpoint, width, height),
    )
    # Trim oldest entries beyond MAX_PER_USER for this user
    old_ids = c.execute(
        "SELECT id FROM history WHERE user_id = ? ORDER BY ts DESC LIMIT -1 OFFSET ?",
        (user_id, MAX_PER_USER),
    ).fetchall()
    for row in old_ids:
        _delete_one(c, row["id"])
    c.commit()


def get_entries(user_id: Optional[int] = None) -> list[dict]:
    """Return entries newest-first. user_id=None → all users."""
    c = _db.get()
    if user_id is not None:
        rows = c.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY ts DESC", (user_id,)
        ).fetchall()
    else:
        rows = c.execute("SELECT * FROM history ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]


def delete_entry(entry_id: str) -> bool:
    c = _db.get()
    deleted = _delete_one(c, entry_id)
    if deleted:
        c.commit()
    return deleted


def clear_all() -> int:
    """Delete all history entries and their image files. Returns count deleted."""
    c = _db.get()
    rows = c.execute("SELECT id FROM history").fetchall()
    count = 0
    for row in rows:
        if _delete_one(c, row["id"]):
            count += 1
    c.commit()
    return count


def _delete_one(c, entry_id: str) -> bool:
    row = c.execute("SELECT file_path FROM history WHERE id = ?", (entry_id,)).fetchone()
    if not row:
        return False
    c.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    fp = Path(row["file_path"])
    if fp.exists():
        try:
            fp.unlink()
        except OSError:
            pass
    return True
