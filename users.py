import json
from typing import Optional

import database as _db

ROLE_ADMIN = "admin"
ROLE_USER  = "user"


def _row_to_dict(row) -> dict:
    return {
        "id":        row["telegram_id"],
        "username":  row["username"],
        "role":      row["role"],
        "gen_count": row["gen_count"],
    }


def all_users() -> list[dict]:
    c = _db.get()
    rows = c.execute("SELECT * FROM users ORDER BY username COLLATE NOCASE").fetchall()
    return [_row_to_dict(r) for r in rows]


def find(telegram_id: int | None = None, username: str = "") -> Optional[dict]:
    c = _db.get()
    if telegram_id is not None:
        row = c.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()
        if row:
            return _row_to_dict(row)
    if username:
        username = username.lstrip("@")
        row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            return _row_to_dict(row)
    return None


def is_allowed(telegram_id: int, username: str = "") -> bool:
    return find(telegram_id, username) is not None


def is_admin(telegram_id: int, username: str = "") -> bool:
    u = find(telegram_id, username)
    return u is not None and u["role"] == ROLE_ADMIN


def add(username: str, role: str = ROLE_USER) -> dict:
    username = username.lstrip("@")
    c = _db.get()
    c.execute(
        "INSERT OR IGNORE INTO users(username, role) VALUES(?, ?)",
        (username, role),
    )
    c.commit()
    return find(username=username)


def remove(username: str) -> bool:
    username = username.lstrip("@")
    c = _db.get()
    cur = c.execute("DELETE FROM users WHERE username = ?", (username,))
    c.commit()
    return cur.rowcount > 0


def set_role(username: str, role: str) -> bool:
    username = username.lstrip("@")
    c = _db.get()
    cur = c.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
    c.commit()
    return cur.rowcount > 0


def admin_count() -> int:
    c = _db.get()
    return c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]


def sync_id(telegram_id: int, username: str) -> None:
    username = username.lstrip("@")
    if not username:
        return
    c = _db.get()
    c.execute(
        "UPDATE users SET telegram_id = ? WHERE username = ? AND (telegram_id IS NULL OR telegram_id != ?)",
        (telegram_id, username, telegram_id),
    )
    c.commit()


def increment_gen_count(telegram_id: int) -> None:
    c = _db.get()
    c.execute("UPDATE users SET gen_count = gen_count + 1 WHERE telegram_id = ?", (telegram_id,))
    c.commit()


def get_gen_settings(telegram_id: int) -> dict:
    c = _db.get()
    rows = c.execute(
        "SELECT key, value FROM gen_settings WHERE telegram_id = ?", (telegram_id,)
    ).fetchall()
    result = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            result[row["key"]] = row["value"]
    return result


def set_gen_setting(telegram_id: int, key: str, value) -> bool:
    if not find(telegram_id=telegram_id):
        return False
    c = _db.get()
    if value is None:
        c.execute(
            "DELETE FROM gen_settings WHERE telegram_id = ? AND key = ?",
            (telegram_id, key),
        )
    else:
        c.execute(
            "INSERT OR REPLACE INTO gen_settings(telegram_id, key, value) VALUES(?, ?, ?)",
            (telegram_id, key, json.dumps(value)),
        )
    c.commit()
    return True


def reset_gen_settings(telegram_id: int) -> bool:
    if not find(telegram_id=telegram_id):
        return False
    c = _db.get()
    c.execute("DELETE FROM gen_settings WHERE telegram_id = ?", (telegram_id,))
    c.commit()
    return True
