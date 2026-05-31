"""
Single SQLite connection shared across all modules.
Call database.init() once at startup.
"""
import json
import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).parent / "bot.db"
_conn: sqlite3.Connection | None = None


def get() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init() -> None:
    c = get()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER UNIQUE,
            username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            role        TEXT    NOT NULL DEFAULT 'user',
            gen_count   INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS gen_settings (
            telegram_id INTEGER NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT,
            PRIMARY KEY (telegram_id, key)
        );
        CREATE TABLE IF NOT EXISTS models (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS loras (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS upscale_models (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS history (
            id         TEXT    PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            username   TEXT    NOT NULL DEFAULT '',
            prompt     TEXT    NOT NULL,
            file_path  TEXT    NOT NULL,
            ts         TEXT    NOT NULL,
            mode       TEXT    NOT NULL DEFAULT 'text2img',
            checkpoint TEXT    NOT NULL DEFAULT '',
            width      INTEGER NOT NULL DEFAULT 512,
            height     INTEGER NOT NULL DEFAULT 512
        );
    """)
    _migrate(c)


def _migrate(c: sqlite3.Connection) -> None:
    base = Path(__file__).parent

    # users.json → users + gen_settings
    uj = base / "users.json"
    if uj.exists() and c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        data = json.loads(uj.read_text(encoding="utf-8"))
        for u in data.get("users", []):
            uname = u.get("username", "")
            if not uname:
                continue
            c.execute(
                "INSERT OR IGNORE INTO users(telegram_id, username, role, gen_count) VALUES(?,?,?,?)",
                (u.get("id"), uname, u.get("role", "user"), u.get("gen_count", 0)),
            )
            tid = u.get("id")
            if tid:
                for k, v in u.get("gen_settings", {}).items():
                    if v is not None:
                        c.execute(
                            "INSERT OR REPLACE INTO gen_settings(telegram_id,key,value) VALUES(?,?,?)",
                            (tid, k, json.dumps(v)),
                        )
        c.commit()
        uj.rename(uj.with_suffix(".json.bak"))

    # models.json → models
    mj = base / "models.json"
    if mj.exists() and c.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0:
        data = json.loads(mj.read_text(encoding="utf-8"))
        for name in data.get("models", []):
            c.execute("INSERT OR IGNORE INTO models(name) VALUES(?)", (name,))
        c.commit()
        mj.rename(mj.with_suffix(".json.bak"))

    # display_name column for models / loras / upscale_models
    for _tbl in ("models", "loras", "upscale_models"):
        _cols = [r[1] for r in c.execute(f"PRAGMA table_info({_tbl})").fetchall()]
        if "display_name" not in _cols:
            c.execute(f"ALTER TABLE {_tbl} ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
        if _tbl == "models" and "workflow" not in _cols:
            c.execute("ALTER TABLE models ADD COLUMN workflow TEXT NOT NULL DEFAULT 'sd15'")
        if _tbl == "loras" and "trigger" not in _cols:
            c.execute("ALTER TABLE loras ADD COLUMN trigger TEXT NOT NULL DEFAULT ''")
    c.commit()

    # history.json is dropped intentionally:
    # old entries used Telegram file_id (not disk paths) — they are invalid after migration.
    hj = base / "history.json"
    if hj.exists():
        hj.rename(hj.with_suffix(".json.bak"))
