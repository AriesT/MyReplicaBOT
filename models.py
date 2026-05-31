import config
import database as _db


def _short(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def _ensure_default() -> None:
    c = _db.get()
    if c.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0:
        c.execute("INSERT OR IGNORE INTO models(name) VALUES(?)", (config.CHECKPOINT,))
        c.commit()


def all_models() -> list[str]:
    _ensure_default()
    c = _db.get()
    return [r["name"] for r in c.execute("SELECT name FROM models ORDER BY name").fetchall()]


def labels() -> dict[str, str]:
    """Return {name: display_label} for every model. Falls back to _short(name)."""
    _ensure_default()
    c = _db.get()
    return {
        r["name"]: (r["display_name"] or _short(r["name"]))
        for r in c.execute("SELECT name, display_name FROM models ORDER BY name").fetchall()
    }


def workflows() -> dict[str, str]:
    """Return {name: workflow_type} for every model."""
    _ensure_default()
    c = _db.get()
    return {
        r["name"]: (r["workflow"] or "sd15")
        for r in c.execute("SELECT name, workflow FROM models").fetchall()
    }


def get_workflow(name: str) -> str:
    c = _db.get()
    row = c.execute("SELECT workflow FROM models WHERE name = ?", (name,)).fetchone()
    return (row["workflow"] or "sd15") if row else "sd15"


def set_workflow(name: str, wf: str) -> bool:
    c = _db.get()
    if not c.execute("SELECT 1 FROM models WHERE name = ?", (name,)).fetchone():
        return False
    c.execute("UPDATE models SET workflow = ? WHERE name = ?", (wf, name))
    c.commit()
    return True


def add_model(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    c = _db.get()
    try:
        c.execute("INSERT INTO models(name) VALUES(?)", (name,))
        c.commit()
        return True
    except Exception:
        return False


def remove_model(name: str) -> bool:
    c = _db.get()
    cur = c.execute("DELETE FROM models WHERE name = ?", (name,))
    c.commit()
    if cur.rowcount == 0:
        return False
    _ensure_default()
    return True


def set_display_name(name: str, display_name: str) -> bool:
    """Set the display label for a model without touching the actual filename."""
    display_name = display_name.strip()
    c = _db.get()
    if not c.execute("SELECT 1 FROM models WHERE name = ?", (name,)).fetchone():
        return False
    c.execute("UPDATE models SET display_name = ? WHERE name = ?", (display_name, name))
    c.commit()
    return True
