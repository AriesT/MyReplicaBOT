import database as _db


def _short(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def all_upscale_models() -> list[str]:
    c = _db.get()
    return [r["name"] for r in c.execute("SELECT name FROM upscale_models ORDER BY name").fetchall()]


def labels() -> dict[str, str]:
    """Return {name: display_label} for every upscale model. Falls back to _short(name)."""
    c = _db.get()
    return {
        r["name"]: (r["display_name"] or _short(r["name"]))
        for r in c.execute("SELECT name, display_name FROM upscale_models ORDER BY name").fetchall()
    }


def add_upscale_model(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    c = _db.get()
    try:
        c.execute("INSERT INTO upscale_models(name) VALUES(?)", (name,))
        c.commit()
        return True
    except Exception:
        return False


def remove_upscale_model(name: str) -> bool:
    c = _db.get()
    cur = c.execute("DELETE FROM upscale_models WHERE name = ?", (name,))
    c.commit()
    return cur.rowcount > 0


def set_display_name(name: str, display_name: str) -> bool:
    """Set the display label for an upscale model without touching the actual filename."""
    display_name = display_name.strip()
    c = _db.get()
    if not c.execute("SELECT 1 FROM upscale_models WHERE name = ?", (name,)).fetchone():
        return False
    c.execute("UPDATE upscale_models SET display_name = ? WHERE name = ?", (display_name, name))
    c.commit()
    return True
