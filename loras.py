import database as _db


def _short(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def all_loras() -> list[str]:
    c = _db.get()
    return [r["name"] for r in c.execute("SELECT name FROM loras ORDER BY name").fetchall()]


def labels() -> dict[str, str]:
    """Return {name: display_label} for every lora. Falls back to _short(name)."""
    c = _db.get()
    return {
        r["name"]: (r["display_name"] or _short(r["name"]))
        for r in c.execute("SELECT name, display_name FROM loras ORDER BY name").fetchall()
    }


def add_lora(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    c = _db.get()
    try:
        c.execute("INSERT INTO loras(name) VALUES(?)", (name,))
        c.commit()
        return True
    except Exception:
        return False


def remove_lora(name: str) -> bool:
    c = _db.get()
    cur = c.execute("DELETE FROM loras WHERE name = ?", (name,))
    c.commit()
    return cur.rowcount > 0


def set_display_name(name: str, display_name: str) -> bool:
    """Set the display label for a lora without touching the actual filename."""
    display_name = display_name.strip()
    c = _db.get()
    if not c.execute("SELECT 1 FROM loras WHERE name = ?", (name,)).fetchone():
        return False
    c.execute("UPDATE loras SET display_name = ? WHERE name = ?", (display_name, name))
    c.commit()
    return True


def get_trigger(name: str) -> str:
    """Return the trigger word(s) for a LoRA, or empty string."""
    c = _db.get()
    row = c.execute("SELECT trigger FROM loras WHERE name = ?", (name,)).fetchone()
    return (row["trigger"] or "") if row else ""


def set_trigger(name: str, trigger: str) -> bool:
    """Set trigger word(s) for a LoRA."""
    trigger = trigger.strip()
    c = _db.get()
    if not c.execute("SELECT 1 FROM loras WHERE name = ?", (name,)).fetchone():
        return False
    c.execute("UPDATE loras SET trigger = ? WHERE name = ?", (trigger, name))
    c.commit()
    return True


def triggers() -> dict[str, str]:
    """Return {name: trigger} for every lora that has a trigger set."""
    c = _db.get()
    return {
        r["name"]: (r["trigger"] or "")
        for r in c.execute("SELECT name, trigger FROM loras").fetchall()
    }


def all_loras_meta() -> list[dict]:
    """Return list of {name, display_name, trigger} for all loras."""
    c = _db.get()
    rows = c.execute(
        "SELECT name, display_name, trigger FROM loras ORDER BY name"
    ).fetchall()
    return [
        {
            "name":         r["name"],
            "display_name": r["display_name"] or _short(r["name"]),
            "trigger":      r["trigger"] or "",
        }
        for r in rows
    ]
