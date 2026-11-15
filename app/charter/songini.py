"""Lectura / escritura de song.ini."""

from __future__ import annotations

from pathlib import Path


def read_ini(path: str | Path) -> dict[str, str]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if "[" in text:
                break
        except UnicodeDecodeError:
            continue
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("[", ";", "#")) or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def write_ini(path: str | Path, values: dict[str, object]) -> None:
    lines = ["[song]"]
    for k, v in values.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = "True" if v else "False"
        lines.append(f"{k} = {v}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
