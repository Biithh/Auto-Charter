"""Resumen de una carpeta de canciones: cuantas son oficiales (y de que juego o pack) y cuantas de la comunidad.
Solo lee los song.ini, no cambia nada. Uso: python -m charter.resumen_biblioteca [carpeta]"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from .calibration.reference import OFFICIAL_CHARTERS, find_songs
from .songini import read_ini


def _clean(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()  # los charters ponen colores tipo <color=#f00>


def summarize(root: Path) -> str:
    songs = official = 0
    charters: Counter = Counter()
    packs: Counter = Counter()  # de donde vienen las oficiales (icon / playlist del song.ini o la carpeta)
    folders: Counter = Counter()
    for folder in find_songs(root, skip_ai=False):
        try:
            ini = read_ini(folder / "song.ini")
        except OSError:
            continue
        songs += 1
        charter = _clean(ini.get("charter") or ini.get("frets") or "") or "(sin charter)"
        top = folder.relative_to(root).parts[0] if folder != root else root.name
        folders[top] += 1
        if any(o in charter.lower() for o in OFFICIAL_CHARTERS):
            official += 1
            packs[_clean(ini.get("playlist") or ini.get("icon") or top) or top] += 1
        else:
            charters[charter] += 1
    if not songs:
        return f"No hay canciones (song.ini + notes.mid/notes.chart) en {root}"
    pct = lambda n: f"{n} ({n / songs:.0%})"
    out = [f"Carpeta: {root}", f"Canciones: {songs}",
           f"Oficiales (Harmonix, Neversoft y otros estudios): {pct(official)}",
           f"  de ellas, de DLC: {sum(n for k, n in packs.items() if 'dlc' in k.lower())}",
           f"De la comunidad: {pct(songs - official)} de {len(charters)} charters distintos", "",
           "Oficiales por juego o pack:"]
    out += [f"  {n:5d}  {k}" for k, n in packs.most_common(30)]
    out += ["", "Charters de la comunidad con mas canciones:"]
    out += [f"  {n:5d}  {k}" for k, n in charters.most_common(25)]
    out += ["", "Canciones por carpeta principal:"]
    out += [f"  {n:5d}  {k}" for k, n in folders.most_common(30)]
    return "\n".join(out)


def main() -> None:
    from PySide6.QtWidgets import QApplication, QFileDialog, QPlainTextEdit

    app = QApplication(sys.argv)
    root = sys.argv[1] if len(sys.argv) > 1 else QFileDialog.getExistingDirectory(None, "Carpeta de canciones")
    if not root:
        return
    text = summarize(Path(root))
    (Path(__file__).resolve().parents[2] / "resumen_biblioteca.txt").write_text(text, encoding="utf-8")
    view = QPlainTextEdit(text)
    view.setReadOnly(True)
    view.setWindowTitle("Resumen de la biblioteca (guardado en resumen_biblioteca.txt)")
    view.resize(760, 640)
    view.show()
    app.exec()


if __name__ == "__main__":
    main()
