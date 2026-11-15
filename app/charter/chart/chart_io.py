"""Lectura de notes.chart (formato de Clone Hero / Moonscraper) al modelo interno.

Solo lo necesario para aprender de la biblioteca: tempo, compases, secciones, guitarra, rhythm,
bajo, teclado y bateria (con platillos y doble bombo). La voz no se lee de .chart.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .midi_io import resolve_hopos
from .model import DIFFICULTIES, OPEN, RES, Chart, DrumNote, FretNote, Phrase, TempoMap

TRACKS = {"Single": "guitar", "DoubleRhythm": "rhythm", "DoubleBass": "bass", "Keyboard": "keys"}
DIFF_PREFIX = {"Easy": "easy", "Medium": "medium", "Hard": "hard", "Expert": "expert"}


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"\[([^\]]+)\]\s*\{(.*?)\}", text, re.S):
        out[m.group(1).strip()] = [l.strip() for l in m.group(2).splitlines() if l.strip()]
    return out


def read_chart(path: str | Path) -> Chart:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    secs = _sections(text)
    res = 192
    for line in secs.get("Song", []):
        k, _, v = line.partition("=")
        if k.strip() == "Resolution":
            try:
                res = int(float(v.strip()))
            except ValueError:
                pass
    scale = RES / res

    def tk(x: str) -> int:
        return int(round(int(x) * scale))

    tempos, sigs = [], []
    for line in secs.get("SyncTrack", []):
        m = re.match(r"(\d+)\s*=\s*(\w+)\s+(\d+)(?:\s+(\d+))?", line)
        if not m:
            continue
        t, kind, a, b = tk(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if kind == "B" and a > 0:
            tempos.append((t, int(round(60_000_000_000 / a))))
        elif kind == "TS":
            den = 2 ** int(b) if b is not None else 4
            sigs.append((t, a, den))
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))
    if not sigs or sigs[0][0] != 0:
        sigs.insert(0, (0, 4, 4))
    chart = Chart(tempo=TempoMap(tempos=tempos, time_sigs=sigs))
    end = 0
    for line in secs.get("Events", []):
        m = re.match(r'(\d+)\s*=\s*E\s+"?section\s+([^"]+)"?', line)
        if m:
            chart.sections.append((tk(m.group(1)), m.group(2).strip()))
    hopo_thr = int(round(65 / 192 * RES))
    for name, lines in secs.items():
        diff = next((DIFF_PREFIX[p] for p in DIFF_PREFIX if name.startswith(p)), None)
        if diff is None:
            continue
        inst_name = name[len(next(p for p in DIFF_PREFIX if name.startswith(p))):]
        notes_by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)
        sp, solos = [], []
        solo_start = None
        for line in lines:
            m = re.match(r"(\d+)\s*=\s*(\w)\s+(\S+)(?:\s+(\d+))?", line)
            if not m:
                continue
            t, kind = tk(m.group(1)), m.group(2)
            end = max(end, t)
            if kind == "N":
                notes_by_tick[t].append((int(m.group(3)), tk(m.group(4) or "0")))
            elif kind == "S" and m.group(3) == "2":
                sp.append(Phrase(t, t + tk(m.group(4) or "0")))
            elif kind == "E":
                if m.group(3) == "solo":
                    solo_start = t
                elif m.group(3) == "soloend" and solo_start is not None:
                    solos.append(Phrase(solo_start, t))
        if inst_name == "Drums":
            dn = []
            for t in sorted(notes_by_tick):
                vals = dict(notes_by_tick[t])
                for v in vals:
                    if v == 0:
                        dn.append(DrumNote(t, 0))
                    elif v == 32 and diff == "expert":
                        dn.append(DrumNote(t, 0, double_kick=True))
                    elif 1 <= v <= 4:
                        pad = v if v <= 3 else 4
                        cym = pad >= 2 and (64 + pad) in vals
                        dn.append(DrumNote(t, pad, cymbal=cym))
            chart.drums.notes[diff] = dn
            if diff == "expert":
                chart.drums.star_power = sp
            continue
        key = TRACKS.get(inst_name)
        if key is None:
            continue
        ft = getattr(chart, key)
        notes = []
        for t in sorted(notes_by_tick):
            vals = notes_by_tick[t]
            frets = sorted(set(v for v, _ in vals if v <= 4))
            flags = set(v for v, _ in vals)
            length = max((l for v, l in vals if v <= 4 or v == 7), default=0)
            if 7 in flags and not frets:
                frets = [OPEN]
            if not frets:
                continue
            fn = FretNote(t, tuple(frets), length if length > RES // 3 else 0)
            fn.tap = 6 in flags
            if 5 in flags:
                fn.hopo = "force"  # se resuelve abajo
            notes.append(fn)
        # en .chart el "forzado" invierte lo natural
        natural = resolve_hopos([FretNote(n.tick, n.frets) for n in notes], hopo_thr)
        for n, nat in zip(notes, natural):
            n.hopo = (not nat) if n.hopo == "force" else None
        ft.notes[diff] = notes
        if diff == "expert":
            ft.star_power = sp
            ft.solos = solos
    chart.end_tick = end + RES
    return chart


def load_chart(folder: str | Path, hopo_frequency: int | None = None) -> Chart:
    """notes.mid si existe; si no, notes.chart."""
    from .midi_io import read_midi

    folder = Path(folder)
    if (folder / "notes.mid").exists():
        return read_midi(folder / "notes.mid", hopo_frequency)
    return read_chart(folder / "notes.chart")
