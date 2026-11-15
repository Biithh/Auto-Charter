"""Evalua el detector de pulsos contra el mapa de tempo de charts de referencia."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import paths

AUDIO_EXT = (".ogg", ".opus", ".mp3", ".wav", ".flac")
SKIP = ("crowd", "preview")


def mix_reference_audio(folder: Path, sr: int = 44100) -> np.ndarray:
    """Mezcla todas las pistas de una cancion de referencia (song + stems)."""
    import tempfile

    from ..audio import decode_to_wav, load

    total = None
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in AUDIO_EXT or any(s in f.stem.lower() for s in SKIP):
            continue
        with tempfile.TemporaryDirectory() as td:
            wav = decode_to_wav(f, Path(td) / "a.wav", sr)
            y, _ = load(wav, mono=True)
        if total is None:
            total = y
        else:
            n = max(len(total), len(y))
            total = np.pad(total, (0, n - len(total))) + np.pad(y, (0, n - len(y)))
    return total


def chart_beats(folder: Path) -> np.ndarray:
    from ..chart.midi_io import read_midi
    from ..chart.model import RES
    from ..songini import read_ini

    ini = read_ini(folder / "song.ini")
    delay = float(ini.get("delay", "0") or 0) / 1000.0
    chart = read_midi(folder / "notes.mid")
    tm = chart.tempo
    out = []
    t = 0
    while t < chart.end_tick:
        out.append(tm.tick_to_sec(t) - delay)
        t += RES
    return np.array(out)


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = 0.07) -> float:
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    ref = ref[(ref >= est[0] - 1) & (ref <= est[-1] + 1)]
    hits = 0
    j = 0
    used = np.zeros(len(est), bool)
    for r in ref:
        i = np.searchsorted(est, r)
        best = None
        for k in (i - 1, i):
            if 0 <= k < len(est) and not used[k] and abs(est[k] - r) <= tol:
                if best is None or abs(est[k] - r) < abs(est[best] - r):
                    best = k
        if best is not None:
            used[best] = True
            hits += 1
    p = hits / len(est)
    r = hits / max(len(ref), 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def main(folders: list[str]) -> None:
    from ..pipeline.beats import build_tempo_map, track_beats
    from ..chart.model import RES

    scores = []
    for f in folders:
        folder = Path(f)
        y = mix_reference_audio(folder)
        grid = track_beats(y, 44100, log=lambda *_: None)
        tm, _ = build_tempo_map(grid, 0.0)
        # pulsos del mapa generado (incluye la extrapolacion inicial)
        est = []
        t = 0
        end = tm.sec_to_tick(len(y) / 44100)
        while t < end:
            est.append(tm.tick_to_sec(t))
            t += RES
        est = np.array(est)
        ref = chart_beats(folder)
        fm_raw = f_measure(ref, grid.beats)
        fm_map = f_measure(ref, est)
        # pulsos a doble o mitad de velocidad cuentan aparte
        fm_double = f_measure(ref, np.sort(np.concatenate([est, (est[1:] + est[:-1]) / 2])))
        scores.append(fm_map)
        print(f"{folder.name[:50]:50s} raw={fm_raw:.3f} map={fm_map:.3f} x2={fm_double:.3f} const={grid.constant_bpm}")
    print("MEDIA", np.mean(scores))


if __name__ == "__main__":
    paths.setup_env()
    main(sys.argv[1:])
