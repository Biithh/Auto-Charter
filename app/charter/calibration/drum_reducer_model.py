"""Modelo aprendido de reduccion de bateria (Experto -> Dificil / Medio / Facil).

Uso: python -m charter.calibration.drum_reducer_model <carpeta_de_charts> [--eval]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from .. import paths
from ..chart.model import RES

MODEL_PATH = paths.DATA / "drum_reducer.pkl"
DIFFS = ("hard", "medium", "easy")


def drum_features(notes, tm) -> np.ndarray:
    """Una fila por golpe del Experto (sin doble bombo)."""
    notes = [n for n in notes if not n.double_kick]
    ticks = np.array(sorted(set(n.tick for n in notes)), dtype=float)
    secs = np.array([tm.tick_to_sec(t) for t in ticks])
    by_tick: dict[int, list] = {}
    for n in notes:
        by_tick.setdefault(n.tick, []).append(n)
    # ultima aparicion de cada pieza para medir su ritmo
    rows = []
    last_same: dict[tuple, float] = {}
    tick_index = {int(t): i for i, t in enumerate(ticks)}
    # siguientes apariciones de cada pieza
    next_same: dict[int, float] = {}
    seen_next: dict[tuple, int] = {}
    for n in reversed(notes):
        k = (n.pad, n.cymbal)
        next_same[id(n)] = (seen_next[k] - n.tick) / RES if k in seen_next else 8.0
        seen_next[k] = n.tick
    for n in notes:
        i = tick_index[n.tick]
        group = by_tick[n.tick]
        start, num, den = tm.ts_at(n.tick)
        meas = num * RES * 4 // den
        rel = (n.tick - start) % meas
        pos = (n.tick % RES) / RES
        k = (n.pad, n.cymbal)
        prev_same = (n.tick - last_same[k]) / RES if k in last_same else 8.0
        last_same[k] = n.tick
        lo = np.searchsorted(ticks, n.tick - RES)
        hi = np.searchsorted(ticks, n.tick + RES, side="right")
        ioi_p = (ticks[i] - ticks[i - 1]) / RES if i > 0 else 8.0
        ioi_n = (ticks[i + 1] - ticks[i]) / RES if i + 1 < len(ticks) else 8.0
        pads = [0] * 8
        for g in group:
            idx = g.pad if not (g.cymbal and g.pad >= 2) else g.pad + 3
            pads[min(idx, 7)] = 1
        rows.append([
            n.pad, float(n.cymbal), pos, float(pos == 0), float(abs(pos - 0.5) < 1e-6), rel / RES, float(rel == 0),
            float(rel % (2 * RES) == 0), min(prev_same, 8), min(next_same[id(n)], 8), min(ioi_p, 8), min(ioi_n, 8),
            hi - lo, len(group), *pads,
            60.0 / max(tm.tick_to_sec(n.tick + RES) - secs[i], 1e-3),
        ])
    return np.array(rows, dtype=np.float32), notes


def train(folder: str, log=print, evaluate: bool = False) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from ..chart.chart_io import load_chart
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS, find_songs

    data = {d: {"X": [], "y": [], "w": [], "g": []} for d in DIFFS}
    for gi, f in enumerate(find_songs(folder)):
        try:
            ini = read_ini(f / "song.ini")
            chart = load_chart(f)
        except Exception:  # noqa: BLE001
            continue
        dt = chart.drums
        if len(dt.notes["expert"]) < 100 or any(len(dt.notes[d]) < 20 for d in DIFFS):
            continue
        official = any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS)
        X, notes = drum_features(dt.notes["expert"], chart.tempo)
        sel = np.sort(np.random.default_rng(gi).choice(len(notes), min(len(notes), 900), replace=False))
        for d in DIFFS:
            tgt = set((n.tick, n.pad, n.cymbal) for n in dt.notes[d])
            tgt_pad = set((n.tick, n.pad) for n in dt.notes[d])
            y = np.array([(n.tick, n.pad, n.cymbal) in tgt or (n.tick, n.pad) in tgt_pad for n in notes])
            data[d]["X"].append(X[sel])
            data[d]["y"].append(y[sel])
            data[d]["w"].append(np.full(len(sel), 2.0 if official else 1.0))
            data[d]["g"].append(np.full(len(sel), gi))
    models = {}
    for d in DIFFS:
        if not data[d]["X"]:
            continue
        X = np.concatenate(data[d]["X"])
        y = np.concatenate(data[d]["y"]).astype(int)
        w = np.concatenate(data[d]["w"])
        g = np.concatenate(data[d]["g"])
        if evaluate:
            f1s = []
            for tr, te in GroupKFold(n_splits=4).split(X, y, g):
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
                m.fit(X[tr], y[tr], sample_weight=w[tr])
                p = m.predict(X[te])
                tp = ((p == 1) & (y[te] == 1)).sum()
                prec = tp / max((p == 1).sum(), 1)
                rec = tp / max((y[te] == 1).sum(), 1)
                f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
            log(f"bateria {d}: F1 golpes conservados (validacion cruzada) = {np.mean(f1s):.3f}")
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
        m.fit(X, y, sample_weight=w)
        models[d] = m
    return models


def save(models: dict, path: Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(models, f)


_CACHE = None


def load_models() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for p in (paths.CONFIG_DIR / "drum_reducer.pkl", MODEL_PATH):
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        _CACHE = pickle.load(f)
                    break
                except Exception:  # noqa: BLE001
                    continue
    return _CACHE


if __name__ == "__main__":
    paths.setup_env()
    ev = "--eval" in sys.argv
    models = train(sys.argv[1], log=print, evaluate=ev)
    save(models)
    print("modelo guardado en", MODEL_PATH)
