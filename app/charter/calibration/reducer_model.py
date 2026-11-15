"""Modelo aprendido de reduccion de dificultad (Experto -> Dificil / Medio / Facil).

Se entrena con charts de referencia (los oficiales pesan mas): aprende que notas del Experto se
mantienen en cada dificultad y a que carril van. Uso:
  python -m charter.calibration.reducer_model <carpeta_de_charts> [--eval]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from .. import paths
from ..chart.model import OPEN, RES

MODEL_PATH = paths.DATA / "reducer.pkl"
DIFFS = ("hard", "medium", "easy")
MAX_LANE = {"hard": 4, "medium": 3, "easy": 2}
MAX_PER_TRACK = 600


def note_features(notes, tm) -> np.ndarray:
    """Caracteristicas de cada nota del Experto (solo del chart, sin audio)."""
    n = len(notes)
    ticks = np.array([x.tick for x in notes], dtype=float)
    secs = np.array([tm.tick_to_sec(t) for t in ticks])
    feats = []
    lanes = [min((f for f in x.frets if f != OPEN), default=0) for x in notes]
    tops = [max((f for f in x.frets if f != OPEN), default=0) for x in notes]
    sizes = [len(x.frets) for x in notes]
    for i, x in enumerate(notes):
        start, num, den = tm.ts_at(x.tick)
        meas = num * RES * 4 // den
        rel = (x.tick - start) % meas
        pos = (x.tick % RES) / RES
        ioi_p = (ticks[i] - ticks[i - 1]) / RES if i > 0 else 8.0
        ioi_n = (ticks[i + 1] - ticks[i]) / RES if i + 1 < n else 8.0
        sp = secs[i] - secs[i - 1] if i > 0 else 4.0
        sn = secs[i + 1] - secs[i] if i + 1 < n else 4.0
        lo = np.searchsorted(ticks, ticks[i] - RES)
        hi = np.searchsorted(ticks, ticks[i] + RES, side="right")
        lo2 = np.searchsorted(ticks, ticks[i] - 2 * RES)
        hi2 = np.searchsorted(ticks, ticks[i] + 2 * RES, side="right")
        ls = np.searchsorted(secs, secs[i] - 2.0)
        hs = np.searchsorted(secs, secs[i] + 2.0, side="right")
        same_prev = float(i > 0 and notes[i - 1].frets == x.frets)
        same_next = float(i + 1 < n and notes[i + 1].frets == x.frets)
        lane_d = lanes[i] - lanes[i - 1] if i > 0 else 0
        lane_dn = lanes[i + 1] - lanes[i] if i + 1 < n else 0
        win = slice(max(0, i - 8), min(n, i + 9))
        feats.append([
            pos, float(pos == 0), float(abs(pos - 0.5) < 1e-6), float(abs(pos * 4 - round(pos * 4)) < 1e-6),
            float(abs(pos * 3 - round(pos * 3)) < 1e-6 and pos != 0),
            rel / RES, float(rel == 0), float(rel % (2 * RES) == 0),
            min(ioi_p, 8), min(ioi_n, 8), min(sp, 4), min(sn, 4),
            hi - lo, hi2 - lo2, (hs - ls) / 4.0,
            sizes[i], x.sustain / RES, float(OPEN in x.frets),
            lanes[i], tops[i], lane_d, lane_dn, same_prev, same_next,
            float(max(tops[win])), float(min(lanes[win])),
            60.0 / max(tm.tick_to_sec(x.tick + RES) - secs[i], 1e-3),
            float(i == 0 or ioi_p >= 2), float(i + 1 == n or ioi_n >= 2),
        ])
    return np.array(feats, dtype=np.float32)


def _collect(folder: str, log=print):
    from ..chart.chart_io import load_chart
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS, find_songs

    data = {d: {"X": [], "keep": [], "lane": [], "chord": [], "w": [], "g": []} for d in DIFFS}
    rng = np.random.default_rng(0)
    for gi, f in enumerate(find_songs(folder)):
        try:
            ini = read_ini(f / "song.ini")
            chart = load_chart(f)
        except Exception:  # noqa: BLE001
            continue
        official = any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS)
        for key in ("guitar", "rhythm", "bass", "keys"):
            ft = getattr(chart, key)
            exp = ft.notes["expert"]
            if len(exp) < 40 or any(len(ft.notes[d]) < 10 for d in DIFFS):
                continue
            X = note_features(exp, chart.tempo)
            # con bibliotecas grandes no hace falta cada nota: una muestra por pista (memoria)
            sel = np.sort(rng.choice(len(exp), min(len(exp), MAX_PER_TRACK), replace=False))
            for d in DIFFS:
                tgt = {n.tick: n for n in ft.notes[d]}
                keep = np.array([n.tick in tgt for n in exp])
                lane = np.array([min((fr for fr in tgt[n.tick].frets if fr != OPEN), default=0) if n.tick in tgt else -1
                                 for n in exp])
                chord = np.array([len(tgt[n.tick].frets) > 1 if n.tick in tgt else False for n in exp])
                dd = data[d]
                dd["X"].append(X[sel])
                dd["keep"].append(keep[sel])
                dd["lane"].append(lane[sel])
                dd["chord"].append(chord[sel])
                dd["w"].append(np.full(len(sel), 2.0 if official else 1.0))
                dd["g"].append(np.full(len(sel), gi))
        log(f"  {f.name}")
    for d in DIFFS:
        for k in data[d]:
            data[d][k] = np.concatenate(data[d][k]) if data[d][k] else np.zeros(0)
    return data


def _lane_features(X: np.ndarray) -> np.ndarray:
    return X


def train(folder: str, log=print, evaluate: bool = False) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    data = _collect(folder, log)
    models = {}
    for d in DIFFS:
        dd = data[d]
        X, y, w, g = dd["X"], dd["keep"].astype(int), dd["w"], dd["g"]
        if len(X) < 500:
            continue
        if evaluate:
            accs = []
            for tr, te in GroupKFold(n_splits=4).split(X, y, g):
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
                m.fit(X[tr], y[tr], sample_weight=w[tr])
                p = m.predict(X[te])
                tp = ((p == 1) & (y[te] == 1)).sum()
                prec = tp / max((p == 1).sum(), 1)
                rec = tp / max((y[te] == 1).sum(), 1)
                accs.append(2 * prec * rec / max(prec + rec, 1e-9))
            log(f"{d}: F1 de notas conservadas (validacion cruzada por cancion) = {np.mean(accs):.3f}")
        keep_m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
        keep_m.fit(X, y, sample_weight=w)
        sel = dd["lane"] >= 0
        Xl = X[sel]
        yl = np.clip(dd["lane"][sel], 0, MAX_LANE[d])
        if evaluate:
            accs = []
            for tr, te in GroupKFold(n_splits=4).split(Xl, yl, g[sel]):
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
                m.fit(Xl[tr], yl[tr], sample_weight=w[sel][tr])
                accs.append((m.predict(Xl[te]) == yl[te]).mean())
            log(f"{d}: acierto de carril = {np.mean(accs):.3f}")
        lane_m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48)
        lane_m.fit(Xl, yl, sample_weight=w[sel])
        chord_m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=32)
        chord_m.fit(Xl, dd["chord"][sel].astype(int), sample_weight=w[sel])
        models[d] = {"keep": keep_m, "lane": lane_m, "chord": chord_m}
    return models


def save(models: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


_CACHE = None


def load_models():
    global _CACHE
    if _CACHE is None:
        for p in (paths.CONFIG_DIR / "reducer.pkl", MODEL_PATH):
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        _CACHE = pickle.load(f)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if _CACHE is None:
            _CACHE = {}
    return _CACHE


if __name__ == "__main__":
    paths.setup_env()
    folder = sys.argv[1]
    ev = "--eval" in sys.argv
    models = train(folder, log=print if ev else (lambda *_: None), evaluate=ev)
    save(models)
    print("modelo guardado en", MODEL_PATH)
