"""Modelos aprendidos de los charts de referencia a partir del AUDIO (guitarra y bajo):

  * "ataque -> nota": que golpes que se oyen en la pista pone un charter como nota en Experto.
  * "acorde": cuantas notas (1, 2 o 3) pone el charter en ese golpe (solo guitarra).

Se entrenan con canciones que traen la pista del instrumento por separado (guitar.ogg, bass.ogg /
rhythm.ogg) y su chart. Con la biblioteca del usuario tambien se pueden usar canciones sin pistas
separadas, separandolas antes con la IA (mas lento).

Uso: python -m charter.calibration.onset_model <carpeta_de_charts> [--eval]
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from .. import paths
from ..pipeline.transcription import CQT_FMIN_MIDI, StemAnalysis, analyze_stem, detect_onsets

MODEL_PATH = paths.DATA / "onset_model.pkl"
FEATURE_VERSION = 2
# etiquetas v2: el pico de la envolvente de ataque llega ~15 ms despues del golpe real, y las notas del chart
# estan en la rejilla mientras que el musico toca con holgura. Con 50 ms sin corregir, el 15-30 % de las notas
# reales se etiquetaban como "no es nota" y el modelo aprendia a poner pocas
LABEL_VERSION = 2
ONSET_LAG = 0.015
LABEL_TOL = 0.07
DEFAULT_THR = 0.3  # medido contra charts oficiales con el modelo antiguo (etiquetas v1)
CAND_K = 1.3
CAND_GAP = 0.04
PITCH_RANGE = {"bass": (28, 64), "guitar": (40, 88)}
# a partir de cuantas pistas de entrenamiento se fia el generador del modelo de guitarra
MIN_TRACKS_GUITAR = 150


def onset_features(sa: StemAnalysis, cands: np.ndarray, beats: np.ndarray, inst: str) -> np.ndarray:
    lo, hi = PITCH_RANGE[inst]
    lo_b, hi_b = lo - CQT_FMIN_MIDI, hi - CQT_FMIN_MIDI
    C = sa.cqt_db
    env = sa.onset_env
    p99 = np.percentile(env, 99) + 1e-9
    from scipy.ndimage import median_filter

    fps = len(env) / max(sa.times[-1], 1e-3) if len(sa.times) else 86.0
    med = median_filter(env, size=max(3, int(fps)))
    rms_sorted = np.sort(sa.rms_db)
    beats = np.asarray(beats)
    rows = []
    for i, t in enumerate(cands):
        f = min(sa.frame(t), C.shape[1] - 2)
        pre = C[lo_b:hi_b, max(f - 4, 0):max(f - 1, 1)].max(axis=1)
        post = C[lo_b:hi_b, f + 1:f + 6].max(axis=1)
        p = int(np.argmax(post))
        rise = float(post[p] - pre[p])
        prom = float(post[p] - np.median(post))
        pchange = float(p != int(np.argmax(pre)))
        e = float(env[max(f - 1, 0):f + 2].max())
        rms_pre = float(sa.rms_db[max(f - 4, 0):max(f - 1, 1)].max())
        rms_post = float(sa.rms_db[f:f + 5].max()) if f < len(sa.rms_db) else rms_pre
        dtp = t - cands[i - 1] if i > 0 else 2.0
        dtn = cands[i + 1] - t if i + 1 < len(cands) else 2.0
        dens = np.sum(np.abs(cands - t) < 1.0)
        if len(beats) > 1:
            j = int(np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 2))
            period = beats[j + 1] - beats[j]
            ph = (t - beats[j]) / period
            ph = ph - np.floor(ph)
        else:
            period, ph = 0.5, 0.0

        def dist(div):
            x = ph * div
            return abs(x - round(x)) / div

        low_e = float(post[: (hi_b - lo_b) // 2].max())
        high_e = float(post[(hi_b - lo_b) // 2:].max())
        pct = float(np.searchsorted(rms_sorted, rms_post) / max(len(rms_sorted), 1))
        # polifonia: cuantas notas distintas suenan fuerte justo despues del golpe (acorde o nota suelta)
        strong = post >= post[p] - 9.0
        peaks = strong & np.r_[True, post[1:] >= post[:-1]] & np.r_[post[:-1] >= post[1:], True]
        n_peaks = float(peaks.sum())
        pcs = set(((np.where(peaks)[0] + lo) % 12).tolist())
        rising = float(((post - pre) >= 3.0)[strong].sum())
        rows.append([
            e / p99, e / (med[f] + 1e-9), rise, prom, pchange, rms_post - sa.active_db, rms_post - rms_pre,
            min(dtp, 2.0), min(dtn, 2.0), dens, dist(1), dist(2), dist(4), dist(3), dist(6), period, pct,
            low_e - high_e, float(p + lo), n_peaks, float(len(pcs)), rising,
        ])
    return np.array(rows, dtype=np.float32)


def candidates(sa: StemAnalysis) -> np.ndarray:
    return detect_onsets(sa, k=CAND_K, min_gap=CAND_GAP)


def _match(cands: np.ndarray, ref: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Indice de la nota del chart que corresponde a cada ataque (-1 si ninguna)."""
    lab = np.full(len(cands), -1, dtype=np.int64)
    used = np.zeros(len(ref), bool)
    for i, t in enumerate(cands):
        j = np.searchsorted(ref, t)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(ref) and not used[k] and abs(ref[k] - t) <= tol:
                if best is None or abs(ref[k] - t) < abs(ref[best] - t):
                    best = k
        if best is not None:
            used[best] = True
            lab[i] = best
    return lab


def stem_for(folder: Path, inst: str, chart) -> Path | None:
    exts = (".ogg", ".opus", ".mp3", ".wav", ".flac")
    names = ["guitar"] if inst == "guitar" else ["bass"] + ([] if chart.rhythm.has_notes() else ["rhythm"])
    for n in names:
        for e in exts:
            p = folder / f"{n}{e}"
            if p.exists():
                return p
    return None


def song_data(folder: Path, chart, ini: dict, inst: str, stem_audio: np.ndarray | None = None) -> dict | None:
    """Caracteristicas y etiquetas de una cancion para un instrumento (None si no sirve)."""
    from ..audio import decode_to_wav, load
    from ..chart.model import OPEN, RES

    notes = getattr(chart, inst).notes["expert"]
    if len(notes) < 80:
        return None
    if stem_audio is None:
        stem = stem_for(folder, inst, chart)
        if stem is None:
            return None
        paths.WORK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=paths.WORK) as td:
            y, _ = load(decode_to_wav(stem, Path(td) / "a.wav"))
        stem_audio = y.mean(axis=0)
    delay = float(ini.get("delay", "0") or 0) / 1000
    tm = chart.tempo
    beats = np.array([tm.tick_to_sec(t) - delay for t in range(0, chart.end_tick, RES)])
    sa = analyze_stem(stem_audio, 44100)
    cands = candidates(sa)
    if len(cands) < 50:
        return None
    ref_t = np.array([tm.tick_to_sec(n.tick) - delay for n in notes])
    order = np.argsort(ref_t)
    ref_t = ref_t[order]
    sizes = np.array([len([f for f in notes[k].frets if f != OPEN]) for k in order])
    X = onset_features(sa, cands, beats, inst)
    y, chord = make_labels(cands, ref_t, sizes)
    return {"X": X, "y": y, "chord": chord, "n_ref": len(ref_t), "t": cands.astype(np.float32),
            "ref_t": ref_t.astype(np.float32), "sizes": sizes.astype(np.int8), "spec": compact_spectrum(sa)}


SPEC_FPS = 22050 / 256 / 2  # ~43 fps


def compact_spectrum(sa: StemAnalysis) -> dict:
    """Espectrograma compacto de la pista (CQT 84 notas a ~43 fps en 0-255 + envolvente de ataques), para
    entrenar en el futuro un modelo que pase directamente de audio a chart."""
    C = sa.cqt_db
    n = C.shape[1] // 2 * 2
    C2 = np.maximum(C[:, :n:2], C[:, 1:n:2])
    q = np.clip((C2 + 100.0) * 2.55, 0, 255).astype(np.uint8)  # -100..0 dB -> 0..255
    e = sa.onset_env[: n // 2 * 2]
    e2 = np.maximum(e[0::2], e[1::2]) if len(e) >= 2 else e
    e2 = np.clip(e2 / (np.percentile(e2, 99.5) + 1e-9) * 200, 0, 255).astype(np.uint8)
    return {"cqt": q, "onset": e2}


def make_labels(cands: np.ndarray, ref_t: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Etiquetas de cada ataque: si es nota en el chart (y) y de cuantas notas es el acorde (chord)."""
    m = _match(np.asarray(cands, np.float64) - ONSET_LAG, np.asarray(ref_t, np.float64), LABEL_TOL)
    y = (m >= 0).astype(np.int8)
    chord = np.where(m >= 0, np.clip(sizes[np.maximum(m, 0)], 1, 3), 0).astype(np.int8)
    return y, chord


def collect(folder: str, log=print) -> dict:
    from ..chart.chart_io import load_chart
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS, find_songs

    data = {inst: [] for inst in ("guitar", "bass")}
    for gi, f in enumerate(find_songs(folder)):
        try:
            chart = load_chart(f)
            ini = read_ini(f / "song.ini")
        except Exception:  # noqa: BLE001
            continue
        official = any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS)
        for inst in ("guitar", "bass"):
            d = song_data(f, chart, ini, inst)
            if d is None:
                continue
            d["g"] = gi
            d["w"] = 1.5 if official else 1.0
            data[inst].append(d)
            log(f"  {f.name} [{inst}]: {len(d['y'])} ataques, {int(d['y'].sum())} son notas (de {d['n_ref']})")
    return data


def fit(data: dict, log=print, evaluate: bool = False, max_rows: int = 2_500_000) -> dict:
    """Entrena los modelos con una lista de canciones por instrumento (cada una con X, y, chord, g, w)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(0)
    models: dict = {"version": FEATURE_VERSION, "label_version": LABEL_VERSION, "n_tracks": {}, "thr": {}}
    for inst, songs in data.items():
        if not songs:
            continue
        X = np.concatenate([s["X"] for s in songs])
        y = np.concatenate([s["y"] for s in songs]).astype(int)
        ch = np.concatenate([s["chord"] for s in songs]).astype(int)
        g = np.concatenate([np.full(len(s["y"]), s["g"]) for s in songs])
        w = np.concatenate([np.full(len(s["y"]), s["w"]) for s in songs])
        sep = np.concatenate([np.full(len(s["y"]), bool(s.get("sep", False))) for s in songs])
        if len(y) > max_rows:
            sel = rng.choice(len(y), max_rows, replace=False)
            X, y, ch, g, w, sep = X[sel], y[sel], ch[sel], g[sel], w[sel], sep[sel]
        models["n_tracks"][inst] = len(songs)
        models["thr"][inst] = 0.5
        if evaluate and len(set(g)) >= 4:
            # probabilidades fuera de muestra (canciones que el modelo no ha visto) para elegir el umbral
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=4).split(X, y, g):
                m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48)
                m.fit(X[tr], y[tr], sample_weight=w[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            # el umbral se mide en audio separado por la IA si hay bastante (es lo que vera al generar)
            dom = sep if sep.sum() > 20000 else np.ones(len(y), bool)
            best = (0.0, 0.5, 0.0, 0.0)
            for thr in np.arange(0.2, 0.61, 0.025):
                p = oof[dom] >= thr
                tp = (p & (y[dom] == 1)).sum()
                prec, rec = tp / max(p.sum(), 1), tp / max((y[dom] == 1).sum(), 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                if f1 > best[0]:
                    best = (f1, float(thr), prec, rec)
            f1, thr, prec, rec = best
            models["thr"][inst] = thr
            log(f"{inst}: {len(songs)} pistas | ataque->nota P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                f"(umbral {thr:.2f}, medido en {'audio separado por la IA' if dom is sep else 'pistas originales'})")
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48)
        m.fit(X, y, sample_weight=w)
        models[inst] = m
        if inst == "guitar":
            sel = ch > 0
            if sel.sum() > 500:
                if evaluate and len(set(g[sel])) >= 4:
                    accs = []
                    for tr, te in GroupKFold(n_splits=4).split(X[sel], ch[sel], g[sel]):
                        mc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=32)
                        mc.fit(X[sel][tr], ch[sel][tr], sample_weight=w[sel][tr])
                        accs.append((mc.predict(X[sel][te]) == ch[sel][te]).mean())
                    log(f"guitar: acierto del tamano de acorde (1/2/3 notas) = {np.mean(accs):.3f}")
                mc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=32)
                mc.fit(X[sel], ch[sel], sample_weight=w[sel])
                models["chord"] = mc
    return models


def save(models: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


_CACHE = None


def load_models() -> dict:
    """Modelos de la biblioteca del usuario (config/) si existen; si no, los de serie."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for p in (paths.CONFIG_DIR / "onset_model.pkl", MODEL_PATH):
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        m = pickle.load(f)
                    if m.get("version") == FEATURE_VERSION:
                        _CACHE = m
                        break
                except Exception:  # noqa: BLE001
                    continue
    return _CACHE


def threshold(models: dict, inst: str) -> float:
    """Probabilidad minima para que un ataque sea nota (elegida al entrenar; los modelos antiguos no la traen)."""
    return float(models.get("thr", {}).get(inst, DEFAULT_THR))


def guitar_model_trusted(models: dict) -> bool:
    return "guitar" in models and models.get("n_tracks", {}).get("guitar", 0) >= MIN_TRACKS_GUITAR


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    ev = "--eval" in sys.argv
    models = fit(collect(sys.argv[1], log=print), log=print, evaluate=ev)
    save(models)
    print("modelo guardado en", MODEL_PATH)
