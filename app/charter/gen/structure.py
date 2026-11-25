"""Estructura de la cancion: secciones (para el modo practica), solos y Star Power."""

from __future__ import annotations

import numpy as np

from ..chart.model import DIFFICULTIES, RES, FretTrack, Phrase, TempoMap
from ..i18n import t as tr
from .common import stat


def activity(stem_mono: np.ndarray, sr: int, hop_s: float = 0.1, rel_db: float = -20.0) -> tuple[np.ndarray, np.ndarray]:
    """Actividad (bool) de una pista cada `hop_s` segundos."""
    hop = int(sr * hop_s)
    n = len(stem_mono) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0, bool)
    frames = stem_mono[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    ref = np.percentile(db, 90)
    act = db > ref + rel_db
    from scipy.ndimage import binary_closing, binary_opening

    act = binary_closing(act, np.ones(5))
    act = binary_opening(act, np.ones(3))
    return np.arange(n) * hop_s, act


def find_sections(mix_mono: np.ndarray, sr: int, tm: TempoMap, time_offset: float, vocal_act, end_tick: int,
                  log=print) -> list[tuple[int, str]]:
    """Segmentacion por compases con novedad armonica/timbrica y etiquetado por repeticion y voz."""
    import librosa

    y = librosa.resample(mix_mono, orig_sr=sr, target_sr=22050)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=22050, hop_length=hop, n_mfcc=13)
    frame_t = np.arange(chroma.shape[1]) * hop / 22050 + time_offset
    measures = [m for m in tm.measure_ticks(end_tick)]
    m_secs = np.array([tm.tick_to_sec(m) for m in measures])
    feats = []
    for i in range(len(measures)):
        a = m_secs[i]
        b = m_secs[i + 1] if i + 1 < len(m_secs) else a + 2.0
        sel = (frame_t >= a) & (frame_t < b)
        if not np.any(sel):
            feats.append(np.zeros(12 + 13))
            continue
        c = chroma[:, sel].mean(axis=1)
        m = mfcc[:, sel].mean(axis=1)
        feats.append(np.concatenate([c / (np.linalg.norm(c) + 1e-9), m / 50.0]))
    F = np.array(feats)
    if len(F) < 8:
        return [(measures[0] if measures else 0, "Cancion")]
    # novedad: distancia entre ventanas de 4 compases antes/despues
    w = 4
    nov = np.zeros(len(F))
    for i in range(w, len(F) - w):
        nov[i] = np.linalg.norm(F[i - w:i].mean(axis=0) - F[i:i + w].mean(axis=0))
    # limites: picos de novedad, cada 4 compases como minimo, preferiendo multiplos de 4
    cand = [i for i in range(w, len(F) - w) if nov[i] >= nov[max(i - 2, 0):i + 3].max() and nov[i] > np.percentile(nov, 60)]
    bounds = [0]
    for i in sorted(cand, key=lambda i: -nov[i]):
        if all(abs(i - b) >= 4 for b in bounds):
            bounds.append(i)
    bounds = sorted(bounds)
    # ajustar al multiplo de 4 compases mas cercano si esta a 1 compas
    adj = []
    for b in bounds:
        r = round(b / 4) * 4
        adj.append(r if abs(r - b) <= 1 else b)
    bounds = sorted(set(adj))
    segs = [(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else len(F)) for i in range(len(bounds))]
    segs = [s for s in segs if s[1] > s[0]]
    # etiquetado: agrupar segmentos parecidos
    seg_feat = np.array([F[a:b].mean(axis=0) for a, b in segs])
    labels = -np.ones(len(segs), int)
    cur = 0
    for i in range(len(segs)):
        if labels[i] >= 0:
            continue
        labels[i] = cur
        for j in range(i + 1, len(segs)):
            if labels[j] < 0 and np.linalg.norm(seg_feat[i] - seg_feat[j]) < 0.25:
                labels[j] = cur
        cur += 1
    vt, va = vocal_act
    def vocal_frac(a, b):
        ta = tm.tick_to_sec(measures[a]) - time_offset
        tb = (tm.tick_to_sec(measures[b]) if b < len(measures) else ta + 8) - time_offset
        sel = (vt >= ta) & (vt < tb)
        return float(va[sel].mean()) if np.any(sel) else 0.0
    vfr = [vocal_frac(a, b) for a, b in segs]
    counts = {l: int((labels == l).sum()) for l in set(labels)}
    vocal_labels = [l for l in counts if np.mean([vfr[i] for i in range(len(segs)) if labels[i] == l]) > 0.35]
    chorus = max(vocal_labels, key=lambda l: (counts[l], np.mean([seg_feat[i][12:].mean() for i in range(len(segs)) if labels[i] == l])), default=None)
    names = []
    n_by = {}
    for i, (a, b) in enumerate(segs):
        lab = labels[i]
        if i == 0 and vfr[i] < 0.3:
            name = "Intro"
        elif i == len(segs) - 1 and vfr[i] < 0.3:
            name = "Final"
        elif lab == chorus:
            name = "Estribillo"
        elif vfr[i] > 0.35:
            name = "Estrofa" if counts[lab] > 1 or chorus is None else "Puente"
        else:
            name = "Instrumental"
        n_by[name] = n_by.get(name, 0) + 1
        names.append(name)
    totals = {n: names.count(n) for n in set(names)}
    out = []
    seen = {}
    for (a, b), name in zip(segs, names):
        seen[name] = seen.get(name, 0) + 1
        label = f"{tr(name)} {seen[name]}" if totals[name] > 1 else tr(name)
        out.append((measures[a], label))
    log(f"  secciones: {', '.join(n for _, n in out)}")
    return out


def find_solos(lead_notes_times: list[float], guitar_act, vocal_act, min_len: float = 8.0) -> list[tuple[float, float]]:
    """Solos: zonas sin voz donde la guitarra toca rapido y protagonista."""
    gt, ga = guitar_act
    vt, va = vocal_act
    if len(gt) == 0 or len(lead_notes_times) == 0:
        return []
    times = np.array(lead_notes_times)
    hop = gt[1] - gt[0] if len(gt) > 1 else 0.1
    n = len(gt)
    dens = np.zeros(n)
    for i, t in enumerate(gt):
        dens[i] = np.sum((times >= t - 2) & (times < t + 2)) / 4.0
    no_voc = np.ones(n, bool)
    m = min(len(va), n)
    no_voc[:m] = ~va[:m]
    med = np.median(dens[ga[:n]] if np.any(ga[:n]) else dens)
    cand = no_voc & ga[:n] & (dens > max(med * 1.3, 3.0))
    out = []
    i = 0
    while i < n:
        if cand[i]:
            j = i
            while j < n and (cand[j] or (j + 10 < n and cand[j:j + 10].any())):
                j += 1
            if (j - i) * hop >= min_len:
                out.append((gt[i], gt[min(j, n - 1)]))
            i = j
        else:
            i += 1
    return out


def add_star_power(ft: FretTrack, tm: TempoMap, key: str, avoid: list[Phrase] | None = None) -> None:
    """Frases de Star Power repartidas por la cancion (~2.8 por minuto, como los oficiales)."""
    exp = ft.notes["expert"]
    if len(exp) < 16:
        return
    per_min = stat(f"{key}.sp_per_min", 2.8)
    t0, t1 = tm.tick_to_sec(exp[0].tick), tm.tick_to_sec(exp[-1].tick)
    n_sp = max(2, int(round((t1 - t0) / 60 * per_min)))
    measures = tm.measure_ticks(exp[-1].tick + RES * 8)
    # puntuacion de cada compas como candidato (2 compases)
    cand = []
    easy_ticks = np.array([n.tick for n in ft.notes["easy"]])
    exp_ticks = np.array([n.tick for n in exp])
    for i in range(len(measures) - 2):
        a, b = measures[i], measures[i + 2]
        if avoid and any(p.start < b and a < p.end for p in avoid):
            continue
        ne = int(((exp_ticks >= a) & (exp_ticks < b)).sum())
        nz = int(((easy_ticks >= a) & (easy_ticks < b)).sum())
        if ne >= 4 and nz >= 2:
            cand.append((i, a, b, ne))
    if not cand:
        return
    spacing = (t1 - t0) / n_sp
    chosen = []
    target = t0 + spacing * 0.6
    for _ in range(n_sp):
        best = min(cand, key=lambda c: abs(tm.tick_to_sec(c[1]) - target) - 0.02 * c[3])
        if chosen and tm.tick_to_sec(best[1]) - tm.tick_to_sec(chosen[-1][2]) < spacing * 0.4:
            target += spacing
            continue
        chosen.append(best)
        target = tm.tick_to_sec(best[1]) + spacing
        if target > t1:
            break
    for i, a, b, _ in chosen:
        inside = [n for n in exp if a <= n.tick < b]
        start = inside[0].tick
        last = inside[-1]
        end = last.tick + max(last.sustain, RES // 4)
        ft.star_power.append(Phrase(start, end))
