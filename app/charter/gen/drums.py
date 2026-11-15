"""Generacion de bateria Pro (4 carriles con platillos y toms) a partir de la bateria separada por piezas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chart.model import DIFFICULTIES, RES, DrumNote, DrumTrack, Phrase, TempoMap
from ..pipeline.transcription import Note
from .common import metric_strength, quantize_ticks, secs_between, stat

SR = 22050
HOP = 128  # ~5.8 ms: los golpes de bateria necesitan precision

# Clases internas
KICK, SNARE, TOM_HI, TOM_MID, TOM_FLOOR, HIHAT, RIDE, CRASH = range(8)
PIECE_CLASS = {"kick": KICK, "snare": SNARE, "hh": HIHAT, "ride": RIDE, "crash": CRASH}
# clase -> (pad, platillo)
PAD_OF = {
    KICK: (0, False), SNARE: (1, False), TOM_HI: (2, False), TOM_MID: (3, False), TOM_FLOOR: (4, False),
    HIHAT: (2, True), RIDE: (3, True), CRASH: (4, True),
}
GM_CLASS = {35: KICK, 36: KICK, 37: SNARE, 38: SNARE, 39: SNARE, 40: SNARE, 41: TOM_FLOOR, 43: TOM_FLOOR,
            45: TOM_MID, 47: TOM_MID, 48: TOM_HI, 50: TOM_HI, 42: HIHAT, 44: HIHAT, 46: HIHAT,
            49: CRASH, 52: CRASH, 55: CRASH, 57: CRASH, 51: RIDE, 53: RIDE, 59: RIDE}


@dataclass
class Hit:
    time: float
    cls: int
    strength: float  # 0..1+ relativo a los golpes tipicos de esa pieza
    conf: float = 1.0
    tick: int = 0


def _onsets_for_piece(y: np.ndarray, piece: str) -> list[tuple[float, float, float]]:
    """Devuelve [(tiempo, fuerza_relativa, nivel_db)] de una pieza de la bateria."""
    import librosa

    if np.max(np.abs(y)) < 1e-4:
        return []
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, lag=1, max_size=3)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    n = min(len(env), len(rms_db))
    env, rms_db = env[:n], rms_db[:n]
    norm = np.percentile(env[env > 0], 99) if np.any(env > 0) else 1.0
    envn = env / (norm + 1e-9)
    wait = {"kick": 8, "snare": 8, "toms": 8, "hh": 6, "ride": 8, "crash": 15}.get(piece, 8)
    delta = {"kick": 0.10, "snare": 0.10, "toms": 0.12, "hh": 0.07, "ride": 0.09, "crash": 0.12}.get(piece, 0.1)
    peaks = librosa.util.peak_pick(envn, pre_max=3, post_max=3, pre_avg=12, post_avg=12, delta=delta, wait=wait)
    if len(peaks) == 0:
        return []
    levels = np.array([np.max(rms_db[p: p + 6]) for p in peaks])
    typical = np.percentile(levels, 75)
    out = []
    for p, lv in zip(peaks, levels):
        rel = lv - typical
        if rel < {"kick": -16, "snare": -18, "toms": -14, "hh": -20, "ride": -16, "crash": -14}.get(piece, -16):
            continue  # golpe muy flojo respecto a esa pieza -> fuga de otra pieza
        out.append((p * HOP / SR, float(envn[p]), float(lv)))
    return out


def _split_toms(y: np.ndarray, times: list[float]) -> list[int]:
    """Clasifica cada golpe de tom en agudo / medio / base segun su tono."""
    import librosa

    if not times:
        return []
    cents = []
    for t in times:
        a = int(t * SR)
        seg = y[a: a + int(0.08 * SR)]
        if len(seg) < 256:
            cents.append(0.0)
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / SR)
        band = (freqs > 50) & (freqs < 600)
        if not np.any(band) or spec[band].sum() == 0:
            cents.append(0.0)
            continue
        # frecuencia dominante en la banda grave (el "tono" del tom)
        cents.append(float(freqs[band][np.argmax(spec[band])]))
    c = np.log(np.array(cents) + 1.0)
    if len(c) < 6 or np.ptp(c) < 0.15:
        return [TOM_MID] * len(times) if len(c) else []
    # k-means 1D con k=3 (o 2 si hay poca variedad)
    k = 3 if np.ptp(c) > 0.45 and len(c) >= 12 else 2
    centers = np.percentile(c, np.linspace(10, 90, k))
    for _ in range(30):
        lab = np.argmin(np.abs(c[:, None] - centers[None, :]), axis=1)
        for j in range(k):
            if np.any(lab == j):
                centers[j] = c[lab == j].mean()
    order = np.argsort(centers)  # de grave a agudo
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    out = []
    for l in lab:
        r = rank[l]
        if k == 3:
            out.append([TOM_FLOOR, TOM_MID, TOM_HI][r])
        else:
            out.append([TOM_FLOOR, TOM_HI][r])
    return out


def detect_hits(pieces: dict[str, np.ndarray], mt3_drums: list[Note], log=print) -> list[Hit]:
    """Golpes de todas las piezas, validados entre si y con la transcripcion de YourMT3."""
    hits: list[Hit] = []
    for piece, y in pieces.items():
        ons = _onsets_for_piece(y, piece)
        if piece == "toms":
            classes = _split_toms(y, [t for t, _, _ in ons])
        else:
            classes = [PIECE_CLASS[piece]] * len(ons)
        for (t, s, lv), c in zip(ons, classes):
            hits.append(Hit(t, c, s))
    hits.sort(key=lambda h: h.time)
    # apoyo de YourMT3 (votacion): si el modelo tambien oye esa pieza cerca, mas confianza
    mt = sorted(((n.onset, GM_CLASS.get(n.pitch, -1)) for n in mt3_drums), key=lambda x: x[0])
    mt_t = np.array([x[0] for x in mt]) if mt else np.zeros(0)
    fam = {KICK: "k", SNARE: "s", TOM_HI: "t", TOM_MID: "t", TOM_FLOOR: "t", HIHAT: "c", RIDE: "c", CRASH: "c"}
    for h in hits:
        if len(mt_t) == 0:
            h.conf = 0.7
            continue
        j = np.searchsorted(mt_t, h.time)
        sup = False
        for k in range(max(j - 3, 0), min(j + 3, len(mt))):
            if abs(mt[k][0] - h.time) <= 0.05 and mt[k][1] >= 0 and fam[mt[k][1]] == fam[h.cls]:
                sup = True
        h.conf = 1.0 if sup else 0.55
    # quitar golpes debiles sin apoyo
    hits = [h for h in hits if h.conf >= 1.0 or h.strength >= 0.3]
    log(f"  bateria: {len(hits)} golpes detectados")
    return hits


def _limit_hands(by_tick: dict[int, list[Hit]]) -> None:
    """Como mucho 2 golpes de manos a la vez (+ bombo)."""
    prio = {CRASH: 5, SNARE: 4, RIDE: 3, TOM_HI: 3, TOM_MID: 3, TOM_FLOOR: 3, HIHAT: 2}
    for t, hs in by_tick.items():
        kicks = [h for h in hs if h.cls == KICK][:1]
        hands = {}
        for h in hs:
            if h.cls == KICK:
                continue
            pad = PAD_OF[h.cls][0]
            # mismo pad: platillo y tom en el mismo carril -> nos quedamos con el mas fuerte
            if pad not in hands or h.strength > hands[pad].strength:
                hands[pad] = h
        hand_list = sorted(hands.values(), key=lambda h: (-prio.get(h.cls, 1), -h.strength))[:2]
        by_tick[t] = kicks + hand_list


def generate_drum_track(hits: list[Hit], tm: TempoMap, time_offset: float, log=print) -> tuple[DrumTrack, dict]:
    dt = DrumTrack()
    if len(hits) < 30:
        return dt, {"hits": len(hits)}
    for h in hits:
        h.time += time_offset
    ticks = quantize_ticks(np.array([h.time for h in hits]), tm, allow_32nd=True)
    by_tick: dict[int, list[Hit]] = {}
    for h, t in zip(hits, ticks):
        h.tick = int(t)
        lst = by_tick.setdefault(h.tick, [])
        if not any(x.cls == h.cls for x in lst):
            lst.append(h)
    _limit_hands(by_tick)
    # velocidad maxima por pieza
    expert: list[DrumNote] = []
    last_by_pad: dict[tuple[int, bool], int] = {}
    kick_ticks = []
    for t in sorted(by_tick):
        for h in by_tick[t]:
            pad, cym = PAD_OF[h.cls]
            k = (pad, cym)
            if k in last_by_pad and secs_between(tm, last_by_pad[k], t) < 0.06:
                continue
            last_by_pad[k] = t
            if pad == 0:
                kick_ticks.append(t)
            expert.append(DrumNote(t, pad, cymbal=cym, conf=h.conf))
    # doble bombo: en rafagas rapidas, los bombos alternos pasan a Expert+
    kick_set = sorted(kick_ticks)
    double = set()
    for a, b in zip(kick_set, kick_set[1:]):
        if secs_between(tm, a, b) < 0.13 and a not in double:
            double.add(b)
    final_expert = []
    for n in expert:
        if n.pad == 0 and n.tick in double:
            final_expert.append(DrumNote(n.tick, 0, double_kick=True))
        else:
            final_expert.append(n)
    # densidad de experto: como los oficiales
    dur = max(secs_between(tm, final_expert[0].tick, final_expert[-1].tick), 1.0)
    target = stat("drums.expert.nps_p90", 9.9)
    normal = [n for n in final_expert if not n.double_kick]
    if len(normal) / dur > target * 1.15:
        # quitar charles debiles fuera de tiempo
        scored = sorted(normal, key=lambda n: (metric_strength(n.tick, tm) + (0.5 if n.pad in (0, 1) else 0)
                                               + (0.4 if n.pad == 4 and n.cymbal else 0)))
        n_drop = int(len(normal) - target * 1.15 * dur)
        drop = set(id(n) for n in scored[:n_drop] if n.pad == 2 and n.cymbal)
        final_expert = [n for n in final_expert if id(n) not in drop]
        log(f"  bateria: densidad ajustada ({len(drop)} charles quitados)")
    dt.notes["expert"] = final_expert
    # el modelo aprendido parte siempre del Experto (como en los charts oficiales)
    dt.notes["hard"] = _reduce_drums(final_expert, tm, "hard")
    dt.notes["medium"] = _reduce_drums(final_expert, tm, "medium")
    dt.notes["easy"] = _reduce_drums(final_expert, tm, "easy")
    dt.fills = _make_fills(final_expert, tm)
    info = {"hits": len(final_expert), "double_kick": sum(n.double_kick for n in final_expert)}
    return dt, info


def _reduce_drums_learned(expert: list[DrumNote], tm: TempoMap, diff: str) -> list[DrumNote] | None:
    from ..calibration.drum_reducer_model import drum_features, load_models

    m = load_models().get(diff)
    if m is None:
        return None
    X, notes = drum_features(expert, tm)
    if len(notes) == 0:
        return []
    p = m.predict_proba(X)[:, 1]
    thr = {"hard": 0.5, "medium": 0.45, "easy": 0.4}[diff]
    by_tick: dict[int, list[tuple[float, DrumNote]]] = {}
    for prob, n in zip(p, notes):
        if prob >= thr:
            by_tick.setdefault(n.tick, []).append((prob, n))
    out = []
    max_hands = 1 if diff == "easy" else 2
    for t in sorted(by_tick):
        group = by_tick[t]
        kick = [x for x in group if x[1].pad == 0]
        hands = sorted([x for x in group if x[1].pad != 0], key=lambda x: -x[0])[:max_hands]
        for prob, n in kick[:1] + hands:
            out.append(DrumNote(n.tick, n.pad, n.cymbal, False, n.review, n.conf))
    return out


def _reduce_drums(src: list[DrumNote], tm: TempoMap, diff: str) -> list[DrumNote]:
    learned = _reduce_drums_learned(src, tm, diff)
    if learned is not None:
        return learned
    notes = [DrumNote(n.tick, n.pad, n.cymbal, False, n.review, n.conf) for n in src if not n.double_kick]
    by_tick: dict[int, list[DrumNote]] = {}
    for n in notes:
        by_tick.setdefault(n.tick, []).append(n)
    out: list[DrumNote] = []
    ticks = sorted(by_tick)
    if diff == "hard":
        # sin rafagas imposibles: bombo a como mucho corcheas rapidas, manos a semicorcheas
        last = {}
        for t in ticks:
            for n in by_tick[t]:
                k = (n.pad, n.cymbal)
                min_gap = (0.16 if n.pad == 0 else 0.11)
                if k in last and secs_between(tm, last[k], t) < min_gap:
                    continue
                last[k] = t
                out.append(n)
        return out
    if diff == "medium":
        last_any = -10 ** 9
        for t in ticks:
            group = by_tick[t]
            strong = metric_strength(t, tm)
            gap_ok = secs_between(tm, last_any, t) >= 0.17 if last_any > -10 ** 8 else True
            if not gap_ok:
                continue
            hands = [n for n in group if n.pad != 0]
            kick = [n for n in group if n.pad == 0]
            # bombo solo en tiempos fuertes o si no hay manos
            if kick and (strong >= 0.75 or not hands):
                out.append(kick[0])
            if hands:
                # como mucho 1 mano salvo platillo + caja en tiempo fuerte
                hands.sort(key=lambda n: (-(n.pad == 4 and n.cymbal), -(n.pad == 1)))
                out.append(hands[0])
                if len(hands) > 1 and strong >= 0.85:
                    out.append(hands[1])
            last_any = t
        return out
    # easy
    last_any = -10 ** 9
    for t in ticks:
        group = by_tick[t]
        strong = metric_strength(t, tm)
        if last_any > -10 ** 8 and secs_between(tm, last_any, t) < 0.33:
            continue
        if strong < 0.5:
            continue
        hands = [n for n in group if n.pad != 0]
        kick = [n for n in group if n.pad == 0]
        if hands:
            hands.sort(key=lambda n: (-(n.pad == 4 and n.cymbal), -(n.pad == 1)))
            out.append(hands[0])
            if kick and strong >= 1.0:
                out.append(kick[0])
        elif kick:
            out.append(kick[0])
        last_any = t
    return out


def _make_fills(notes: list[DrumNote], tm: TempoMap) -> list[Phrase]:
    """Zonas de "fill" para activar Star Power: el compas antes de un crash fuerte, cada ~8 compases."""
    crashes = [n.tick for n in notes if n.pad == 4 and n.cymbal]
    measure = RES * tm.ts_at(0)[1] * 4 // tm.ts_at(0)[2]
    fills = []
    last_end = -10 ** 9
    first = notes[0].tick if notes else 0
    for c in crashes:
        if c % measure != 0:
            continue
        start = c - measure
        if start < first + measure * 4 or start - last_end < measure * 8:
            continue
        fills.append(Phrase(start, c + 1))
        last_end = c
    return fills
