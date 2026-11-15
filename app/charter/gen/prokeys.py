"""Pro Keys (teclado real de 25 teclas, C3-C5) a partir de las notas del teclado."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chart.model import DIFFICULTIES, RES, FretTrack, ProKeyNote, ProKeysTrack

RANGES = {0: (48, 64), 2: (50, 65), 4: (52, 67), 5: (53, 69), 7: (55, 71), 9: (57, 72)}
MAX_CHORD = {"expert": 4, "hard": 3, "medium": 2, "easy": 1}


def _best_range(pitches: list[int]) -> int:
    best, best_n = 0, -1
    for r, (lo, hi) in RANGES.items():
        n = sum(lo <= p <= hi for p in pitches)
        if n > best_n:
            best, best_n = r, n
    return best


def _fold(p: int, lo: int, hi: int) -> int:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p if lo <= p <= hi else max(lo, min(hi, p))


def generate_pro_keys_neural(keys5: FretTrack, sa, tm, time_offset: float, canon: np.ndarray | None, beats: np.ndarray) -> ProKeysTrack:
    """Pro Keys con su propio ritmo y sus propias teclas, tal como las pondria un charter (modelo neural_keys)."""
    from ..calibration.neural_keys import FPS, _inputs, chord_tones, hold_end, keys_at, load_models, peak_times, peaks, predict_mean
    from ..calibration.onset_model import compact_spectrum
    from .common import quantize_ticks
    from .frets import Event, _assign_sustains
    from .repeats import average_frames, choose_sources, clip_sustains, copy_beats

    pk = ProKeysTrack()
    nets = load_models()
    info = nets[0][1]
    spec = compact_spectrum(sa)
    x = _inputs(spec["cqt"], spec["onset"])
    p = predict_mean(nets, x)
    on, keys, hold = p["pro_on"], p["keys"], p["hold"][:, 0]
    chord = chord_tones(x)
    if canon is not None:
        on, keys = average_frames([on, keys], FPS, beats, canon)
    frames = peaks(on[:, 0], info["thr_pro"])
    times = peak_times(on[:, 0], frames) + time_offset
    events: list[Event] = []
    for f, t, tick in zip(frames, times, quantize_ticks(times, tm)):
        e = Event(float(t), keys_at(keys, int(f), info, chord), hold_end(hold, int(f)) + time_offset, 1.0, 1.0,
                  extra={"diff": on[max(f - 1, 0): f + 2, 1:].max(axis=0)})
        e.tick = int(tick)
        if events and e.tick - events[-1].tick < RES // 8:  # el mismo golpe partido en dos
            events[-1].pitches = sorted(set(events[-1].pitches + e.pitches))
            continue
        events.append(e)
    if len(events) < 8:
        return pk
    for a, b in zip(events, events[1:]):
        a.end = min(a.end, b.time)
    _assign_sustains(events, tm, "keys")
    for d in DIFFICULTIES:
        k = DIFFICULTIES_ORDER.index(d)
        thr = info["thr_pro_diff"][d]
        chosen = [e for e in events if k == 0 or e.extra["diff"][k - 1] >= thr]
        groups = [_Chord(e.tick, tuple(_limit_chord(e.pitches, MAX_CHORD[d])), e.sus_ticks) for e in chosen]
        if canon is not None:
            groups = copy_beats(groups, choose_sources(groups, canon, lambda g: g.pitches))
            clip_sustains(groups)
        out, ranges = _place_on_keyboard(groups)
        pk.notes[d] = out
        pk.ranges[d] = ranges or [(0, 0)]
    pk.star_power = list(keys5.star_power)
    pk.solos = list(keys5.solos)
    return pk


DIFFICULTIES_ORDER = ("expert", "hard", "medium", "easy")


@dataclass
class _Chord:
    """Todas las teclas de un golpe: las repeticiones se copian golpe a golpe, no tecla a tecla."""

    tick: int
    pitches: tuple[int, ...]
    sustain: int = 0


def _limit_chord(pitches: list[int], k: int) -> list[int]:
    """Acordes por dificultad: nos quedamos con extremos (melodia arriba, bajo abajo)."""
    p = sorted(set(pitches))
    if len(p) <= k:
        return p
    return [p[-1]] if k == 1 else sorted(set([p[0], p[-1]] + p[1:-1][: k - 2]))


def _place_on_keyboard(groups: list[_Chord]) -> tuple[list[ProKeyNote], list[tuple[int, int]]]:
    """Reparte el teclado visible por bloques de 4 compases; lo que no quepa se mueve de octava."""
    block = RES * 16
    per_block: dict[int, list[int]] = {}
    for g in groups:
        per_block.setdefault(g.tick // block * block, []).extend(g.pitches)
    block_range = {b: _best_range(ps) for b, ps in per_block.items()}
    out: list[ProKeyNote] = []
    ranges: list[tuple[int, int]] = []
    cur = None
    for g in groups:
        r = block_range[g.tick // block * block]
        if r != cur:
            # el cambio de rango va un poco antes de la nota (aviso visual)
            start = max(g.tick // block * block - RES, ranges[-1][0] + 1 if ranges else 0, 0)
            ranges.append((min(start, g.tick), r))
            cur = r
        lo, hi = RANGES[r]
        for p in sorted(set(_fold(x, lo, hi) for x in g.pitches)):
            out.append(ProKeyNote(g.tick, p, g.sustain))
    return out, ranges


def generate_pro_keys(keys5: FretTrack, event_pitches: dict[int, list[int]]) -> ProKeysTrack:
    pk = ProKeysTrack()
    if not keys5.has_notes() or not event_pitches:
        return pk
    all_p = [p for ps in event_pitches.values() for p in ps]
    if not all_p:
        return pk
    # transposicion global por octavas para centrar la cancion en C3-C5
    med = np.median(all_p)
    shift = int(round((60 - med) / 12)) * 12
    for d in DIFFICULTIES:
        notes = keys5.notes[d]
        if not notes:
            continue
        # cambios de rango por frases de 4 compases
        out: list[ProKeyNote] = []
        ranges: list[tuple[int, int]] = []
        block = RES * 16
        block_range: dict[int, int] = {}
        for t, pp in event_pitches.items():
            block_range.setdefault((t // block) * block, [])  # type: ignore[arg-type]
            block_range[(t // block) * block] += [p + shift for p in pp]  # type: ignore[operator]
        block_range = {b: _best_range(ps) for b, ps in block_range.items()}  # type: ignore[arg-type]
        cur_range = None
        for n in notes:
            ps = sorted(set(p + shift for p in event_pitches.get(n.tick, [])))
            if not ps:
                continue
            r = block_range.get((n.tick // block) * block, _best_range(ps))
            if r != cur_range:
                # el cambio de rango va un poco antes de la nota (aviso visual)
                start = max((n.tick // block) * block - RES, ranges[-1][0] + 1 if ranges else 0, 0)
                ranges.append((min(start, n.tick), r))
                cur_range = r
            lo, hi = RANGES[cur_range]
            folded = sorted(set(_fold(p, lo, hi) for p in ps))
            # acordes por dificultad: nos quedamos con extremos (melodia arriba, bajo abajo)
            k = MAX_CHORD[d]
            if len(folded) > k:
                if k == 1:
                    folded = [folded[-1]]
                else:
                    keep = [folded[0], folded[-1]] + folded[1:-1][: k - 2]
                    folded = sorted(set(keep))
            for p in folded:
                out.append(ProKeyNote(n.tick, p, n.sustain, review=n.review))
        pk.notes[d] = out
        pk.ranges[d] = ranges or [(0, 0)]
    pk.star_power = list(keys5.star_power)
    pk.solos = list(keys5.solos)
    return pk
