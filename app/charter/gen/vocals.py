"""Generacion de la pista de voz (YARG): notas con tono, silabas, deslizamientos y notas habladas."""

from __future__ import annotations

import re

import numpy as np

from ..chart.model import RES, Phrase, TempoMap, VocalNote, VocalTrack
from ..pipeline.lyrics import Lyrics, Word, shown_words

PITCH_HOP = 0.01  # 10 ms
# vocales largas ("salvacioooon"): el alineador suele cerrar la palabra antes de que acabe la nota sostenida. Si tras
# la palabra la voz sigue con tono y no empieza otra palabra, la palabra se alarga hasta donde se apaga la voz
HOLD_EXTEND = True
HOLD_MAX_S = 4.0


def track_pitch(vocals_mono: np.ndarray, sr: int, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tono de la voz con CREPE (modelo 'full'). Devuelve (tiempos, midi, periodicidad)."""
    import librosa
    import torch
    import torchcrepe

    y = librosa.resample(vocals_mono, orig_sr=sr, target_sr=16000).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.from_numpy(y)[None]
    hop = int(16000 * PITCH_HOP)
    f0, per = torchcrepe.predict(audio, 16000, hop, 50.0, 1100.0, "full", batch_size=1024, device=dev,
                                 return_periodicity=True, decoder=torchcrepe.decode.viterbi)
    f0 = f0[0].cpu().numpy()
    per = per[0].cpu().numpy()
    per = torchcrepe.filter.median(torch.from_numpy(per)[None], 3)[0].numpy()
    # silencio: sin energia
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, per, rms = f0[:n], per[:n], rms[:n]
    loud = 20 * np.log10(rms + 1e-9)
    thr = np.percentile(loud[loud > -80], 30) if np.any(loud > -80) else -60
    per = np.where(loud < thr - 6, 0.0, per)
    midi = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
    times = np.arange(n) * PITCH_HOP
    return times, midi, per


def syllabify(word: str, language: str) -> list[str]:
    """Divide una palabra en silabas (diccionarios de separacion de Pyphen)."""
    import pyphen

    lang = {"es": "es_ES", "en": "en_US", "pt": "pt_PT", "fr": "fr_FR", "it": "it_IT", "de": "de_DE"}.get(language, "es_ES")
    try:
        dic = pyphen.Pyphen(lang=lang, left=1, right=1)
    except Exception:  # noqa: BLE001
        return [word]
    parts = dic.inserted(word).split("-")
    parts = [p for p in parts if p]
    # una "silaba" sin vocal se une a la siguiente si abre la palabra (la t de "two", la y de "you")
    # y a la anterior en otro caso (la s de "pills")
    while len(parts) > 1 and not re.search(r"[aeiouáéíóúüàèìòùâêîôûäëïö]", parts[0].lower()):
        parts[:2] = [parts[0] + parts[1]]
    merged: list[str] = []
    for p in parts:
        if merged and not re.search(r"[aeiouyáéíóúüàèìòùâêîôûäëïöÿ]", p.lower()):
            merged[-1] += p
        else:
            merged.append(p)
    return merged or [word]


def _tuning_offset(midi, per) -> float:
    """Cuanto se aparta la cancion de la afinacion estandar (en semitonos): si esta 35 centesimas alta,
    redondear cada nota por separado la deja un semitono arriba o abajo segun el momento."""
    frac = midi[per >= 0.6] - np.round(midi[per >= 0.6])
    return float(np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi)) if len(frac) else 0.0


def _segment_pitch(times, midi, per, t0, t1, voiced_thr=0.45):
    """Partes estables de tono dentro de [t0, t1]: [(inicio, fin, nota_midi)]."""
    i0, i1 = int(t0 / PITCH_HOP), int(t1 / PITCH_HOP) + 1
    seg_m = midi[i0:i1]
    seg_p = per[i0:i1]
    if len(seg_m) == 0:
        return [], 0.0
    voiced = seg_p >= voiced_thr
    vfrac = float(voiced.mean())
    if voiced.sum() < 3:
        return [], vfrac
    notes = np.round(seg_m)
    out = []
    cur_start = None
    cur_vals = []
    for k in range(len(notes)):
        if not voiced[k]:
            continue
        val = seg_m[k]
        if cur_start is None:
            cur_start, cur_vals = k, [val]
            continue
        med = np.median(cur_vals)
        if abs(val - med) > 1.2 and len(cur_vals) >= 6:
            # comprobar que el cambio se mantiene (no es un vibrato)
            look = seg_m[k:k + 6][voiced[k:k + 6]]
            if len(look) >= 4 and abs(np.median(look) - med) > 1.2:
                out.append((cur_start, k, int(round(med))))
                cur_start, cur_vals = k, [val]
                continue
        cur_vals.append(val)
    if cur_start is not None:
        out.append((cur_start, len(notes), int(round(np.median(cur_vals)))))
    res = [(t0 + a * PITCH_HOP, t0 + b * PITCH_HOP, p) for a, b, p in out if (b - a) * PITCH_HOP >= 0.06]
    return res, vfrac


def _held_end(per, end: float, limit: float, voiced_thr: float = 0.45) -> float:
    """Hasta donde sigue sonando con tono la voz despues de `end` (sin pasar de `limit`)."""
    i, lim = int(end / PITCH_HOP), min(int(limit / PITCH_HOP), len(per))
    last = None
    for k in range(i, lim):
        if per[k] >= voiced_thr:
            last = k
        elif (k - (last if last is not None else i)) > 8:
            break
    return max(end, (last + 1) * PITCH_HOP) if last is not None else end


def _octave_fold(pitches: list[int]) -> list[int]:
    """Lleva las notas al rango 36-84 sin romper la melodia."""
    out = []
    for p in pitches:
        while p < 36:
            p += 12
        while p > 84:
            p -= 12
        out.append(p)
    return out


def _make_phrases(notes: list[VocalNote], tm: TempoMap, q: int) -> list[Phrase]:
    """Frases como en los charts oficiales (~3 s): se corta en silencios y cambios de linea."""
    phrases: list[Phrase] = []
    if not notes:
        return phrases
    start = 0
    for i in range(1, len(notes) + 1):
        if i < len(notes):
            prev, cur = notes[i - 1], notes[i]
            gap = tm.tick_to_sec(cur.tick) - tm.tick_to_sec(prev.tick + prev.length)
            length = tm.tick_to_sec(cur.tick) - tm.tick_to_sec(notes[start].tick)
            new_line = getattr(cur, "line", 0) != getattr(prev, "line", 0)
            cut = gap > 0.9 or (new_line and gap > 0.25) or (length > 6.0 and gap > 0.2) or length > 10.0
            if not cut or cur.lyric == "+":
                continue
        a = notes[start]
        b = notes[i - 1]
        phrases.append(Phrase(max(a.tick - q, 0), b.tick + b.length + q))
        start = i
    for a, b in zip(phrases, phrases[1:]):
        if a.end > b.start:
            a.end = max(a.start + q, b.start - 1)
    return phrases


def generate_vocal_track(lyr: Lyrics, times, midi, per, tm: TempoMap, time_offset: float, log=print) -> tuple[VocalTrack, dict]:
    vt = VocalTrack()
    q = RES // 8  # cuantizacion fina (fusas) para no perder precision
    midi = midi - _tuning_offset(midi, per)
    starts = [w.start for l in lyr.lines for w in l.words]
    for li, line in enumerate(lyr.lines):
        words: list[Word] = line.words
        if not words:
            continue
        line_notes: list[VocalNote] = []
        for wi, (w, shown) in enumerate(zip(words, shown_words(line))):
            syls = syllabify(shown, lyr.language)
            if HOLD_EXTEND:
                nxt = min((t for t in starts if t > w.start + 1e-3), default=w.end + HOLD_MAX_S)
                end = _held_end(per, w.end, min(nxt - 0.05, w.end + HOLD_MAX_S))
                if end > w.end:
                    w = Word(w.text, w.start, end, w.score)
            dur = max(w.end - w.start, 0.08)
            # 1) si el cantante cambia de nota tantas veces como silabas tiene la palabra, cada silaba
            #    empieza donde empieza su nota (lo mas parecido a como se canta de verdad)
            wsegs, wv = _segment_pitch(times, midi, per, w.start, w.end + 0.05)
            if len(syls) > 1 and len(wsegs) >= len(syls) and wv >= 0.3:
                scores = []
                for k in range(len(wsegs) - 1):
                    gap = wsegs[k + 1][0] - wsegs[k][1]
                    jump = abs(wsegs[k + 1][2] - wsegs[k][2])
                    scores.append(gap * 10 + jump * 0.3)
                cuts = sorted(np.argsort(scores)[::-1][: len(syls) - 1])
                groups, start = [], 0
                for c in cuts:
                    groups.append(wsegs[start:c + 1])
                    start = c + 1
                groups.append(wsegs[start:])
                for si, (s, grp) in enumerate(zip(syls, groups)):
                    text = s if si == len(syls) - 1 else s + "-"
                    for gi, (a, b, p) in enumerate(grp[:3]):
                        vn = VocalNote(0, 0, p, text if gi == 0 else "+")
                        vn.extra_times = (a, b)  # type: ignore[attr-defined]
                        line_notes.append(vn)
                continue
            # 2) si no, tiempo de cada silaba proporcional a sus letras (las vocales pesan mas)
            weights = np.array([max(1, len(re.findall(r"[aeiouáéíóúü]", s)) * 2 + len(s)) for s in syls], float)
            bounds = w.start + np.concatenate([[0], np.cumsum(weights) / weights.sum()]) * dur
            for si, s in enumerate(syls):
                s0, s1 = float(bounds[si]), float(bounds[si + 1])
                segs, vfrac = _segment_pitch(times, midi, per, s0, s1)
                last_syl_of_word = si == len(syls) - 1
                text = s if last_syl_of_word else s + "-"
                if not segs or vfrac < 0.3:
                    # hablado / gritado: sin tono
                    pitch = line_notes[-1].pitch if line_notes else 60
                    line_notes.append(VocalNote(0, 0, pitch, text + "#", unpitched=True))
                    line_notes[-1].extra_times = (s0, s1)  # type: ignore[attr-defined]
                    continue
                first = True
                for (a, b, p) in segs[:3]:
                    lyric = text if first else "+"
                    vn = VocalNote(0, 0, p, lyric)
                    vn.extra_times = (a if first else a, b)  # type: ignore[attr-defined]
                    if first:
                        vn.extra_times = (min(a, s0 + 0.05) if a - s0 < 0.15 else a, b)  # type: ignore[attr-defined]
                    line_notes.append(vn)
                    first = False
        if not line_notes:
            continue
        pitches = _octave_fold([n.pitch for n in line_notes])
        for n, p in zip(line_notes, pitches):
            n.pitch = p
        # a ticks, sin solaparse
        placed: list[VocalNote] = []
        for n in line_notes:
            a, b = n.extra_times  # type: ignore[attr-defined]
            ta = int(round(tm.sec_to_tick(a + time_offset) / q) * q)
            tb = int(round(tm.sec_to_tick(b + time_offset) / q) * q)
            if placed and ta < placed[-1].tick + placed[-1].length:
                prev = placed[-1]
                if ta <= prev.tick:
                    ta = prev.tick + q
                prev.length = max(q // 2, ta - prev.tick - (0 if n.lyric == "+" else q // 2))
            tb = max(tb, ta + q)
            placed.append(VocalNote(ta, tb - ta, n.pitch, n.lyric, n.unpitched))
        for n in placed:
            n.line = li  # type: ignore[attr-defined]
        vt.notes.extend(placed)
    vt.notes.sort(key=lambda n: n.tick)
    vt.phrases = _make_phrases(vt.notes, tm, q)
    # star power: 1 de cada ~4 frases
    for i, p in enumerate(vt.phrases):
        if i % 4 == 2:
            vt.star_power.append(Phrase(p.start, p.end))
    info = {"notes": len(vt.notes), "phrases": len(vt.phrases),
            "source": lyr.source, "unpitched": int(sum(n.unpitched for n in vt.notes))}
    return vt, info
