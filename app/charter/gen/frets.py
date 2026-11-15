"""Generacion de charts de 5 carriles (guitarra, bajo, rhythm, teclado) a partir de notas transcritas.

Pasos:
  1. eventos: notas simultaneas -> un evento (nota o acorde) con fuerza y confianza
  2. cuantizacion a la rejilla del tempo
  3. limpieza y control de densidad para un Experto "jugable" (como los oficiales)
  4. acordes (pocos acordes de 3), carriles por frase (contorno melodico), sustains reales
  5. HOPOs / taps / abiertas
  6. reduccion a Dificil, Medio y Facil
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chart.midi_io import resolve_hopos
from ..chart.model import DIFFICULTIES, OPEN, RES, FretNote, FretTrack, Phrase, TempoMap
from ..pipeline.transcription import Note, StemAnalysis
from .common import metric_strength, nms_select, quantize_ticks, secs_between, stat

HOPO_THR_TICKS = 170  # umbral natural de HOPO de YARG / CH (hopo_frequency = 170)
# probabilidad minima (modelo de dificultades) para que una nota de Experto se quede en cada dificultad
REDUCE_THR = {"hard": 0.55, "medium": 0.48, "easy": 0.30}  # ajustado en el banco de pruebas (cantidad ~ oficial)
# acordes: por debajo de esta separacion (s) con la nota vecina, nota simple; y si el modelo de acordes puede
# poner un acorde aunque YourMT3 solo oiga una nota (la quinta de un power chord suena muy flojita)
CHORD_FAST_GAP = 0.08  # medido en el banco de pruebas: 0,15 quitaba acordes reales en riffs rapidos
CHORD_TRUST_MODEL = False
# guitarra: notas del modelo neuronal (aprendido del espectrograma de miles de guitarras separadas) si esta entrenado
USE_NEURAL = True
MIN_SEP_TICKS = RES // 4  # dos golpes mas juntos que esto son el mismo golpe partido en dos
THIN_NEURAL_GAP_ONLY = True  # con el modelo neuronal, de la limpieza de Experto solo queda la separacion minima
NEURAL_THR_ADD = -0.03  # guitarra: mas bajo = mas notas (se escapan menos rasgueos seguidos, pero sobran mas)
CHORD_NEURAL = True  # el tamano de cada acorde tambien lo dice el modelo neuronal
CHORD_NEURAL_BIAS = 0.0  # >0 favorece acordes frente a notas sueltas
CHORD_LONG_STRUM = True  # nota larga en la que suenan varias notas (rasgueo que se queda sonando) -> acorde
# botones: peso de lo que oye el modelo neuronal (si la nota sube, baja o repite) frente al tono transcrito
LANE_NEURAL_DEBIAS = True
LANE_SPREAD = 0.8  # en frases de pocas notas distintas, cuanto se abren los botones por cada semitono
LANE_NEURAL_MOVE = 7.0
LANE_NEURAL_LANE = 2.0
LOWER_LANES = "learned"  # botones de Dificil/Medio/Facil: "learned" (modelo de dificultades) o "scaled" (los de Experto comprimidos)
REDUCE_NEURAL = 0.5  # peso del modelo neuronal al decidir que notas quedan en Dificil, Medio y Facil (0 = no)
REDUCE_NEURAL_THR_ADD = 0.0  # mas alto = menos notas en las dificultades bajas
# partes repetidas (estribillos, riffs que vuelven): se oyen juntas (media de todas las veces que suenan) y
# cada repeticion copia las notas de la primera vez, como en los charts oficiales
REPEAT_AVERAGE = True
REPEAT_COPY = True
HARM_LANES = 3.0  # coste de no poner una nota en el boton que lleva en las otras partes con los mismos acordes
# golpes del bajo que la IA deja en la guitarra: la guitarra suena tantos dB por debajo del bajo, ataca flojo
# (menos de esta fraccion de sus ataques fuertes) y el bajo ataca a la vez el doble -> no es nota de guitarra
# notas largas: si el sonido tarda en caer los primeros 10 dB mas de RING_SLOW_S, se deja sonar hasta que cae
# RING_DROP_DB (rasgueos que se dejan sonar hasta que se apagan)
RING_DROP_DB = 20.0
# notas largas: cuanto del tope de los charts oficiales se usa (1 = percentil 90, 0 = mediana). Medido en el banco: con 1
# la cantidad sale como en los charts reales (10,7 % frente a 11,0 %); bajarlo no acierta mas y hace menos constante el ritmo
SUSTAIN_P90_WEIGHT = 1.0
# el mismo ritmo en el mismo punto del compas lleva siempre la misma duracion: si la mayoria son largas, todas; si no, ninguna
SUSTAIN_VOTE = True
SUSTAIN_VOTE_MIN = 3
SUSTAIN_MIN_NEXT = RES * 5 // 8  # no es nota larga si la siguiente llega antes de esto (banco: bajo 0,625 -> 0,639, guitarra igual)
RING_SLOW_S = 0.8
BASS_BLEED_DB = 8.0
BASS_BLEED_ATTACK = 0.35
# ...o, justo tras el golpe, la guitarra suena tantos dB por debajo del bajo y de su propio nivel normal, aunque ataque
# fuerte (en el banco quita 7 de unas 11.000 notas, 1 real; el arranque de bajo solo de Hervidero se quitaba a medias)
BASS_BLEED_QUIET_DB = 20.0
# la regla antigua no quita una nota si el modelo neuronal esta asi de seguro de que es guitarra (>1 = sin proteccion)
BASS_BLEED_KEEP_PROB = 0.6  # banco: notas F1 0,9010 -> 0,9025; recupera los rasgueos de Hervidero (prob 0,82-0,93)
BASS_BLEED_QUIET_SELF_DB = 15.0
DOUBLE_STRUM_FRAC = 0.6  # dos golpes en el mismo tick: el segundo pasa a la semicorchea siguiente si estan asi de separados
DOUBLE_STRUM_MIN = 0.25
QUANT_MAX_S = None  # segundos: si la rejilla aleja la nota mas que esto de donde suena, se pone en la division fina
QUANT_FINE = 16  # divisiones por pulso de esa division fina
REPEAT_LOWER = "first"  # Dificil/Medio/Facil: copia de la primera vez ("first"), del pulso mas representativo ("medoid") o nada
CQT_FMIN_MIDI_ = 24


@dataclass
class Event:
    time: float
    pitches: list[int]
    end: float
    strength: float
    votes: float
    tick: int = 0
    sus_ticks: int = 0
    lanes: tuple[int, ...] = ()
    legato: bool = False
    review: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> int:
        return min(self.pitches)


# ------------------------------------------------------------------ eventos
def build_events_learned(notes: list[Note], sa: StemAnalysis, beats: np.ndarray, inst: str,
                         model, thr: float = 0.5) -> list[Event]:
    """Eventos decididos por el modelo aprendido de charts oficiales: que ataques del audio son notas.

    El tono de cada evento sale de las notas transcritas (YourMT3) que suenan en ese momento; si no hay,
    del pico de energia de la pista en el rango del instrumento.
    """
    from ..calibration.onset_model import ONSET_LAG, PITCH_RANGE, candidates, onset_features

    cands = candidates(sa)
    if len(cands) == 0:
        return []
    X = onset_features(sa, cands, beats, inst)
    prob = model.predict_proba(X[:, :model.n_features_in_])[:, 1]
    env = sa.onset_env / (np.percentile(sa.onset_env, 99) + 1e-9)
    notes = sorted(notes, key=lambda n: n.onset)
    on_arr = np.array([n.onset for n in notes]) if notes else np.zeros(0)
    off_arr = np.array([n.offset for n in notes]) if notes else np.zeros(0)
    lo, hi = PITCH_RANGE[inst]
    events: list[Event] = []
    for t, p in zip(cands, prob):
        if p < thr:
            continue
        f = sa.frame(t)
        strength = float(np.max(env[max(f - 2, 0): f + 3]))
        near = np.where(np.abs(on_arr - t) <= 0.06)[0] if len(on_arr) else np.zeros(0, int)
        if len(near) == 0 and len(on_arr):
            near = np.where((on_arr < t) & (off_arr > t + 0.02))[0]
        # la duracion es la mas larga entre la de YourMT3 y la que se mide en el audio (un rasgueo que se queda
        # sonando es una nota larga aunque YourMT3 no la oiga entera)
        if len(near):
            pitches = sorted(set(notes[k].pitch for k in near))
            end = max(float(max(off_arr[near])), _decay_end(sa, t))
        else:
            post = sa.cqt_db[lo - CQT_FMIN_MIDI_: hi - CQT_FMIN_MIDI_, f + 1:f + 6]
            if post.size == 0:
                continue
            pitches = [int(np.argmax(post.max(axis=1))) + lo]
            end = _decay_end(sa, t)
        # el pico de ataque llega un poco despues del golpe real
        events.append(Event(float(t) - ONSET_LAG, pitches, end, strength, float(max(p, 0.5)),
                            extra={"prob": float(p), "cand_t": float(t)}))
    # rescate: acordes largos y claros que YourMT3 oye justo donde el modelo duda (rasgueos que se quedan sonando,
    # con ataque suave). Dos detectores distintos de acuerdo y nada mas alrededor: es una nota
    ev_t = np.array([e.time for e in events]) if events else np.zeros(0)
    k = 0
    while k < len(notes):
        g = [notes[k]]
        while k + len(g) < len(notes) and notes[k + len(g)].onset - notes[k].onset < 0.08:
            g.append(notes[k + len(g)])
        k += len(g)
        t = g[0].onset
        dur = max(n.offset for n in g) - t
        if len(set(n.pitch for n in g)) < 2 or dur < 0.8 or np.mean([n.votes for n in g]) < 0.6:
            continue
        if len(ev_t) and np.min(np.abs(ev_t - t)) < 0.25:
            continue
        j = np.where(np.abs(cands - t) <= 0.06)[0]
        if len(j) == 0 or prob[j].max() < 0.5 * thr:
            continue
        jb = j[np.argmax(prob[j])]
        ct = float(cands[jb])
        f = sa.frame(ct)
        events.append(Event(ct - ONSET_LAG, sorted(set(n.pitch for n in g)), max(t + dur, _decay_end(sa, ct)),
                            float(np.max(env[max(f - 2, 0): f + 3])), 0.6,
                            extra={"prob": float(prob[jb]), "cand_t": ct, "rescued": True}))
    # notas que todos los pases del modelo oyen pero cuyo ataque no se detecto (ligados suaves)
    events.sort(key=lambda e: e.time)
    ev_t = np.array([e.time for e in events]) if events else np.zeros(0)
    for n in notes:
        if n.votes < 1.0:
            continue
        if len(ev_t) and np.min(np.abs(ev_t - n.onset)) < 0.08:
            continue
        if len(cands) and np.min(np.abs(cands - n.onset)) < 0.05:
            continue  # el modelo ya vio ese ataque y lo descarto
        f = sa.frame(n.onset)
        events.append(Event(n.onset, [n.pitch], n.offset, float(np.max(env[max(f - 2, 0): f + 3])), 0.75,
                            legato=True, extra={"prob": 0.5}))
    events.sort(key=lambda e: e.time)
    for a, b in zip(events, events[1:]):
        a.end = min(a.end, b.time)
    return events


def build_events_neural(notes: list[Note], sa: StemAnalysis, net, beats: np.ndarray | None = None,
                        canon: np.ndarray | None = None, instrument: str = "guitar") -> list[Event]:
    """Eventos del modelo neuronal: en que momentos pone nota un charter, mirando el espectrograma de la pista.

    El tono sale de las notas transcritas (YourMT3) que suenan en ese momento; si no hay, del pico de energia de
    la pista en el rango de la guitarra.
    """
    from ..calibration import neural_keys
    from ..calibration.neural_onset import FPS, _inputs, load_models, peak_times, peaks, predict_all
    from ..calibration.onset_model import ONSET_LAG, PITCH_RANGE, candidates, compact_spectrum

    keys = instrument == "keys"
    if keys:
        load_models, predict_all = neural_keys.load_models, neural_keys.predict_all
    elif instrument == "bass":
        from ..calibration.neural_bass import load_models
    model, info = net
    spec = compact_spectrum(sa)
    nets = load_models() or [net]
    prob_all, chord_p, move_p, lane_p = predict_all(nets, _inputs(spec["cqt"], spec["onset"]))
    buttons = any("botones" in i for _, i in nets)
    if canon is not None and beats is not None:
        from .repeats import average_frames

        prob_all, chord_p, move_p, lane_p = average_frames([prob_all, chord_p, move_p, lane_p], FPS, beats, canon)
    prob = prob_all[:, 0]
    frames = peaks(prob, info["thr"] + (NEURAL_THR_ADD if instrument == "guitar" else 0.0))
    times = peak_times(prob, frames)
    cands = candidates(sa)
    env = sa.onset_env / (np.percentile(sa.onset_env, 99) + 1e-9)
    notes = sorted(notes, key=lambda n: n.onset)
    on_arr = np.array([n.onset for n in notes]) if notes else np.zeros(0)
    off_arr = np.array([n.offset for n in notes]) if notes else np.zeros(0)
    lo, hi = (48, 84) if keys else PITCH_RANGE["bass" if instrument == "bass" else "guitar"]
    events: list[Event] = []
    for fr, t in zip(frames, times):
        f = sa.frame(t)
        strength = float(np.max(env[max(f - 1, 0): f + 4]))
        near = np.where(np.abs(on_arr - t) <= 0.06)[0] if len(on_arr) else np.zeros(0, int)
        if len(near) == 0 and len(on_arr):
            near = np.where((on_arr < t) & (off_arr > t + 0.02))[0]
        if len(near):
            pitches = sorted(set(notes[k].pitch for k in near))
            # en el teclado la duracion ya la dice el modelo; la energia de un piano tarda mucho en apagarse
            end = float(max(off_arr[near])) if keys else max(float(max(off_arr[near])), _decay_end(sa, t))
        else:
            post = sa.cqt_db[lo - CQT_FMIN_MIDI_: hi - CQT_FMIN_MIDI_, f + 1:f + 6]
            if post.size == 0:
                continue
            pitches = [int(np.argmax(post.max(axis=1))) + lo]
            end = _decay_end(sa, t)
        # el ataque detectado mas cercano (para las caracteristicas del modelo de acordes)
        j = int(np.argmin(np.abs(cands - (t + ONSET_LAG)))) if len(cands) else -1
        ct = float(cands[j]) if j >= 0 and abs(cands[j] - (t + ONSET_LAG)) <= 0.04 else float(t) + ONSET_LAG
        events.append(Event(float(t), pitches, end, strength, float(max(prob[fr], 0.5)),
                            extra={"prob": float(prob[fr]), "cand_t": ct, "nn_chord": chord_p[fr].tolist(),
                                   "nn_thr": info.get("thr_diff", {}),
                                   # probabilidad de que esta nota se quede en Dificil, Medio y Facil
                                   "nn_diff": prob_all[max(fr - 1, 0): fr + 2, 1:].max(axis=0).tolist()}))
        if buttons:
            events[-1].extra["nn_move"] = move_p[fr].tolist()
            events[-1].extra["nn_lane"] = lane_p[fr].tolist()
    for a, b in zip(events, events[1:]):
        a.end = min(a.end, b.time)
    return events


def _song_repeats(sa: StemAnalysis, beats: np.ndarray, net) -> np.ndarray:
    """Para cada pulso, el primer pulso de la cancion que suena igual (ver repeats.py)."""
    from ..calibration.neural_onset import FPS, _inputs, embed
    from ..calibration.onset_model import compact_spectrum
    from .repeats import find_repeats

    spec = compact_spectrum(sa)
    return find_repeats(sa, beats, embed(net[0], _inputs(spec["cqt"], spec["onset"])), FPS)


def _clone_event(tm: TempoMap):
    def clone(e: Event, tick: int) -> Event:
        return Event(tm.tick_to_sec(tick), list(e.pitches), e.end + tm.tick_to_sec(tick) - tm.tick_to_sec(e.tick),
                     e.strength, e.votes, tick, e.sus_ticks, e.lanes, e.legato, e.review, dict(e.extra))

    return clone


def _copy_lanes(events: list[Event], canon: np.ndarray) -> None:
    """Los botones de cada repeticion, exactamente los de la primera vez (las notas ya son copia)."""
    by_tick = {e.tick: e for e in events}
    for e in events:
        b = e.tick // RES
        if b < len(canon) and canon[b] != b:
            src = by_tick.get(e.tick - (b - int(canon[b])) * RES)
            if src is not None and len(src.lanes) == len(e.lanes):
                e.lanes = src.lanes


def _decay_end(sa: StemAnalysis, t: float, drop_db: float = 10.0, max_s: float = 8.0) -> float:
    """Hasta cuando sigue sonando un golpe: hasta que el nivel cae `drop_db` por debajo del pico tras el ataque
    (o al ruido de fondo). Un rasgueo que se queda sonando decae despacio; uno apagado (palm mute), enseguida."""
    f = sa.frame(t)
    r = sa.rms_db
    if f >= len(r) - 2:
        return t + 0.15
    peak = float(np.max(r[f: f + 8]))
    floor = max(peak - drop_db, sa.noise_floor_db + 6)
    stop = min(len(r), f + int(max_s / max(sa.times[1] - sa.times[0], 1e-3)))
    below = np.where(r[f + 4: stop] < floor)[0]
    g = f + 4 + (int(below[0]) if len(below) else stop - f - 4)
    if RING_DROP_DB > drop_db and sa.times[min(g, len(sa.times) - 1)] - t >= RING_SLOW_S:
        # cae despacio: es un rasgueo que se deja sonar (guitarra espanola, acustica): dura hasta que se apaga
        floor = max(peak - RING_DROP_DB, sa.noise_floor_db + 6)
        below = np.where(r[g: stop] < floor)[0]
        g = g + (int(below[0]) if len(below) else stop - g)
    return float(sa.times[min(g, len(sa.times) - 1)])


def build_events(notes: list[Note], sa: StemAnalysis, tol: float = 0.035) -> list[Event]:
    """Eventos = ataques de las notas transcritas + re-ataques reales de la pista.

    Los modelos de transcripcion suelen juntar en una sola nota larga los rasgueos repetidos de un
    mismo acorde (riffs con palm mute, corcheas sobre la misma nota). Cada ataque que se oye en la
    pista mientras esa nota sigue sonando se convierte en un evento nuevo con las mismas notas.
    """
    from ..pipeline.transcription import detect_onsets, pitch_energy

    notes = sorted(notes, key=lambda n: n.onset)
    events: list[Event] = []
    i = 0
    env = sa.onset_env / (np.percentile(sa.onset_env, 99) + 1e-9)

    def strength_at(t: float) -> float:
        f = sa.frame(t)
        return float(np.max(env[max(f - 2, 0): f + 3])) if len(env) else 0.5

    while i < len(notes):
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j].onset - notes[i].onset <= tol:
            group.append(notes[j])
            j += 1
        t = float(np.median([n.onset for n in group]))
        pitches = sorted(set(n.pitch for n in group))
        events.append(Event(t, pitches, float(max(n.offset for n in group)), strength_at(t),
                            float(np.mean([n.votes for n in group]))))
        i = j
    # re-ataques
    onsets = detect_onsets(sa)
    ev_times = np.array([e.time for e in events]) if events else np.zeros(0)
    on_arr = np.array([n.onset for n in notes])
    off_arr = np.array([n.offset for n in notes])
    extra = []
    for t in onsets:
        # 45 ms: un rasgueo doble ("pa-pam") son dos ataques muy seguidos y deben ser dos notas
        if len(ev_times) and np.min(np.abs(ev_times - t)) < 0.045:
            continue
        act = np.where((on_arr + 0.045 < t) & (off_arr - 0.03 > t))[0]
        if len(act) == 0:
            continue
        s = strength_at(t)
        if s < 0.12:
            continue
        # solo las notas que de verdad se vuelven a tocar (su energia sube en ese ataque); si no, la mas
        # fuerte. Asi un re-ataque no hereda todas las notas que siguen sonando (acordes falsos)
        f = sa.frame(t)
        rises = []
        for k in act:
            e_post = pitch_energy(sa, notes[k].pitch, t + 0.01, t + 0.06)
            e_pre = pitch_energy(sa, notes[k].pitch, max(t - 0.05, 0), t - 0.005)
            rises.append((float(np.max(e_post)) - float(np.max(e_pre)) if len(e_post) and len(e_pre) else 0.0,
                          float(np.max(e_post)) if len(e_post) else -100.0, notes[k].pitch))
        rising = [p for r, lv, p in rises if r >= 2.5]
        pitches = sorted(set(rising)) if rising else [max(rises, key=lambda x: x[1])[2]]
        votes = float(min(0.9, np.mean([notes[k].votes for k in act])))
        extra.append(Event(float(t), pitches, float(np.min(off_arr[act])), s, votes,
                           extra={"reattack": True}))
    events = sorted(events + extra, key=lambda e: e.time)
    # cada evento termina como muy tarde cuando empieza el siguiente
    for a, b in zip(events, events[1:]):
        a.end = min(a.end, b.time)
    return events


def _clean_events(events: list[Event], sa: StemAnalysis, instrument: str) -> list[Event]:
    """Quita eventos dudosos: muy flojos, sin ataque y con pocos votos (restos de separacion)."""
    out = []
    for k, e in enumerate(events):
        prev = out[-1] if out else None
        legato = False
        if prev is not None and e.time - prev.time < 0.25 and e.strength < 0.25 and prev.end >= e.time - 0.03:
            legato = True  # ligado (hammer-on / pull-off): ataque flojo pero nota real
        if e.votes < 0.5:
            continue
        if e.strength < 0.08 and e.votes < 0.75 and not legato:
            continue
        e.legato = legato
        # notas extremadamente agudas o graves para el instrumento -> casi siempre errores
        lo, hi = {"bass": (23, 67), "guitar": (36, 90), "rhythm": (36, 90), "keys": (21, 108)}.get(instrument, (0, 127))
        e.pitches = [p for p in e.pitches if lo <= p <= hi]
        if not e.pitches:
            continue
        out.append(e)
    return out


def _simplify_pitches(e: Event, instrument: str) -> None:
    """Reduce un acorde a lo que importa para el juego: raiz + forma (2 o 3 notas)."""
    ps = sorted(set(e.pitches))
    if instrument == "bass":
        e.pitches = [ps[0]]
        return
    # quitar duplicados de octava de la raiz y notas muy separadas (armonicos)
    root = ps[0]
    pcs = []
    kept = [root]
    for p in ps[1:]:
        if (p - root) % 12 == 0:
            continue
        if p - root > 24:
            continue
        if (p - root) % 12 in pcs:
            continue
        pcs.append((p - root) % 12)
        kept.append(p)
    e.pitches = kept


# ------------------------------------------------------------------ densidad
def _thin_expert(events: list[Event], tm: TempoMap, instrument: str, log) -> list[Event]:
    """Experto jugable: limita la velocidad y la densidad como en los charts oficiales."""
    if not events:
        return events
    key = "bass" if instrument == "bass" else ("keys" if instrument == "keys" else "guitar")
    # 1) separacion minima absoluta (~16 notas/s: deja pasar rasgueos dobles, no rafagas de ruido)
    min_gap = 0.055
    ticks = [e.tick for e in events]
    scores = [_importance(e, tm) for e in events]
    keep = nms_select(ticks, scores, RES // 8, min_gap, tm)
    events = [events[i] for i in keep]
    if THIN_NEURAL_GAP_ONLY and events and "nn_chord" in events[0].extra:
        # el modelo neuronal ya sabe cuantas notas pone un charter: los topes de densidad de una cancion
        # "tipica" le quitaban notas reales en las canciones rapidas (banco de pruebas: F1 0,85 -> 0,89)
        return events
    # 2) ventanas de 2 s: como mucho lo que tienen los tramos densos de una cancion oficial tipica
    dense = stat(f"{key}.expert.nps_p90.p50", 5.5)
    cap_window = max(8, int(round(min(dense * 1.35, 8.0) * 2)))
    events = _window_cap(events, tm, 2.0, cap_window)
    # 3) ventanas de 8 s (una frase): una parte "normal" no puede ir todo el rato al maximo
    cap8 = max(20, int(round(min(dense * 1.1, 6.5) * 8)))
    events = _window_cap(events, tm, 8.0, cap8)
    # 4) densidad media de toda la cancion: como una oficial tipica (con margen)
    target = stat(f"{key}.expert.nps.p50", 3.9) * 1.35
    dur = max(events[-1].time - events[0].time, 1.0)
    nps = len(events) / dur
    if nps > target:
        n_keep = int(target * dur)
        order = np.argsort([-_importance(e, tm) for e in events])[:n_keep]
        events = [events[i] for i in sorted(order)]
        log(f"  densidad reducida de {nps:.1f} a {len(events) / dur:.1f} notas/s (tope ~{target:.1f})")
    return events


def _window_cap(events: list[Event], tm: TempoMap, win: float, cap: int) -> list[Event]:
    times = np.array([e.time for e in events])
    alive = np.ones(len(events), bool)
    changed = True
    while changed:
        changed = False
        idx = np.where(alive)[0]
        t = times[idx]
        j = 0
        for i in range(len(idx)):
            while t[i] - t[j] > win:
                j += 1
            if i - j + 1 > cap:
                cand = idx[j:i + 1]
                worst = min(cand, key=lambda k: _importance(events[k], tm))
                alive[worst] = False
                changed = True
                break
    return [e for e, a in zip(events, alive) if a]


def _importance(e: Event, tm: TempoMap) -> float:
    m = metric_strength(e.tick, tm)
    chord = 0.15 if len(e.pitches) > 1 else 0.0
    sus = 0.25 if e.sus_ticks > 0 else 0.0
    return m * 0.9 + min(e.strength, 1.5) * 0.6 + e.votes * 0.5 + chord + sus


def _drop_bass_bleed(events: list[Event], sa: StemAnalysis, bass_sa: StemAnalysis) -> list[Event]:
    """Quita las notas que son el bajo colandose en la guitarra separada: donde la guitarra suena mucho mas
    flojo que el bajo (o casi no suena), su ataque es debil y el bajo ataca a la vez con mucha mas fuerza."""
    g_env = sa.onset_env / (np.percentile(sa.onset_env, 99) + 1e-9)
    b_env = bass_sa.onset_env / (np.percentile(bass_sa.onset_env, 99) + 1e-9)
    win = sa.frame(1.0)  # nivel de cada pista en el segundo de alrededor

    def level(an: StemAnalysis, t: float) -> float:
        f0 = an.frame(t)
        return float(np.max(an.rms_db[f0:max(an.frame(t + 0.3), f0 + 1)])) if f0 < len(an.rms_db) else -180.0

    normal = float(np.median([level(sa, e.time) for e in events])) if events else 0.0
    out = []
    for e in events:
        f = sa.frame(e.time)
        a, z = max(f - win, 0), f + win
        g_lvl = float(np.mean(sa.rms_db[a:z]))
        b_lvl = float(np.mean(bass_sa.rms_db[a:min(z, len(bass_sa.rms_db))])) if a < len(bass_sa.rms_db) else -120.0
        g_att = float(g_env[max(f - 1, 0):f + 3].max())
        fb = min(f, len(b_env) - 1)
        b_att = float(b_env[max(fb - 2, 0):fb + 3].max())
        if (b_lvl - g_lvl >= BASS_BLEED_DB and g_att < BASS_BLEED_ATTACK and b_att >= 2 * g_att
                and e.extra.get("prob", 0.0) < BASS_BLEED_KEEP_PROB):
            continue
        g_now = level(sa, e.time)
        if g_now <= level(bass_sa, e.time) - BASS_BLEED_QUIET_DB and g_now <= normal - BASS_BLEED_QUIET_SELF_DB:
            continue
        out.append(e)
    return out


# ------------------------------------------------------------------ sustains
def _assign_sustains(events: list[Event], tm: TempoMap, key: str) -> None:
    p50 = stat(f"{key}.expert.sustain_ratio.p50", 0.10)
    p90 = stat(f"{key}.expert.sustain_ratio.p90", 0.22)
    max_ratio = p50 + SUSTAIN_P90_WEIGHT * (p90 - p50)
    cands = []
    # Como en los charts oficiales (45 medidos): una nota larga dura desde 3/8 de pulso y, si suena hasta la
    # siguiente, llega casi hasta ella (hueco mediano 0,01 pulsos). Antes se exigian 0,9 pulsos y se cortaba
    # una semicorchea antes, y se perdian casi todas las notas largas de canciones como R U Mine
    min_len = int(RES * 0.375)
    gap = RES // 24
    for i, e in enumerate(events):
        nxt = events[i + 1].tick if i + 1 < len(events) else e.tick + RES * 8
        end_tick = int(tm.sec_to_tick(e.end))
        dur_s = e.end - e.time
        e.sus_ticks = 0
        if dur_s < 0.3 or end_tick - e.tick < min_len or nxt - e.tick < SUSTAIN_MIN_NEXT:
            continue  # (en los oficiales, el 95 % de las notas largas tiene la siguiente a medio pulso o mas)
        if end_tick >= nxt - RES // 4:
            length = nxt - gap - e.tick  # suena hasta la siguiente nota
        else:
            length = int(round((end_tick - e.tick) / (RES // 8))) * (RES // 8)
        length = min(length, nxt - gap - e.tick)
        if length >= min_len:
            cands.append((dur_s, i, length))
    cands.sort(reverse=True)
    n_max = int(max_ratio * len(events)) + 1
    for dur_s, i, length in cands[:n_max]:
        events[i].sus_ticks = length
    if SUSTAIN_VOTE:
        _vote_sustains(events)


def _vote_sustains(events: list[Event]) -> None:
    """Que una nota no pase de larga a corta porque si dentro de un ritmo que se repite."""
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, e in enumerate(events[:-1]):
        prev = e.tick - events[i - 1].tick if i else -1
        groups.setdefault((e.tick % (RES * 4), events[i + 1].tick - e.tick, prev), []).append(i)
    for idx in groups.values():
        if len(idx) < SUSTAIN_VOTE_MIN:
            continue
        long_ones = [events[i].sus_ticks for i in idx if events[i].sus_ticks > 0]
        length = int(np.median(long_ones)) if len(long_ones) * 2 >= len(idx) else 0
        for i in idx:
            events[i].sus_ticks = min(length, events[i + 1].tick - RES // 24 - events[i].tick)


# ------------------------------------------------------------------ acordes y carriles
def _chord_shape(e: Event, allow3: bool) -> int:
    """Numero de notas del acorde en el juego: 1, 2 o 3."""
    if len(e.pitches) <= 1:
        return 1
    if len(e.pitches) >= 3 and allow3:
        return 3
    return 2


def _decide_chords(events: list[Event], tm: TempoMap, key: str) -> list[int]:
    """Cuantas notas tiene cada evento, respetando proporciones de los charts oficiales."""
    sizes = []
    for i, e in enumerate(events):
        s = _chord_shape(e, True)
        if "chord_pred" in e.extra:
            # lo que haria un charter segun el modelo, sin pasar de las notas que realmente suenan
            if CHORD_TRUST_MODEL:
                s = min(e.extra["chord_pred"], 2 if len(e.pitches) <= 1 else len(e.pitches))
            else:
                s = min(e.extra["chord_pred"], max(len(e.pitches), 1)) if len(e.pitches) > 1 else 1
        if CHORD_NEURAL and "nn_chord" in e.extra:
            pc = np.asarray(e.extra["nn_chord"], float) + np.array([0.0, CHORD_NEURAL_BIAS, CHORD_NEURAL_BIAS])
            s = int(np.argmax(pc)) + 1
            if CHORD_LONG_STRUM and s == 1 and len(e.pitches) >= 2 and e.sus_ticks >= RES:
                s = 2
        # acordes en pasajes muy rapidos -> nota simple (son injugables en serie)
        prev_gap = secs_between(tm, events[i - 1].tick, e.tick) if i > 0 else 1.0
        next_gap = secs_between(tm, e.tick, events[i + 1].tick) if i + 1 < len(events) else 1.0
        if s > 1 and min(prev_gap, next_gap) < CHORD_FAST_GAP:
            s = 1
        # acordes de 3 solo en golpes sueltos (con aire antes y despues), nunca en serie
        if s == 3 and min(prev_gap, next_gap) < 0.3:
            s = 2
        sizes.append(s)
    sizes = np.array(sizes)
    n = len(events)
    if n == 0:
        return []
    # tope de acordes: como una cancion oficial tipica (mediana, como mucho un tercio de las notas)
    max_chord = min(stat(f"{key}.expert.chord_ratio.p50", 0.28) * 1.15, 0.33)
    max_c3 = min(max(stat(f"{key}.expert.chord3_ratio.p50", 0.007) * 2, 0.01), 0.025)
    # demasiados acordes de 3 -> se quedan solo los mas acentuados (golpes, notas largas)
    c3 = np.where(sizes == 3)[0]
    if len(c3) > max_c3 * n:
        sc = [(_importance(events[i], tm) + (0.5 if events[i].sus_ticks else 0)) for i in c3]
        keep = set(c3[np.argsort(sc)[::-1][: int(max_c3 * n)]])
        for i in c3:
            if i not in keep:
                sizes[i] = 2
    chords = np.where(sizes > 1)[0]
    if len(chords) > max_chord * n:
        sc = [_importance(events[i], tm) for i in chords]
        keep = set(chords[np.argsort(sc)[::-1][: int(max_chord * n)]])
        for i in chords:
            if i not in keep:
                sizes[i] = 1
    return sizes.tolist()


def _phrases(events: list[Event], gap_beats: float = 1.5) -> list[list[int]]:
    out, cur = [], []
    for i, e in enumerate(events):
        if cur and (e.tick - events[cur[-1]].tick) > gap_beats * RES:
            out.append(cur)
            cur = []
        cur.append(i)
        if len(cur) >= 24:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _target_step(dp: int) -> int:
    a = abs(dp)
    if a == 0:
        return 0
    if a <= 2:
        return 1
    if a <= 4:
        return 2 if a == 4 else 1
    if a <= 7:
        return 2
    if a <= 12:
        return 3
    return 4


def _assign_lanes(events: list[Event], sizes: list[int], key: str, targets: dict[int, int] | None = None,
                  debias: bool = True) -> None:
    """Asigna carriles por frase con programacion dinamica (respeta el contorno melodico)."""
    lane_prior = np.array(stat_list(f"{key}.expert.lanes", [0.24, 0.23, 0.24, 0.19, 0.10])[:5])
    lane_prior = lane_prior / lane_prior.sum()
    prior_cost = -np.log(lane_prior + 1e-3) * 0.6
    cache: dict[tuple, list[int]] = {}
    if debias and LANE_NEURAL_DEBIAS and events and all("nn_lane" in e.extra for e in events):
        # el modelo tiende a ir al centro (en la duda, rojo o amarillo): se corrige con lo que usa en toda la
        # cancion y se lleva al reparto de botones de los charts oficiales
        P = np.array([e.extra["nn_lane"] for e in events])
        adj = P / (P.mean(0) + 1e-3) * lane_prior
        adj = adj / adj.sum(1, keepdims=True)
        for e, a in zip(events, adj):
            e.extra["nn_lane"] = a.tolist()
    for ph in _phrases(events):
        roots = [events[i].root for i in ph]
        sig = tuple(r - roots[0] for r in roots) + tuple(sizes[i] for i in ph) + (roots[0] % 12,)
        tg = [targets.get(i) for i in ph] if targets else None
        if tg is not None:
            sig += tuple(-1 if t is None else t for t in tg)
        nn = [events[i].extra.get("nn_move") for i in ph]
        nn_l = [events[i].extra.get("nn_lane") for i in ph]
        if not all(x is not None for x in nn):
            nn = nn_l = None
        if sig in cache and len(ph) >= 3:
            lanes0 = cache[sig]  # el mismo riff repetido lleva los mismos botones
        else:
            lanes0 = _viterbi_lanes(roots, [sizes[i] for i in ph], prior_cost, nn, nn_l, tg)
            cache[sig] = lanes0
        for i, l0 in zip(ph, lanes0):
            s = sizes[i]
            e = events[i]
            if s == 1:
                e.lanes = (l0,)
            elif s == 2:
                width = 2 if len(e.pitches) > 1 and (e.pitches[1] - e.pitches[0]) >= 7 and l0 <= 2 else 1
                e.lanes = (l0, min(l0 + width, 4))
            else:
                e.lanes = (l0, min(l0 + 1, 4), min(l0 + 2, 4)) if l0 <= 2 else (2, 3, 4)
            e.lanes = tuple(sorted(set(e.lanes)))


def stat_list(path: str, default: list[float]) -> list[float]:
    from .common import calibration

    cur = calibration()
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur if isinstance(cur, list) else default


def _viterbi_lanes(roots: list[int], sizes: list[int], prior_cost: np.ndarray,
                   nn_move: list | None = None, nn_lane: list | None = None,
                   targets: list | None = None) -> list[int]:
    n = len(roots)
    pmin, pmax = min(roots), max(roots)
    span = max(pmax - pmin, 1)
    L = 5
    INF = 1e9
    cost = np.full((n, L), INF)
    back = np.zeros((n, L), dtype=int)

    def emit(i: int, l: int) -> float:
        s = sizes[i]
        if l + (s - 1) > 4:
            return INF
        # posicion relativa en la frase (graves a la izquierda, agudos a la derecha)
        if span <= 4:
            exp = 2.0 + (roots[i] - (pmin + pmax) / 2) * LANE_SPREAD
        else:
            exp = (roots[i] - pmin) / span * 4.0
        c = abs(l - exp) * 0.35 + prior_cost[l]
        if s >= 2:
            c += 0.1 * l  # acordes un poco hacia la izquierda, como en los oficiales
        if nn_lane is not None and LANE_NEURAL_LANE:
            c += -np.log(nn_lane[i][l] + 1e-3) * LANE_NEURAL_LANE
        if targets is not None and targets[i] is not None and l != targets[i]:
            c += HARM_LANES  # la misma nota que en las otras partes con los mismos acordes, al mismo boton
        return c

    for l in range(L):
        cost[0, l] = emit(0, l)
    for i in range(1, n):
        dp = roots[i] - roots[i - 1]
        tgt = _target_step(dp)
        for l in range(L):
            e = emit(i, l)
            if e >= INF:
                continue
            best, arg = INF, 0
            for k in range(L):
                if cost[i - 1, k] >= INF:
                    continue
                dl = l - k
                if dp == 0:
                    t = 0.0 if dl == 0 else 3.0 + abs(dl)
                else:
                    t = 0.0
                    if np.sign(dl) != np.sign(dp):
                        t += 2.5 if dl == 0 else 4.0
                    t += 0.6 * abs(abs(dl) - tgt)
                if nn_move is not None and LANE_NEURAL_MOVE:
                    # lo que oye el modelo neuronal: si esta nota sube, baja o repite respecto a la anterior
                    t += -np.log(nn_move[i][int(np.sign(dl)) + 1] + 1e-3) * LANE_NEURAL_MOVE
                c = cost[i - 1, k] + t
                if c < best:
                    best, arg = c, k
            cost[i, l] = best + e
            back[i, l] = arg
    lanes = [int(np.argmin(cost[-1]))]
    for i in range(n - 1, 0, -1):
        lanes.append(int(back[i, lanes[-1]]))
    return lanes[::-1]


# ------------------------------------------------------------------ HOPO / tap / open
def _hopo_flags(notes: list[FretNote], events: list[Event], tm: TempoMap, max_ratio: float,
                allow_taps: bool, solos: list[Phrase]) -> None:
    """Decide que notas son HOPO y lo deja explicito (asi no depende del tempo elegido)."""
    natural = resolve_hopos(notes, HOPO_THR_TICKS)
    want = []
    for i, n in enumerate(notes):
        if i == 0 or n.is_chord:
            want.append(False)
            continue
        prev = notes[i - 1]
        gap_s = secs_between(tm, prev.tick, n.tick)
        same = set(n.frets) == set(prev.frets) or (prev.is_chord and set(n.frets) <= set(prev.frets))
        e = events[i]
        cand = not same and (gap_s <= 0.17 or (e.legato and gap_s <= 0.3))
        want.append(cand)
    # limitar proporcion: se quedan los mas ligados (ataque flojo)
    idx = [i for i, w in enumerate(want) if w]
    if len(idx) > max_ratio * len(notes):
        idx_sorted = sorted(idx, key=lambda i: (not events[i].legato, events[i].strength))
        keep = set(idx_sorted[: int(max_ratio * len(notes))])
        want = [w and (i in keep) for i, w in enumerate(want)]
    for i, n in enumerate(notes):
        if want[i] != natural[i]:
            n.hopo = want[i]
        else:
            n.hopo = None
    if not allow_taps:
        return
    # taps: solo en solos, rafagas muy rapidas y ligadas de 6+ notas
    run: list[int] = []
    for i in range(1, len(notes) + 1):
        ok = i < len(notes) and want[i] and events[i].legato and any(p.start <= notes[i].tick < p.end for p in solos) \
            and secs_between(tm, notes[i - 1].tick, notes[i].tick) <= 0.12
        if ok:
            run.append(i)
        else:
            if len(run) >= 6:
                for k in run:
                    notes[k].tap = True
                    notes[k].hopo = None
            run = []


def _apply_opens(events: list[Event], sizes: list[int], instrument: str) -> None:
    """Notas abiertas: solo en el bajo, nota mas grave de la cancion repetida (pedal), muy pocas."""
    if instrument != "bass" or len(events) < 20:
        return
    roots = np.array([e.root for e in events])
    lowest = int(np.percentile(roots, 1))
    run: list[int] = []
    marked = 0
    limit = int(0.05 * len(events))
    for i, e in enumerate(events + [None]):
        if e is not None and sizes[i] == 1 and e.root <= lowest and e.root in (26, 28, 23, 24):
            run.append(i)
            continue
        if len(run) >= 4:
            # solo si en la frase hay notas mas agudas ocupando el verde
            neighbors = [events[k] for k in range(max(run[0] - 8, 0), min(run[-1] + 8, len(events)))]
            if any(n.lanes and n.lanes[0] == 0 and n.root > lowest for n in neighbors) and marked + len(run) <= limit:
                for k in run:
                    events[k].lanes = (OPEN,)
                marked += len(run)
        run = []


# ------------------------------------------------------------------ reduccion
REDUCE = {
    #          min beats, min secs, max lanes, max chord, chord prob, strum only
    "hard": (0.25, 0.12, 5, 2, 0.75, False),
    "medium": (0.5, 0.18, 4, 2, 0.45, True),
    "easy": (0.5, 0.30, 3, 1, 0.0, True),
}


def _reduce_learned(expert: list[FretNote], tm: TempoMap, diff: str, models: dict,
                    events: list[Event] | None = None) -> list[FretNote] | None:
    """Reduccion con el modelo aprendido de los charts de referencia (que notas quedan y en que carril)."""
    from ..calibration.reducer_model import MAX_LANE, note_features

    min_beats, min_secs, max_lanes, max_chord, chord_keep, strum_only = REDUCE[diff]
    m = models.get(diff)
    if not m or len(expert) < 8:
        return None
    X = note_features(expert, tm)
    p_keep = m["keep"].predict_proba(X)[:, 1]
    thr = REDUCE_THR[diff]
    if REDUCE_NEURAL and events and len(events) == len(expert) and all("nn_diff" in e.extra for e in events):
        # el modelo neuronal oye la cancion: sabe que golpes destacan y cuales son relleno
        from ..calibration.neural_onset import DIFFS

        k = DIFFS.index(diff) - 1
        nn_thr = events[0].extra.get("nn_thr", {}).get(diff, 0.5)
        pn = np.array([e.extra["nn_diff"][k] for e in events])
        p_keep = (1 - REDUCE_NEURAL) * p_keep + REDUCE_NEURAL * pn
        thr = (1 - REDUCE_NEURAL) * thr + REDUCE_NEURAL * nn_thr + REDUCE_NEURAL_THR_ADD
    # primera nota tras un silencio: siempre
    for i, n in enumerate(expert):
        if i == 0 or n.tick - expert[i - 1].tick >= RES * 2:
            p_keep[i] = max(p_keep[i], 0.9)
    ticks = [n.tick for n in expert]
    keep = nms_select(ticks, list(p_keep), int(min_beats * RES), min_secs, tm)
    keep = [i for i in keep if p_keep[i] >= thr]
    lanes = m["lane"].predict(X[keep]) if keep else []
    p_chord = m["chord"].predict_proba(X[keep])[:, 1] if keep else []
    out = []
    for k, i in enumerate(keep):
        n = expert[i]
        if OPEN in n.frets and diff == "hard":
            frets = (OPEN,)
        else:
            lo = int(min(lanes[k], MAX_LANE[diff]))
            exp_lo = min((f for f in n.frets if f != OPEN), default=None)
            if LOWER_LANES == "scaled" and exp_lo is not None:
                # mismo dibujo que Experto (sube donde sube, repite donde repite), en menos botones
                lo = int(round(exp_lo * MAX_LANE[diff] / 4))
            elif LOWER_LANES == "clip" and exp_lo is not None:
                lo = int(min(exp_lo, MAX_LANE[diff]))
            frets = [lo]
            exp_frets = [f for f in n.frets if f != OPEN]
            if len(exp_frets) > 1 and max_chord >= 2 and p_chord[k] >= 0.5:
                width = max(exp_frets) - min(exp_frets)
                top = min(lo + max(1, min(width, 2)), MAX_LANE[diff])
                if top > lo:
                    frets.append(top)
                    if diff == "hard" and len(exp_frets) >= 3 and top + 1 <= 4 and width >= 2:
                        pass  # en Dificil como mucho 2 notas
            frets = tuple(sorted(set(frets)))
        out.append(FretNote(n.tick, frets, n.sustain, hopo=False if strum_only else None, review=n.review))
    return out


def _reduce(expert: list[FretNote], events: list[Event], tm: TempoMap, diff: str, key: str) -> list[FretNote]:
    min_beats, min_secs, max_lanes, max_chord, chord_keep, strum_only = REDUCE[diff]
    if not expert:
        return []
    from ..calibration.reducer_model import load_models

    learned = _reduce_learned(expert, tm, diff, load_models(), events)
    if learned is not None:
        out = learned
        # en las dificultades faciles una nota larga tiene que serlo de verdad (en tu biblioteca, el largo
        # mediano pasa de 0,8 pulsos en Experto a 1 en Dificil, 1,5 en Medio y 2 en Facil)
        min_sus = int(RES * {"hard": 0.5, "medium": 0.75, "easy": 1.0}[diff])
        for a, b in zip(out, out[1:]):
            if a.sustain and a.tick + a.sustain > b.tick - RES // 24:
                a.sustain = max(0, b.tick - RES // 24 - a.tick)
        for a in out:
            if a.sustain and a.sustain < min_sus:
                a.sustain = 0
        if strum_only:
            nat = resolve_hopos([FretNote(x.tick, x.frets) for x in out], HOPO_THR_TICKS)
            for x, h in zip(out, nat):
                x.hopo = False if h else None
        return out
    target_ratio = stat(f"diff_ratios.{key}.{diff}.p50", {"hard": 0.82, "medium": 0.57, "easy": 0.36}[diff])
    ticks = [n.tick for n in expert]
    scores = []
    for i, n in enumerate(expert):
        s = metric_strength(n.tick, tm) * 1.2 + min(events[i].strength, 1.5) * 0.4
        s += 0.3 if n.sustain else 0.0
        s += 0.2 if n.is_chord else 0.0
        # primera nota despues de un silencio: muy importante
        if i == 0 or n.tick - expert[i - 1].tick >= RES * 2:
            s += 0.6
        scores.append(s)
    keep = nms_select(ticks, scores, int(min_beats * RES), min_secs, tm)
    # ajustar al numero objetivo de notas (proporcion tipica de los oficiales)
    target = int(round(target_ratio * len(expert)))
    if len(keep) > target * 1.08:
        keep = sorted(sorted(keep, key=lambda i: -scores[i])[:int(target * 1.08)])
    out: list[FretNote] = []
    lanes_used_max = max((max(f for f in n.frets if f != OPEN) for n in expert if any(f != OPEN for f in n.frets)), default=4)
    for rank, i in enumerate(keep):
        n = expert[i]
        frets = [f for f in n.frets if f != OPEN]
        is_open = OPEN in n.frets
        if is_open and diff in ("medium", "easy"):
            frets = [0]
            is_open = False
        # comprimir carriles
        if frets and max_lanes < 5:
            scale = (max_lanes - 1) / max(lanes_used_max, 1)
            frets = sorted(set(min(int(round(f * scale)), max_lanes - 1) for f in frets))
        # acordes
        if len(frets) > max_chord:
            frets = [frets[0], frets[-1]] if max_chord == 2 else [frets[0]]
        if len(frets) == 2 and chord_keep < 1.0:
            # en medio/facil quedan menos acordes: solo en tiempos fuertes o notas largas
            strong = metric_strength(n.tick, tm) >= 0.75 or n.sustain > 0
            if not strong or (rank % 100) / 100 >= chord_keep:
                frets = [frets[0]]
        if not frets:
            frets = [0]
        fn = FretNote(n.tick, (OPEN,) if is_open else tuple(frets), n.sustain, hopo=False if strum_only else None,
                      review=n.review)
        out.append(fn)
    # sustains no deben pisar la siguiente nota
    for a, b in zip(out, out[1:]):
        if a.sustain and a.tick + a.sustain > b.tick - RES // 4:
            a.sustain = max(0, b.tick - RES // 4 - a.tick)
            if a.sustain < RES // 2:
                a.sustain = 0
    # en Medio y Facil nada de HOPOs: forzar rasgueo donde el juego pondria HOPO natural
    if strum_only:
        nat = resolve_hopos([FretNote(x.tick, x.frets) for x in out], HOPO_THR_TICKS)
        for x, h in zip(out, nat):
            x.hopo = False if h else None
    else:
        _fix_lane_repeats(out, expert, keep)
    return out


def _fix_lane_repeats(out: list[FretNote], expert: list[FretNote], keep: list[int]) -> None:
    """En Dificil, que dos notas distintas seguidas no queden en el mismo carril por la reduccion."""
    for k in range(1, len(out)):
        a, b = expert[keep[k - 1]], expert[keep[k]]
        if a.frets != b.frets and out[k].frets == out[k - 1].frets and len(out[k].frets) == 1:
            f = out[k].frets[0]
            direction = 1 if (b.frets[0] > a.frets[0]) else -1
            nf = f + direction
            if 0 <= nf <= 4:
                out[k].frets = (nf,)


# ------------------------------------------------------------------ API
def generate_fret_track(notes: list[Note], sa: StemAnalysis, tm: TempoMap, instrument: str, time_offset: float,
                        solos_hint: list[tuple[float, float]] | None = None, log=print,
                        bass_sa: StemAnalysis | None = None) -> tuple[FretTrack, dict]:
    """Genera las 4 dificultades. `time_offset` = silencio anadido al principio (s). `bass_sa`: la pista del
    bajo (para no chartear en la guitarra los golpes del bajo que la IA deja en ella)."""
    key = "bass" if instrument == "bass" else ("keys" if instrument == "keys" else "guitar")
    ft = FretTrack()
    from ..calibration.onset_model import guitar_model_trusted, load_models, threshold

    # Medido contra charts oficiales: el modelo aprendido mejora mucho el bajo (F1 0.34 -> 0.72). En guitarra,
    # con pocas canciones de entrenamiento el metodo de votacion + re-ataques daba mas (0.77 frente a 0.72),
    # asi que el modelo de guitarra solo se usa cuando se ha entrenado con una biblioteca grande.
    models = load_models()
    model = None
    if instrument == "bass":
        model = models.get("bass")
    elif instrument in ("guitar", "rhythm") and guitar_model_trusted(models):
        model = models.get("guitar")
    inst_m = "bass" if instrument == "bass" else "guitar"
    end_tick = int(tm.sec_to_tick(sa.times[-1] + time_offset)) if len(sa.times) else RES * 400
    # pulsos en tiempo del audio original (sin el silencio anadido)
    beats = np.array([tm.tick_to_sec(t) - time_offset for t in range(0, end_tick + RES, RES)])
    net = None
    if instrument in ("guitar", "rhythm") and USE_NEURAL:
        from ..calibration.neural_onset import load_model

        net = load_model()
    elif instrument == "keys" and USE_NEURAL:
        from ..calibration.neural_keys import load_model

        net = load_model()
    elif instrument == "bass" and USE_NEURAL:
        from ..calibration.neural_bass import load_model

        net = load_model()
    canon = harm = None
    if net is not None and HARM_LANES:
        from .repeats import find_harmonic_repeats

        harm = find_harmonic_repeats(sa, beats)
        log(f"  partes con los mismos acordes (aunque cambie el ritmo): {np.mean(harm != np.arange(len(harm))):.0%} de la cancion")
    if net is not None and (REPEAT_AVERAGE or REPEAT_COPY):
        from .repeats import stats

        canon = _song_repeats(sa, beats, net)
        n_parts, frac = stats(canon)
        log(f"  partes repetidas: {n_parts} grupos de pulsos que suenan igual ({frac:.0%} de la cancion es repeticion)")
    if net is not None:
        events = build_events_neural(notes, sa, net, beats, canon if REPEAT_AVERAGE else None, instrument)
        log(f"  {len(events)} notas elegidas por el modelo neuronal")
        events = [e for e in events if e.pitches]
        if bass_sa is not None and BASS_BLEED_DB:
            n0 = len(events)
            events = _drop_bass_bleed(events, sa, bass_sa)
            if n0 - len(events):
                log(f"  {n0 - len(events)} notas quitadas: eran golpes del bajo que se colaban en la guitarra")
    elif model is not None:
        events = build_events_learned(notes, sa, beats, inst_m, model, thr=threshold(models, inst_m))
        log(f"  {len(events)} notas elegidas por el modelo aprendido de charts")
        events = [e for e in events if e.pitches]
    else:
        events = build_events(notes, sa)
        events = _clean_events(events, sa, instrument)
    for e in events:
        e.time += time_offset
        e.end += time_offset
        _simplify_pitches(e, instrument)
    if len(events) < 8:
        return ft, {"events": 0}
    ticks = quantize_ticks(np.array([e.time for e in events]), tm)
    for e, t in zip(events, ticks):
        e.tick = int(t)
        # si la rejilla esta mal en ese tramo (pulso detectado algo desviado), la semicorchea mas cercana puede quedar
        # lejos del golpe: entonces se pone donde suena, en la division fina mas cercana
        if QUANT_MAX_S is not None and abs(tm.tick_to_sec(e.tick) - e.time) > QUANT_MAX_S:
            fine = RES // QUANT_FINE
            e.tick = int(round(tm.sec_to_tick(e.time) / fine) * fine)
    # rasgueo doble ("pa-pam"): dos ataques fuertes que la rejilla junta en el mismo tick. El segundo pasa a la
    # semicorchea siguiente solo si de verdad estan separados casi una semicorchea; si no, es el mismo rasgueo
    # (las cuerdas suenan una tras otra y dan varios ataques) y se juntan. Nunca a la fusa: los charts oficiales
    # casi no las usan y partian un rasgueo de 3 en 4 notas
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        if b.tick != a.tick:
            continue
        sixteenth = tm.tick_to_sec(a.tick + RES // 4) - tm.tick_to_sec(a.tick)
        nxt = events[i + 1].tick if i + 1 < len(events) else b.tick + RES
        if (b.time - a.time >= DOUBLE_STRUM_FRAC * sixteenth and b.strength >= DOUBLE_STRUM_MIN and a.strength >= 0.2
                and a.tick + RES // 4 < nxt):
            b.tick = a.tick + RES // 4
    # separacion minima: una semicorchea (en canciones muy lentas, donde la semicorchea dura mas de 0,2 s,
    # se permite media). Lo que quede mas junto es el mismo golpe partido en dos: se queda el mas fuerte
    kept: list[Event] = []
    for e in events:
        if kept and e.tick != kept[-1].tick:
            p = kept[-1]
            sixteenth = tm.tick_to_sec(p.tick + RES // 4) - tm.tick_to_sec(p.tick)
            min_ticks = RES // 8 if sixteenth > 0.2 else MIN_SEP_TICKS
            if e.tick - p.tick < min_ticks - 1:
                if e.strength > p.strength:
                    e.tick = p.tick  # el golpe empieza donde empezo el rasgueo
                    kept[-1] = e
                else:
                    p.end = max(p.end, e.end)
                continue
        kept.append(e)
    events = kept
    # fusionar eventos que caen en el mismo tick
    merged: list[Event] = []
    for e in events:
        if merged and merged[-1].tick == e.tick:
            m = merged[-1]
            m.pitches = sorted(set(m.pitches + e.pitches))
            m.end = max(m.end, e.end)
            m.strength = max(m.strength, e.strength)
            m.votes = max(m.votes, e.votes)
        else:
            merged.append(e)
    events = merged
    for e in events:
        e.extra["raw"] = sorted(set(e.pitches))
        _simplify_pitches(e, instrument)
    _assign_sustains(events, tm, key)
    events = _thin_expert(events, tm, instrument, log)
    _assign_sustains(events, tm, key)
    chord_m = models.get("chord") if instrument in ("guitar", "rhythm") else None
    if chord_m is not None and events:
        # el modelo de acordes (aprendido de charts) dice cuantas notas pone un charter en cada golpe
        from ..calibration.onset_model import onset_features

        times0 = np.array([e.extra.get("cand_t", e.time - time_offset) for e in events])
        Xc = onset_features(sa, times0, beats, "guitar")
        pred = chord_m.predict(Xc[:, :chord_m.n_features_in_])
        for e, k in zip(events, pred):
            e.extra["chord_pred"] = int(k)
    sizes = _decide_chords(events, tm, key)
    if instrument == "bass":
        sizes = [1 if s > 1 and np.random.default_rng(i).random() > 0.15 else s for i, s in enumerate(sizes)]
        sizes = [min(s, 2) for s in sizes]
    copy = canon is not None and REPEAT_COPY
    if copy:
        from .repeats import choose_sources, clip_sustains, copy_beats

        for e, sz in zip(events, sizes):
            e.extra["size"] = sz
        events = copy_beats(events, choose_sources(events, canon, lambda e: e.extra["size"]), _clone_event(tm))
        sizes = [e.extra["size"] for e in events]
        clip_sustains(events, attr="sus_ticks", min_len=int(RES * 0.375))
    _assign_lanes(events, sizes, key)
    if harm is not None:
        from .repeats import lane_targets

        tg = lane_targets([(e.tick, e.root, min(e.lanes)) for e in events], harm)
        if tg:
            _assign_lanes(events, sizes, key, tg, debias=False)
    _apply_opens(events, sizes, instrument)
    if copy:
        _copy_lanes(events, canon)
    expert = [FretNote(e.tick, e.lanes, e.sus_ticks, conf=e.votes) for e in events]
    # solos (zonas que nos indican desde fuera: sin voz y guitarra protagonista)
    for a, b in (solos_hint or []):
        ft.solos.append(Phrase(int(tm.sec_to_tick(a + time_offset)), int(tm.sec_to_tick(b + time_offset))))
    hopo_max = stat(f"{key}.expert.hopo_ratio.p90", 0.35)
    _hopo_flags(expert, events, tm, min(hopo_max, 0.35), allow_taps=instrument in ("guitar",), solos=ft.solos)
    ft.notes["expert"] = expert
    for d in ("hard", "medium", "easy"):
        red = _reduce(expert, events, tm, d, key)
        if d == "hard":
            hard_events = [events[[n.tick for n in expert].index(x.tick)] for x in red]
            _hopo_flags(red, hard_events, tm, stat(f"{key}.hard.hopo_ratio.p90", 0.25), allow_taps=False,
                        solos=ft.solos)
        ft.notes[d] = red
    if copy:
        # al final, cada dificultad: la repeticion se toca exactamente igual que la primera vez
        for d in DIFFICULTIES:
            if d != "expert" and REPEAT_LOWER == "none":
                continue
            src = canon if d != "expert" and REPEAT_LOWER == "first" else choose_sources(ft.notes[d], canon, lambda n: n.frets)
            ft.notes[d] = copy_beats(ft.notes[d], src)
            clip_sustains(ft.notes[d])
    info = {
        "events": len(events),
        "nps": len(expert) / max(secs_between(tm, expert[0].tick, expert[-1].tick), 1.0) if expert else 0,
        "chord_ratio": float(np.mean([n.is_chord for n in expert])) if expert else 0,
        "event_pitches": {e.tick: e.extra.get("raw", e.pitches) for e in events},
    }
    if instrument == "keys":
        info["canon"], info["beats"] = canon, beats  # para que Pro Keys repita las mismas partes
    return ft, info

