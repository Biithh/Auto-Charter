"""Transcripcion de notas con YourMT3+ (varias pasadas en ensemble) + analisis de la pista.

Para evitar las notas inventadas por restos de la separacion, cada nota necesita:
  * aparecer en varias pasadas independientes del modelo (votacion), y
  * tener energia real en la pista separada a su altura (comparada con el ruido de fondo).
La duracion (sustain) se mide con la energia real de la nota, no con lo que dice el modelo.
"""

from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths

TR_SR = 16000
CQT_SR = 22050
CQT_HOP = 256  # ~11.6 ms
CQT_FMIN_MIDI = 24  # C1
CQT_BINS = 84  # hasta B7


@dataclass
class Note:
    onset: float
    offset: float
    pitch: int
    program: int = 0
    is_drum: bool = False
    votes: float = 1.0


@dataclass
class StemAnalysis:
    """Informacion de la pista separada para validar notas."""

    cqt_db: np.ndarray  # [84, frames] energia en dB por semitono
    onset_env: np.ndarray  # fuerza de ataque por frame
    rms_db: np.ndarray  # volumen por frame
    noise_floor_db: float
    active_db: float  # volumen tipico cuando el instrumento toca
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def frame(self, t: float) -> int:
        return int(np.clip(round(t * CQT_SR / CQT_HOP), 0, self.cqt_db.shape[1] - 1))


# ------------------------------------------------------------------ YourMT3
_MODEL = None


def _load_yourmt3():
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("MT3_CHECKPOINT_DIR", str(paths.MODELS / "mt3"))
        import torch
        from mt3_infer import load_model

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = load_model("yourmt3", device=dev)
    return _MODEL


def unload() -> None:
    global _MODEL
    _MODEL = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _midi_to_notes(path: Path) -> list[Note]:
    import mido

    mf = mido.MidiFile(str(path))
    notes: list[Note] = []
    tempo = 500000
    for tr in mf.tracks:
        t = 0.0
        program = 0
        open_: dict[tuple[int, int], tuple[float, int]] = {}
        for msg in tr:
            t += mido.tick2second(msg.time, mf.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "program_change":
                program = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                open_[(msg.channel, msg.note)] = (t, program)
            elif msg.type in ("note_off", "note_on"):
                k = (msg.channel, msg.note)
                if k in open_:
                    st, prog = open_.pop(k)
                    notes.append(Note(st, max(t, st + 0.01), msg.note, prog, is_drum=msg.channel == 9))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def run_yourmt3(audio_16k: np.ndarray) -> list[Note]:
    model = _load_yourmt3()
    midi = model.transcribe(audio_16k.astype(np.float32), sr=TR_SR)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.mid"
        midi.save(str(p))
        return _midi_to_notes(p)


def ensemble_notes(runs: list[list[Note]], tol: float = 0.05, min_votes: float = 0.5,
                   octave_merge: bool = False) -> list[Note]:
    """Une varias transcripciones: una nota sobrevive si aparece en suficientes pasadas."""
    if not runs:
        return []
    n_runs = len(runs)
    pool = sorted([(n.onset, n.pitch, i, n) for i, r in enumerate(runs) for n in r if not n.is_drum],
                  key=lambda x: (x[0], x[1]))
    used = [False] * len(pool)
    out: list[Note] = []
    for a in range(len(pool)):
        if used[a]:
            continue
        t0, p0, r0, n0 = pool[a]
        group = [a]
        seen_runs = {r0}
        b = a + 1
        while b < len(pool) and pool[b][0] - t0 <= tol:
            tb, pb, rb, nb = pool[b]
            same = pb == p0 or (octave_merge and abs(pb - p0) == 12)
            if not used[b] and same and rb not in seen_runs:
                group.append(b)
                seen_runs.add(rb)
            b += 1
        votes = len(seen_runs) / n_runs
        for g in group:
            used[g] = True
        if votes + 1e-9 < min_votes:
            continue
        members = [pool[g][3] for g in group]
        onset = float(np.median([m.onset for m in members]))
        offset = float(np.median([m.offset for m in members]))
        progs = [m.program for m in members]
        out.append(Note(onset, offset, p0, max(set(progs), key=progs.count), votes=votes))
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


# ------------------------------------------------------------------ analisis de pista
def analyze_stem(audio_mono: np.ndarray, sr: int) -> StemAnalysis:
    import librosa

    y = librosa.resample(audio_mono, orig_sr=sr, target_sr=CQT_SR) if sr != CQT_SR else audio_mono
    C = np.abs(librosa.cqt(y, sr=CQT_SR, hop_length=CQT_HOP, fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                           n_bins=CQT_BINS, bins_per_octave=12))
    cqt_db = librosa.amplitude_to_db(C, ref=1.0, top_db=None)
    # flujo espectral tipo "superflux" (lag + filtro de maximo): ignora el vibrato y marca cada ataque
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=CQT_HOP))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=CQT_SR, n_mels=138, fmin=27.5, fmax=16000)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=CQT_SR, hop_length=CQT_HOP, lag=2,
                                         max_size=3)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=CQT_HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    n = min(cqt_db.shape[1], len(onset), len(rms_db))
    cqt_db, onset, rms_db = cqt_db[:, :n], onset[:n], rms_db[:n]
    loud = rms_db[rms_db > -90]
    active = float(np.percentile(loud, 75)) if len(loud) else -20.0
    # ruido de fondo: lo mas bajo de la pista, pero nunca pegado al nivel normal (instrumentos que
    # suenan todo el rato, como el bajo, no tienen "silencios" de los que medirlo)
    noise = float(min(np.percentile(loud, 3), active - 24.0)) if len(loud) else -80.0
    return StemAnalysis(cqt_db, onset, rms_db, noise, active, np.arange(n) * CQT_HOP / CQT_SR)


def pitch_energy(sa: StemAnalysis, pitch: int, t0: float, t1: float) -> np.ndarray:
    """Energia (dB) de la nota (fundamental + 2 armonicos) entre t0 y t1."""
    b = pitch - CQT_FMIN_MIDI
    f0, f1 = sa.frame(t0), max(sa.frame(t1), sa.frame(t0) + 1)
    rows = [r for r in (b, b + 12, b + 19) if 0 <= r < sa.cqt_db.shape[0]]
    if not rows:
        return np.full(f1 - f0, -100.0)
    return np.max(sa.cqt_db[rows, f0:f1], axis=0)


def validate_and_measure(notes: list[Note], sa: StemAnalysis, min_rel_db: float = -18.0) -> list[Note]:
    """Quita notas sin energia real y recalcula su duracion con la envolvente de la pista.

    min_rel_db: cuantos dB por debajo del volumen tipico del instrumento se admite una nota.
    """
    out = []
    for n in notes:
        f = sa.frame(n.onset)
        win = sa.rms_db[max(f - 2, 0): f + 6]
        loud = float(np.max(win)) if len(win) else -100
        if loud < sa.active_db + min_rel_db or loud < sa.noise_floor_db + 6:
            continue  # demasiado flojo: casi seguro que es resto de otra pista
        # sustain real: mientras la energia de la nota siga alta
        e = pitch_energy(sa, n.pitch, n.onset, max(n.offset, n.onset + 0.05) + 1.0)
        if len(e) == 0:
            continue
        peak = float(np.max(e[: min(len(e), 8)]))
        thr = max(peak - 14.0, sa.noise_floor_db + 4.0)
        below = np.where(e[3:] < thr)[0]
        end_frame = (below[0] + 3) if len(below) else len(e)
        measured = n.onset + end_frame * CQT_HOP / CQT_SR
        # nos quedamos con la duracion mas corta entre la del modelo y la medida (mas conservador)
        n.offset = max(n.onset + 0.03, min(n.offset, measured))
        out.append(n)
    return out


def detect_onsets(sa: StemAnalysis, k: float = 1.8, min_gap: float = 0.05) -> np.ndarray:
    """Ataques con umbral adaptativo: un pico cuenta si destaca sobre lo que suena alrededor (1 s)."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    env = sa.onset_env
    if len(env) == 0:
        return np.zeros(0)
    fps = CQT_SR / CQT_HOP
    med = median_filter(env, size=max(3, int(fps * 1.0)))
    floor = np.percentile(env, 55)
    # sin ataques en silencio (por debajo del ruido de la pista)
    quiet = sa.rms_db[: len(env)] < sa.noise_floor_db + 6
    height = np.maximum(med * k, floor)
    height[quiet[: len(height)]] = np.inf
    pk, _ = find_peaks(env, distance=max(1, int(min_gap * fps)), height=height)
    return pk / fps


def onset_support(notes: list[Note], onsets: np.ndarray, tol: float = 0.06) -> np.ndarray:
    """Para cada nota, si hay un ataque real cerca en la pista."""
    if len(onsets) == 0:
        return np.zeros(len(notes), bool)
    out = np.zeros(len(notes), bool)
    for i, n in enumerate(notes):
        j = np.searchsorted(onsets, n.onset)
        for k in (j - 1, j):
            if 0 <= k < len(onsets) and abs(onsets[k] - n.onset) <= tol:
                out[i] = True
    return out


def transcribe_pitched(stem: np.ndarray, sr: int, log=print, check_pause=lambda: None,
                       passes: str = "full") -> dict:
    """Transcribe una pista (estereo) con varias pasadas.

    Devuelve dict con 'mono' (ensemble), 'left', 'right' (por canal) y 'analysis'.
    """
    import librosa

    mono = stem.mean(axis=0) if stem.ndim == 2 else stem
    m16 = librosa.resample(mono, orig_sr=sr, target_sr=TR_SR)
    runs = []
    log("  transcripcion pasada 1 (mezcla mono)")
    runs.append(run_yourmt3(m16))
    check_pause()
    left = right = None
    if stem.ndim == 2 and passes == "full":
        l16 = librosa.resample(stem[0], orig_sr=sr, target_sr=TR_SR)
        r16 = librosa.resample(stem[1], orig_sr=sr, target_sr=TR_SR)
        log("  transcripcion pasada 2 (canal izquierdo)")
        left = run_yourmt3(l16)
        check_pause()
        log("  transcripcion pasada 3 (canal derecho)")
        right = run_yourmt3(r16)
        check_pause()
        runs += [left, right]
        # pasada desplazada 23 ms: el modelo trocea el audio en segmentos, asi cambian los cortes
        log("  transcripcion pasada 4 (desplazada)")
        shift = int(0.023 * TR_SR)
        shifted = run_yourmt3(np.concatenate([np.zeros(shift, np.float32), m16]))
        for n in shifted:
            n.onset -= shift / TR_SR
            n.offset -= shift / TR_SR
        runs.append(shifted)
        check_pause()
    return {"runs": runs, "left": left, "right": right, "mono_run": runs[0]}


def transcribe_drums(stem: np.ndarray, sr: int, log=print) -> list[Note]:
    import librosa

    mono = stem.mean(axis=0) if stem.ndim == 2 else stem
    m16 = librosa.resample(mono, orig_sr=sr, target_sr=TR_SR)
    notes = run_yourmt3(m16)
    return [n for n in notes if n.is_drum]
