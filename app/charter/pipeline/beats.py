"""Deteccion de pulsos y compases (Beat This!, ensemble de 3 modelos) y mapa de tempo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..chart.model import RES, TempoMap

FPS = 50  # Beat This trabaja a 50 frames por segundo
ONNX_REPO = "https://huggingface.co/ashudesai/songbird-models/resolve/main/{name}.onnx"
ENSEMBLE = ("final0", "final1", "final2")
# ajuste de los pulsos a las notas (fit_grid_to_notes)
FIT_NOTE_BIAS = 0.01  # el modelo pone las notas un poco antes que los charts oficiales (s)
FIT_PRE = 0.05  # una nota un poco antes del pulso es de ese pulso
FIT_RANGE = 0.14  # como mucho, algo mas de media semicorchea (en pulsos)
FIT_STEP = 0.005
FIT_CAP = 0.04  # una nota que no encaja (s) no debe mandar sobre las demas
FIT_SMOOTH = 1.5  # coste de cambiar de un pulso al siguiente, en notas
FIT_PREFER_SMALL = 0.03
FIT_GOOD = 0.012  # con un error medio asi (s), basta un desplazamiento unico
FIT_KEEP_CONSTANT = 0.018  # un tempo fijo solo se abandona si no encaja de verdad
PHASE_MIN_FIT = 0.88  # fraccion de pulsos que debe caer en la rejilla regular para considerar el tempo fijo


def _onnx_path(name: str) -> Path:
    return paths.MODELS / "beat_this" / f"{name}.onnx"


def ensure_models(log=print) -> None:
    import requests

    for name in ENSEMBLE:
        p = _onnx_path(name)
        if p.exists() and p.stat().st_size > 1_000_000:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        log(f"Descargando modelo de pulso {name}...")
        with requests.get(ONNX_REPO.format(name=name), stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = p.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(p)


class _OnnxBeatModel:
    def __init__(self, path: Path):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:  # noqa: BLE001
            pass
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(str(path), so, providers=providers)
        except Exception:  # noqa: BLE001
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, chunk):
        import torch

        beat, down = self.sess.run(None, {"input_spectrogram": chunk.cpu().numpy().astype(np.float32)})
        return {"beat": torch.from_numpy(beat), "downbeat": torch.from_numpy(down)}


def _torch_checkpoint(name: str) -> Path | None:
    p = paths.MODELS / "torch" / "hub" / "checkpoints" / f"beat_this-{name}.ckpt"
    return p if p.exists() else None


def _windowed_predict(spect: np.ndarray, model, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Prediccion por ventanas solapadas promediando con pesos triangulares."""
    import torch

    n = len(spect)
    acc_b = np.zeros(n)
    acc_d = np.zeros(n)
    wsum = np.zeros(n)
    starts = list(range(0, max(n - win, 0) + 1, hop))
    if not starts or starts[-1] + win < n:
        starts.append(max(n - win, 0))
    taper = np.minimum(np.arange(1, win + 1), np.arange(win, 0, -1)).astype(float)
    taper = np.minimum(taper / (win * 0.25), 1.0)
    for s in starts:
        chunk = spect[s:s + win]
        out = model(torch.from_numpy(chunk[None].astype(np.float32)))
        b = out["beat"][0].numpy() if hasattr(out["beat"], "numpy") else np.asarray(out["beat"][0])
        d = out["downbeat"][0].numpy() if hasattr(out["downbeat"], "numpy") else np.asarray(out["downbeat"][0])
        w = taper[: len(chunk)].copy()
        if s == 0:
            w[: win // 2] = 1.0
        if s + win >= n:
            w[len(chunk) // 2:] = 1.0
        acc_b[s:s + len(chunk)] += b * w
        acc_d[s:s + len(chunk)] += d * w
        wsum[s:s + len(chunk)] += w
    wsum[wsum == 0] = 1
    return acc_b / wsum, acc_d / wsum


def beat_activations(audio_mono: np.ndarray, sr: int, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve las probabilidades (sigmoid) de pulso y de compas a 50 fps, promediando 3 modelos."""
    import soxr
    import torch
    from beat_this.preprocessing import LogMelSpect

    sig = soxr.resample(audio_mono, in_rate=sr, out_rate=22050) if sr != 22050 else audio_mono
    spect = LogMelSpect(device="cpu")(torch.tensor(sig, dtype=torch.float32))
    beats, downs = [], []
    if all(_torch_checkpoint(n) for n in ENSEMBLE):
        # modelos originales en PyTorch: ventanas de 30 s (lo ideal)
        from beat_this.inference import Spect2Frames

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name in ENSEMBLE:
            s2f = Spect2Frames(str(_torch_checkpoint(name)), device=dev)
            b, d = s2f(spect.to(dev))
            beats.append(torch.sigmoid(b).cpu().numpy())
            downs.append(torch.sigmoid(d).cpu().numpy())
            del s2f
    else:
        # version ONNX: solo admite ~6 s de contexto, se usan ventanas muy solapadas
        ensure_models(log)
        sp = spect.numpy()
        for name in ENSEMBLE:
            model = _OnnxBeatModel(_onnx_path(name))
            b, d = _windowed_predict(sp, model, win=296, hop=74)
            beats.append(1 / (1 + np.exp(-b)))
            downs.append(1 / (1 + np.exp(-d)))
            del model
    return np.mean(beats, axis=0), np.mean(downs, axis=0)


def _peaks(act: np.ndarray, thr: float = 0.5, min_dist: int = 7) -> np.ndarray:
    """Picos locales por encima del umbral, con distancia minima (en frames)."""
    from scipy.signal import find_peaks

    idx, _ = find_peaks(act, height=thr, distance=min_dist)
    # refinamiento sub-frame por interpolacion parabolica
    out = []
    for i in idx:
        if 0 < i < len(act) - 1:
            a, b, c = act[i - 1], act[i], act[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den != 0 else 0.0
            out.append(i + float(np.clip(off, -0.5, 0.5)))
        else:
            out.append(float(i))
    return np.array(out) / FPS


@dataclass
class BeatGrid:
    beats: np.ndarray  # tiempos (s) de cada pulso
    downbeat_idx: np.ndarray  # indices de pulsos que son inicio de compas
    constant_bpm: float | None  # si la cancion tiene tempo fijo


def track_beats(audio_mono: np.ndarray, sr: int, log=print) -> BeatGrid:
    act_b, act_d = beat_activations(audio_mono, sr, log)
    beats = _peaks(act_b, 0.5)
    downs = _peaks(act_d, 0.5)
    if len(beats) < 8:
        # cancion sin pulso claro: tempo fijo estimado a 120
        dur = len(audio_mono) / sr
        beats = np.arange(0, dur, 0.5)
        return BeatGrid(beats, np.arange(0, len(beats), 4), 120.0)
    beats = _repair_local(_repair_beats(beats))
    # asignar compases: cada downbeat al pulso mas cercano
    nearest = [beats[int(np.argmin(np.abs(beats - d)))] for d in downs]
    return _grid_from(beats, np.array([b for b, d in zip(nearest, downs) if abs(b - d) < 0.07]))


def _repair_beats(beats: np.ndarray) -> np.ndarray:
    """Rellena pulsos perdidos y quita pulsos dobles usando el intervalo local."""
    ibi = np.diff(beats)
    med = np.median(ibi)
    out = [beats[0]]
    for b in beats[1:]:
        gap = b - out[-1]
        if gap < 0.55 * med:
            continue  # pulso espurio
        n = int(round(gap / med))
        if n >= 2 and abs(gap / n - med) < 0.15 * med:
            start = out[-1]
            for k in range(1, n):
                out.append(start + gap * k / n)
        out.append(b)
    return np.array(out)


def _repair_local(beats: np.ndarray) -> np.ndarray:
    """Como _repair_beats, pero con el pulso de la zona (no el de toda la cancion): en intros lentas o sin ritmo
    claro el detector se salta pulsos o mete de mas, y el tempo saltaria (56-165 BPM donde va a 92)."""
    ibi = np.diff(beats)
    out = [float(beats[0])]
    for i in range(1, len(beats)):
        p = float(np.percentile(ibi[max(i - 9, 0): i + 8], 35))
        gap = beats[i] - out[-1]
        if gap < 0.6 * p and i + 1 < len(beats) and abs(beats[i + 1] - out[-1] - p) < abs(gap - p):
            continue  # pulso de mas
        n = int(round(gap / p))
        if n >= 2 and abs(gap / n - p) < 0.2 * p:
            out.extend(out[-1] + gap * k / n for k in range(1, n))  # pulsos perdidos
        out.append(float(beats[i]))
    return np.array(out)


def _constant_by_phase(beats: np.ndarray, tol: float = 0.035) -> tuple[float, float] | None:
    """Tempo fijo aunque algunos pulsos esten mal (a _constant_tempo le basta un pulso de mas para descuadrarse,
    porque cuenta pulsos): se busca la rejilla regular a la que caen casi todos, midiendo tiempos."""
    p0 = float(np.median(np.diff(beats)))

    def fit(p: float) -> tuple[float, float, float, float]:
        ph = beats / p
        phi = float(np.angle(np.mean(np.exp(2j * np.pi * ph)))) / (2 * np.pi)
        r = ((ph - phi + 0.5) % 1 - 0.5) * p
        inl = np.abs(r) <= tol
        return float(np.mean(inl)), float(np.std(r[inl])) if inl.any() else 1.0, phi * p, float(np.mean(np.abs(r)))

    ps = p0 * np.linspace(0.985, 1.015, 1201)
    fits = [fit(p) for p in ps]
    best = max(range(len(ps)), key=lambda k: (fits[k][0], -fits[k][3]))
    period = float(ps[best])
    frac, std, t0, _ = fits[best]
    if frac < PHASE_MIN_FIT or std > 0.018:
        return None
    # todos los puntos de la rejilla deben tener su pulso (no vale media cancion a otro tempo que caiga por casualidad)
    grid = np.arange(np.ceil((beats[0] - t0) / period), np.floor((beats[-1] - t0) / period) + 1) * period + t0
    if np.mean(np.abs(grid[:, None] - beats[None, :]).min(axis=1) <= tol) < PHASE_MIN_FIT:
        return None
    bpm = 60.0 / period
    for cand in (round(bpm), round(bpm * 2) / 2, round(bpm, 1)):
        c = fit(60.0 / cand)
        if cand > 0 and c[0] >= frac - 0.02:
            return float(cand), c[2]
    return float(bpm), t0


def _grid_from(beats: np.ndarray, down_t: np.ndarray) -> BeatGrid:
    """Rejilla final a partir de los pulsos reparados y los inicios de compas detectados (en segundos)."""
    const = _constant_tempo(beats) or _constant_by_phase(beats)
    if const is not None:
        bpm, t0 = const
        period = 60.0 / bpm
        # rehacer la rejilla perfectamente regular sobre toda la cancion
        n0 = int(np.floor((beats[0] - t0) / period + 0.5))
        n1 = int(np.floor((beats[-1] - t0) / period + 0.5))
        beats = t0 + period * np.arange(n0, n1 + 1)
    down_idx = np.unique([int(np.argmin(np.abs(beats - d))) for d in down_t]).astype(int) if len(down_t) else np.zeros(0, int)
    return BeatGrid(beats, _regularize_downbeats(len(beats), down_idx), const[0] if const else None)


def refine_grid(grid: BeatGrid) -> BeatGrid:
    """Aplica las reparaciones a una rejilla ya calculada (guardada de antes); si ya es regular no cambia."""
    if grid.constant_bpm is not None:
        return grid
    return _grid_from(_repair_local(grid.beats), grid.beats[grid.downbeat_idx])


def _constant_tempo(beats: np.ndarray) -> tuple[float, float] | None:
    """Si los pulsos encajan en un tempo fijo (cancion de estudio con claqueta), lo devuelve."""
    idx = np.arange(len(beats))
    A = np.vstack([idx, np.ones_like(idx)]).T.astype(float)
    (period, t0), *_ = np.linalg.lstsq(A, beats, rcond=None)
    resid = beats - (idx * period + t0)
    # robusto: ignorar el 5% peor (silencios, finales con ritardando)
    keep = np.abs(resid) <= np.percentile(np.abs(resid), 95)
    (period, t0), *_ = np.linalg.lstsq(A[keep], beats[keep], rcond=None)
    resid = beats - (idx * period + t0)
    std = np.std(resid[keep])
    if std > 0.018 or np.percentile(np.abs(resid), 90) > 0.035:
        return None
    bpm = 60.0 / period
    # muchas canciones estan a BPM entero o con un decimal
    for cand in (round(bpm), round(bpm * 2) / 2, round(bpm, 1)):
        p = 60.0 / cand
        r = beats - (idx * p + t0)
        t0c = t0 + np.median(r)
        r = beats - (idx * p + t0c)
        if np.std(r[keep]) <= std * 1.15 + 0.002 and np.percentile(np.abs(r), 90) < 0.035:
            return float(cand), float(t0c)
    return float(bpm), float(t0)


def _regularize_downbeats(n_beats: int, down_idx: np.ndarray) -> np.ndarray:
    """Elige el compas mas probable (normalmente 4/4) y su fase, respetando cambios largos."""
    if len(down_idx) < 2:
        return np.arange(0, n_beats, 4)
    gaps = np.diff(down_idx)
    vals, counts = np.unique(gaps, return_counts=True)
    bpb = int(vals[np.argmax(counts)])
    if bpb not in (2, 3, 4, 5, 6, 7):
        bpb = 4
    # fase: la que mas downbeats detectados explica
    best_phase = max(range(bpb), key=lambda ph: np.sum((down_idx - ph) % bpb == 0))
    return np.arange(best_phase, n_beats, bpb)


def _grid_error(rel: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Distancia (s) de cada nota a la semicorchea o al tresillo mas cercano. `rel`: pulsos desde el pulso de la nota."""
    e4 = np.abs(rel * 4 - np.round(rel * 4)) / 4
    e3 = np.abs(rel * 3 - np.round(rel * 3)) / 3
    return np.minimum(e4, e3) * period


def fit_grid_to_notes(grid: BeatGrid, notes: np.ndarray, log=print) -> BeatGrid:
    """Ajusta los pulsos a las notas que suenan. El detector de pulsos puede ir hasta media semicorchea desplazado (en
    canciones tocadas sin claqueta, o en las que se ha tomado por fijo un tempo que no lo es) y al cuantizar todas las
    notas se mueven 50-90 ms: se nota al jugar y hacia caer el F1 de Santana de 0,84 a 0,65."""
    notes = np.sort(np.asarray(notes, np.float64)) + FIT_NOTE_BIAS
    beats = grid.beats
    if len(notes) < 16 or len(beats) < 8:
        return grid
    k = np.clip(np.searchsorted(beats, notes + FIT_PRE, side="right") - 1, 0, len(beats) - 2)
    period = np.diff(beats)
    p_med = float(np.median(period))
    shifts = np.arange(-FIT_RANGE, FIT_RANGE + 1e-9, FIT_STEP) * p_med
    # coste de cada desplazamiento en cada pulso: lo lejos que quedan sus notas de la rejilla
    cost = np.zeros((len(beats), len(shifts)))
    for j, d in enumerate(shifts):
        rel = (notes - d - beats[k]) / period[k]
        np.add.at(cost[:, j], k, np.minimum(_grid_error(rel, period[k]), FIT_CAP))
    before = float(np.mean(np.minimum(_grid_error((notes - beats[k]) / period[k], period[k]), FIT_CAP)))
    # primero, un desplazamiento unico para toda la cancion: si basta, se queda el tempo que hubiera
    total = cost.sum(0) + FIT_PREFER_SMALL * np.abs(shifts) * len(notes)
    best = int(np.argmin(total))
    glob = float(cost[:, best].sum() / len(notes))
    if glob <= FIT_GOOD or (grid.constant_bpm is not None and glob <= before and glob <= FIT_KEEP_CONSTANT):
        if abs(shifts[best]) < 0.001:
            return grid
        log(f"Pulsos ajustados a las notas: {shifts[best] * 1000:+.0f} ms (error medio {before * 1000:.0f} -> {glob * 1000:.0f} ms)")
        return BeatGrid(beats + shifts[best], grid.downbeat_idx, grid.constant_bpm)
    # si no, un desplazamiento por pulso que cambie poco a poco
    n, m = cost.shape
    step = np.abs(shifts[:, None] - shifts[None, :]) * FIT_SMOOTH
    acc = cost[0] + FIT_PREFER_SMALL * np.abs(shifts)
    back = np.zeros((n, m), int)
    for i in range(1, n):
        tot = acc[:, None] + step
        back[i] = np.argmin(tot, axis=0)
        acc = tot[back[i], np.arange(m)] + cost[i] + FIT_PREFER_SMALL * np.abs(shifts)
    path = np.zeros(n, int)
    path[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    fitted = beats + shifts[path]
    if np.any(np.diff(fitted) < 0.5 * period.min()):
        return grid
    per = np.diff(fitted)
    after = float(np.mean(np.minimum(_grid_error((notes - fitted[k]) / per[k], per[k]), FIT_CAP)))
    if after >= before:
        return grid
    log(f"Pulsos ajustados a las notas, poco a poco: error medio {before * 1000:.0f} -> {after * 1000:.0f} ms")
    return BeatGrid(fitted, grid.downbeat_idx, None)


def build_tempo_map(grid: BeatGrid, lead_in: float) -> tuple[TempoMap, float]:
    """Construye el TempoMap con el tick 0 = segundo 0 del audio final (con `lead_in` de silencio).

    Devuelve (tempo_map, lead_in_real).
    """
    beats = grid.beats + lead_in
    down = grid.downbeat_idx
    bpb = int(np.median(np.diff(down))) if len(down) > 1 else 4
    fd = int(down[0]) if len(down) else 0
    beats = beats[fd:]  # el mapa arranca en el primer compas; lo anterior se extrapola
    t_d = float(beats[0])
    if grid.constant_bpm is not None:
        p0 = 60.0 / grid.constant_bpm
    else:
        p0 = float(np.median(np.diff(beats[: min(len(beats), 9)])))
    measure = bpb * p0
    m_true = int(np.floor(t_d / measure))
    head = t_d - m_true * measure
    tempos = []
    if m_true >= 1 and head < 0.25 * measure:
        # el sobrante es pequeno: el primer compas dura un poco mas
        tempos.append((0, int(round((head + measure) / bpb * 1e6))))
        n_meas = m_true
    else:
        # compas inicial "de relleno" con el tiempo sobrante, el resto al tempo real
        tempos.append((0, int(round(max(head, 0.05) / bpb * 1e6))))
        n_meas = m_true + 1
    if n_meas > 1:
        tempos.append((bpb * RES, int(round(p0 * 1e6))))
    start_tick = n_meas * bpb * RES
    if grid.constant_bpm is not None:
        tempos.append((start_tick, int(round(p0 * 1e6))))
    else:
        # tempo variable: un cambio de tempo por pulso, anclado a cada pulso detectado
        for i in range(len(beats) - 1):
            us = int(round((beats[i + 1] - beats[i]) * 1e6))
            tempos.append((start_tick + i * RES, us))
    tempos = _dedupe(tempos)
    tm = TempoMap(tempos=tempos, time_sigs=[(0, bpb, 4)])
    return tm, lead_in


def _dedupe(tempos):
    out = []
    for t, us in tempos:
        if out and out[-1][0] == t:
            out[-1] = (t, us)
        elif out and abs(out[-1][1] - us) <= 1:
            continue
        else:
            out.append((t, us))
    return out
