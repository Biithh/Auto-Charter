"""Partes repetidas de la cancion (estribillos, riffs que vuelven...): los charts oficiales las tocan igual cada vez.

Pulso a pulso se busca que otros pulsos de la cancion suenan igual durante al menos un compas seguido
(lo que oye el modelo neuronal + el espectro de la pista). Los pulsos que suenan igual forman un grupo:
  - lo que oye el modelo se promedia entre todas las veces que suena el grupo (mas fiable que una sola vez),
  - y todas las veces se tocan igual: la copia sale de la vez que mas se parece a las demas.

Medido en 102 canciones de la biblioteca que el modelo no vio al entrenar: de los pulsos que se copian, el
chart oficial tiene el mismo ritmo en el 89 %. En el banco de pruebas, las partes que se tocan igual pasan del
20 % al 43 % de los pulsos (en los oficiales, el 50 %) sin perder aciertos.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np

from ..chart.model import RES
from ..pipeline.transcription import CQT_HOP, CQT_SR, StemAnalysis

REPEAT_THR = 0.78  # parecido minimo (0..1) durante REPEAT_WIN pulsos seguidos (mas bajo = mas copias y mas fallos)
REPEAT_WIN = 4  # un compas de 4/4
REPEAT_MIN_LAG = 4  # un pulso no se compara con los de su mismo compas
NSUB = 4  # trozos por pulso al comparar
SUB = 12  # posiciones por pulso para el dibujo de ataques
ATTACK_W = 1.0  # peso del dibujo de ataques frente al tono
# partes con los mismos acordes aunque cambie el ritmo (el estribillo con otro rasgueo): ahi la misma nota va en
# el mismo boton. En el banco, los oficiales lo hacen en el 87 % de los casos
HARM_WIN = 8
HARM_THR = 0.9


def _spectrum_features(sa: StemAnalysis, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fps = CQT_SR / CQT_HOP
    C = np.maximum(sa.cqt_db, sa.noise_floor_db)
    env = sa.onset_env / (np.percentile(sa.onset_env, 99) + 1e-9)
    n = C.shape[1]
    F, act = [], []
    for a, b in zip(beats, beats[1:]):
        f0, f1 = int(np.clip(a * fps, 0, n - 1)), int(np.clip(b * fps, 0, n))
        if f1 - f0 < NSUB:
            F.append(np.zeros(84 * NSUB + SUB))
            act.append(False)
            continue
        edges = np.linspace(f0, f1, NSUB + 1).astype(int)
        parts = []
        for x, y in zip(edges, edges[1:]):
            c = C[:, x:max(y, x + 1)].mean(1)
            parts.append(c - c.mean())
        v = np.concatenate(parts)
        v = v / (np.linalg.norm(v) + 1e-9)
        pos = (a + (b - a) * np.arange(SUB) / SUB) * fps
        on = np.array([env[int(p): int(p) + 3].max() if 0 <= int(p) < len(env) - 3 else 0.0 for p in pos])
        o = on - on.mean()
        o = o / (np.linalg.norm(o) + 1e-9)
        F.append(np.concatenate([v, o * ATTACK_W]))
        act.append(bool(sa.rms_db[f0:f1].mean() > sa.noise_floor_db + 12))
    F = np.array(F)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9), np.array(act, bool)


def _model_features(H: np.ndarray, fps: float, beats: np.ndarray) -> np.ndarray:
    Hc = H - H.mean(0)
    n = len(Hc)
    F = []
    for a, b in zip(beats, beats[1:]):
        f0, f1 = int(np.clip(a * fps, 0, n - 1)), int(np.clip(b * fps, 0, n))
        if f1 - f0 < NSUB:
            F.append(np.zeros(Hc.shape[1] * NSUB))
            continue
        edges = np.linspace(f0, f1, NSUB + 1).astype(int)
        v = np.concatenate([Hc[x:max(y, x + 1)].mean(0) for x, y in zip(edges, edges[1:])])
        F.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(F)


def find_repeats(sa: StemAnalysis, beats: np.ndarray, H: np.ndarray, fps: float) -> np.ndarray:
    """Para cada pulso, el primer pulso de la cancion que suena igual (el mismo si no se repite).

    `beats`: tiempo (s, audio original) de cada pulso; `H`: estado interno del modelo neuronal por trama."""
    nb = len(beats) - 1
    canon = np.arange(nb)
    if nb < REPEAT_WIN * 2:
        return canon
    Fs, act = _spectrum_features(sa, beats)
    F = np.concatenate([_model_features(H, fps, beats), Fs], 1)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    return canon_from_similarity(F @ F.T, act, REPEAT_THR)


def find_harmonic_repeats(sa: StemAnalysis, beats: np.ndarray) -> np.ndarray:
    """Como find_repeats pero solo con los acordes que suenan (croma de la pista), sin mirar el ritmo."""
    fps = CQT_SR / CQT_HOP
    C = np.maximum(sa.cqt_db, sa.noise_floor_db)
    n = C.shape[1]
    X, act = [], []
    for a, b in zip(beats, beats[1:]):
        f0, f1 = int(np.clip(a * fps, 0, n - 1)), int(np.clip(b * fps, 0, n))
        if f1 - f0 < 2:
            X.append(np.zeros(12))
            act.append(False)
            continue
        c = 10 ** (C[:, f0:f1].mean(1) / 20)
        ch = np.array([c[i::12].sum() for i in range(12)])
        X.append(ch / (np.linalg.norm(ch) + 1e-9))
        act.append(bool(sa.rms_db[f0:f1].mean() > sa.noise_floor_db + 12))
    act = np.array(act, bool)
    if len(X) < HARM_WIN * 2 or not act.any():
        return np.arange(len(X))
    X = np.array(X)
    X = X - X[act].mean(0)  # lo que distingue un acorde de otro en esta cancion
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return canon_from_similarity(X @ X.T, act, HARM_THR, HARM_WIN)


def lane_targets(items: list[tuple[int, int, int]], harm: np.ndarray, min_share: float = 0.4) -> dict[int, int]:
    """items: (tick, nota mas grave, boton). En cada grupo de pulsos con los mismos acordes, el boton que mas
    lleva cada nota; solo si aparece en compases distintos. Devuelve indice -> boton."""
    from collections import Counter

    g = groups(harm)
    count: dict[tuple, Counter] = defaultdict(Counter)
    where: dict[tuple, list] = defaultdict(list)
    for i, (tick, root, lane) in enumerate(items):
        b = tick // RES
        if b >= len(harm) or int(harm[b]) not in g:
            continue
        k = (int(harm[b]), root)
        count[k][lane] += 1
        where[k].append((i, b // 4))
    out = {}
    for k, cnt in count.items():
        if len({bar for _, bar in where[k]}) < 2:
            continue
        lane, n = cnt.most_common(1)[0]
        if n / sum(cnt.values()) >= min_share:
            for i, _ in where[k]:
                out[i] = lane
    return out


def canon_from_similarity(S: np.ndarray, act: np.ndarray, thr: float, win: int | None = None) -> np.ndarray:
    """Grupos de pulsos repetidos a partir del parecido entre cada par de pulsos."""
    nb = len(S)
    parent = np.arange(nb)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    W = win or REPEAT_WIN
    for L in range(REPEAT_MIN_LAG, nb - W):
        d = np.diagonal(S, L)
        c = np.concatenate([[0.0], np.cumsum(d)])
        good = ((c[W:] - c[:-W]) / W >= thr).astype(float)  # ventana de W pulsos desde i
        if not good.any():
            continue
        cover = np.convolve(good, np.ones(W))[: len(d)] > 0  # pulsos dentro de alguna ventana buena
        for i in np.where(cover & act[: len(d)] & act[L:L + len(d)])[0]:
            a, b = root(int(i)), root(int(i) + L)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(nb)])


def groups(canon: np.ndarray) -> dict[int, list[int]]:
    g: dict[int, list[int]] = defaultdict(list)
    for b, c in enumerate(canon):
        g[int(c)].append(b)
    return {c: ms for c, ms in g.items() if len(ms) > 1}


def average_frames(arrs: list[np.ndarray], fps: float, beats: np.ndarray, canon: np.ndarray) -> list[np.ndarray]:
    """Media de cada salida del modelo (por trama) entre todas las veces que suena la misma parte, en la misma
    posicion dentro del pulso. Todas las repeticiones quedan con la misma media."""
    n = arrs[0].shape[0]
    t = np.arange(n) / fps
    x = np.interp(t, beats, np.arange(len(beats)))
    b = np.floor(x).astype(int)
    valid = (t >= beats[0]) & (t < beats[-1]) & (b < len(canon))
    out = [a.copy() for a in arrs]
    bc = np.where(valid, canon[np.clip(b, 0, len(canon) - 1)], -1)
    for c, ms in groups(canon).items():
        sel = np.where(bc == c)[0]
        if not len(sel):
            continue
        u = x[sel] - b[sel]
        acc = [np.zeros((len(sel),) + a.shape[1:], np.float64) for a in arrs]
        for m in ms:
            fm = (beats[m] + u * (beats[m + 1] - beats[m])) * fps
            i0 = np.clip(np.floor(fm).astype(int), 0, n - 2)
            w = np.clip(fm - i0, 0, 1)[:, None]
            for k, a in enumerate(arrs):
                acc[k] += a[i0] * (1 - w) + a[i0 + 1] * w
        for k in range(len(arrs)):
            out[k][sel] = acc[k] / len(ms)
    return out


def choose_sources(items: list, canon: np.ndarray, key=lambda it: 0) -> np.ndarray:
    """De cada grupo de pulsos que suenan igual, el que mejor representa a todos (el que mas se parece a los
    demas): la primera vez que suena una parte suele ser mas floja (intro) y copiarla quitaba notas."""
    by_beat: dict[int, set] = defaultdict(set)
    for it in items:
        by_beat[it.tick // RES].add((it.tick % RES, key(it)))
    src = np.array(canon)
    for c, ms in groups(canon).items():
        pats = [by_beat.get(m, set()) for m in ms]

        def sim(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if a | b else 1.0

        score = [sum(sim(p, q) for q in pats) for p in pats]
        best = ms[int(np.argmax(score))]  # en empate, la primera vez
        for m in ms:
            src[m] = best
    return src


def copy_beats(items: list, canon: np.ndarray, clone=None) -> list:
    """Cada pulso repetido se queda con una copia de las notas del pulso `canon[b]` (desplazadas los pulsos que
    haga falta). `items` tiene `.tick`; `clone(item, nuevo_tick)` hace la copia."""
    clone = clone or (lambda it, tk: dataclasses.replace(it, tick=tk))
    nb = len(canon)
    by_beat: dict[int, list] = defaultdict(list)
    for it in items:
        by_beat[it.tick // RES].append(it)
    out, copied = [], set()
    for it in items:
        b = it.tick // RES
        if b >= nb or canon[b] == b:
            out.append(it)
    for b in range(nb):
        c = int(canon[b])
        if c != b:
            for it in by_beat.get(c, []):
                cp = clone(it, it.tick + (b - c) * RES)
                copied.add(id(cp))
                out.append(cp)
    out.sort(key=lambda it: it.tick)
    # donde una repeticion empieza pegada a una nota que no es copia: se queda la copia
    clean = []
    for it in out:
        if clean and it.tick - clean[-1].tick < RES // 8:
            if id(clean[-1]) in copied and id(it) not in copied:
                continue
            clean.pop()
        clean.append(it)
    return clean


def clip_sustains(notes: list, gap: int = RES // 24, min_len: int = RES // 4, attr: str = "sustain") -> None:
    """Tras copiar, una nota larga no puede pisar la siguiente."""
    for a, b in zip(notes, notes[1:]):
        s = getattr(a, attr)
        if s and a.tick + s > b.tick - gap:
            s = max(0, b.tick - gap - a.tick)
            setattr(a, attr, s if s >= min_len else 0)


def stats(canon: np.ndarray) -> tuple[int, float]:
    """(numero de partes que se repiten, proporcion de pulsos que son repeticion de otro)."""
    g = groups(canon)
    return len(g), float(np.mean(canon != np.arange(len(canon)))) if len(canon) else 0.0
