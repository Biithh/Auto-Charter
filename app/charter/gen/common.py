"""Utilidades comunes de generacion: calibracion, cuantizacion, importancia metrica."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .. import paths
from ..chart.model import RES, TempoMap


@lru_cache(maxsize=1)
def calibration() -> dict:
    """Estadisticas de los charts de referencia (incluidas con el programa o del usuario)."""
    user = paths.CONFIG_DIR / "reference_stats.json"
    base = paths.DATA / "reference_stats.json"
    for p in (user, base):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["aggregate"]
            except Exception:  # noqa: BLE001
                continue
    return {}


def stat(path: str, default: float) -> float:
    """Lee una estadistica tipo 'guitar.expert.nps.p50'."""
    cur = calibration()
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


# Divisiones de pulso permitidas (en ticks a RES=480)
GRID_STRAIGHT = (RES // 4,)  # semicorcheas
GRID_TRIPLET = (RES // 3, RES // 6)


def quantize_ticks(times: np.ndarray, tm: TempoMap, allow_32nd: bool = False) -> np.ndarray:
    """Cuantiza tiempos (s) a la rejilla musical, eligiendo binario o ternario por pulso."""
    raw = np.array([tm.sec_to_tick(t) for t in times])
    out = np.zeros(len(raw), dtype=np.int64)
    beat_of = np.floor(raw / RES + 0.5 / 4).astype(int)
    for b in np.unique(beat_of):
        idx = np.where(beat_of == b)[0]
        x = raw[idx]
        q4 = np.round(x / (RES / 4)) * (RES / 4)
        q3 = np.round(x / (RES / 3)) * (RES / 3)
        q6 = np.round(x / (RES / 6)) * (RES / 6)
        e4 = np.abs(x - q4).mean()
        e3 = np.abs(x - q3).mean()
        e6 = np.abs(x - q6).mean()
        best = q4
        # el ternario solo si encaja claramente mejor (evita tresillos fantasma)
        if e3 < e4 * 0.55 and e3 < RES * 0.04:
            best = q3
        elif e6 < e4 * 0.5 and e6 < RES * 0.03 and len(idx) >= 3:
            best = q6
        if allow_32nd:
            q8 = np.round(x / (RES / 8)) * (RES / 8)
            e8 = np.abs(x - q8).mean()
            if e8 < min(e4, e3) * 0.5 and len(idx) >= 4:
                best = q8
        out[idx] = best.astype(np.int64)
    return out


def metric_strength(tick: int, tm: TempoMap) -> float:
    """Importancia metrica de una posicion (1 = inicio de compas)."""
    start, num, den = tm.ts_at(tick)
    measure = int(num * RES * 4 / den)
    rel = (tick - start) % measure
    if rel == 0:
        return 1.0
    if rel % RES == 0:
        return 0.85 if (rel // RES) % 2 == 0 else 0.75
    if rel % (RES // 2) == 0:
        return 0.55
    if rel % (RES // 4) == 0:
        return 0.35
    if rel % (RES // 3) == 0:
        return 0.4
    return 0.2


def secs_between(tm: TempoMap, t0: int, t1: int) -> float:
    return tm.tick_to_sec(t1) - tm.tick_to_sec(t0)


def nms_select(ticks: list[int], scores: list[float], min_gap_ticks: np.ndarray | int,
               min_gap_secs: float, tm: TempoMap) -> list[int]:
    """Seleccion voraz por importancia respetando una separacion minima. Devuelve indices."""
    order = np.argsort(-np.asarray(scores))
    chosen: list[int] = []
    chosen_ticks: list[int] = []
    gap_arr = np.broadcast_to(np.asarray(min_gap_ticks), (len(ticks),))
    for i in order:
        t = ticks[i]
        ok = True
        pos = np.searchsorted(chosen_ticks, t)
        for k in (pos - 1, pos):
            if 0 <= k < len(chosen_ticks):
                o = chosen_ticks[k]
                if abs(o - t) < gap_arr[i] or abs(secs_between(tm, min(o, t), max(o, t))) < min_gap_secs:
                    ok = False
                    break
        if ok:
            chosen_ticks.insert(pos, t)
            chosen.append(int(i))
    return sorted(chosen, key=lambda i: ticks[i])
