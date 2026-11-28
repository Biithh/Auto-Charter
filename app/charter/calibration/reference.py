"""Analisis de charts de referencia (oficiales y de la comunidad).

Saca estadisticas de jugabilidad (densidad, acordes, HOPOs, sustains, carriles,
star power...) que luego se usan para calibrar los charts generados.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..chart.chart_io import load_chart
from ..chart.midi_io import resolve_hopos
from ..chart.model import DIFFICULTIES, OPEN, RES, Chart
from ..songini import read_ini

OFFICIAL_CHARTERS = ("harmonix", "neversoft", "activision", "vicarious", "budcat", "pandemic", "beenox", "freestylegames")


def _made_by_charter(song_dir: Path) -> bool:
    """Charts hechos por este programa: no se aprende de ellos (seria aprender de nuestros propios fallos)."""
    try:
        return "charter (ia)" in (song_dir / "song.ini").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def find_songs(folder: str | Path, limit: int | None = None, skip_ai: bool = True):
    """Busca carpetas con notes.mid o notes.chart + song.ini (recursivo, sin repetir)."""
    import os

    folder = Path(folder)
    count = 0
    if folder.is_file():
        # lista de carpetas (una por linea)
        for line in folder.read_text(encoding="utf-8").splitlines():
            if line.strip() and not (skip_ai and _made_by_charter(Path(line.strip()))):
                yield Path(line.strip())
                count += 1
                if limit and count >= limit:
                    return
        return
    for root, _dirs, files in os.walk(folder):
        low = {f.lower() for f in files}
        if ("notes.mid" in low or "notes.chart" in low) and "song.ini" in low:
            if skip_ai and _made_by_charter(Path(root)):
                continue
            yield Path(root)
            count += 1
            if limit and count >= limit:
                return


def _grid_class(tick: int) -> str:
    pos = tick % RES
    for name, div in (("4", 1), ("8", 2), ("16", 4), ("12", 3), ("24", 6), ("32", 8)):
        step = RES / div
        if abs(pos / step - round(pos / step)) < 0.02:
            return name
    return "other"


def fret_stats(chart: Chart, key: str, hopo_thr: int) -> dict | None:
    ft = getattr(chart, key)
    if not ft.has_notes():
        return None
    tm = chart.tempo
    out = {}
    exp = ft.notes["expert"]
    if len(exp) < 20:
        return None
    t_first = tm.tick_to_sec(exp[0].tick)
    t_last = tm.tick_to_sec(exp[-1].tick)
    dur = max(t_last - t_first, 1.0)
    for d in DIFFICULTIES:
        notes = ft.notes[d]
        if not notes:
            continue
        n = len(notes)
        secs = np.array([tm.tick_to_sec(x.tick) for x in notes])
        chord_sizes = [len([f for f in x.frets if f != OPEN]) for x in notes]
        hopos = resolve_hopos(notes, hopo_thr)
        ioi_beats = np.diff([x.tick for x in notes]) / RES if n > 1 else np.array([1.0])
        ioi_secs = np.diff(secs) if n > 1 else np.array([1.0])
        # densidad por ventanas de 4 segundos (solo ventanas con notas)
        bins = np.arange(t_first, t_last + 4, 4.0)
        hist, _ = np.histogram(secs, bins=bins)
        active = hist[hist > 0] / 4.0
        lanes = np.zeros(6)
        for x in notes:
            for f in x.frets:
                lanes[5 if f == OPEN else f] += 1
        grid = {}
        for x in notes:
            g = _grid_class(x.tick)
            grid[g] = grid.get(g, 0) + 1
        sus = [x.sustain / RES for x in notes if x.sustain > 0]
        out[d] = {
            "n": n,
            "nps": n / dur,
            "nps_p50": float(np.percentile(active, 50)) if len(active) else 0,
            "nps_p90": float(np.percentile(active, 90)) if len(active) else 0,
            "nps_max": float(active.max()) if len(active) else 0,
            "chord_ratio": float(np.mean([c > 1 for c in chord_sizes])),
            "chord3_ratio": float(np.mean([c >= 3 for c in chord_sizes])),
            "hopo_ratio": float(np.mean(hopos)),
            "tap_ratio": float(np.mean([x.tap for x in notes])),
            "open_ratio": float(np.mean([OPEN in x.frets for x in notes])),
            "sustain_ratio": len(sus) / n,
            "sustain_beats_p50": float(np.median(sus)) if sus else 0,
            "min_ioi_beats_p05": float(np.percentile(ioi_beats, 5)),
            "min_ioi_secs_p05": float(np.percentile(ioi_secs, 5)),
            "ioi_secs_p50": float(np.median(ioi_secs)),
            "lanes": (lanes / lanes.sum()).round(4).tolist(),
            "max_lane": int(max((f for x in notes for f in x.frets if f != OPEN), default=0)),
            "grid": {k: round(v / n, 4) for k, v in grid.items()},
        }
    out["sp_phrases"] = len(ft.star_power)
    out["sp_per_min"] = len(ft.star_power) / (dur / 60)
    out["solos"] = len(ft.solos)
    out["duration"] = dur
    out["bpm_median"] = float(np.median([tm.bpm_at(x.tick) for x in exp]))
    return out


def drum_stats(chart: Chart) -> dict | None:
    dt = chart.drums
    if not dt.has_notes() or len(dt.notes["expert"]) < 20:
        return None
    tm = chart.tempo
    exp = dt.notes["expert"]
    t_first = tm.tick_to_sec(exp[0].tick)
    t_last = tm.tick_to_sec(exp[-1].tick)
    dur = max(t_last - t_first, 1.0)
    out = {}
    for d in DIFFICULTIES:
        notes = [x for x in dt.notes[d] if not x.double_kick]
        if not notes:
            continue
        ticks = sorted(set(x.tick for x in notes))
        pads = np.zeros(5)
        cym = 0
        for x in notes:
            pads[x.pad] += 1
            cym += x.cymbal
        per_tick = len(notes) / max(len(ticks), 1)
        out[d] = {
            "n": len(notes),
            "nps": len(notes) / dur,
            "hits_per_tick": per_tick,
            "pads": (pads / pads.sum()).round(4).tolist(),
            "cymbal_ratio": cym / len(notes),
            "double_kick": sum(1 for x in dt.notes[d] if x.double_kick),
        }
    out["fills"] = len(dt.fills)
    out["sp_phrases"] = len(dt.star_power)
    out["duration"] = dur
    return out


def vocal_stats(chart: Chart) -> dict | None:
    vt = chart.vocals
    if not vt.has_notes():
        return None
    tm = chart.tempo
    notes = vt.notes
    lens = [tm.tick_to_sec(n.tick + n.length) - tm.tick_to_sec(n.tick) for n in notes]
    pitches = [n.pitch for n in notes if not n.unpitched and n.lyric != "+"]
    phr_lens = [tm.tick_to_sec(p.end) - tm.tick_to_sec(p.start) for p in vt.phrases]
    return {
        "n": len(notes),
        "unpitched_ratio": float(np.mean([n.unpitched for n in notes])),
        "slide_ratio": float(np.mean([n.lyric.startswith("+") for n in notes])),
        "note_len_p50": float(np.median(lens)),
        "pitch_range": [int(min(pitches)), int(max(pitches))] if pitches else [0, 0],
        "phrases": len(vt.phrases),
        "phrase_len_p50": float(np.median(phr_lens)) if phr_lens else 0,
        "sp_phrases": len(vt.star_power),
    }


def analyze_song(folder: Path) -> dict | None:
    ini = read_ini(folder / "song.ini")
    try:
        hopo = int(float(ini.get("hopo_frequency", "170") or 170))
    except ValueError:
        hopo = 170
    if ini.get("eighthnote_hopo", "").lower() in ("1", "true"):
        hopo = 250
    try:
        chart = load_chart(folder, hopo)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "folder": folder.name}
    charter = ini.get("charter", ini.get("frets", "")).strip()
    hopo_thr = int(round(hopo * RES / 480))
    res = {
        "folder": folder.name,
        "artist": ini.get("artist", ""),
        "name": ini.get("name", ""),
        "charter": charter,
        "official": any(o in charter.lower() for o in OFFICIAL_CHARTERS),
        "tracks": {},
    }
    for key in ("guitar", "rhythm", "bass", "keys"):
        st = fret_stats(chart, key, hopo_thr)
        if st:
            res["tracks"][key] = st
    st = drum_stats(chart)
    if st:
        res["tracks"]["drums"] = st
    st = vocal_stats(chart)
    if st:
        res["tracks"]["vocals"] = st
    res["sections"] = len(chart.sections)
    res["has_pro_keys"] = chart.pro_keys.has_notes()
    return res


def aggregate(songs: list[dict]) -> dict:
    """Combina estadisticas de muchas canciones. Las oficiales pesan el doble."""
    agg: dict = {"n_songs": len(songs), "n_official": sum(s.get("official", False) for s in songs)}
    for key in ("guitar", "rhythm", "bass", "keys"):
        per_diff = {}
        for d in DIFFICULTIES:
            rows, weights = [], []
            for s in songs:
                tr = s.get("tracks", {}).get(key)
                if tr and d in tr:
                    rows.append(tr[d])
                    weights.append(2.0 if s.get("official") else 1.0)
            if not rows:
                continue
            w = np.array(weights)
            stat = {}
            for field in ("nps", "nps_p50", "nps_p90", "nps_max", "chord_ratio", "chord3_ratio", "hopo_ratio",
                          "tap_ratio", "open_ratio", "sustain_ratio", "sustain_beats_p50", "min_ioi_secs_p05",
                          "min_ioi_beats_p05", "ioi_secs_p50"):
                vals = np.array([r[field] for r in rows], dtype=float)
                stat[field] = {
                    "mean": float(np.average(vals, weights=w)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                }
            stat["lanes"] = np.average(np.array([r["lanes"] for r in rows]), axis=0, weights=w).round(4).tolist()
            stat["max_lane_mode"] = int(np.bincount([r["max_lane"] for r in rows]).argmax())
            per_diff[d] = stat
        if per_diff:
            sp = [s["tracks"][key]["sp_per_min"] for s in songs if key in s.get("tracks", {})]
            per_diff["sp_per_min"] = float(np.median(sp))
            agg[key] = per_diff
    # Ratios entre dificultades (cuantas notas quedan respecto a experto)
    ratios = {}
    for key in ("guitar", "bass", "keys", "drums"):
        r = {}
        for d in ("hard", "medium", "easy"):
            vals = []
            for s in songs:
                tr = s.get("tracks", {}).get(key)
                if tr and d in tr and "expert" in tr and tr["expert"]["n"]:
                    vals.append(tr[d]["n"] / tr["expert"]["n"])
            if vals:
                r[d] = {"p50": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                        "p90": float(np.percentile(vals, 90))}
        ratios[key] = r
    agg["diff_ratios"] = ratios
    drows = [s["tracks"]["drums"] for s in songs if "drums" in s.get("tracks", {})]
    if drows:
        agg["drums"] = {
            d: {
                "nps_p50": float(np.median([r[d]["nps"] for r in drows if d in r])),
                "nps_p90": float(np.percentile([r[d]["nps"] for r in drows if d in r], 90)),
                "cymbal_ratio": float(np.median([r[d]["cymbal_ratio"] for r in drows if d in r])),
                "pads": np.mean([r[d]["pads"] for r in drows if d in r], axis=0).round(4).tolist(),
            }
            for d in DIFFICULTIES
            if any(d in r for r in drows)
        }
        agg["drums"]["fills_per_song"] = float(np.median([r["fills"] for r in drows]))
    vrows = [s["tracks"]["vocals"] for s in songs if "vocals" in s.get("tracks", {})]
    if vrows:
        agg["vocals"] = {
            k: float(np.median([r[k] for r in vrows]))
            for k in ("unpitched_ratio", "slide_ratio", "note_len_p50", "phrase_len_p50")
        }
    return agg


def analyze_folder(folder: str | Path, out_json: str | Path | None = None, log=print, limit=None) -> dict:
    songs = []
    for f in find_songs(folder, limit):
        r = analyze_song(f)
        if r and "error" not in r:
            songs.append(r)
            log(f"  analizada: {f.name}")
        elif r:
            log(f"  error en {f.name}: {r['error']}")
    agg = aggregate(songs)
    result = {"aggregate": agg, "songs": songs}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    res = analyze_folder(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(res["aggregate"], indent=1))
