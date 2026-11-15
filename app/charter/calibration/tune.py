"""Ajuste rapido de la guitarra con las canciones ya analizadas por el banco de pruebas (solo CPU, minutos).

Regenera la guitarra de cada cancion con la transcripcion guardada y la compara con su chart real: notas
acertadas (F1), cantidad, notas largas, acordes y HOPOs por dificultad. Sirve para probar cambios del
generador sin volver a separar ni transcribir.

Uso: python -m charter.calibration.tune [carpeta_trabajo ...]   (por defecto, todas las eval_* de trabajo/)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from .. import paths


class _Rep:
    def log(self, m):
        pass

    def progress(self, f):
        pass


def load_case(work: Path) -> dict | None:
    """Datos guardados de una cancion del banco: transcripcion, tempo y chart real."""
    from ..audio import load
    from ..chart.chart_io import load_chart
    from ..pipeline import runner
    from ..songini import read_ini

    job = work / "referencia.txt"
    if not (work / "pasadas_guitar.pkl").exists():
        return None
    if not job.exists() and not _backfill(work):
        return None
    ref_dir = Path(job.read_text(encoding="utf-8").strip())
    info_p = work / "lead_in.txt"
    lead = float(info_p.read_text()) if info_p.exists() else 0.0
    stems = {p.stem: p for p in (work / "stems").glob("*.wav")}
    mix, sr = load(work / "mix.wav")
    trans = runner._transcribe_all(stems, mix, sr, _Rep(), lambda: None, work, {"guitar"})
    if not trans["notes"].get("guitar"):
        return None
    from ..pipeline.beats import refine_grid

    grid = refine_grid(pickle.load(open(work / "beats.pkl", "rb")))
    ini = read_ini(ref_dir / "song.ini")
    return {"name": ref_dir.name, "trans": trans, "grid": grid, "lead": lead, "ref": load_chart(ref_dir),
            "delay": float(ini.get("delay", "0") or 0) / 1000}


def _backfill(work: Path) -> bool:
    """Carpetas de evaluacion antiguas: busca su cancion (banco de pruebas o referencias) y su silencio inicial."""
    from ..pipeline.runner import slug
    from .library_train import HOLDOUT_FILE

    name = work.name[len("eval_"):]
    pool = [Path(x) for x in HOLDOUT_FILE.read_text(encoding="utf-8").splitlines()] if HOLDOUT_FILE.exists() else []
    pool += list((paths.ROOT / "referencias").glob("*")) if (paths.ROOT / "referencias").exists() else []
    ref = next((p for p in pool if p.name[:40] == name and (p / "song.ini").exists()), None)
    if ref is None or not (work / "meta.pkl").exists():
        return False
    meta = pickle.load(open(work / "meta.pkl", "rb"))
    info_p = paths.ROOT / "salida_eval" / slug(f"{meta.artist} - {meta.title}") / "charter_info.json"
    if not info_p.exists():
        return False
    lead = json.loads(info_p.read_text(encoding="utf-8")).get("lead_in", 0.0)
    (work / "referencia.txt").write_text(str(ref.resolve()), encoding="utf-8")
    (work / "lead_in.txt").write_text(str(lead), encoding="utf-8")
    return True


def _repeat_runs(pats: list, minrun: int = 4) -> set:
    """Pares de pulsos (i, i+L) dentro de tramos de >= minrun pulsos seguidos tocados exactamente igual."""
    n = len(pats)
    out = set()
    for L in range(4, n):
        run = []
        for i in range(n - L + 1):
            if i < n - L and pats[i] == pats[i + L]:
                run.append(i)
                continue
            if len(run) >= minrun and sum(bool(pats[k]) for k in run) >= len(run) / 2:
                out.update((k, k + L) for k in run if pats[k])
            run = []
    return out


def repeat_scores(case: dict, notes: list, tm, lead: float) -> tuple[float, float, float]:
    """Repeticiones, pulso a pulso con nuestra rejilla: (de las partes que el chart oficial repite exactamente,
    cuantas repetimos igual nosotros; pulsos con notas que son repeticion en el oficial; y en el nuestro)."""
    from ..chart.model import RES

    ref = case["ref"]
    rn = ref.guitar.notes["expert"]
    rtk = np.array([n.tick for n in rn])
    otk = np.array([n.tick for n in notes])
    nb = int(otk.max() // RES) + 2 if len(otk) else 0
    ref_p, our_p = [], []
    q = RES // 4
    for k in range(nb):
        a, b = tm.tick_to_sec(k * RES) - lead, tm.tick_to_sec((k + 1) * RES) - lead
        r0 = int(round(ref.tempo.sec_to_tick(a + case["delay"]) / q) * q)
        r1 = int(round(ref.tempo.sec_to_tick(b + case["delay"]) / q) * q)
        idx = np.where((rtk >= r0) & (rtk < r1))[0]
        ref_p.append(tuple((int(rtk[i] - r0), rn[i].frets) for i in idx))
        idx = np.where((otk >= k * RES) & (otk < (k + 1) * RES))[0]
        our_p.append(tuple((int(otk[i] - k * RES), notes[i].frets) for i in idx))
    off, ours = _repeat_runs(ref_p), _repeat_runs(our_p)
    same = float(np.mean([our_p[i] == our_p[j] and bool(our_p[i]) for i, j in off])) if off else 0.0

    def cover(pairs, pats):
        full = sum(bool(p) for p in pats)
        return len({j for _, j in pairs}) / max(full, 1)

    return same, cover(off, ref_p), cover(ours, our_p)


def score(case: dict) -> dict:
    from ..chart.model import DIFFICULTIES, OPEN
    from ..gen.frets import generate_fret_track
    from ..pipeline.beats import build_tempo_map
    from .eval_full import onset_f1

    tm, lead = build_tempo_map(case["grid"], case["lead"])
    src = "guitar_lead" if case["trans"]["notes"].get("guitar_lead") else "guitar"
    ft, _ = generate_fret_track(case["trans"]["notes"][src], case["trans"]["analysis"][src], tm, "guitar", lead,
                                log=lambda m: None)
    ref = case["ref"]
    out = {}
    for d in DIFFICULTIES:
        rn, on = ref.guitar.notes[d], ft.notes[d]
        if not rn or not on:
            continue
        rt = [ref.tempo.tick_to_sec(n.tick) - case["delay"] for n in rn]
        ot = [tm.tick_to_sec(n.tick) - lead for n in on]
        f1, p, r = onset_f1(rt, ot)

        def chords(ns):
            return float(np.mean([len([f for f in n.frets if f != OPEN]) > 1 for n in ns]))

        # de las notas que coinciden (+-60 ms), en cuantas acertamos si es acorde o nota suelta
        ot_a = np.array(ot)
        hit = []
        for n, t in zip(rn, rt):
            j = int(np.argmin(np.abs(ot_a - t)))
            if abs(ot_a[j] - t) <= 0.06:
                hit.append((len([f for f in n.frets if f != OPEN]) > 1) == (len([f for f in on[j].frets if f != OPEN]) > 1))
        # contorno: entre dos notas seguidas que acertamos, si el chart sube, baja o se queda en el mismo boton
        # y nosotros hacemos lo mismo (lo que hace que se note que estas tocando la melodia)
        pairs = []
        prev = None
        for n, t in zip(rn, rt):
            j = int(np.argmin(np.abs(ot_a - t)))
            ok = abs(ot_a[j] - t) <= 0.06 and OPEN not in n.frets and OPEN not in on[j].frets
            cur = (min(n.frets), min(on[j].frets), j) if ok else None
            if prev and cur and cur[2] != prev[2]:
                pairs.append((np.sign(cur[0] - prev[0]), np.sign(cur[1] - prev[1])))
            prev = cur
        pa = np.array(pairs) if pairs else np.zeros((0, 2))
        cont = float(np.mean(pa[:, 0] == pa[:, 1])) if len(pa) else 0.0
        out[d] = {"f1": f1, "p": p, "r": r, "cantidad": len(on) / len(rn), "contorno": cont,
                  "acierto_acorde": float(np.mean(hit)) if hit else 0.0,
                  "largas_ref": float(np.mean([n.sustain > 0 for n in rn])),
                  "largas": float(np.mean([n.sustain > 0 for n in on])),
                  "acordes_ref": chords(rn), "acordes": chords(on)}
    if "expert" in out:
        same, cov_ref, cov = repeat_scores(case, ft.notes["expert"], tm, lead)
        out["expert"].update({"repes": same, "repes_ref": cov_ref, "repes_propias": cov})
    return out


def summarize(results: list[dict]) -> dict:
    keys = ("f1", "p", "r", "cantidad", "largas_ref", "largas", "acordes_ref", "acordes", "acierto_acorde", "contorno")
    out = {}
    for d in ("expert", "hard", "medium", "easy"):
        rows = [r[d] for r in results if d in r]
        if rows:
            out[d] = {k: float(np.mean([x[k] for x in rows])) for k in keys + ("repes", "repes_ref", "repes_propias")
                      if k in rows[0]}
            out[d]["n"] = len(rows)
    return out


def print_summary(s: dict, title: str = "") -> None:
    if title:
        print(title)
    for d, v in s.items():
        print(f"  {d:7} F1 {v['f1']:.3f} (P {v['p']:.2f} R {v['r']:.2f}) cantidad {v['cantidad']:.2f}x | "
              f"largas {v['largas']:.2f} (real {v['largas_ref']:.2f}) | acordes {v['acordes']:.2f} "
              f"(real {v['acordes_ref']:.2f}, aciertos {v['acierto_acorde']:.2f}) | botones {v['contorno']:.2f} | {v['n']} canciones",
              flush=True)
        if "repes" in v:
            print(f"          repeticiones: el oficial repite igual el {v['repes_ref']:.0%} de los pulsos, nosotros el "
                  f"{v['repes_propias']:.0%}; de lo que el oficial repite, lo repetimos igual el {v['repes']:.0%}", flush=True)


def cases(folders: list[str] | None = None) -> list[dict]:
    dirs = [Path(f) for f in folders] if folders else sorted(paths.WORK.glob("eval_*"))
    out = []
    for w in dirs:
        try:
            c = load_case(w)
        except Exception as e:  # noqa: BLE001
            print(f"  {w.name}: no se puede cargar ({e})")
            c = None
        if c:
            out.append(c)
    return out


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    cs = cases(sys.argv[1:] or None)
    print(f"{len(cs)} canciones")
    res = [score(c) for c in cs]
    for c, r in zip(cs, res):
        e = r.get("expert", {})
        print(f"  {c['name'][:40]:40} Experto F1 {e.get('f1', 0):.3f} cantidad {e.get('cantidad', 0):.2f}x")
    print_summary(summarize(res), "MEDIA")
    print(json.dumps(summarize(res)))
