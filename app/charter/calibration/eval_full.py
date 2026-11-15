"""Evaluacion completa: genera el chart de una cancion de referencia (mezclando sus pistas) y lo compara
con el chart oficial. Uso: python -m charter.calibration.eval_full <carpeta_ref> [<carpeta_ref> ...]"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import paths


def onset_f1(ref: list[float], est: list[float], tol: float = 0.06) -> tuple[float, float, float]:
    ref = np.sort(np.asarray(ref))
    est = np.sort(np.asarray(est))
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(est), bool)
    tp = 0
    for r in ref:
        i = np.searchsorted(est, r)
        best = None
        for k in (i - 1, i, i + 1):
            if 0 <= k < len(est) and not used[k] and abs(est[k] - r) <= tol:
                if best is None or abs(est[k] - r) < abs(est[best] - r):
                    best = k
        if best is not None:
            used[best] = True
            tp += 1
    p, r = tp / len(est), tp / len(ref)
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def compare(ref_dir: Path, out_dir: Path, lead_in: float) -> dict:
    from ..chart.midi_io import read_midi
    from ..chart.model import DIFFICULTIES
    from ..songini import read_ini

    ini = read_ini(ref_dir / "song.ini")
    delay = float(ini.get("delay", "0") or 0) / 1000
    ref = read_midi(ref_dir / "notes.mid")
    ours = read_midi(out_dir / "notes.mid")
    res = {}
    for key in ("guitar", "bass", "drums", "keys"):
        r_obj, o_obj = getattr(ref, key), getattr(ours, key)
        if not r_obj.has_notes():
            continue
        row = {}
        for d in DIFFICULTIES:
            rn = r_obj.notes[d]
            on = o_obj.notes[d] if o_obj.has_notes() else []
            if key == "drums":
                rt = sorted(set(ref.tempo.tick_to_sec(n.tick) - delay for n in rn if not n.double_kick))
                ot = sorted(set(ours.tempo.tick_to_sec(n.tick) - lead_in for n in on if not n.double_kick))
            else:
                rt = [ref.tempo.tick_to_sec(n.tick) - delay for n in rn]
                ot = [ours.tempo.tick_to_sec(n.tick) - lead_in for n in on]
            f1, p, r = onset_f1(rt, ot)
            row[d] = {"f1": round(f1, 3), "p": round(p, 3), "r": round(r, 3), "n_ref": len(rt), "n_ours": len(ot)}
            if key != "drums" and rn and on:
                row[d]["chords_ref"] = round(float(np.mean([n.is_chord for n in rn])), 3)
                row[d]["chords_ours"] = round(float(np.mean([n.is_chord for n in on])), 3)
                row[d]["sus_ref"] = round(float(np.mean([n.sustain > 0 for n in rn])), 3)
                row[d]["sus_ours"] = round(float(np.mean([n.sustain > 0 for n in on])), 3)
        res[key] = row
    if ref.vocals.has_notes():
        rt = [ref.tempo.tick_to_sec(n.tick) - delay for n in ref.vocals.notes if n.lyric != "+"]
        ot = [ours.tempo.tick_to_sec(n.tick) - lead_in for n in ours.vocals.notes if n.lyric != "+"]
        f1, p, r = onset_f1(rt, ot, 0.1)
        res["vocals"] = {"f1": round(f1, 3), "p": round(p, 3), "r": round(r, 3), "n_ref": len(rt), "n_ours": len(ot)}
    return res


def _mix_stereo(folder: Path) -> np.ndarray:
    import tempfile

    from ..audio import decode_to_wav, load
    from .eval_beats import AUDIO_EXT, SKIP

    total = None
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in AUDIO_EXT or any(s in f.stem.lower() for s in SKIP):
            continue
        with tempfile.TemporaryDirectory() as td:
            y, _ = load(decode_to_wav(f, Path(td) / "a.wav"))
        if total is None:
            total = y
        else:
            n = max(total.shape[1], y.shape[1])
            total = np.pad(total, ((0, 0), (0, n - total.shape[1]))) + np.pad(y, ((0, 0), (0, n - y.shape[1])))
    return total


def main(folders: list[str]) -> None:
    from ..audio import save
    from ..pipeline.runner import JobConfig, process_song
    from ..songini import read_ini
    from .eval_beats import mix_reference_audio

    out_root = paths.ROOT / "salida_eval"
    out_root.mkdir(exist_ok=True)
    all_res = {}
    for f in folders:
        ref_dir = Path(f)
        name = ref_dir.name
        work = paths.WORK / f"eval_{name[:40]}"
        work.mkdir(parents=True, exist_ok=True)
        mix_path = work / "referencia_mezcla.wav"
        if not mix_path.exists():
            y = _mix_stereo(ref_dir)
            peak = np.max(np.abs(y)) or 1.0
            if peak > 0.99:
                y = y / peak * 0.99
            save(mix_path, y, 44100)
        ini = read_ini(ref_dir / "song.ini")
        cfg = JobConfig(audio_path=str(mix_path), output_dir=str(out_root), title=ini.get("name", name),
                        artist=ini.get("artist", ""), album=ini.get("album", ""))

        def emit(ev):
            if ev.get("type") in ("log", "warning", "error"):
                print(ev.get("msg"), flush=True)

        out_dir = process_song(cfg, work, emit)
        info = json.loads((out_dir / "charter_info.json").read_text(encoding="utf-8"))
        # para el ajuste rapido (calibration.tune): de que cancion es esta carpeta y el silencio inicial
        (work / "referencia.txt").write_text(str(ref_dir.resolve()), encoding="utf-8")
        (work / "lead_in.txt").write_text(str(info.get("lead_in", 0.0)), encoding="utf-8")
        res = compare(ref_dir, out_dir, info.get("lead_in", 0.0))
        all_res[name] = res
        print("RESULTADO", name, json.dumps(res, indent=1), flush=True)
    (out_root / "evaluacion.json").write_text(json.dumps(all_res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    main(sys.argv[1:])
