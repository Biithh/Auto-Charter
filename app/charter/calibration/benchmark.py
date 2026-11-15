"""Banco de pruebas: genera charts de canciones que el entrenamiento nunca ha visto (las apartadas en
config/entrenamiento/banco_pruebas.txt) a partir de la mezcla de sus pistas, y los compara con sus charts.

Uso: python -m charter.calibration.benchmark [numero_de_canciones]   (por defecto 12; usa la GPU, ~8 min/cancion)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import paths


def pick(n: int) -> list[Path]:
    from .library_train import HOLDOUT_FILE

    if not HOLDOUT_FILE.exists():
        raise SystemExit("Primero hay que entrenar con la biblioteca (crea la lista de canciones de prueba).")
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS

    out = []
    for line in HOLDOUT_FILE.read_text(encoding="utf-8").splitlines():
        f = Path(line.strip())
        if f.is_dir() and (f / "notes.mid").exists() and any(
                (f / f"{s}{e}").exists() for s in ("guitar", "bass") for e in (".ogg", ".opus", ".mp3")):
            out.append(f)

    def official(f: Path) -> bool:
        try:
            return any(o in read_ini(f / "song.ini").get("charter", "").lower() for o in OFFICIAL_CHARTERS)
        except Exception:  # noqa: BLE001
            return False

    # primero los charts oficiales (la referencia mas fiable)
    out.sort(key=lambda f: not official(f))
    return out[:n]


def summary(results: dict) -> dict:
    rows: dict = {}
    for res in results.values():
        for inst, v in res.items():
            if inst == "vocals":
                rows.setdefault(("vocals", ""), []).append((v["f1"], 1.0))
                continue
            for d, x in v.items():
                if x["n_ref"]:
                    rows.setdefault((inst, d), []).append((x["f1"], x["n_ours"] / x["n_ref"]))
    return {f"{i} {d}".strip(): {"f1": round(float(np.mean([a for a, _ in v])), 3),
                                  "cantidad": round(float(np.mean([b for _, b in v])), 2), "canciones": len(v)}
            for (i, d), v in sorted(rows.items())}


def main(n: int) -> None:
    from .eval_full import main as eval_main

    songs = pick(n)
    print(f"Banco de pruebas: {len(songs)} canciones que el entrenamiento no ha visto", flush=True)
    eval_main([str(s) for s in songs])
    res = json.loads((paths.ROOT / "salida_eval" / "evaluacion.json").read_text(encoding="utf-8"))
    summ = summary(res)
    (paths.ROOT / "salida_eval" / "resumen_banco.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\nRESUMEN (F1: 1 = identico al chart real; cantidad: notas nuestras / notas del chart)")
    for k, v in summ.items():
        print(f"  {k:16} F1 {v['f1']:.3f}   cantidad {v['cantidad']:.2f}x   ({v['canciones']} canciones)")


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
