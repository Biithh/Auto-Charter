"""Modelo de bajo: la misma red que la guitarra (neural_onset), entrenada con los charts de bajo de la biblioteca.

Uso: python -m charter.calibration.neural_bass [minutos]   (usa la GPU; antes, bass_spectra)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import paths
from . import neural_onset as no
from .bass_spectra import BASS_SPEC_DIR

MODEL_PATH = paths.CONFIG_DIR / "bajo_neuronal.pt"
PREV_PATH = paths.CONFIG_DIR / "bajo_neuronal_anterior.pt"
PREV2_PATH = paths.CONFIG_DIR / "bajo_neuronal_anterior2.pt"
_CACHE: dict = {}


def load_dataset(log=print) -> tuple[list, list]:
    train, val_sep, val_stem = [], [], []
    hold = no._holdout_titles()
    for p in sorted(BASS_SPEC_DIR.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        try:
            d = np.load(p)
            folder = str(d["folder"])
            if no._title(Path(folder)) in hold:
                continue
            x = no._inputs(d["cqt"], d["onset"])
            n = x.shape[1]
            t = d["ref_t"].astype(np.float64)
            refs = [t] + [d[f"t_{k}"].astype(np.float64) for k in no.DIFFS[1:]]
            on, ch = no._targets(n, refs, d["sizes"])
            mv, ln = no._button_targets(n, t, d["lane"].astype(np.int64))
        except Exception:  # noqa: BLE001
            continue
        if n < no.CROP or on[:, 0].sum() < 50:
            continue
        sep = bool(d["sep"])
        item = {"x": x, "on": on.astype(np.float16), "ch": ch.astype(np.int8), "mv": mv.astype(np.int8),
                "ln": ln.astype(np.int8), "fast": no._fast_frames(n, t), "refs": refs, "name": folder, "sep": sep}
        if not no._is_val(folder):
            train.append(item)
        else:
            (val_sep if sep else val_stem).append(item)
    # se valida con bajos separados por la IA, como los que vera al generar
    val = val_sep if len(val_sep) >= 20 else val_sep + val_stem
    log(f"  {len(train)} canciones para entrenar ({sum(i['sep'] for i in train)} separadas por la IA), "
        f"{len(val)} para validar ({sum(i['sep'] for i in val)} separadas por la IA)")
    return train, val


def train(minutes: float = 60, log=print, **kw) -> dict:
    return no.train(minutes, log=log, inst=sys.modules[__name__], **kw)


def load_model():
    if "m" not in _CACHE:
        _CACHE["m"] = no._load(MODEL_PATH)
    return _CACHE["m"]


def load_models() -> list:
    """El modelo actual y los de los entrenamientos anteriores (si son del mismo tamano): juntos aciertan mas."""
    if "all" not in _CACHE:
        main = load_model()
        out = [main] if main else []
        for path in (PREV_PATH, PREV2_PATH) if main else ():
            prev = no._load(path)
            if prev is not None and prev[1].get("ancho") == main[1].get("ancho"):
                out.append(prev)
        _CACHE["all"] = out
    return _CACHE["all"]


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    print(json.dumps(train(float(sys.argv[1]) if len(sys.argv) > 1 else 60, log=lambda m: print(m, flush=True), width=1.5,
                           batch_size=32, eval_every=1000)))
