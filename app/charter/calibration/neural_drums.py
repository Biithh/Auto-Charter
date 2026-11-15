"""Modelo de bateria: la misma red que la guitarra (neural_onset), con 8 salidas por trama: bombo, caja, y para
amarillo, azul y verde si es tom o platillo. Aprende de los charts de bateria de la biblioteca (Pro Drums).

Uso: python -m charter.calibration.neural_drums [minutos]   (usa la GPU; antes, drum_spectra)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths
from . import neural_onset as no
from .drum_spectra import CLASSES, DRUM_SPEC_DIR

MODEL_PATH = paths.CONFIG_DIR / "bateria_neuronal.pt"
N_CLS = len(CLASSES)
CROP = no.CROP
SEP_WEIGHT = 2.0
PRO_ONLY = (2, 3, 4, 5, 6, 7)  # salidas que solo se aprenden de charts Pro (los demas no marcan platillos)
NAMES = ("bombo", "caja", "tom amarillo", "charles", "tom azul", "ride", "tom verde", "crash")
_CACHE: dict = {}


def _item(p: Path, d) -> dict | None:
    x = no._inputs(d["cqt"], d["onset"])
    if x.shape[1] < CROP:
        return None
    refs = [d[f"c{k}"].astype(np.float64) for k in range(N_CLS)]
    if sum(len(r) for r in refs) < 150:
        return None
    return {"x": x, "refs": refs, "pro": bool(d["pro"]), "sep": bool(d["sep"]), "name": str(d["folder"])}


def load_dataset(log=print) -> tuple[list, list]:
    train, val_sep, val_stem = [], [], []
    hold = no._holdout_titles()
    for p in sorted(DRUM_SPEC_DIR.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        try:
            d = np.load(p)
            folder = str(d["folder"])
            if no._title(Path(folder)) in hold:
                continue
            it = _item(p, d)
        except Exception:  # noqa: BLE001
            continue
        if it is None:
            continue
        if not no._is_val(folder):
            train.append(it)
        elif it["pro"]:
            (val_sep if it["sep"] else val_stem).append(it)
    # se valida con baterias separadas por la IA (como al generar); si aun hay pocas, tambien con pistas originales
    val = val_sep if len(val_sep) >= 20 else val_sep + val_stem[:max(0, 40 - len(val_sep))]
    log(f"  {len(train)} canciones para entrenar ({sum(i['sep'] for i in train)} separadas por la IA, "
        f"{sum(i['pro'] for i in train)} Pro), {len(val)} para validar ({sum(i['sep'] for i in val)} separadas)")
    return train, val


def _target(n: int, s: int, refs: list[np.ndarray]) -> np.ndarray:
    on = np.zeros((CROP, N_CLS), np.float32)
    for k, t in enumerate(refs):
        f = np.round(t * no.FPS).astype(int) - s
        f = f[(f >= -1) & (f <= CROP)]
        for g, v in ((f - 1, 0.25), (f + 1, 0.25), (f, 1.0)):
            ok = (g >= 0) & (g < CROP)
            on[g[ok], k] = np.maximum(on[g[ok], k], v)
    return on


def predict(model, x: np.ndarray) -> np.ndarray:
    """Probabilidad por trama y clase (n, 8)."""
    import torch

    dev = next(model.parameters()).device
    xf = x.astype(np.float32) / 255.0
    n = xf.shape[1]
    win, hop = 4096, 3584
    acc = np.zeros((n, N_CLS), np.float32)
    cnt = np.zeros(n, np.float32)
    model.eval()
    with torch.no_grad():
        s = 0
        while True:
            e = min(s + win, n)
            o = torch.sigmoid(model(torch.from_numpy(xf[:, s:e][None]).to(dev))[0].float())[0].cpu().numpy()
            acc[s:e] += o
            cnt[s:e] += 1
            if e == n:
                break
            s += hop
    return acc / cnt[:, None]


def evaluate(model, val: list, thrs=(0.3, 0.4, 0.5, 0.6)) -> dict:
    from .eval_full import onset_f1

    res = {t: [[] for _ in range(N_CLS)] for t in thrs}
    allf = {t: [] for t in thrs}
    for it in val:
        pr = predict(model, it["x"])
        for t in thrs:
            ours_all = []
            for k in range(N_CLS):
                ours = no.peak_times(pr[:, k], no.peaks(pr[:, k], t))
                ours_all += list(ours)
                if len(it["refs"][k]) >= 10:
                    res[t][k].append(onset_f1(list(it["refs"][k]), list(ours))[0])
            ref_all = np.unique(np.round(np.concatenate(it["refs"]), 2))
            allf[t].append(onset_f1(list(ref_all), sorted(set(np.round(ours_all, 2))))[0])
    thr = [max(thrs, key=lambda t: np.mean(res[t][k]) if res[t][k] else 0) for k in range(N_CLS)]
    per = [float(np.mean(res[thr[k]][k])) if res[thr[k]][k] else 0.0 for k in range(N_CLS)]
    best_all = max(thrs, key=lambda t: np.mean(allf[t]))
    # media ponderada por cuantas canciones usan cada clase
    wts = [len(res[thr[k]][k]) for k in range(N_CLS)]
    return {"f1": float(np.average(per, weights=wts)) if sum(wts) else 0.0, "f1_clases": [round(v, 3) for v in per],
            "umbrales": thr, "golpes": float(np.mean(allf[best_all]))}


def train(minutes: float = 60, log=print, width: float = 1.5, batch_size: int = 32, eval_every: int = 1000,
          lr: float = 1e-3, init: Path | None = None, stop_file: Path | None = None, checkpoint: Path | None = None) -> dict:
    import torch
    from torch import nn

    seed = int(time.time()) % 10000
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, val = load_dataset(log)
    if len(tr) < 50:
        raise SystemExit("Hacen falta mas espectrogramas de bateria (drum_spectra).")
    model = no._net(N_CLS, width).to(dev)
    if init is not None and init.exists():
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["state"])
        log(f"  se sigue entrenando {init.name}")
    log(f"  modelo de ancho {width}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parametros")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=no.WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    lens = np.array([it["x"].shape[1] for it in tr], np.float64)
    pick = lens * np.array([SEP_WEIGHT if it["sep"] else 1.0 for it in tr])
    pick /= pick.sum()
    pro_mask = torch.zeros(N_CLS, device=dev)
    pro_mask[list(PRO_ONLY)] = 1
    deadline = time.time() + minutes * 60
    t0 = time.time()
    step, best, best_state, best_info = 0, -1.0, None, {}
    while time.time() < deadline and not (stop_file is not None and stop_file.exists()):
        batch = rng.choice(len(tr), batch_size, p=pick)
        xs, ons, pros = [], [], []
        for i in batch:
            it = tr[i]
            s = int(rng.integers(0, it["x"].shape[1] - CROP + 1))
            x = it["x"][:, s:s + CROP].astype(np.float32) / 255.0
            x[:84] = np.clip(x[:84] + rng.normal(0, 0.04) + np.linspace(-1, 1, 84)[:, None] * rng.normal(0, 0.05), 0, 1)
            for _ in range(int(rng.integers(0, 3))):
                w = int(rng.integers(2, 9))
                f0 = int(rng.integers(0, 84 - w))
                x[f0:f0 + w] = 0
            xs.append(x)
            ons.append(_target(it["x"].shape[1], s, it["refs"]))
            pros.append(1.0 if it["pro"] else 0.0)
        xb = torch.from_numpy(np.stack(xs)).to(dev)
        ob = torch.from_numpy(np.stack(ons)).to(dev)
        pb = torch.tensor(pros, device=dev)[:, None, None]
        frac = min((time.time() - t0) / (minutes * 60), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + np.cos(np.pi * frac))) + 2e-5
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(xb)[0]
        o = o.float()
        w = 1 + 4 * (ob >= 0.99).float()
        # sin Pro no se sabe si el amarillo/azul/verde es tom o platillo: esas salidas no se aprenden de ahi
        w = w * (1 - pro_mask * (1 - pb))
        loss = (bce(o, ob) * w).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % eval_every == 0:
            r = evaluate(model, val)
            model.train()
            log(f"  paso {step}: perdida {loss.item():.3f} | validacion F1 {r['f1']:.3f} (golpes {r['golpes']:.3f}) "
                f"{dict(zip(NAMES, r['f1_clases']))} | {(deadline - time.time()) / 60:.0f} min restantes")
            if r["f1"] > best:
                best, best_info = r["f1"], {**r, "pasos": step, "ancho": width, "canciones": len(tr)}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint is not None:
                    torch.save({"state": best_state, "info": best_info}, checkpoint)
    if best_state is None:
        r = evaluate(model, val)
        best_info = {**r, "pasos": step, "ancho": width, "canciones": len(tr)}
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best = r["f1"]
    old = load_model()
    if old is not None and old[1].get("f1", 0) > best + 0.002:
        log(f"Modelo de bateria: el anterior sigue siendo mejor ({old[1]['f1']:.3f} frente a {best:.3f}), se queda")
        return old[1]
    if MODEL_PATH.exists():
        import shutil

        shutil.copyfile(MODEL_PATH, MODEL_PATH.with_name("bateria_neuronal_anterior.pt"))
    torch.save({"state": best_state, "info": best_info}, MODEL_PATH)
    _CACHE.clear()
    log(f"Modelo de bateria guardado: {best_info}")
    return best_info


def load_model():
    if "m" not in _CACHE:
        m = None
        if MODEL_PATH.exists():
            try:
                import torch

                ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                net = no._net(N_CLS, float(ck["info"].get("ancho", 1.5)))
                net.load_state_dict(ck["state"])
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                m = (net.to(dev).eval(), ck["info"])
            except Exception:  # noqa: BLE001
                m = None
        _CACHE["m"] = m
    return _CACHE["m"]


def hits_from_audio(y: np.ndarray, sr: int) -> list[tuple[float, int, float]] | None:
    """Golpes (tiempo, clase, probabilidad) de la pista de bateria con el modelo, o None si no hay modelo."""
    from .drum_spectra import drum_spectrum

    m = load_model()
    if m is None:
        return None
    sp = drum_spectrum(y, sr)
    pr = predict(m[0], no._inputs(sp["cqt"], sp["onset"]))
    out = []
    for k in range(N_CLS):
        thr = float(m[1]["umbrales"][k]) if "umbrales" in m[1] else 0.5
        fr = no.peaks(pr[:, k], thr)
        for f, t in zip(fr, no.peak_times(pr[:, k], fr)):
            out.append((float(t), k, float(pr[f, k])))
    out.sort()
    return out


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    print(json.dumps(train(float(sys.argv[1]) if len(sys.argv) > 1 else 60, log=lambda m: print(m, flush=True))))
