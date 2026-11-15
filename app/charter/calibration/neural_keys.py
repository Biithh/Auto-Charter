"""Modelo neuronal de teclado: mira el espectrograma de la pista de teclado y dice donde pone nota un charter, tanto
en 5 carriles como en Pro Keys (y que teclas toca, de C3 a C5). Es como neural_onset.py para la guitarra, con las
salidas de Pro Keys y una entrada directa del espectro, porque aqui importa la nota exacta.

Uso: python -m charter.calibration.neural_keys [minutos]   (usa la GPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths
from .neural_onset import (CROP, DIFFS, FAST_BOOST, FPS, STEM_SPEC_DIR, WEIGHT_DECAY, _button_targets, _fast_frames, _holdout_titles,
                           _inputs, _is_val, _title, peak_times, peaks)

KEYS_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_teclas"
STATS_CACHE = paths.CONFIG_DIR / "entrenamiento" / "estadisticas_cache.json"
MODEL_PATH = paths.CONFIG_DIR / "teclas_neuronal.pt"
PREV_PATH = paths.CONFIG_DIR / "teclas_neuronal_anterior.pt"
PREV2_PATH = paths.CONFIG_DIR / "teclas_neuronal_anterior2.pt"
USE_PREV = True  # 12-11: el nuevo junto al anterior acierta mas que solo (5 carriles 0,761 frente a 0,758, Pro 0,758 frente a 0,755)
USE_PREV2 = False  # el de hace dos entrenamientos no suma (actual + anterior 0,758)
TONES_PATH = paths.CONFIG_DIR / "teclas_tonos.pt"
PRO_LOW = 48  # C3, la tecla mas grave de Pro Keys
PRO_KEYS = 25
NEG_SHARE = 0.12  # parte de cada lote con pistas sin teclado: restos de otros instrumentos no son notas
NEG_DB = (10.0, 35.0)  # esos restos suenan mucho mas bajos que un teclado de verdad
MAX_NEGATIVES = 500
# que oye el modelo de una cancion separada por la IA: la pista de piano o piano + resto (ver keys_spectra.separate_all)
USE_MIX = True
# al generar solo se oyen pistas separadas por la IA: sus canciones y sus restos pesan mas aunque haya pocas
KEYS_SEP_WEIGHT = 6.0
REAL_NEG_SHARE = 0.5


def _net(width: float = 1.5):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                     nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())

            c1, c2, c3 = int(24 * width), int(48 * width), int(64 * width)
            hid = int(128 * width)
            self.conv = nn.Sequential(block(1, c1), nn.MaxPool2d((2, 1)), block(c1, c2), nn.MaxPool2d((2, 1)),
                                      block(c2, c3), nn.MaxPool2d((3, 1)), nn.Dropout(0.15))
            self.env = nn.Conv1d(1, 16, 5, padding=2)
            # las convoluciones reducen la frecuencia; esta entrada conserva cada semitono para saber la tecla
            self.raw = nn.Linear(84, int(64 * width))
            self.proj = nn.Linear(c3 * 7 + 16 + int(64 * width), int(192 * width))
            self.rnn = nn.GRU(int(192 * width), hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            self.onset = nn.Linear(2 * hid, len(DIFFS))
            self.chord = nn.Linear(2 * hid, 3)
            self.move = nn.Linear(2 * hid, 3)
            self.lane = nn.Linear(2 * hid, 5)
            self.pro_onset = nn.Linear(2 * hid, len(DIFFS))
            self.pro_keys = nn.Linear(2 * hid, PRO_KEYS)
            self.hold = nn.Linear(2 * hid, 1)

        def features(self, x):  # x: (B, 85, T) en 0..1
            c = self.conv(x[:, None, :84])
            b, ch, f, t = c.shape
            c = c.reshape(b, ch * f, t).transpose(1, 2)
            e = torch.relu(self.env(x[:, 84:85])).transpose(1, 2)
            r = torch.relu(self.raw(x[:, :84].transpose(1, 2)))
            h, _ = self.rnn(torch.relu(self.proj(torch.cat([c, e, r], 2))))
            return h

        def forward(self, x):
            h = self.features(x)
            return (self.onset(h), self.chord(h), self.move(h), self.lane(h), self.pro_onset(h), self.pro_keys(h),
                    self.hold(h)[..., 0])

    return Net()


def _onset_target(n: int, refs: list[np.ndarray]) -> np.ndarray:
    on = np.zeros((n, len(refs)), np.float32)
    for k, ref_t in enumerate(refs):
        f = np.round(np.asarray(ref_t, np.float64) * FPS).astype(int)
        f = f[(f >= 0) & (f < n)]
        for g in (f - 1, f + 1):
            g = g[(g >= 0) & (g < n)]
            on[g, k] = 0.25
        on[f, k] = 1.0
    return on


def _hold_target(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    h = np.zeros(n, np.uint8)
    for a, b in zip(starts, ends):
        if b - a >= 0.1:
            h[max(int(a * FPS), 0): max(int(b * FPS), 0)] = 1
    return h


def _item(p: Path, d) -> dict | None:
    x = _inputs(d["cqt_mix"], d["onset_mix"]) if USE_MIX and "cqt_mix" in d else _inputs(d["cqt"], d["onset"])
    n = x.shape[1]
    if n < CROP:
        return None
    has5, haspro = bool(d["has5"]), bool(d["haspro"])
    refs5 = [d[f"k5_{k}"].astype(np.float64) for k in DIFFS]
    refspro = [d[f"pr_{k}"].astype(np.float64) for k in DIFFS]
    on = _onset_target(n, refs5 + refspro)
    k5_t, sizes, lanes = d["k5_t"].astype(np.float64), d["k5_size"], d["k5_lane"]
    order = np.argsort(k5_t)
    k5_t, sizes, lanes = k5_t[order], sizes[order], lanes[order]
    ch = np.full(n, -1, np.int8)
    f = np.round(k5_t * FPS).astype(int)
    ok = (f >= 0) & (f < n)
    ch[f[ok]] = np.clip(sizes[ok], 1, 3) - 1
    mv, ln = _button_targets(n, k5_t, lanes.astype(np.int64))
    # teclas de Pro Keys: solo donde hay nota (es la pregunta "de las notas que hay aqui, cuales son")
    pk_f = np.round(d["pk_t"].astype(np.float64) * FPS).astype(int)
    pk_i = d["pk_p"].astype(int) - PRO_LOW
    ok = (pk_f >= 0) & (pk_f < n) & (pk_i >= 0) & (pk_i < PRO_KEYS)
    hold = _hold_target(n, d["k5_t"], d["k5_end"]) if has5 else _hold_target(n, d["pk_t"], d["pk_end"])
    return {"x": x, "on": on.astype(np.float16), "ch": ch, "mv": mv.astype(np.int8), "ln": ln.astype(np.int8),
            "fast": _fast_frames(n, np.unique(np.round(np.concatenate([refs5[0], refspro[0]]), 2))),  # sin contar dos veces la misma nota
            "pk_f": pk_f[ok].astype(np.int32), "pk_i": pk_i[ok].astype(np.int8), "hold": hold, "has5": has5,
            "haspro": haspro, "refs5": refs5, "refspro": refspro, "name": str(d["folder"]), "sep": bool(d["sep"])}


def _negatives(log) -> list[dict]:
    """Pistas de guitarra de canciones sin teclado: al generar, el teclado separado de esas canciones trae restos."""
    try:
        stats = json.loads(STATS_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    no_keys = {v["folder"] for v in stats.values()
               if not (v.get("tracks", {}).get("keys") or v.get("has_pro_keys"))}
    out: list[dict] = []
    # los restos de verdad de una cancion sin teclado, separada por la IA (las de validacion no se tocan)
    for p in sorted(KEYS_SPEC_DIR.glob("*_ianeg.npz")):
        try:
            d = np.load(p)
            if not _is_val(str(d["folder"])):
                out.append({"x": _inputs(d["cqt_mix"], d["onset_mix"]) if USE_MIX else _inputs(d["cqt"], d["onset"]), "real": True})
        except Exception:  # noqa: BLE001
            continue
    files = sorted(STEM_SPEC_DIR.glob("*_guitar.npz"))
    for p in np.random.default_rng(7).permutation(files):
        if len(out) >= MAX_NEGATIVES:
            break
        try:
            d = np.load(p)
            if Path(str(d["folder"])).name not in no_keys or _is_val(str(d["folder"])):
                continue
            x = _inputs(d["cqt"], d["onset"])
        except Exception:  # noqa: BLE001
            continue
        if x.shape[1] >= CROP:
            out.append({"x": x})
    log(f"  {len(out)} canciones sin teclado como ejemplo de lo que no son notas")
    return out


def load_dataset(log=print) -> tuple[list, list, list]:
    train, val_sep, val_stem = [], [], []
    hold = _holdout_titles()
    for p in sorted(KEYS_SPEC_DIR.glob("*.npz")):
        if p.name.endswith((".tmp.npz", "_ianeg.npz")):
            continue
        try:
            d = np.load(p)
            if _title(Path(str(d["folder"]))) in hold:
                continue
            it = _item(p, d)
        except Exception:  # noqa: BLE001
            continue
        if it is None:
            continue
        if not _is_val(it["name"]):
            train.append(it)
        else:
            (val_sep if it["sep"] else val_stem).append(it)
    # se valida con teclados separados por la IA, como los que vera al generar
    val = val_sep if len(val_sep) >= 20 else val_sep + val_stem
    log(f"  {len(train)} canciones para entrenar ({sum(i['sep'] for i in train)} separadas por la IA), "
        f"{len(val)} para validar ({sum(i['sep'] for i in val)} separadas por la IA)")
    return train, val, _negatives(log)


def predict(model, x: np.ndarray, device=None) -> dict:
    """Por trama: 'on' (n, dificultades), 'chord', 'move', 'lane', 'pro_on' (n, dificultades), 'keys' (n, 25), 'hold'."""
    import torch

    device = device or next(model.parameters()).device
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    model.eval()
    win, hop = 4096, 3584
    n = x.shape[1]
    acc: dict = {}
    cnt = np.zeros(n, np.float32)
    with torch.no_grad():
        s = 0
        while True:
            e = min(s + win, n)
            o, c, mv, ln, po, pk, hd = model(torch.from_numpy(x[:, s:e][None]).to(device))
            parts = {"on": torch.sigmoid(o), "chord": torch.softmax(c, -1), "move": torch.softmax(mv, -1),
                     "lane": torch.softmax(ln, -1), "pro_on": torch.sigmoid(po), "keys": torch.sigmoid(pk),
                     "hold": torch.sigmoid(hd)[..., None]}
            for k, v in parts.items():
                v = v[0].float().cpu().numpy()
                if k not in acc:
                    acc[k] = np.zeros((n, v.shape[1]), np.float32)
                acc[k][s:e] += v
            cnt[s:e] += 1
            if e == n:
                break
            s += hop
    return {k: v / cnt[:, None] for k, v in acc.items()}


def predict_mean(nets: list, x: np.ndarray) -> dict:
    outs = [predict(m, x) for m, _ in nets]
    res = {k: np.mean([o[k] for o in outs], 0) for k in outs[0]}
    tones = load_tones()
    if tones is not None:
        res["keys"] = predict_tones(tones[0], x)
    return res


def predict_all(nets: list, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Lo mismo que neural_onset.predict_all, para que el generador de 5 carriles sirva tal cual."""
    p = predict_mean(nets, x)
    return p["on"], p["chord"], p["move"], p["lane"]


def _pitch_f1(pred: np.ndarray, it: dict, thr: float) -> tuple[int, int, int]:
    """Aciertos de teclas en las tramas que tienen nota en el chart: (aciertos, nuestras, del chart)."""
    tp = fp = fn = 0
    by_f: dict[int, set[int]] = {}
    for f, i in zip(it["pk_f"], it["pk_i"]):
        by_f.setdefault(int(f), set()).add(int(i))
    for f, ref in by_f.items():
        ours = set(np.where(pred[f] >= thr)[0].tolist()) or {int(np.argmax(pred[f]))}
        tp += len(ours & ref)
        fp += len(ours - ref)
        fn += len(ref - ours)
    return tp, fp, fn


def evaluate(model, val: list, thrs: list[float], k: int = 0) -> dict:
    from .eval_full import onset_f1

    r5 = {t: [] for t in thrs}
    rp = {t: [] for t in thrs}
    for it in val:
        p = predict(model, it["x"])
        for res, on, refs, has in ((r5, p["on"], it["refs5"], it["has5"]), (rp, p["pro_on"], it["refspro"], it["haspro"])):
            if not has or len(refs[k]) == 0:
                continue
            prob = on[:, k]
            for t in thrs:
                res[t].append(onset_f1(refs[k], peak_times(prob, peaks(prob, t)))[0])
    return {"k5": {t: float(np.mean(v)) if v else 0.0 for t, v in r5.items()},
            "pro": {t: float(np.mean(v)) if v else 0.0 for t, v in rp.items()}}


def train(minutes: float = 40, log=print, width: float = 1.5, batch_size: int = 32, eval_every: int = 1000,
          seed: int | None = None, lr: float = 1e-3, stop_file: Path | None = None, checkpoint: Path | None = None,
          init: Path | None = None) -> dict:
    import torch
    from torch import nn

    if seed is None:
        seed = int(time.time()) % 10000
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, val, negs = load_dataset(log)
    if len(tr) < 50:
        raise SystemExit("Hacen falta mas espectrogramas de teclado (keys_spectra).")
    model = _net(width).to(dev)
    if init is not None:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["state"])
        log(f"  se sigue entrenando {init.name}")
    log(f"  modelo de ancho {width}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parametros")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    pick = np.array([it["x"].shape[1] * (KEYS_SEP_WEIGHT if it["sep"] else 1.0) for it in tr], np.float64)
    real_negs = [it for it in negs if "real" in it]
    pick /= pick.sum()
    deadline = time.time() + minutes * 60
    t0 = time.time()
    step, best, best_state = 0, -1.0, None
    thrs = [0.3, 0.4, 0.5, 0.6]
    diff_w = torch.tensor([1.0, 0.6, 0.6, 0.6], device=dev)
    while time.time() < deadline and not (stop_file is not None and stop_file.exists()):
        xs, ons, chs, mvs, lns, pks, hds, m5, mp, fss = [], [], [], [], [], [], [], [], [], []
        for _ in range(batch_size):
            neg = bool(negs) and rng.random() < NEG_SHARE
            if neg and real_negs and rng.random() < REAL_NEG_SHARE:
                it = real_negs[int(rng.integers(len(real_negs)))]
            else:
                it = negs[int(rng.integers(len(negs)))] if neg else tr[int(rng.choice(len(tr), p=pick))]
            s = int(rng.integers(0, it["x"].shape[1] - CROP + 1))
            x = it["x"][:, s:s + CROP].astype(np.float32) / 255.0
            gain = -rng.uniform(*NEG_DB) * 0.01 if neg and "real" not in it else rng.normal(0, 0.04)
            x[:84] = np.clip(x[:84] + gain, 0, 1)
            shift = int(rng.integers(-2, 3))
            if shift:
                x[:84] = np.roll(x[:84], shift, 0)
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[84 + shift:84] = 0
            x[:84] = np.clip(x[:84] + np.linspace(-1, 1, 84)[:, None] * rng.normal(0, 0.05), 0, 1)
            for _b in range(int(rng.integers(0, 3))):
                w = int(rng.integers(2, 9))
                f0 = int(rng.integers(0, 84 - w))
                x[f0:f0 + w] = 0
            xs.append(x)
            pk = np.full((CROP, PRO_KEYS), -1, np.int8)  # -1: sin nota en el chart, no cuenta
            if neg:
                ons.append(np.zeros((CROP, 8), np.float32))
                fss.append(np.zeros(CROP, np.float32))
                chs.append(np.full(CROP, -1, np.int64))
                mvs.append(np.full(CROP, -1, np.int64))
                lns.append(np.full(CROP, -1, np.int64))
                hds.append(np.zeros(CROP, np.float32))
                m5.append(1.0)
                mp.append(1.0)
            else:
                ons.append(it["on"][s:s + CROP].astype(np.float32))
                fss.append(it["fast"][s:s + CROP].astype(np.float32))
                chs.append(it["ch"][s:s + CROP].astype(np.int64))
                mvs.append(it["mv"][s:s + CROP].astype(np.int64))
                lns.append(it["ln"][s:s + CROP].astype(np.int64))
                hds.append(it["hold"][s:s + CROP].astype(np.float32))
                m5.append(float(it["has5"]))
                mp.append(float(it["haspro"]))
                sel = (it["pk_f"] >= s) & (it["pk_f"] < s + CROP)
                f = it["pk_f"][sel] - s
                # al subir de tono, la tecla sube igual; la que se sale del teclado vuelve a entrar una octava abajo o arriba
                i = it["pk_i"][sel].astype(int) + shift
                i = np.where(i >= PRO_KEYS, i - 12, np.where(i < 0, i + 12, i))
                pk[f] = 0
                pk[f, i] = 1
            pks.append(pk)
        xb = torch.from_numpy(np.stack(xs)).to(dev)
        ob = torch.from_numpy(np.stack(ons)).to(dev)
        cb = torch.from_numpy(np.stack(chs)).to(dev)
        mb = torch.from_numpy(np.stack(mvs)).to(dev)
        lb = torch.from_numpy(np.stack(lns)).to(dev)
        kb = torch.from_numpy(np.stack(pks).astype(np.float32)).to(dev)
        hb = torch.from_numpy(np.stack(hds)).to(dev)
        mask5 = torch.tensor(m5, device=dev)[:, None, None]
        maskp = torch.tensor(mp, device=dev)[:, None, None]
        frac = min((time.time() - t0) / (minutes * 60), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + np.cos(np.pi * frac))) + 2e-5
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o, c, mv, ln, po, pk_out, hd = model(xb)
        o, c, mv, ln, po, pk_out, hd = o.float(), c.float(), mv.float(), ln.float(), po.float(), pk_out.float(), hd.float()
        on5, onp = ob[..., :4], ob[..., 4:]
        # las notas muy seguidas son las que mas se escapan (ver neural_onset.FAST_GAP)
        fast = 1 + (FAST_BOOST - 1) * torch.from_numpy(np.stack(fss)).to(dev)[..., None] * torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        loss = (bce(o, on5) * (1 + 4 * (on5 >= 0.99).float()) * fast * diff_w * mask5).mean()
        loss = loss + (bce(po, onp) * (1 + 4 * (onp >= 0.99).float()) * fast * diff_w * maskp).mean()
        for out, tg, n_cls, wt in ((c, cb, 3, 0.3), (mv, mb, 3, 0.3), (ln, lb, 5, 0.15)):
            if (tg >= 0).any():
                loss = loss + wt * ce(out.reshape(-1, n_cls), tg.reshape(-1))
        seen = kb >= 0
        if seen.any():
            loss = loss + 0.5 * bce(pk_out[seen], kb[seen]).mean()
        loss = loss + 0.2 * bce(hd, hb).mean()
        opt.zero_grad()
        step += 1
        if not torch.isfinite(loss):
            continue
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if step % eval_every == 0:
            r = evaluate(model, val, thrs)
            b5, bp = max(r["k5"], key=r["k5"].get), max(r["pro"], key=r["pro"].get)
            score = (r["k5"][b5] + r["pro"][bp]) / 2
            log(f"  paso {step}: perdida {loss.item():.3f} | validacion 5 carriles {r['k5'][b5]:.3f} (umbral {b5}), "
                f"Pro Keys {r['pro'][bp]:.3f} (umbral {bp}) | {(deadline - time.time()) / 60:.0f} min restantes")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint is not None:
                    torch.save({"state": best_state, "info": {"score": best, "pasos": step, "ancho": width}}, checkpoint)
            model.train()
    if best_state is None:
        log("Modelo de teclado: parado antes de poder medirlo, no se guarda")
        return {}
    model.load_state_dict(best_state)
    fine = [round(x, 2) for x in np.arange(0.2, 0.71, 0.05)]
    thr5, f15, thrp, f1p = {}, {}, {}, {}
    for k, d in enumerate(DIFFS):
        r = evaluate(model, val, fine, k)
        thr5[d] = float(max(r["k5"], key=r["k5"].get))
        f15[d] = round(r["k5"][thr5[d]], 3)
        thrp[d] = float(max(r["pro"], key=r["pro"].get))
        f1p[d] = round(r["pro"][thrp[d]], 3)
    by_thr = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        tot = np.zeros(3)
        for it in val:
            if it["haspro"]:
                tot += _pitch_f1(predict(model, it["x"])["keys"], it, t)
        by_thr[t] = 2 * tot[0] / max(2 * tot[0] + tot[1] + tot[2], 1)
    thr_keys = float(max(by_thr, key=by_thr.get))
    info = {"thr": thr5["expert"], "thr_diff": thr5, "val_f1": f15["expert"], "val_f1_diff": f15, "thr_pro": thrp["expert"],
            "thr_pro_diff": thrp, "val_f1_pro": f1p["expert"], "val_f1_pro_diff": f1p, "thr_keys": thr_keys,
            "val_f1_teclas": round(float(by_thr[thr_keys]), 3), "botones": 1, "canciones": len(tr), "pasos": step, "ancho": width}
    log(f"  teclas de Pro Keys: F1 {by_thr[thr_keys]:.3f} en las notas de validacion (umbral {thr_keys})")
    info["presencia"] = presence_threshold(model, val, thr5["expert"], log)
    old = _load(MODEL_PATH)
    if old is not None:
        r = evaluate(old[0], val, [old[1]["thr"]])
        old_f1 = r["k5"][old[1]["thr"]]
        if old_f1 > f15["expert"] + 0.002:
            log(f"Modelo de teclado: el anterior sigue siendo mejor ({old_f1:.3f} frente a {f15['expert']:.3f}), se queda")
            return old[1]
    _CACHE.clear()
    if MODEL_PATH.exists():
        import shutil

        if PREV_PATH.exists():
            shutil.copyfile(PREV_PATH, PREV2_PATH)
        shutil.copyfile(MODEL_PATH, PREV_PATH)
    torch.save({"state": model.state_dict(), "info": info}, MODEL_PATH)
    log(f"Modelo de teclado guardado: 5 carriles {f15}, Pro Keys {f1p}, umbrales {thr5} {thrp} ({len(tr)} canciones)")
    return info


def _rate(model, x: np.ndarray, thr: float) -> float:
    """Notas de 5 carriles por minuto que pone el modelo en una pista."""
    on = predict(model, x)["on"][:, 0]
    return len(peaks(on, thr)) / max(x.shape[1] / FPS / 60, 0.1)


def presence_threshold(model, val: list, thr: float, log=print) -> float:
    """Cuantas notas por minuto hacen falta para decir que la cancion tiene teclado: entre lo que sale en canciones
    que no lo tienen (sus restos) y lo que sale en las que si."""
    pos = [_rate(model, it["x"], thr) for it in val if it["has5"]]
    neg = []
    for p in sorted(KEYS_SPEC_DIR.glob("*_ianeg.npz")):
        try:
            d = np.load(p)
            if _is_val(str(d["folder"])):
                neg.append(_rate(model, _inputs(d["cqt_mix"], d["onset_mix"]) if USE_MIX else _inputs(d["cqt"], d["onset"]), thr))
        except Exception:  # noqa: BLE001
            continue
    if len(neg) < 10:  # con tan pocas el umbral sale absurdo; 40 es lo medido con 35 canciones con teclado y 30 sin el
        return 40.0
    if not pos or not neg:
        return 15.0
    best = max(np.arange(2.0, 80.0, 2.0), key=lambda t: np.mean(np.array(pos) >= t) + np.mean(np.array(neg) < t))
    log(f"  presencia: notas por minuto con teclado {np.percentile(pos, [10, 50]).round(0).tolist()} (p10, p50), sin teclado "
        f"{np.percentile(neg, [50, 90]).round(0).tolist()} (p50, p90, {len(neg)} canciones); desde {best:.0f} se chartea "
        f"(detecta el {np.mean(np.array(pos) >= best):.0%}, se equivoca en el {np.mean(np.array(neg) >= best):.0%} sin teclado)")
    return float(best)


def notes_from_model(sa, log=print) -> tuple[list, float]:
    """Notas del teclado segun el modelo (tiempo, tecla y duracion) y notas por minuto. Hace el papel de la
    transcripcion, que en un teclado no aporta: el modelo ya dice la tecla y es mucho mas rapido."""
    from ..pipeline.transcription import Note
    from .onset_model import compact_spectrum

    nets = load_models()
    info = nets[0][1]
    spec = compact_spectrum(sa)
    p = predict_mean(nets, _inputs(spec["cqt"], spec["onset"]))
    prob = p["on"][:, 0]
    frames = peaks(prob, info["thr"])
    times = peak_times(prob, frames)
    notes = []
    for f, t in zip(frames, times):
        for pitch in keys_at(p["keys"], int(f), info):
            notes.append(Note(float(t), hold_end(p["hold"][:, 0], int(f)), pitch, votes=float(prob[f])))
    return notes, len(frames) / max(len(prob) / FPS / 60, 0.1)


# Acordes: el espectro de la pista, sumado por notas (do, do#...) en ~1,5 s, se compara con los 24 acordes mayores y
# menores. Las teclas del acorde que suena pesan un poco mas al elegir; las que no son del acorde, un poco menos.
CHORD_BOOST = 0.0  # 0 = no se usa (se ajusta con la prueba de Pro Keys de punta a punta)
CHORD_WIN = 64  # tramas (~1,5 s)
CHORD_MIN_CONF = 0.0  # parecido minimo con el acorde para fiarse de el


def _chord_templates() -> np.ndarray:
    t = np.zeros((24, 12), np.float32)
    for r in range(12):
        t[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1  # mayor
        t[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1  # menor
    return t


def chord_tones(x: np.ndarray) -> np.ndarray:
    """(n, 12): 1 en las notas del acorde que suena en cada trama (todo 0 si no esta claro). x: entrada del modelo."""
    q = x[:84].astype(np.float32) * (1.0 if x.dtype == np.uint8 else 255.0)
    n = q.shape[1]
    amp = 10 ** ((q / 2.55 - 100.0) / 20.0)  # de 0..255 (dB) a amplitud; la CQT empieza en C1, asi que fila i = nota i % 12
    chroma = amp.reshape(7, 12, n).sum(0)
    ker = np.ones(CHORD_WIN, np.float32) / CHORD_WIN
    chroma = np.stack([np.convolve(c, ker, mode="same") for c in chroma])
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    tpl = _chord_templates()
    sim = (tpl / np.linalg.norm(tpl, axis=1, keepdims=True)) @ chroma  # (24, n)
    best = sim.argmax(0)
    out = tpl[best]
    out[sim.max(0) < CHORD_MIN_CONF] = 0
    return out


def keys_at(keys: np.ndarray, f: int, info: dict, chord: np.ndarray | None = None) -> list[int]:
    """Teclas (MIDI) que toca el modelo en la trama f; al menos una. `chord`: salida de chord_tones."""
    k = keys[max(f - 1, 0): f + 2].max(axis=0)
    if chord is not None and CHORD_BOOST and chord[f].any():
        inside = chord[f][(np.arange(PRO_KEYS) + PRO_LOW) % 12] > 0
        k = np.clip(k * np.where(inside, 1 + CHORD_BOOST, 1 - CHORD_BOOST), 0, 1)
    tones = load_tones()
    idx = np.where(k >= (TONE_THR if tones is not None else info["thr_keys"]))[0]
    return [int(i) + PRO_LOW for i in (idx if len(idx) else [int(np.argmax(k))])]


def hold_end(hold: np.ndarray, f: int, thr: float = 0.5) -> float:
    """Hasta cuando se mantiene la nota que empieza en la trama f (segundos)."""
    below = np.where(hold[f + 1:] < thr)[0]
    return float(f + 1 + (below[0] if len(below) else len(hold) - f - 1)) / FPS


_CACHE: dict = {}


def _load(path: Path):
    if not path.exists():
        return None
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = _net(float(ck["info"].get("ancho", 1.5)))
        net.load_state_dict(ck["state"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return net.to(dev).eval(), ck["info"]
    except Exception:  # noqa: BLE001
        return None


def load_model():
    if "m" not in _CACHE:
        _CACHE["m"] = _load(MODEL_PATH)
    return _CACHE["m"]


def load_models() -> list:
    """El modelo actual y los de los entrenamientos anteriores (si son del mismo tamano): juntos aciertan mas."""
    if "all" not in _CACHE:
        main = load_model()
        out = [main] if main else []
        for path in ((PREV_PATH, PREV2_PATH) if USE_PREV2 else (PREV_PATH,)) if main and USE_PREV else ():
            prev = _load(path)
            if prev is not None and prev[1].get("ancho") == main[1].get("ancho"):
                out.append(prev)
        _CACHE["all"] = out
    return _CACHE["all"]


# ---------------------------------------------------------------------------- que teclas son
# Mira el espectrograma alrededor de cada golpe sin reducir la frecuencia: cada semitono sigue siendo un semitono.
TONE_PAST, TONE_FUT = 10, 18  # tramas que ve alrededor del golpe (incluye margen para que las capas vean contexto real)
TONE_SONG_CAP = 300  # golpes por cancion como mucho, para que no manden las canciones larguisimas
TONE_THR = 0.7  # con este umbral los acordes salen del tamano de los reales (1,56 teclas por golpe frente a 1,48)
TONE_SEP_SHARE = 0.15  # parte de cada lote con canciones separadas por la IA (hay muy pocas con Pro Keys)


def _tone_net():
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def b(i, o, k):
                return nn.Sequential(nn.Conv2d(i, o, k, padding=(k[0] // 2, k[1] // 2)), nn.BatchNorm2d(o), nn.GELU())

            self.conv = nn.Sequential(b(1, 32, (5, 3)), b(32, 32, (5, 3)), b(32, 48, (25, 3)), b(48, 48, (1, 5)), b(48, 16, (1, 1)))
            self.head = nn.Sequential(nn.Linear(16 * 2 * 84, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, PRO_KEYS))

        def forward(self, x):  # x: (B, 84, TONE_PAST + TONE_FUT): el golpe esta en la trama TONE_PAST
            c = self.conv(x[:, None])
            a = c[..., TONE_PAST - 4: TONE_PAST + 4].amax(-1)
            b_ = c[..., TONE_PAST + 4: TONE_PAST + 12].amax(-1)
            return self.head(torch.cat([a, b_], 1).flatten(1))

    return Net()


def predict_tones(net, x: np.ndarray) -> np.ndarray:
    """Probabilidad de cada tecla (n, 25) en todas las tramas de la pista."""
    import torch
    import torch.nn.functional as F

    dev = next(net.parameters()).device
    q = (x[:84].astype(np.float32) / 255.0) if x.dtype == np.uint8 else x[:84].astype(np.float32)
    n = q.shape[1]
    win, margin = 4096, 16
    out = np.zeros((n, PRO_KEYS), np.float32)
    net.eval()
    with torch.no_grad():
        for s in range(0, n, win):
            a, b = max(s - margin, 0), min(s + win + margin, n)
            c = net.conv(torch.from_numpy(q[:, a:b])[None, None].to(dev))
            m = c.shape[-1]
            p = F.max_pool2d(F.pad(c, (4, 11)), (1, 8), stride=1)  # p[t] mira t-4..t+3, p[t + 8] mira t+4..t+11
            feat = torch.cat([p[..., :m], p[..., 8: 8 + m]], 1).permute(3, 0, 1, 2).flatten(1)
            pr = torch.sigmoid(torch.cat([net.head(feat[i:i + 2048]) for i in range(0, m, 2048)])).cpu().numpy()
            k = s - a
            out[s:min(s + win, n)] = pr[k:k + min(win, n - s)]
    return out


def _tone_patches(d, cqt: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray] | None:
    t, pit = d["pk_t"].astype(np.float64), d["pk_p"].astype(int) - PRO_LOW
    ok = (pit >= 0) & (pit < PRO_KEYS)
    by: dict[int, list[int]] = {}
    for f, i in zip(np.round(t[ok] * FPS).astype(int), pit[ok]):
        by.setdefault(int(f), []).append(int(i))
    frames = [f for f in by if f >= TONE_PAST and f + TONE_FUT <= cqt.shape[1]]
    if not frames:
        return None
    frames = rng.permutation(frames)[:TONE_SONG_CAP]
    xs = np.stack([cqt[:, f - TONE_PAST: f + TONE_FUT] for f in frames]).astype(np.uint8)
    ys = np.zeros((len(frames), PRO_KEYS), np.uint8)
    for j, f in enumerate(frames):
        ys[j, by[int(f)]] = 1
    return xs, ys


def train_tones(minutes: float = 30, log=print, seed: int | None = None, batch_size: int = 128) -> dict:
    """Entrena la red de teclas con los mismos espectrogramas que el modelo de teclado. Se valida en separadas por la IA."""
    import torch
    from torch import nn

    seed = int(time.time()) % 10000 if seed is None else seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hold = _holdout_titles()
    sets: dict[str, tuple[list, list]] = {"pista": ([], []), "ia": ([], []), "val": ([], [])}
    for path in sorted(KEYS_SPEC_DIR.glob("*.npz")):
        if path.name.endswith((".tmp.npz", "_ianeg.npz")):
            continue
        try:
            d = np.load(path)
            folder = str(d["folder"])
            if not bool(d["haspro"]) or _title(Path(folder)) in hold:
                continue
            sep = bool(d["sep"])
            if _is_val(folder) and not sep:
                continue
            r = _tone_patches(d, d["cqt_mix"] if USE_MIX and sep else d["cqt"], rng)
        except Exception:  # noqa: BLE001
            continue
        if r is not None:
            name = "val" if _is_val(folder) else ("ia" if sep else "pista")
            sets[name][0].append(r[0])
            sets[name][1].append(r[1])
    log("  teclas: canciones " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()))
    (xp, yp), (xi, yi), (xv, yv) = [(np.concatenate(sets[k][0]), np.concatenate(sets[k][1])) for k in ("pista", "ia", "val")]
    del sets
    log(f"  teclas: golpes {len(xp)} pista, {len(xi)} separadas, {len(xv)} validacion")
    net = _tone_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=dev))
    thrs = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    def evaluate_tones() -> dict:
        net.eval()
        with torch.no_grad():
            pr = np.concatenate([torch.sigmoid(net(torch.from_numpy(xv[i:i + 512].astype(np.float32) / 255).to(dev))).cpu().numpy()
                                 for i in range(0, len(xv), 512)])
        net.train()
        top = pr.argmax(1)
        rows = np.arange(len(pr))
        chroma = np.zeros((len(pr), 12), bool)
        for k in range(PRO_KEYS):
            chroma[:, k % 12] |= yv[:, k] > 0
        f1 = {}
        for th in thrs:
            q = pr >= th
            q[rows, top] = True
            f1[th] = float(2 * (q & (yv > 0)).sum() / max(q.sum() + yv.sum(), 1))
        best_thr = max(f1, key=f1.get)
        return {"thr": best_thr, "val_f1": round(f1[best_thr], 3), "val_tecla": round(float(yv[rows, top].mean()), 3),
                "val_sin_octava": round(float(chroma[rows, top % 12].mean()), 3)}

    t0 = time.time()
    deadline = t0 + minutes * 60
    step, best, best_info, best_state = 0, -1.0, {}, None
    n_sep = int(batch_size * TONE_SEP_SHARE)
    while time.time() < deadline:
        frac = (time.time() - t0) / (minutes * 60)
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.5 * (1 + np.cos(np.pi * frac))) + 1e-5
        ip, ii = rng.integers(0, len(xp), batch_size - n_sep), rng.integers(0, len(xi), n_sep)
        x = np.concatenate([xp[ip], xi[ii]]).astype(np.float32) / 255
        y = np.concatenate([yp[ip], yi[ii]]).astype(np.float32)
        for j in range(len(x)):
            sh = int(rng.integers(-2, 3))
            if sh:
                x[j] = np.roll(x[j], sh, 0)
                if sh > 0:
                    x[j, :sh] = 0
                else:
                    x[j, 84 + sh:] = 0
                # al subir de tono la tecla sube igual; la que se sale vuelve a entrar una octava abajo o arriba
                keys = np.where(y[j] > 0)[0] + sh
                y[j] = 0
                y[j, np.where(keys >= PRO_KEYS, keys - 12, np.where(keys < 0, keys + 12, keys))] = 1
            w = int(rng.integers(0, 7))
            if w:
                f0 = int(rng.integers(0, 84 - w))
                x[j, f0:f0 + w] = 0
        x = np.clip(x + rng.normal(0, 0.04, (len(x), 1, 1)) + np.linspace(-1, 1, 84)[None, :, None] * rng.normal(0, 0.05, (len(x), 1, 1)), 0, 1)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = crit(net(torch.from_numpy(x.astype(np.float32)).to(dev)).float(), torch.from_numpy(y).to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 2000 == 0:
            r = evaluate_tones()
            log(f"  teclas, paso {step}: perdida {loss.item():.3f} | {r} | {(deadline - time.time()) / 60:.0f} min restantes")
            if r["val_f1"] > best:
                best, best_info = r["val_f1"], {**r, "pasos": step}
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_state is None:
        return {}
    old = _load_tones()
    if old is not None and old[1].get("val_f1", 0) > best:
        log(f"  teclas: la red anterior sigue siendo mejor ({old[1]['val_f1']} frente a {best}), se queda")
        return old[1]
    torch.save({"state": best_state, "info": best_info}, TONES_PATH)
    _CACHE.pop("tonos", None)
    log(f"  teclas: guardada {best_info}")
    return best_info


def _load_tones():
    if not TONES_PATH.exists():
        return None
    try:
        import torch

        ck = torch.load(TONES_PATH, map_location="cpu", weights_only=False)
        net = _tone_net()
        net.load_state_dict(ck["state"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return net.to(dev).eval(), ck["info"]
    except Exception:  # noqa: BLE001
        return None


def load_tones():
    if "tonos" not in _CACHE:
        _CACHE["tonos"] = _load_tones()
    return _CACHE["tonos"]


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    if sys.argv[1:2] == ["tonos"]:
        print(json.dumps(train_tones(float(sys.argv[2]) if len(sys.argv) > 2 else 30, log=lambda m: print(m, flush=True))))
        raise SystemExit
    print(json.dumps(train(float(sys.argv[1]) if len(sys.argv) > 1 else 40, log=lambda m: print(m, flush=True))))
