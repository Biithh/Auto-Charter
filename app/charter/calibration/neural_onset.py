"""Modelo neuronal de notas: mira el espectrograma de la guitarra separada y dice en que momentos pone nota
un charter (y de cuantas notas es el acorde). Se entrena con los espectrogramas que guarda el entrenamiento
con la biblioteca (config/entrenamiento/espectros) y las notas de sus charts.

Uso: python -m charter.calibration.neural_onset [minutos]   (por defecto 40; usa la GPU)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths

SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros"
# guitarras originales de la biblioteca con restos del resto de la cancion (ver stem_spectra.py): muchas mas
# canciones; la separada por la IA pesa mas al elegir ejemplos y es la unica que se usa para validar
STEM_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_pistas"
SEP_WEIGHT = 2.0
LABEL_DIR = paths.CONFIG_DIR / "entrenamiento" / "audio_v2"
MODEL_PATH = paths.CONFIG_DIR / "notas_neuronal.pt"
# el modelo del entrenamiento anterior: al generar se promedian los dos (cada entrenamiento sale un poco distinto
# y juntos aciertan mas que cada uno por separado)
PREV_PATH = paths.CONFIG_DIR / "notas_neuronal_anterior.pt"
PREV2_PATH = paths.CONFIG_DIR / "notas_neuronal_anterior2.pt"  # y el de antes de ese
FPS = 22050 / 256 / 2
N_IN = 85  # 84 notas de CQT + envolvente de ataques
CROP = 640  # ~15 s por ejemplo
SPEC_AUG = True
MIX_AUG = False  # aumentos de "master comercial" (_mix_aug); 26-11 en prueba
WEIGHT_DECAY = 1e-2
MIN_DIST = 2  # tramas minimas entre dos notas (~46 ms)
# las notas muy seguidas son las que mas se escapan ("pa-pa-paaam" se quedaba en "pa-paaam"): pesan mas al entrenar
FAST_GAP = 0.2  # segundos hasta la nota vecina
FAST_BOOST = 1.0  # peso extra de las notas rapidas al entrenar; con 3.0 la guitarra y el teclado salieron peores
DIFFS = ("expert", "hard", "medium", "easy")  # una salida por dificultad: que notas pone el charter en cada una
DIFF_DIR = paths.CONFIG_DIR / "entrenamiento" / "etiquetas_dificultad"
BTN_DIR = paths.CONFIG_DIR / "entrenamiento" / "etiquetas_botones"
MOVES = ("baja", "igual", "sube")  # boton de cada nota de Experto respecto a la anterior


def _net(n_on: int = len(DIFFS), width: float = 1.0):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                     nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())

            # convoluciones en (frecuencia, tiempo): reducen solo la frecuencia, el tiempo queda a ~23 ms
            c1, c2, c3 = int(24 * width), int(48 * width), int(64 * width)
            hid = int(128 * width)
            self.conv = nn.Sequential(block(1, c1), nn.MaxPool2d((2, 1)), block(c1, c2), nn.MaxPool2d((2, 1)),
                                      block(c2, c3), nn.MaxPool2d((3, 1)), nn.Dropout(0.15))
            self.env = nn.Conv1d(1, 16, 5, padding=2)
            self.proj = nn.Linear(c3 * 7 + 16, int(192 * width))
            self.rnn = nn.GRU(int(192 * width), hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            self.onset = nn.Linear(2 * hid, n_on)
            self.chord = nn.Linear(2 * hid, 3)
            # botones: si la nota sube, baja o repite boton respecto a la anterior, y en que boton cae
            self.move = nn.Linear(2 * hid, 3)
            self.lane = nn.Linear(2 * hid, 5)

        def forward(self, x):  # x: (B, 85, T) en 0..1
            c = self.conv(x[:, None, :84])  # (B, 64, 7, T)
            b, ch, f, t = c.shape
            c = c.reshape(b, ch * f, t).transpose(1, 2)
            e = torch.relu(self.env(x[:, 84:85])).transpose(1, 2)
            h, _ = self.rnn(torch.relu(self.proj(torch.cat([c, e], 2))))
            return self.onset(h), self.chord(h), self.move(h), self.lane(h)

    return Net()


def _mix_aug(x: np.ndarray, tr: list, rng) -> None:
    """Como suena la guitarra separada de un master comercial (y no de las pistas del juego): se cuela algo de
    otra cancion, otra dinamica y otra ecualizacion. CQT en 0..1 = -100..0 dB. Las guitarras separadas de Arde
    Bogota no tienen mas ruido que las del banco (percentil 50: 0,63-0,75 frente a 0,73-0,76), asi que sin ruido."""
    q = x[:84]
    if rng.random() < 0.5:  # fuga de otros instrumentos: otra pista 10-25 dB por debajo (en dB, sumar ~ maximo)
        o = tr[int(rng.integers(len(tr)))]["x"]
        s = int(rng.integers(0, o.shape[1] - CROP + 1))
        np.maximum(q, o[:84, s:s + CROP].astype(np.float32) / 255.0 - rng.uniform(0.10, 0.25), out=q)
    if rng.random() < 0.5:  # mas o menos diferencia entre lo fuerte y lo flojo
        m = float(q.max())
        q[:] = m - (m - q) * rng.uniform(0.85, 1.15)
    if rng.random() < 0.5:  # ecualizacion suave cualquiera (no solo inclinada)
        q += np.interp(np.arange(84), np.linspace(0, 83, 6), rng.normal(0, 0.06, 6))[:, None]
    np.clip(q, 0, 1, out=q)


def _inputs(cqt: np.ndarray, onset: np.ndarray) -> np.ndarray:
    n = min(cqt.shape[1], len(onset))
    return np.concatenate([cqt[:, :n], onset[None, :n]], 0).astype(np.uint8)


def _targets(n: int, refs: list[np.ndarray], sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Objetivo por trama: (n, dificultades) con 1 donde hay nota (0,25 en las tramas vecinas) y el tamano
    del acorde de Experto (-1 donde no hay nota)."""
    on = np.zeros((n, len(DIFFS)), np.float32)
    ch = np.full(n, -1, np.int64)
    for k, ref_t in enumerate(refs):
        for j, t in enumerate(ref_t):
            f = int(round(t * FPS))
            if 0 <= f < n:
                on[f, k] = 1.0
                if k == 0:
                    ch[f] = int(np.clip(sizes[j], 1, 3)) - 1
                for g in (f - 1, f + 1):
                    if 0 <= g < n:
                        on[g, k] = max(on[g, k], 0.25)
    return on, ch


def _fast_frames(n: int, ref_t: np.ndarray) -> np.ndarray:
    """1 en las tramas de las notas que tienen otra muy cerca (se escapan mas: ver FAST_GAP)."""
    t = np.sort(np.asarray(ref_t, np.float64))
    fast = np.zeros(n, np.uint8)
    if len(t) < 2:
        return fast
    gaps = np.diff(t)
    near = np.r_[False, gaps < FAST_GAP] | np.r_[gaps < FAST_GAP, False]
    f = np.round(t[near] * FPS).astype(int)
    fast[f[(f >= 0) & (f < n)]] = 1
    return fast


def _button_targets(n: int, times: np.ndarray, lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Objetivo de botones por trama: movimiento respecto a la nota anterior y boton mas bajo (-1 = nada)."""
    mv = np.full(n, -1, np.int64)
    ln = np.full(n, -1, np.int64)
    for j, (t, l) in enumerate(zip(times, lanes)):
        f = int(round(t * FPS))
        if not 0 <= f < n or l < 0:
            continue
        ln[f] = int(l)
        if j > 0 and lanes[j - 1] >= 0 and times[j] - times[j - 1] < 2.0:
            mv[f] = int(np.sign(l - lanes[j - 1])) + 1
    return mv, ln


def _button_labels(folder: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Tiempo y boton mas bajo de cada nota de Experto del chart (-1 si es abierta)."""
    cache = BTN_DIR / name
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["lane"]
    from ..chart.chart_io import load_chart
    from ..chart.model import OPEN
    from ..songini import read_ini

    try:
        chart = load_chart(folder)
        delay = float(read_ini(folder / "song.ini").get("delay", "0") or 0) / 1000
    except Exception:  # noqa: BLE001
        return None
    ns = sorted(chart.guitar.notes["expert"], key=lambda n: n.tick)
    t = np.array([chart.tempo.tick_to_sec(n.tick) - delay for n in ns], np.float64)
    lane = np.array([-1 if OPEN in n.frets or not n.frets else min(n.frets) for n in ns], np.int64)
    BTN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t, lane=lane)
    return t, lane


def _diff_times(folder: Path, name: str) -> list[np.ndarray] | None:
    """Tiempos de las notas de Dificil, Medio y Facil del chart (se guardan para no releer la biblioteca)."""
    cache = DIFF_DIR / name
    if cache.exists():
        d = np.load(cache)
        return [d[k] for k in DIFFS[1:]]
    from ..chart.chart_io import load_chart
    from ..songini import read_ini

    try:
        chart = load_chart(folder)
        delay = float(read_ini(folder / "song.ini").get("delay", "0") or 0) / 1000
    except Exception:  # noqa: BLE001
        return None
    out = [np.sort(np.array([chart.tempo.tick_to_sec(n.tick) - delay for n in chart.guitar.notes[d]], np.float64))
           for d in DIFFS[1:]]
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k: v for k, v in zip(DIFFS[1:], out)})
    return out


def _is_val(folder: str) -> bool:
    return int(hashlib.sha1(folder.encode("utf-8")).hexdigest(), 16) % 10 == 0


def _title(folder: Path) -> str:
    from ..songini import read_ini

    try:
        name = read_ini(folder / "song.ini").get("name", "") or folder.name
    except Exception:  # noqa: BLE001
        name = folder.name
    name = name.split(" - ")[-1] if " - " in name and name == folder.name else name
    return "".join(ch for ch in name.lower().split("(")[0] if ch.isalnum())


def _holdout_titles() -> set[str]:
    """Titulos de las canciones del banco de pruebas: la misma cancion en otra carpeta (otro pack) tampoco
    se usa para entrenar, para que la nota del banco sea limpia."""
    from .library_train import HOLDOUT_FILE

    if not HOLDOUT_FILE.exists():
        return set()
    return {_title(Path(x.strip())) for x in HOLDOUT_FILE.read_text(encoding="utf-8").splitlines() if x.strip()}


def load_dataset(log=print) -> tuple[list, list]:
    train, val = [], []
    hold = _holdout_titles()
    skipped = 0
    files = sorted(SPEC_DIR.glob("*.npz")) + sorted(STEM_SPEC_DIR.glob("*_guitar.npz"))
    for p in files:
        stem = p.parent == STEM_SPEC_DIR
        lab = LABEL_DIR / p.name
        if not stem and not lab.exists():
            continue
        try:
            d = np.load(p)
            a = d if stem else np.load(lab)
            x = _inputs(d["cqt"], d["onset"])
            folder = str(d["folder"])
            if _title(Path(folder)) in hold:
                skipped += 1
                continue
            lower = _diff_times(Path(folder), _label_name(p))
            if lower is None:
                continue
            refs = [a["ref_t"].astype(np.float64)] + lower
            on, ch = _targets(x.shape[1], refs, a["sizes"])
            btn = _button_labels(Path(folder), _label_name(p))
            if btn is None:
                continue
            mv, ln = _button_targets(x.shape[1], *btn)
        except Exception:  # noqa: BLE001
            continue
        if x.shape[1] < CROP or on[:, 0].sum() < 50:
            continue
        item = {"x": x, "on": on.astype(np.float16), "ch": ch.astype(np.int8), "mv": mv.astype(np.int8),
                "ln": ln.astype(np.int8), "fast": _fast_frames(x.shape[1], refs[0]), "refs": refs, "name": folder,
                "sep": not stem}
        if not _is_val(folder):
            train.append(item)
        elif not stem:
            val.append(item)  # se valida con guitarras separadas por la IA, como las que vera al generar
    n_sep = sum(it["sep"] for it in train)
    log(f"  {len(train)} canciones para entrenar ({n_sep} separadas por la IA, {len(train) - n_sep} con la pista "
        f"original), {len(val)} para validar ({skipped} apartadas: estan en el banco)")
    return train, val


def _label_name(p: Path) -> str:
    """Nombre de las etiquetas en cache (las de pistas originales, aparte de las separadas por la IA)."""
    return ("pista_" + p.name) if p.parent == STEM_SPEC_DIR else p.name


def peaks(prob: np.ndarray, thr: float) -> np.ndarray:
    """Tramas con nota: maximos locales por encima del umbral, separados al menos MIN_DIST tramas."""
    p = np.asarray(prob)
    cand = np.where((p >= thr) & (p >= np.r_[0, p[:-1]]) & (p > np.r_[p[1:], 0]))[0]
    keep: list[int] = []
    for f in cand[np.argsort(-p[cand])]:
        if all(abs(f - k) >= MIN_DIST for k in keep):
            keep.append(int(f))
    return np.array(sorted(keep), int)


def peak_times(prob: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Tiempo de cada pico con interpolacion parabolica (precision por debajo de una trama)."""
    out = []
    for f in frames:
        a, b, c = (prob[f - 1] if f > 0 else prob[f]), prob[f], (prob[f + 1] if f + 1 < len(prob) else prob[f])
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        out.append((f + float(np.clip(off, -0.5, 0.5))) / FPS)
    return np.array(out)


def predict(model, x: np.ndarray, device=None) -> tuple[np.ndarray, ...]:
    """Por trama: probabilidad de nota (n, dificultades), de acorde (1, 2, 3+ notas), de movimiento de boton
    (baja, igual, sube) y de boton (verde..naranja)."""
    import torch

    device = device or next(model.parameters()).device
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    model.eval()
    on_all, ch_all = [], []
    win, hop = 4096, 3584  # trozos largos con solape para no cortar el contexto
    n = x.shape[1]
    with torch.no_grad():
        acc_on = None
        acc_ch = np.zeros((n, 3), np.float32)
        acc_mv = np.zeros((n, 3), np.float32)
        acc_ln = np.zeros((n, 5), np.float32)
        cnt = np.zeros(n, np.float32)
        s = 0
        while True:
            e = min(s + win, n)
            xb = torch.from_numpy(x[:, s:e][None]).to(device)
            o, c, mv, ln = model(xb)
            po = torch.sigmoid(o)[0].float().cpu().numpy()
            if acc_on is None:
                acc_on = np.zeros((n, po.shape[1]), np.float32)
            acc_on[s:e] += po
            acc_ch[s:e] += torch.softmax(c, -1)[0].float().cpu().numpy()
            acc_mv[s:e] += torch.softmax(mv, -1)[0].float().cpu().numpy()
            acc_ln[s:e] += torch.softmax(ln, -1)[0].float().cpu().numpy()
            cnt[s:e] += 1
            if e == n:
                break
            s += hop
    c = cnt[:, None]
    return acc_on / c, acc_ch / c, acc_mv / c, acc_ln / c


def embed(model, x: np.ndarray) -> np.ndarray:
    """Lo que 'oye' el modelo en cada trama (su estado interno, 256 valores): dos trozos que suenan igual para
    el modelo daran las mismas notas. Sirve para encontrar las partes repetidas de la cancion."""
    import torch

    device = next(model.parameters()).device
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    model.eval()
    got = []
    hook = model.rnn.register_forward_hook(lambda mod, inp, out: got.append(out[0][0].float().cpu().numpy()))
    win, hop = 4096, 3584
    n = x.shape[1]
    acc, cnt = None, np.zeros(n, np.float32)
    try:
        with torch.no_grad():
            s = 0
            while True:
                e = min(s + win, n)
                got.clear()
                model(torch.from_numpy(x[:, s:e][None]).to(device))
                if acc is None:
                    acc = np.zeros((n, got[0].shape[1]), np.float32)
                acc[s:e] += got[0]
                cnt[s:e] += 1
                if e == n:
                    break
                s += hop
    finally:
        hook.remove()
    return acc / cnt[:, None]


def _val_moves(model, val) -> float:
    """Aciertos del movimiento de boton (sube / baja / igual) en las notas de validacion."""
    hit = []
    for it in val:
        mv = predict(model, it["x"])[2]
        idx = np.where(it["mv"] >= 0)[0]
        hit.extend(np.argmax(mv[idx], 1) == it["mv"][idx])
    return float(np.mean(hit)) if hit else 0.0


def _val_f1(model, val, thrs, k: int = 0) -> dict:
    from .eval_full import onset_f1

    res = {t: [] for t in thrs}
    for it in val:
        if len(it["refs"][k]) == 0:
            continue
        prob = predict(model, it["x"])[0][:, k]
        for t in thrs:
            res[t].append(onset_f1(it["refs"][k], peak_times(prob, peaks(prob, t)))[0])
    return {t: float(np.mean(v)) for t, v in res.items()}


def train(minutes: float = 40, log=print, width: float = 1.0, batch_size: int = 24, eval_every: int = 400,
          seed: int | None = None, init: Path | None = None, lr: float = 1e-3, stop_file: Path | None = None,
          checkpoint: Path | None = None, inst=None) -> dict:
    """`init`: sigue entrenando ese modelo (con `lr` mas bajo) en vez de empezar de cero.
    `stop_file`: si aparece ese archivo se termina ya y se guarda lo mejor conseguido hasta entonces.
    `checkpoint`: copia del mejor modelo durante el entrenamiento, por si se va la luz o se cierra el proceso.
    `inst`: otro instrumento con esta misma red (el modulo que da load_dataset, load_model y las rutas de sus modelos)."""
    import torch
    from torch import nn

    if seed is None:  # cada entrenamiento distinto: el modelo nuevo y el anterior se equivocan en cosas distintas
        seed = int(time.time()) % 10000
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, val = (inst.load_dataset if inst else load_dataset)(log)
    model_path, prev_path, prev2_path = (inst.MODEL_PATH, inst.PREV_PATH, inst.PREV2_PATH) if inst else (MODEL_PATH, PREV_PATH, PREV2_PATH)
    if len(tr) < 50:
        raise SystemExit("Hacen falta mas espectrogramas: entrena antes con la biblioteca y separacion con IA.")
    model = _net(width=width).to(dev)
    if init is not None:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["state"])
        log(f"  se sigue entrenando {init.name}")
    log(f"  modelo de ancho {width}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parametros")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    lens = np.array([it["x"].shape[1] for it in tr], np.float64)
    pick_p = lens * np.array([SEP_WEIGHT if it["sep"] else 1.0 for it in tr])
    pick_p = pick_p / pick_p.sum()
    deadline = time.time() + minutes * 60
    total_steps = None
    step, best, best_state = 0, -1.0, None
    t0 = time.time()
    thrs = [0.3, 0.4, 0.5, 0.6]
    while time.time() < deadline and not (stop_file is not None and stop_file.exists()):
        batch = rng.choice(len(tr), batch_size, p=pick_p)
        xs, ons, chs, mvs, lns, fss = [], [], [], [], [], []
        for i in batch:
            it = tr[i]
            s = int(rng.integers(0, it["x"].shape[1] - CROP + 1))
            x = it["x"][:, s:s + CROP].astype(np.float32) / 255.0
            # aumentos: volumen distinto y desplazamiento de tono de hasta 2 semitonos
            x[:84] = np.clip(x[:84] + rng.normal(0, 0.04), 0, 1)
            k = int(rng.integers(-2, 3))
            if k:
                x[:84] = np.roll(x[:84], k, 0)
                if k > 0:
                    x[:k] = 0
                else:
                    x[84 + k:84] = 0
            if SPEC_AUG:
                # otro sonido de guitarra (graves/agudos inclinados) y bandas de frecuencia tapadas: asi no se
                # aprende de memoria el sonido de cada cancion
                x[:84] = np.clip(x[:84] + np.linspace(-1, 1, 84)[:, None] * rng.normal(0, 0.05), 0, 1)
                for _ in range(int(rng.integers(0, 3))):
                    w = int(rng.integers(2, 9))
                    f0 = int(rng.integers(0, 84 - w))
                    x[f0:f0 + w] = 0
            if MIX_AUG:
                _mix_aug(x, tr, rng)
            xs.append(x)
            ons.append(it["on"][s:s + CROP].astype(np.float32))
            chs.append(it["ch"][s:s + CROP].astype(np.int64))
            mvs.append(it["mv"][s:s + CROP].astype(np.int64))
            lns.append(it["ln"][s:s + CROP].astype(np.int64))
            fss.append(it["fast"][s:s + CROP].astype(np.float32))
        xb = torch.from_numpy(np.stack(xs)).to(dev)
        ob = torch.from_numpy(np.stack(ons)).to(dev)
        cb = torch.from_numpy(np.stack(chs)).to(dev)
        mb = torch.from_numpy(np.stack(mvs)).to(dev)
        lb = torch.from_numpy(np.stack(lns)).to(dev)
        frac = min((time.time() - t0) / (minutes * 60), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + np.cos(np.pi * frac))) + 2e-5
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o, c, mv, ln = model(xb)
        o, c, mv, ln = o.float(), c.float(), mv.float(), ln.float()
        w = 1 + 4 * (ob >= 0.99).float()  # las notas son pocas frente a las tramas sin nota
        w[..., 0] = w[..., 0] * (1 + (FAST_BOOST - 1) * torch.from_numpy(np.stack(fss)).to(dev))
        loss = (bce(o, ob) * w * torch.tensor([1.0, 0.6, 0.6, 0.6], device=dev)).mean()
        if (cb >= 0).any():  # sin ninguna nota en el lote, la perdida de acordes no existe (daria NaN)
            loss = loss + 0.3 * ce(c.reshape(-1, 3), cb.reshape(-1))
        if (mb >= 0).any():
            loss = loss + 0.3 * ce(mv.reshape(-1, 3), mb.reshape(-1))
        if (lb >= 0).any():
            loss = loss + 0.15 * ce(ln.reshape(-1, 5), lb.reshape(-1))
        opt.zero_grad()
        step += 1
        if not torch.isfinite(loss):
            continue
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if step % eval_every == 0:
            f1 = _val_f1(model, val, thrs)
            bt = max(f1, key=f1.get)
            log(f"  paso {step}: perdida {loss.item():.3f} | validacion F1 {f1[bt]:.3f} (umbral {bt}) | "
                f"{(deadline - time.time()) / 60:.0f} min restantes")
            if f1[bt] > best:
                best, best_state = f1[bt], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint is not None:
                    torch.save({"state": best_state, "info": {"val_f1": best, "pasos": step, "ancho": width}}, checkpoint)
            model.train()
    if best_state is None:  # parado antes de la primera medicion: no hay nada que guardar
        log("Modelo neuronal: parado antes de poder medirlo, no se guarda")
        return load_model()[1] if load_model() else {}
    model.load_state_dict(best_state)
    fine = [round(x, 2) for x in np.arange(0.2, 0.71, 0.05)]
    thr_d, f1_d = {}, {}
    for k, d in enumerate(DIFFS):
        f1 = _val_f1(model, val, fine, k)
        thr_d[d] = float(max(f1, key=f1.get))
        f1_d[d] = round(f1[thr_d[d]], 3)
    moves = round(_val_moves(model, val), 3)
    info = {"thr": thr_d["expert"], "thr_diff": thr_d, "val_f1": f1_d["expert"], "val_f1_diff": f1_d,
            "botones": moves, "canciones": len(tr), "pasos": step, "ancho": width}
    log(f"  botones: acierta si sube, baja o repite en el {moves:.0%} de las notas de validacion")
    # el modelo anterior se mide con las mismas canciones de validacion: si era mejor, se queda el anterior
    old = (inst.load_model if inst else load_model)()
    if old is not None and "botones" in old[1]:
        old_f1 = _val_f1(old[0], val, [old[1]["thr"]])[old[1]["thr"]]
        if old_f1 > f1_d["expert"] + 0.002:
            log(f"Modelo neuronal: el anterior sigue siendo mejor ({old_f1:.3f} frente a {f1_d['expert']:.3f}), se queda")
            return old[1]
    _CACHE.clear()
    if inst:
        inst._CACHE.clear()
    if model_path.exists():
        import shutil

        if prev_path.exists():
            shutil.copyfile(prev_path, prev2_path)
        shutil.copyfile(model_path, prev_path)
    torch.save({"state": model.state_dict(), "info": info}, model_path)
    log(f"Modelo neuronal guardado: validacion F1 {f1_d} con umbrales {thr_d} ({len(tr)} canciones)")
    return info


_CACHE: dict = {}


def predict_all(nets: list, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Media de las predicciones de varios modelos (los botones, solo de los que los saben)."""
    outs = [(predict(m, x), "botones" in info) for m, info in nets]
    n = min(o[0][0].shape[0] for o in outs)
    mean = [np.mean([o[0][k][:n] for o in outs], 0) for k in (0, 1)]
    btn = [o[0] for o in outs if o[1]] or [o[0] for o in outs]
    mean += [np.mean([o[k][:n] for o in btn], 0) for k in (2, 3)]
    return tuple(mean)


def load_models() -> list:
    """El modelo actual y, si existe, el del entrenamiento anterior (con la misma forma de salida)."""
    if "all" in _CACHE:
        return _CACHE["all"]
    out = []
    main = load_model()
    if main is not None:
        out.append(main)
        for path, use in ((PREV_PATH, USE_PREV), (PREV2_PATH, USE_PREV and USE_PREV2)):
            prev = _load(path) if use else None
            # solo si es del mismo tamano: con uno mas pequeno (y peor) la media sale peor que el actual solo
            if prev is not None and prev[1].get("ancho", 1.0) == main[1].get("ancho", 1.0):
                out.append(prev)
    _CACHE["all"] = out
    return out


# modelos grandes juntos en el banco de pruebas: uno solo Experto 0.894, dos 0.895, tres 0.900 (Medio 0.774/0.780/0.782)
USE_PREV = True
USE_PREV2 = True


def load_model():
    """Modelo entrenado (o None si no hay). Se carga una vez."""
    if "m" not in _CACHE:
        _CACHE["m"] = _load(MODEL_PATH)
    return _CACHE["m"]


def _load(path: Path):
    m = None
    if path.exists():
        try:
            import torch

            ck = torch.load(path, map_location="cpu", weights_only=False)
            net = _net(int(ck["state"]["onset.weight"].shape[0]), float(ck["info"].get("ancho", 1.0)))
            net.load_state_dict(ck["state"], strict=False)  # modelos antiguos: sin la parte de botones
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = (net.to(dev).eval(), ck["info"])
        except Exception:  # noqa: BLE001
            m = None
    return m


if __name__ == "__main__":
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    print(json.dumps(train(float(sys.argv[1]) if len(sys.argv) > 1 else 40, log=lambda m: print(m, flush=True),
                           width=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)))
