"""Modelo de botones por frases: mira hasta 192 notas seguidas de la guitarra (lo que oye el modelo de notas en
cada una, el espectro del ataque, el tamano del acorde y la separacion entre notas) y dice, para cada nota, el boton
mas bajo y si sube, baja o repite respecto a la anterior. Sustituye a lo que dice el modelo de notas nota a nota
(nn_lane / nn_move) antes de asignar los botones. Se entrena aparte (scratchpad train_btn.py) con las notas de
los charts de la biblioteca."""

from __future__ import annotations

import numpy as np

from .. import paths

MODEL_PATH = paths.CONFIG_DIR / "botones_frases.pt"
L = 192
N_IN = 96 + 3 + 2
_CACHE: dict = {}


def note_features(x: np.ndarray, probs, frames) -> np.ndarray:
    """(N, 96): espectro en el ataque (max de 5 tramas) + movimiento, boton y acorde del modelo de notas.
    x: entradas del modelo de notas (85, T) uint8; probs: (on, acorde, movimiento, boton) por trama."""
    _, ch, mv, ln = probs
    n = x.shape[1]
    out = np.zeros((len(frames), 96), np.float32)
    for k, f in enumerate(frames):
        f = int(np.clip(f, 0, n - 1))
        out[k, :85] = x[:, max(f - 1, 0):f + 4].max(1) / 255.0
        out[k, 85:88] = mv[min(f, len(mv) - 1)]
        out[k, 88:93] = ln[min(f, len(ln) - 1)]
        out[k, 93:96] = ch[min(f, len(ch) - 1)]
    return out


def build(t, f, size) -> np.ndarray:
    n = len(t)
    x = np.zeros((n, N_IN), np.float32)
    x[:, :96] = f
    x[np.arange(n), 96 + np.clip(np.asarray(size), 1, 3) - 1] = 1
    x[:, 99] = np.log(np.clip(np.diff(t, prepend=t[0] - 2.0), 0.01, 4.0))
    x[:, 100] = np.log(np.clip(np.diff(t, append=t[-1] + 2.0), 0.01, 4.0))
    return x


def _net():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, d=256, layers=4):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(N_IN, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Embedding(2048, d)
            enc = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True,
                                             activation="gelu")
            self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.lane = nn.Linear(d, 5)
            self.move = nn.Linear(d, 3)
            self.register_buffer("mask", torch.ones(N_IN))  # entradas que usa (lo guarda el entrenamiento)

        def forward(self, x, pad=None):
            h = self.inp(x * self.mask) + self.pos(torch.arange(x.shape[1], device=x.device) % 2048)[None]
            h = self.enc(h, src_key_padding_mask=pad)
            return self.lane(h), self.move(h)

    return Net()


def load_model():
    if "m" not in _CACHE:
        _CACHE["m"] = None
        if MODEL_PATH.exists():
            try:
                import torch

                ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                net = _net()
                net.load_state_dict(ck["state"], strict=False)  # los primeros no guardaban "mask"
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _CACHE["m"] = net.to(dev).eval()
            except Exception:  # noqa: BLE001
                _CACHE["m"] = None
    return _CACHE["m"]


def predict(net, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """probabilidades de boton (N, 5) y movimiento (N, 3) de toda la cancion, en trozos de L con solape"""
    import torch

    dev = next(net.parameters()).device
    n = len(x)
    pl, pm, c = np.zeros((n, 5)), np.zeros((n, 3)), np.zeros(n)
    s = 0
    with torch.no_grad():
        while n:
            e = min(s + L, n)
            a, b = net(torch.from_numpy(x[s:e][None]).to(dev))
            pl[s:e] += torch.softmax(a, -1)[0].float().cpu().numpy()
            pm[s:e] += torch.softmax(b, -1)[0].float().cpu().numpy()
            c[s:e] += 1
            if e == n:
                break
            s += L // 2
    return pl / np.maximum(c, 1)[:, None], pm / np.maximum(c, 1)[:, None]
