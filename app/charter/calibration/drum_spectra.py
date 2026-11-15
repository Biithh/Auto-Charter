"""Espectrogramas para el modelo de bateria: de las canciones que traen drums*.* (+ restos del resto, como al separar)
y de canciones separadas por la IA, que son las que vera al generar.

Para la bateria se usa un espectrograma mel de 84 bandas (30 Hz - 16 kHz) en lugar del CQT: los platillos viven
muy por encima de la nota mas aguda del CQT. Se guarda con la clave "cqt" para reutilizar la misma red.

Solo lee la biblioteca; todo se guarda en config/entrenamiento/espectros_bateria.

Uso: python -m charter.calibration.drum_spectra pistas <canciones.json> [procesos]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import paths
from .keys_spectra import _disk_ok, _save
from .stem_spectra import BLEED_DB, _audio_files, _init, _wait_disk

DRUM_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_bateria"
MIN_NOTES = 150
# salidas del modelo: (pad, platillo). pad 0 = bombo, 1 = caja, 2/3/4 = amarillo/azul/verde
CLASSES = ((0, False), (1, False), (2, False), (2, True), (3, False), (3, True), (4, False), (4, True))
SR = 44100
HOP = 1024  # 43,07 tramas por segundo, como el resto de modelos


def drum_spectrum(y: np.ndarray, sr: int = SR) -> dict:
    """Mel de 84 bandas en 0-255 + envolvente de ataques, con la misma forma que compact_spectrum."""
    import librosa

    y = np.asarray(y, np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=84, fmin=30, fmax=16000)
    db = librosa.power_to_db(m, ref=np.max(m) + 1e-12)
    q = np.clip((db + 80.0) * (255 / 80), 0, 255).astype(np.uint8)
    env = librosa.onset.onset_strength(S=db, sr=SR, hop_length=HOP)
    e = np.clip(env / (np.percentile(env, 99.5) + 1e-9) * 200, 0, 255).astype(np.uint8)
    n = min(q.shape[1], len(e))
    return {"cqt": q[:, :n], "onset": e[:n]}


def chart_labels(folder: Path) -> dict | None:
    """Golpes de Experto en segundos, por clase (bombo, caja, amarillo/azul/verde tom o platillo)."""
    from ..chart.chart_io import load_chart
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS

    chart = load_chart(folder)
    ns = chart.drums.notes["expert"]
    if len(ns) < MIN_NOTES:
        return None
    ini = read_ini(folder / "song.ini")
    pro = str(ini.get("pro_drums", "")).strip().lower() in ("true", "1")
    delay = float(ini.get("delay", "0") or 0) / 1000
    sec = chart.tempo.tick_to_sec
    out: dict = {}
    for k, (pad, cym) in enumerate(CLASSES):
        out[f"c{k}"] = np.sort(np.array([sec(n.tick) - delay for n in ns
                                         if n.pad == pad and (pad < 2 or bool(n.cymbal) == cym)], np.float32))
    out["pro"] = np.bool_(pro)
    out["official"] = np.bool_(any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS))
    out["folder"] = str(folder)
    return out


def song_spectrum(folder: Path, key: str) -> dict | None:
    """Bateria original (todas las pistas drums*) + restos del resto de la cancion, con sus etiquetas."""
    from ..audio import decode_to_wav, load

    files = _audio_files(folder)
    drum_files = {n: p for n, p in files.items() if n.startswith("drums")}
    if not drum_files:
        return None
    labels = chart_labels(folder)
    if labels is None:
        return None
    paths.WORK.mkdir(parents=True, exist_ok=True)

    def add(a: np.ndarray | None, b: np.ndarray) -> np.ndarray:
        if a is None:
            return b
        n = max(len(a), len(b))
        return np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b)))

    with tempfile.TemporaryDirectory(dir=paths.WORK) as td:
        def mono(p: Path) -> tuple[np.ndarray, int]:
            a, sr = load(decode_to_wav(p, Path(td) / "a.wav"))
            return a.mean(axis=0), sr

        y = rest = None
        sr = SR
        for name, p in files.items():
            if name.startswith(("crowd", "preview")):
                continue
            a, sr = mono(p)
            if name in drum_files:
                y = add(y, a)
            else:
                rest = add(rest, a)
    if rest is not None:
        gain = 10 ** (float(np.random.default_rng(int(key[:8], 16)).uniform(*BLEED_DB)) / 20)
        y = add(y, rest * gain)
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return {**drum_spectrum(y, sr), "sep": np.bool_(False), **labels}


def _work(folder: str) -> str:
    try:
        return _work_inner(Path(folder))
    except OSError:
        return "disco"


def _work_inner(f: Path) -> str:
    from .library_train import _key

    for attempt in range(3):
        if not _disk_ok(f) and not _wait_disk(f):
            return "disco"
        key = _key(f)
        out = DRUM_SPEC_DIR / f"{key}_pista.npz"
        none = out.with_suffix(".none")
        if out.exists() or none.exists():
            return "ya"
        try:
            d = song_spectrum(f, key)
            break
        except Exception as e:  # noqa: BLE001
            if isinstance(e, OSError) or "ffmpeg no pudo" in str(e) or not _disk_ok(f):
                if attempt == 2:
                    return "disco"
                time.sleep(20)
                continue
            none.write_text(str(e)[:200], encoding="utf-8")
            return "error"
    if d is None:
        if not _disk_ok(f):
            return "disco"
        none.write_text("", encoding="utf-8")
        return "no"
    _save(out, d)
    return "ok"


def extract_all(songs: list[Path], workers: int = 2, log=print, deadline: float | None = None) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    DRUM_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  {len(songs)} canciones con bateria en la lista")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(_work, str(f)) for f in songs]
        for k, fu in enumerate(as_completed(futs)):
            try:
                r = fu.result()
            except OSError:
                r = "disco"
            counts[r] = counts.get(r, 0) + 1
            if (k + 1) % 100 == 0:
                rate = (time.time() - t0) / (k + 1)
                log(f"  {k + 1}/{len(songs)} ({counts}) ~{rate * (len(songs) - k - 1) / 60:.0f} min restantes")
            if deadline and time.time() > deadline:
                for fu2 in futs:
                    fu2.cancel()
                break
    log(f"  espectrogramas de bateria: {counts}")
    return counts.get("ok", 0)


if __name__ == "__main__":
    _init()
    lista = [Path(x) for x in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))]
    if sys.argv[1] == "pistas":
        dl = time.time() + float(sys.argv[4]) * 60 if len(sys.argv) > 4 else None
        extract_all(lista, int(sys.argv[3]) if len(sys.argv) > 3 else 2, log=lambda m: print(m, flush=True), deadline=dl)
