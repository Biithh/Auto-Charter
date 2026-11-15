"""Espectrogramas para el modelo de bajo: de las canciones que traen bass.* o rhythm.* (+ restos del resto, como al separar) y de
canciones separadas por la IA, que son las que vera al generar.

Solo lee la biblioteca; todo se guarda en config/entrenamiento/espectros_bajo.

Uso: python -m charter.calibration.bass_spectra pistas <canciones.json> [procesos]
     python -m charter.calibration.bass_spectra ia <canciones.json> <minutos>
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
from .keys_spectra import _disk_ok, _save, _spec
from .stem_spectra import BLEED_DB, _audio_files, _init, _wait_disk

BASS_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_bajo"
DIFFS = ("expert", "hard", "medium", "easy")
MIN_NOTES = 80


def chart_labels(folder: Path) -> dict | None:
    """Notas de bajo del chart en segundos, en las 4 dificultades, con su tamano y su boton en Experto."""
    from ..chart.chart_io import load_chart
    from ..chart.model import OPEN
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS

    chart = load_chart(folder)
    ns = sorted(chart.bass.notes["expert"], key=lambda n: n.tick)
    if len(ns) < MIN_NOTES:
        return None
    ini = read_ini(folder / "song.ini")
    delay = float(ini.get("delay", "0") or 0) / 1000
    sec = chart.tempo.tick_to_sec
    out: dict = {f"t_{d}": np.sort(np.array([sec(n.tick) - delay for n in chart.bass.notes[d]], np.float32)) for d in DIFFS[1:]}
    out["ref_t"] = np.array([sec(n.tick) - delay for n in ns], np.float32)
    out["sizes"] = np.array([len([f for f in n.frets if f != OPEN]) for n in ns], np.int8)
    out["lane"] = np.array([-1 if OPEN in n.frets or not n.frets else min(n.frets) for n in ns], np.int8)
    out["official"] = np.bool_(any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS))
    out["folder"] = str(folder)
    return out


def song_spectrum(folder: Path, key: str) -> dict | None:
    """Bajo original + restos del resto de la cancion, con sus etiquetas."""
    from ..audio import decode_to_wav, load

    from ..chart.chart_io import load_chart

    files = _audio_files(folder)
    # como en Rock Band, el bajo suele venir en rhythm.*: si el chart no tiene guitarra ritmica, rhythm es el bajo
    stem = "bass" if "bass" in files else "rhythm"
    if stem not in files or (stem == "rhythm" and load_chart(folder).rhythm.has_notes()):
        return None
    labels = chart_labels(folder)
    if labels is None:
        return None
    paths.WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.WORK) as td:
        def mono(p: Path) -> np.ndarray:
            return load(decode_to_wav(p, Path(td) / "a.wav"))[0].mean(axis=0)

        y = mono(files[stem])
        rest = None
        for name, p in files.items():
            if name == stem or name.startswith(("crowd", "preview")):
                continue
            a = mono(p)
            n = max(len(a), 0 if rest is None else len(rest))
            rest = np.pad(a, (0, n - len(a))) if rest is None else np.pad(rest, (0, n - len(rest))) + np.pad(a, (0, n - len(a)))
    if rest is not None:
        gain = 10 ** (float(np.random.default_rng(int(key[:8], 16)).uniform(*BLEED_DB)) / 20)
        n = max(len(y), len(rest))
        y = np.pad(y, (0, n - len(y))) + np.pad(rest, (0, n - len(rest))) * gain
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return {**_spec(y), "sep": np.bool_(False), "pista": stem, **labels}


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
        key = _key(f)  # sin disco la clave saldria distinta
        out = BASS_SPEC_DIR / f"{key}_pista.npz"
        none = out.with_suffix(".none")
        if out.exists() or none.exists():
            return "ya"
        try:
            d = song_spectrum(f, key)
            break
        except Exception as e:  # noqa: BLE001
            # disco desconectado un momento (ffmpeg solo dice que no puede abrir el archivo): se espera y se reintenta
            if isinstance(e, OSError) or "ffmpeg no pudo" in str(e) or not _disk_ok(f):
                if attempt == 2:
                    return "disco"
                time.sleep(20)
                if not _disk_ok(f) and not _wait_disk(f):
                    return "disco"
                continue
            none.write_text(str(e)[:200], encoding="utf-8")
            return "error"
    if d is None:
        if not _disk_ok(f):  # sin disco tampoco se ve la pista: no es que no la tenga
            return "disco"
        none.write_text("", encoding="utf-8")
        return "no"
    _save(out, d)
    return "ok"


def extract_all(songs: list[Path], workers: int = 3, log=print) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    BASS_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  {len(songs)} canciones con bajo en la lista")
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
    log(f"  espectrogramas de bajo: {counts}")
    return counts.get("ok", 0)


def separate_all(songs: list[Path], deadline: float, log=print) -> None:
    """Separa canciones con la IA hasta la hora limite y se queda con la pista de bajo."""
    from .library_train import _key, _QuickSeparator

    BASS_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    sep = _QuickSeparator(log)
    done = 0
    t0 = time.time()
    try:
        for f in songs:
            if time.time() > deadline:
                break
            try:
                out = BASS_SPEC_DIR / f"{_key(f)}_ia.npz"
                if out.exists():
                    continue
                labels = chart_labels(f)
                if labels is None:
                    continue
                res = sep.separate(f, insts=("bass",))
                if "bass" not in res:
                    continue
                _save(out, {**_spec(res["bass"]), "sep": np.bool_(True), **labels})
                done += 1
                if done % 10 == 0:
                    log(f"  {done} separadas, {(time.time() - t0) / done:.0f} s cada una")
            except Exception as e:  # noqa: BLE001
                log(f"  {f.name}: {e}")
    finally:
        sep.close()
    log(f"  separacion con IA terminada: {done} canciones")


if __name__ == "__main__":
    _init()
    lista = [Path(x) for x in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))]
    if sys.argv[1] == "pistas":
        extract_all(lista, int(sys.argv[3]) if len(sys.argv) > 3 else 3, log=lambda m: print(m, flush=True))
    elif sys.argv[1] == "ia":
        separate_all(lista, time.time() + float(sys.argv[3]) * 60, log=lambda m: print(m, flush=True))
