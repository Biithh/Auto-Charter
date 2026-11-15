"""Espectrogramas para el modelo neuronal a partir de las canciones que traen la pista de guitarra original.

Separar una cancion con la IA cuesta ~40 s de GPU; muchas canciones de la biblioteca ya traen guitar.ogg, y de
ellas sale un espectrograma en unos segundos de CPU. Para que se parezca a lo que vera el modelo al generar
(una guitarra separada por la IA, con restos de los demas instrumentos), a la guitarra se le suma el resto de la
cancion muy bajo (entre -30 y -12 dB, al azar en cada cancion). Si hay rhythm.ogg ademas de bass.ogg, es una
segunda guitarra y va entera: la IA la deja en la misma pista.

Solo lee la biblioteca; todo se guarda en config/entrenamiento/espectros_pistas.

Uso: python -m charter.calibration.stem_spectra <carpeta_biblioteca> [procesos]
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import paths

STEM_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_pistas"
AUDIO_EXT = (".ogg", ".opus", ".mp3", ".wav", ".flac")
BLEED_DB = (-30.0, -12.0)


def _audio_files(folder: Path) -> dict[str, Path]:
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            out.setdefault(p.stem.lower(), p)
    return out


def song_spectrum(folder: Path, key: str) -> dict | None:
    """Espectrograma compacto de la guitarra original (+ restos del resto de la cancion) y sus etiquetas."""
    from ..audio import decode_to_wav, load
    from ..chart.chart_io import load_chart
    from ..chart.model import OPEN
    from ..pipeline.transcription import analyze_stem
    from ..songini import read_ini
    from .onset_model import SPEC_FPS, compact_spectrum
    from .reference import OFFICIAL_CHARTERS

    files = _audio_files(folder)
    if "guitar" not in files:
        return None
    chart = load_chart(folder)
    notes = chart.guitar.notes["expert"]
    if len(notes) < 80:
        return None
    ini = read_ini(folder / "song.ini")
    delay = float(ini.get("delay", "0") or 0) / 1000
    paths.WORK.mkdir(parents=True, exist_ok=True)
    second_guitar = "rhythm" in files and "bass" in files
    with tempfile.TemporaryDirectory(dir=paths.WORK) as td:
        def mono(p: Path) -> np.ndarray:
            return load(decode_to_wav(p, Path(td) / "a.wav"))[0].mean(axis=0)

        def add(a: np.ndarray | None, b: np.ndarray, g: float = 1.0) -> np.ndarray:
            if a is None:
                return b * g
            n = max(len(a), len(b))
            return np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b))) * g

        gtr = mono(files["guitar"])
        if second_guitar:
            gtr = add(gtr, mono(files["rhythm"]))
        rest = None
        for name, p in files.items():
            if name == "guitar" or (name == "rhythm" and second_guitar) or name.startswith(("crowd", "preview")):
                continue
            rest = add(rest, mono(p))
    rng = np.random.default_rng(int(key[:8], 16))
    bleed = float(rng.uniform(*BLEED_DB))
    y = add(gtr, rest, 10 ** (bleed / 20)) if rest is not None else gtr
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    spec = compact_spectrum(analyze_stem(y.astype(np.float32), 44100))
    ref_t = np.array([chart.tempo.tick_to_sec(n.tick) - delay for n in notes])
    order = np.argsort(ref_t)
    sizes = np.array([len([f for f in notes[k].frets if f != OPEN]) for k in order])
    official = any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS)
    return {"cqt": spec["cqt"], "onset": spec["onset"], "fps": np.float32(SPEC_FPS), "folder": str(folder),
            "sep": np.bool_(False), "official": np.bool_(official), "bleed_db": np.float32(bleed if rest is not None else -99),
            "ref_t": ref_t[order].astype(np.float32), "sizes": sizes.astype(np.int8)}


def _wait_disk(f: Path, max_s: float = 180) -> bool:
    """Espera a que la carpeta vuelva a verse (discos USB que se desconectan unos segundos)."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        time.sleep(10)
        try:
            if f.is_dir():
                time.sleep(5)
                return True
        except OSError:
            pass
    return False


def _work(folder: str) -> str:
    from .library_train import _key

    f = Path(folder)
    for attempt in range(3):
        if not f.is_dir() and not _wait_disk(f):
            return "disco"
        key = _key(f)  # sin disco la clave saldria distinta
        out = STEM_SPEC_DIR / f"{key}_guitar.npz"
        none = out.with_suffix(".none")
        if out.exists() or none.exists():
            return "ya"
        try:
            d = song_spectrum(f, key)
            break
        except Exception as e:  # noqa: BLE001
            if isinstance(e, FileNotFoundError) or getattr(e, "winerror", None) in (3, 21, 1167) or not f.is_dir():
                # el disco se ha desconectado un momento: se espera a que vuelva y se reintenta (sin marcarla)
                if attempt == 2 or not _wait_disk(f):
                    return "disco"
                continue
            none.write_text(str(e)[:200], encoding="utf-8")
            return "error"
    if d is None:
        none.write_text("", encoding="utf-8")
        return "no"
    # etiquetas de las otras dificultades y de los botones (asi el entrenamiento no tiene que releer el chart)
    from .neural_onset import _button_labels, _diff_times

    if _diff_times(f, "pista_" + out.name) is None or _button_labels(f, "pista_" + out.name) is None:
        if not f.is_dir():
            return "disco"
        none.write_text("sin etiquetas", encoding="utf-8")
        return "no"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, out)
    return "ok"


def _init():
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")


def candidates(library: Path) -> list[Path]:
    """Canciones de la biblioteca con guitar.* (sin las del banco de pruebas)."""
    from .library_train import is_holdout
    from .reference import find_songs

    out = []
    for f in find_songs(library):
        if is_holdout(f):
            continue
        try:
            if any((f / f"guitar{e}").exists() for e in AUDIO_EXT):
                out.append(f)
        except OSError:
            continue
    return out


def extract_all(library: Path, workers: int = 8, log=print, stop=lambda: False) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    STEM_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    songs = candidates(library)
    log(f"  {len(songs)} canciones traen la pista de guitarra original")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(_work, str(f)) for f in songs]
        for k, fu in enumerate(as_completed(futs)):
            r = fu.result()
            counts[r] = counts.get(r, 0) + 1
            if (k + 1) % 100 == 0:
                rate = (time.time() - t0) / (k + 1)
                log(f"  {k + 1}/{len(songs)} ({counts}) ~{rate * (len(songs) - k - 1) / 60:.0f} min restantes")
            if stop():
                for x in futs:
                    x.cancel()
                break
    log(f"  espectrogramas de pistas originales: {counts}")
    return counts.get("ok", 0)


if __name__ == "__main__":
    _init()
    lib = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    extract_all(lib, n, log=lambda m: print(m, flush=True))
