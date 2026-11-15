"""Espectrogramas y etiquetas para el modelo de teclado (5 carriles y Pro Keys).

De las canciones que traen keys.* sale en unos segundos de CPU: el teclado original con el resto de la cancion muy
bajo, para que se parezca a una pista separada por la IA (como hace stem_spectra.py con la guitarra). Las demas
canciones con teclado, y las de validacion, se separan con la IA (separate_all, usa la GPU).

Solo lee la biblioteca; todo se guarda en config/entrenamiento/espectros_teclas.

Uso: python -m charter.calibration.keys_spectra pistas <carpeta_biblioteca> [procesos]
     python -m charter.calibration.keys_spectra ia <todas_las_canciones.json> <canciones_con_teclado.json> <minutos>
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import paths
from .stem_spectra import AUDIO_EXT, BLEED_DB, _audio_files, _init, _wait_disk

KEYS_SPEC_DIR = paths.CONFIG_DIR / "entrenamiento" / "espectros_teclas"
DIFFS = ("expert", "hard", "medium", "easy")
MIN_NOTES = 80


def chart_labels(folder: Path) -> dict | None:
    """Notas de teclado del chart en segundos: 5 carriles (k5_*) y Pro Keys (pr_*), en las 4 dificultades."""
    from ..chart.chart_io import load_chart
    from ..chart.model import OPEN
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS

    chart = load_chart(folder)
    ini = read_ini(folder / "song.ini")
    delay = float(ini.get("delay", "0") or 0) / 1000
    sec = chart.tempo.tick_to_sec
    out: dict = {}
    k5 = sorted(chart.keys.notes["expert"], key=lambda n: n.tick)
    pk = sorted(chart.pro_keys.notes["expert"], key=lambda n: (n.tick, n.pitch))
    has5, haspro = len(k5) >= MIN_NOTES, len({n.tick for n in pk}) >= MIN_NOTES
    if not has5 and not haspro:
        return None
    for d in DIFFS:
        out[f"k5_{d}"] = np.sort(np.array([sec(n.tick) - delay for n in chart.keys.notes[d]], np.float32))
        out[f"pr_{d}"] = np.array(sorted({sec(n.tick) - delay for n in chart.pro_keys.notes[d]}), np.float32)
    out["k5_size"] = np.array([len([f for f in n.frets if f != OPEN]) for n in k5], np.int8)
    out["k5_lane"] = np.array([-1 if OPEN in n.frets or not n.frets else min(n.frets) for n in k5], np.int8)
    out["k5_end"] = np.array([sec(n.tick + n.sustain) - delay for n in k5], np.float32)
    out["k5_t"] = np.array([sec(n.tick) - delay for n in k5], np.float32)
    out["pk_t"] = np.array([sec(n.tick) - delay for n in pk], np.float32)
    out["pk_p"] = np.array([n.pitch for n in pk], np.int8)
    out["pk_end"] = np.array([sec(n.tick + n.sustain) - delay for n in pk], np.float32)
    out["has5"], out["haspro"] = np.bool_(has5), np.bool_(haspro)
    out["official"] = np.bool_(any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS))
    out["folder"] = str(folder)
    return out


def song_spectrum(folder: Path, key: str) -> dict | None:
    """Teclado original + restos del resto de la cancion, con sus etiquetas."""
    from ..audio import decode_to_wav, load
    from ..pipeline.transcription import analyze_stem
    from .onset_model import compact_spectrum

    files = _audio_files(folder)
    if "keys" not in files:
        return None
    labels = chart_labels(folder)
    if labels is None:
        return None
    paths.WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.WORK) as td:
        def mono(p: Path) -> np.ndarray:
            return load(decode_to_wav(p, Path(td) / "a.wav"))[0].mean(axis=0)

        def add(a: np.ndarray | None, b: np.ndarray, g: float = 1.0) -> np.ndarray:
            if a is None:
                return b * g
            n = max(len(a), len(b))
            return np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b))) * g

        y = mono(files["keys"])
        rest = None
        for name, p in files.items():
            if name != "keys" and not name.startswith(("crowd", "preview")):
                rest = add(rest, mono(p))
    bleed = float(np.random.default_rng(int(key[:8], 16)).uniform(*BLEED_DB))
    if rest is not None:
        y = add(y, rest, 10 ** (bleed / 20))
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    spec = compact_spectrum(analyze_stem(y.astype(np.float32), 44100))
    return {"cqt": spec["cqt"], "onset": spec["onset"], "sep": np.bool_(False), **labels}


def _spec(y: np.ndarray) -> dict:
    from ..pipeline.transcription import analyze_stem
    from .onset_model import compact_spectrum

    return compact_spectrum(analyze_stem(y.astype(np.float32), 44100))


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Parecido entre dos espectrogramas compactos (0..1): cuanto de una pista esta en la otra."""
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n].astype(np.float32).ravel(), b[:, :n].astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def separation_order(all_songs: list[Path], keys_list: list[dict]) -> list[tuple[Path, str]]:
    """Que separar primero con la IA: validacion, canciones sin teclado (para medir sus restos), las que no traen
    pista de teclado (solo se pueden usar asi) y luego las oficiales."""
    from .library_train import is_holdout
    from .neural_onset import _is_val

    with_keys = [r for r in keys_list if not is_holdout(Path(r["folder"]))]
    val = [r for r in with_keys if _is_val(r["folder"])]
    val.sort(key=lambda r: not r["official"])
    train = [r for r in with_keys if not _is_val(r["folder"])]
    no_stem = [r for r in train if not r["keys"]]
    no_stem.sort(key=lambda r: not r["official"])
    official = [r for r in train if r["keys"] and r["official"]]
    return ([(Path(r["folder"]), "ia") for r in val[:40]]
            + [(p, "neg") for p in negative_songs(all_songs, 30)]
            + [(Path(r["folder"]), "ia") for r in no_stem + official + val[40:]])


def negative_songs(all_songs: list[Path], n: int) -> list[Path]:
    """Canciones oficiales de banda completa sin chart de teclado, repartidas entre entrenamiento y validacion."""
    import json

    from .library_train import is_holdout

    stats = json.loads((paths.CONFIG_DIR / "entrenamiento" / "estadisticas_cache.json").read_text(encoding="utf-8"))
    good = {v["folder"] for v in stats.values() if v.get("official") and not v.get("has_pro_keys")
            and not v.get("tracks", {}).get("keys") and v.get("tracks", {}).get("bass") and v.get("tracks", {}).get("guitar")}
    songs = [f for f in all_songs if f.name in good and not is_holdout(f)]
    return list(np.random.default_rng(11).permutation(songs)[:n])


def separate_all(all_songs: list[Path], keys_list: list[dict], deadline: float, log=print) -> None:
    """Separa canciones con la IA hasta la hora limite. Lo que se usa al generar es la pista de piano; se guarda
    tambien piano + resto, porque muchos teclados de los charts son sintetizadores y caen en el resto."""
    from .library_train import _key, _QuickSeparator

    KEYS_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    order = separation_order(all_songs, keys_list)
    log(f"  {len(order)} canciones en la lista de separacion")
    sep = _QuickSeparator(log)
    sims: list[tuple[float, float]] = []
    done = 0
    t0 = time.time()
    try:
        for f, kind in order:
            if time.time() > deadline:
                break
            try:
                key = _key(f)
                out = KEYS_SPEC_DIR / (f"{key}_ia.npz" if kind == "ia" else f"{key}_ianeg.npz")
                if out.exists():
                    continue
                labels = chart_labels(f) if kind == "ia" else {"folder": str(f)}
                if labels is None:
                    continue
                res = sep.separate(f, insts=("piano", "other"))
                if "piano" not in res:
                    continue
                piano = _spec(res["piano"])
                both = _spec(res["piano"] + res["other"]) if "other" in res else piano
                _save(out, {"cqt": piano["cqt"], "onset": piano["onset"], "cqt_mix": both["cqt"], "onset_mix": both["onset"],
                            "sep": np.bool_(True), **labels})
                done += 1
                files = _audio_files(f)
                if kind == "ia" and "keys" in files and len(sims) < 40:
                    from ..audio import decode_to_wav, load

                    wav = decode_to_wav(files["keys"], sep.tmp / "keys_real.wav")
                    real = _spec(load(wav)[0].mean(axis=0))["cqt"]
                    sims.append((_similarity(real, piano["cqt"]), _similarity(real, both["cqt"])))
                if done % 10 == 0:
                    msg = f"  {done} separadas, {(time.time() - t0) / done:.0f} s cada una"
                    if sims:
                        msg += f" | parecido con el teclado real: piano {np.mean([s[0] for s in sims]):.3f}, " \
                               f"piano+resto {np.mean([s[1] for s in sims]):.3f} ({len(sims)} canciones)"
                    log(msg)
            except Exception as e:  # noqa: BLE001
                log(f"  {f.name}: {e}")
    finally:
        sep.close()
    log(f"  separacion con IA terminada: {done} canciones")


def _save(out: Path, d: dict) -> None:
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, out)


def _disk_ok(f: Path) -> bool:
    try:
        return f.is_dir()
    except OSError:  # con el disco desconectado, segun el momento, hasta esto da error
        return False


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
        out = KEYS_SPEC_DIR / f"{key}_pista.npz"
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
        none.write_text("", encoding="utf-8")
        return "no"
    _save(out, d)
    return "ok"


def candidates(library: Path, with_stem: bool) -> list[Path]:
    """Canciones que el entrenamiento puede usar (no las del banco de pruebas), con pista de teclado o sin ella."""
    from .library_train import is_holdout
    from .reference import find_songs

    out = []
    for f in find_songs(library):
        if is_holdout(f):
            continue
        try:
            if any((f / f"keys{e}").exists() for e in AUDIO_EXT) == with_stem:
                out.append(f)
        except OSError:
            continue
    return out


def extract_all(library: Path, workers: int = 5, log=print, songs: list[Path] | None = None) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    KEYS_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    songs = candidates(library, True) if songs is None else songs
    log(f"  {len(songs)} canciones traen la pista de teclado")
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
            if (k + 1) % 50 == 0:
                rate = (time.time() - t0) / (k + 1)
                log(f"  {k + 1}/{len(songs)} ({counts}) ~{rate * (len(songs) - k - 1) / 60:.0f} min restantes")
    log(f"  espectrogramas de teclado: {counts}")
    return counts.get("ok", 0)


if __name__ == "__main__":
    _init()
    if sys.argv[1] == "pistas":
        extract_all(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 5, log=lambda m: print(m, flush=True))
    elif sys.argv[1] == "ia":
        import json

        separate_all([Path(x) for x in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))],
                     json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
                     time.time() + float(sys.argv[4]) * 60, log=lambda m: print(m, flush=True))
