"""Entrenamiento con la biblioteca del usuario (miles de canciones), con limite de tiempo y reanudable.

Fases:
  1. Charts (todas las canciones, .mid o .chart): estadisticas de jugabilidad (densidad, acordes,
     HOPOs, star power...) y modelos de reduccion de dificultad (guitarra/bajo/teclado y bateria).
  2. Audio: canciones que traen la pista separada (guitar.ogg, bass.ogg / rhythm.ogg): modelo de
     "que golpes son notas" y modelo de acordes. Lo analizado se guarda, asi que se puede parar y seguir.
  3. (Opcional, usa la GPU) canciones sin pistas separadas: se separan con la IA y se aprovechan igual.

Uso: python -m charter.calibration.library_train <trabajo.json>
     trabajo.json = {"folder": ..., "hours": 4, "separate": false}
Escribe en stdout una linea JSON por evento (log / progress / done / error).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import paths

TRAIN_DIR = paths.CONFIG_DIR / "entrenamiento"
# v2: guarda los tiempos de ataques y notas (las etiquetas se recalculan al entrenar). La carpeta "audio" antigua
# tenia etiquetas con un desfase que hacia que el modelo pusiera pocas notas: se borra
AUDIO_CACHE = TRAIN_DIR / "audio_v2"
OLD_AUDIO_CACHE = TRAIN_DIR / "audio"
HOLDOUT_FILE = TRAIN_DIR / "banco_pruebas.txt"
# espectrogramas compactos (guitarra separada + su chart) para entrenar en el futuro un modelo audio -> chart
SPEC_DIR = TRAIN_DIR / "espectros"
ROWS_PER_SONG = 900
FIT_RESERVE_S = 3 * 3600  # tiempo que se guarda al final: pistas originales (~1 h) y modelos (el neuronal, 2,5 h)
NEURAL_MINUTES = 150
NEURAL_WIDTH = 1.5  # con miles de canciones, un modelo mas grande aprende mas
# si existe, el entrenamiento deja de analizar y entrena ya con lo que tiene (boton "Terminar ya")
FINISH_FLAG = TRAIN_DIR / "terminar_ya.flag"


class Trainer:
    def __init__(self, folder: str, hours: float, separate: bool, emit):
        self.folder = Path(folder)
        # horas <= 0: sin limite (analiza todo lo que haya y luego entrena)
        self.unlimited = hours <= 0
        self.deadline = float("inf") if self.unlimited else time.time() + hours * 3600
        self.start = time.time()
        FINISH_FLAG.unlink(missing_ok=True)
        self.separate = separate
        from ..pipeline.runner import active_instruments

        self.active = active_instruments()
        # instrumentos con modelo de audio propio (ataque -> nota)
        self.audio_insts = tuple(i for i in ("guitar", "bass") if i in self.active) or ("guitar",)
        self.emit = emit
        AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        if OLD_AUDIO_CACHE.exists():
            import shutil

            shutil.rmtree(OLD_AUDIO_CACHE, ignore_errors=True)

    def log(self, msg: str) -> None:
        self.emit({"type": "log", "msg": msg})

    def progress(self, phase: str, frac: float) -> None:
        self.emit({"type": "progress", "phase": phase, "frac": float(frac),
                   "remaining": -1 if self.unlimited else max(self.deadline - time.time(), 0)})

    def time_left(self) -> float:
        return self.deadline - time.time()

    # ------------------------------------------------------------------ fase 1
    def phase_charts(self, songs: list[Path]) -> None:
        from . import drum_reducer_model, reducer_model
        from .reference import aggregate, analyze_song

        self.log(f"Fase 1: aprendiendo de {len(songs)} charts (densidad, acordes, HOPOs, dificultades)...")
        cache_p = TRAIN_DIR / "estadisticas_cache.json"
        try:
            cache = json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
        results = []
        for i, f in enumerate(songs):
            key = _key(f)
            if key in cache:
                r = cache[key]
            else:
                try:
                    r = analyze_song(f)
                except Exception:  # noqa: BLE001
                    r = None
                cache[key] = r
            if r and "error" not in r:
                results.append(r)
            if i % 50 == 0:
                self.progress("Analizando charts", i / max(len(songs), 1) * 0.5)
            if i % 500 == 0 and i:
                self.log(f"  {i}/{len(songs)} charts analizados")
                cache_p.write_text(json.dumps(cache), encoding="utf-8")
        cache_p.write_text(json.dumps(cache), encoding="utf-8")
        agg = aggregate(results)
        (paths.CONFIG_DIR / "reference_stats.json").write_text(
            json.dumps({"aggregate": agg, "songs": []}, indent=1, ensure_ascii=False), encoding="utf-8")
        self.log(f"  estadisticas guardadas ({agg.get('n_songs', 0)} canciones, {agg.get('n_official', 0)} oficiales)")
        # modelos de dificultad (con una muestra si la biblioteca es enorme, por memoria y tiempo)
        sub = _sample_folder(songs, 2500)
        sig = difficulty_signature(sub)
        tmpdir = _link_dir(sub)
        self.progress("Modelo de dificultades", 0.6)
        mk = TRAIN_DIR / "dificultad_frets.txt"
        if mk.exists() and mk.read_text(encoding="utf-8") == sig and (paths.CONFIG_DIR / "reducer.pkl").exists():
            self.log("  modelo de dificultades (guitarra/bajo/teclado) ya entrenado con estas canciones: se reutiliza")
        else:
            self.log(f"  entrenando el modelo de dificultades (guitarra/bajo/teclado) con {len(sub)} canciones...")
            counter = {"n": 0}

            def tick(*_):
                counter["n"] += 1
                if counter["n"] % 250 == 0:
                    self.log(f"    {counter['n']}/{len(sub)} charts preparados")

            reducer_model.save(reducer_model.train(str(tmpdir), log=tick), paths.CONFIG_DIR / "reducer.pkl")
            mk.write_text(sig, encoding="utf-8")
        self.progress("Modelo de dificultades", 0.8)
        mk = TRAIN_DIR / "dificultad_bateria.txt"
        if "drums" not in self.active:
            self.log("  bateria desactivada: no se entrena su modelo de dificultades")
        elif mk.exists() and mk.read_text(encoding="utf-8") == sig and (paths.CONFIG_DIR / "drum_reducer.pkl").exists():
            self.log("  modelo de dificultades de bateria ya entrenado: se reutiliza")
        else:
            self.log("  entrenando el modelo de dificultades de bateria...")
            drum_reducer_model.save(drum_reducer_model.train(str(tmpdir), log=lambda *_: None),
                                    paths.CONFIG_DIR / "drum_reducer.pkl")
            mk.write_text(sig, encoding="utf-8")
        self.progress("Modelo de dificultades", 1.0)
        self.log("  modelos de dificultad listos")

    # ------------------------------------------------------------------ fase 2/3
    def phase_audio(self, songs: list[Path]) -> None:
        from ..chart.chart_io import load_chart
        from ..songini import read_ini
        from .onset_model import song_data, stem_for
        from .reference import OFFICIAL_CHARTERS

        with_stems, without = [], []
        for f in songs:
            names = ("guitar", "bass", "rhythm") if "bass" in self.audio_insts else ("guitar",)
            if any((f / f"{n}{e}").exists() for n in names for e in (".ogg", ".opus", ".mp3")):
                with_stems.append(f)
            else:
                without.append(f)
        self.log(f"Fase 2: audio. {len(with_stems)} canciones traen pistas separadas"
                 + (f"; despues, con el tiempo que quede, la IA separara las {len(without)} que no las traen y "
                    "volvera a separar las que si (asi aprende con audio como el que vera al generar)"
                    if self.separate else ""))
        # oficiales primero, el resto en orden aleatorio (asi un corte por tiempo deja una muestra variada)
        random.seed(1)
        random.shuffle(with_stems)
        random.shuffle(without)

        def official(f):
            try:
                return any(o in read_ini(f / "song.ini").get("charter", "").lower() for o in OFFICIAL_CHARTERS)
            except Exception:  # noqa: BLE001
                return False

        with_stems.sort(key=lambda f: not official(f))
        queue = [(f, False) for f in with_stems]
        if self.separate:
            # se alternan canciones nuevas (sin pistas) y re-separaciones de las que tienen pistas
            resep = [(f, True) for f in with_stems]
            new = [(f, True) for f in without]
            for k in range(max(len(resep), len(new))):
                queue += new[k:k + 1] + resep[k:k + 1]
        done = 0
        sep = None
        for i, (f, needs_sep) in enumerate(queue):
            if self.time_left() < FIT_RESERVE_S:
                self.log("  se acaba el tiempo: paso a entrenar con lo analizado")
                break
            if FINISH_FLAG.exists():
                self.log("  'Terminar ya': paso a entrenar con lo analizado")
                break
            key = _key(f) + ("_ia" if needs_sep else "")
            if all((AUDIO_CACHE / f"{key}_{inst}.npz").exists() or (AUDIO_CACHE / f"{key}_{inst}.none").exists()
                   for inst in self.audio_insts):
                done += 1
                continue
            try:
                ini = read_ini(f / "song.ini")
                chart = load_chart(f)
                stems_audio = {}
                if needs_sep:
                    if all(len(getattr(chart, inst).notes["expert"]) < 80 for inst in self.audio_insts):
                        for inst in self.audio_insts:
                            (AUDIO_CACHE / f"{key}_{inst}.none").write_text("sin notas")
                        continue
                    if sep is None:
                        sep = _QuickSeparator(self.log)
                    stems_audio = sep.separate(f)
                for inst in self.audio_insts:
                    out = AUDIO_CACHE / f"{key}_{inst}.npz"
                    if out.exists():
                        continue
                    if needs_sep:
                        d = song_data(f, chart, ini, inst, stems_audio.get(inst)) if inst in stems_audio else None
                    else:
                        d = song_data(f, chart, ini, inst) if stem_for(f, inst, chart) else None
                    if d is None:
                        (AUDIO_CACHE / f"{key}_{inst}.none").write_text("")
                        continue
                    sel = _subsample(len(d["y"]), ROWS_PER_SONG, seed=i)
                    w = 1.5 if any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS) else 1.0
                    if needs_sep:
                        w *= 1.3  # mismo tipo de audio que al generar charts
                    np.savez_compressed(out, X=d["X"][sel], t=d["t"][sel], ref_t=d["ref_t"], sizes=d["sizes"],
                                        w=np.float32(w), sep=np.bool_(needs_sep))
                    try:
                        from .onset_model import SPEC_FPS

                        SPEC_DIR.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(SPEC_DIR / f"{key}_{inst}.npz", cqt=d["spec"]["cqt"],
                                            onset=d["spec"]["onset"], fps=np.float32(SPEC_FPS),
                                            folder=str(f), sep=np.bool_(needs_sep), official=np.bool_(w >= 1.5))
                    except Exception as e:  # noqa: BLE001
                        self.log(f"  aviso: no se pudo guardar el espectrograma de {f.name} ({e})")
                done += 1
                if done % 10 == 0 or needs_sep:
                    self.log(f"  {done} canciones con audio analizadas ({f.name})")
            except Exception as e:  # noqa: BLE001
                self.log(f"  aviso: {f.name}: {e}")
                for inst in self.audio_insts:
                    (AUDIO_CACHE / f"{key}_{inst}.none").write_text(str(e)[:200])
            if self.unlimited:
                self.progress("Analizando audio", min((i + 1) / len(queue), 0.99))
            else:
                span = max(self.deadline - FIT_RESERVE_S - self.start, 1)
                self.progress("Analizando audio", min((time.time() - self.start) / span, 0.99))
        if sep is not None:
            sep.close()

    def fit(self) -> None:
        from .onset_model import FEATURE_VERSION, fit, make_labels, save

        self.log("Fase final: entrenando los modelos de audio con todo lo analizado...")
        data = {"guitar": [], "bass": []}
        groups: dict[str, int] = {}
        for p in sorted(AUDIO_CACHE.glob("*.npz")):
            inst = p.stem.split("_")[-1]
            if inst not in data:
                continue
            z = np.load(p)
            if z["X"].shape[1] != 22:
                continue  # analizado con otra version de las caracteristicas
            # la version separada por la IA y la original de una cancion van en el mismo grupo (validacion honesta)
            gi = groups.setdefault(p.stem.split("_")[0], len(groups))
            y, chord = make_labels(z["t"], z["ref_t"], z["sizes"])
            data[inst].append({"X": z["X"], "y": y, "chord": chord, "g": gi, "w": float(z["w"]),
                               "sep": bool(z["sep"])})
        n = {k: len(v) for k, v in data.items()}
        n_sep = {k: sum(d["sep"] for d in v) for k, v in data.items()}
        self.log(f"  pistas analizadas: guitarra {n['guitar']} ({n_sep['guitar']} separadas por la IA), "
                 f"bajo {n['bass']} ({n_sep['bass']} separadas por la IA)")
        if n["guitar"] + n["bass"] < 10:
            self.log("  muy pocas pistas con audio: se mantienen los modelos de audio de serie")
            return
        models = fit(data, log=self.log, evaluate=True, max_rows=2_500_000)
        models["version"] = FEATURE_VERSION
        # si para algun instrumento hay menos ejemplos que en los modelos de serie, se queda el de serie
        import pickle

        from .onset_model import DEFAULT_THR, MODEL_PATH

        try:
            with open(MODEL_PATH, "rb") as f:
                default = pickle.load(f)
        except Exception:  # noqa: BLE001
            default = {}
        for inst in ("guitar", "bass"):
            n_new = models.get("n_tracks", {}).get(inst, 0)
            n_def = default.get("n_tracks", {}).get(inst, 0)
            if n_new < n_def and inst in default:
                models[inst] = default[inst]
                models["n_tracks"][inst] = n_def
                models.setdefault("thr", {})[inst] = default.get("thr", {}).get(inst, DEFAULT_THR)
                if inst == "guitar" and "chord" in default:
                    models["chord"] = default["chord"]
                self.log(f"  {inst}: pocas pistas ({n_new}); se mantiene el modelo de serie ({n_def} pistas)")
        save(models, paths.CONFIG_DIR / "onset_model.pkl")
        self.log("  modelos de audio guardados")

    def phase_stem_spectra(self) -> None:
        """Espectrogramas de las guitarras originales (con restos del resto de la cancion) para el modelo
        neuronal: unos segundos de CPU por cancion, sin separar con la IA."""
        if "guitar" not in self.active or FINISH_FLAG.exists():
            return
        from .stem_spectra import extract_all

        self.log("Guitarras originales para el modelo neuronal (las que traen guitar.ogg)...")
        try:
            # pocos procesos: con muchos, algunos discos USB se desconectan un momento
            extract_all(self.folder, workers=3, log=self.log, stop=FINISH_FLAG.exists)
        except Exception as e:  # noqa: BLE001
            self.log(f"  aviso: no se pudieron preparar las guitarras originales ({e})")

    def fit_neural(self) -> None:
        """Modelo neuronal de notas de guitarra, con los espectrogramas guardados (solo se queda si mejora)."""
        if "guitar" in self.active:
            _fit_neural_impl(self)

    def run(self) -> None:
        from .reference import find_songs

        self.log(f"Buscando charts en {self.folder} ...")
        songs = list(find_songs(self.folder))
        self.log(f"Encontradas {len(songs)} canciones con chart")
        if not songs:
            raise RuntimeError("No se han encontrado canciones (carpetas con notes.mid o notes.chart y song.ini)")
        # banco de pruebas: unas pocas canciones con pistas que nunca se usan para entrenar (para medir de verdad)
        test = [f for f in songs if is_holdout(f)]
        HOLDOUT_FILE.write_text("\n".join(str(f) for f in test), encoding="utf-8")
        songs = [f for f in songs if not is_holdout(f)]
        self.log(f"  {len(test)} canciones apartadas como banco de pruebas (no se usan para entrenar)")
        self.phase_charts(songs)
        self.phase_audio(songs)
        self.phase_stem_spectra()
        self.fit()
        self.fit_neural()
        mins = (time.time() - self.start) / 60
        self.log(f"Entrenamiento terminado en {mins:.0f} min. Las proximas canciones usaran lo aprendido.")


def _fit_neural_impl(trainer) -> None:
    from . import neural_onset

    n = len(list(SPEC_DIR.glob("*.npz")))
    n_orig = len(list(neural_onset.STEM_SPEC_DIR.glob("*_guitar.npz")))
    if n < 100:
        trainer.log(f"  modelo neuronal: hacen falta mas guitarras separadas con IA ({n} de 100)")
        return
    big = n_orig >= 1000
    minutes = NEURAL_MINUTES if big else 40
    trainer.log(f"Entrenando el modelo neuronal de notas con {n} guitarras separadas y {n_orig} originales "
                f"(unos {minutes} min con la GPU) ...")
    try:
        neural_onset.train(minutes, log=trainer.log, width=NEURAL_WIDTH if big else 1.0,
                           batch_size=32 if big else 24, eval_every=1000 if big else 400)
    except Exception as e:  # noqa: BLE001
        trainer.log(f"  no se pudo entrenar el modelo neuronal: {e}")


class _QuickSeparator:
    """Separacion rapida (1 modelo, 1 pasada) para aprovechar canciones sin pistas separadas."""

    def __init__(self, log):
        import tempfile

        # temporal dentro de la carpeta del programa (nunca se escribe en la biblioteca del usuario)
        paths.WORK.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="entrenamiento_sep_", dir=paths.WORK))
        self.log = log
        self.sep = None
        self.autocast = True

    def _model(self):
        # el modelo se carga una sola vez; solapamiento 2 y media precision: varias veces mas rapido que la
        # separacion de las canciones (solapamiento 8) y con una calidad casi igual para aprender de ella
        if self.sep is None:
            from ..pipeline.separation import SIX_STEM_MODEL, _separator

            (self.tmp / "out").mkdir(parents=True, exist_ok=True)
            self.sep = _separator(self.tmp / "out", self.log, overlap=2, autocast=self.autocast)
            self.sep.load_model(model_filename=SIX_STEM_MODEL)
        return self.sep

    def _run(self, wav: Path) -> dict:
        out = {}
        for f in self._model().separate(str(wav)):
            p = Path(f)
            if not p.is_absolute():
                p = self.tmp / "out" / p
            name = p.stem
            out[name.split("_(")[-1].split(")")[0].lower() if "_(" in name else name.lower()] = p
        return out

    def separate(self, folder: Path, insts: tuple[str, ...] = ("guitar", "bass")) -> dict:
        from ..audio import decode_to_wav, load, save
        exts = (".ogg", ".opus", ".mp3", ".wav", ".flac")
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
                 and not p.stem.lower().startswith(("crowd", "preview"))]
        if not files:
            return {}
        if len(files) == 1:
            wav = decode_to_wav(files[0], self.tmp / "mix.wav")
        else:
            # cancion con pistas: se suman todas para tener la mezcla completa (como un mp3 normal)
            mix = None
            for k, p in enumerate(files):
                a = load(decode_to_wav(p, self.tmp / f"pista{k}.wav"))[0]
                (self.tmp / f"pista{k}.wav").unlink(missing_ok=True)
                if mix is None:
                    mix = a
                else:
                    n = max(mix.shape[1], a.shape[1])
                    mix = np.pad(mix, ((0, 0), (0, n - mix.shape[1]))) + np.pad(a, ((0, 0), (0, n - a.shape[1])))
            peak = float(np.abs(mix).max()) or 1.0
            if peak > 0.99:
                mix = mix * (0.99 / peak)
            wav = self.tmp / "mix.wav"
            save(wav, mix)
        res = self._run(wav)
        out = {}
        for inst in insts:
            if inst in res:
                out[inst] = load(res[inst])[0].mean(axis=0)
        if self.autocast and not all(np.isfinite(a).all() for a in out.values()):
            # la media precision ha fallado en esta tarjeta: se repite en precision normal
            self.log("  separacion rapida inestable: sigo en precision normal")
            self.close_model()
            self.autocast = False
            res = self._run(wav)
            out = {inst: load(res[inst])[0].mean(axis=0) for inst in insts if inst in res}
        # limpiar
        for p in (self.tmp / "out").rglob("*"):
            if p.is_file():
                p.unlink()
        for d in sorted((self.tmp / "out").glob("*"), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        return out

    def close_model(self):
        from ..pipeline.separation import _free_gpu

        self.sep = None
        _free_gpu()

    def close(self):
        import shutil

        self.close_model()
        shutil.rmtree(self.tmp, ignore_errors=True)


def difficulty_signature(sub: list[Path]) -> str:
    return hashlib.sha1("\n".join(sorted(_key(s) for s in sub)).encode("utf-8")).hexdigest()


def is_holdout(folder: Path) -> bool:
    """~2 % de las canciones, siempre las mismas (por nombre), quedan fuera del entrenamiento."""
    return int(hashlib.sha1(folder.name.lower().encode("utf-8")).hexdigest()[:8], 16) % 50 == 0


def _key(folder: Path) -> str:
    try:
        st = max((p.stat().st_mtime for p in folder.iterdir() if p.is_file()), default=0)
    except OSError:
        st = 0
    return hashlib.sha1(f"{folder.resolve()}|{int(st)}".encode("utf-8")).hexdigest()[:16]


def _subsample(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _sample_folder(songs: list[Path], k: int) -> list[Path]:
    if len(songs) <= k:
        return songs
    random.seed(2)
    return random.sample(songs, k)


def _link_dir(songs: list[Path]) -> Path:
    """Carpeta "virtual" con la lista de canciones (archivo de rutas) para los entrenadores de dificultad."""
    p = TRAIN_DIR / "lista_canciones.txt"
    p.write_text("\n".join(str(s) for s in songs), encoding="utf-8")
    return p


def main() -> int:
    paths.setup_env()
    import warnings

    warnings.filterwarnings("ignore")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = sys.stdout
    sys.stdout = sys.stderr

    def emit(ev):
        ev = {**ev, "t": time.strftime("%H:%M:%S")}
        out.write(json.dumps(ev, ensure_ascii=False) + "\n")
        out.flush()

    try:
        Trainer(job["folder"], float(job.get("hours", 4)), bool(job.get("separate", False)), emit).run()
        emit({"type": "done"})
        return 0
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "msg": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
