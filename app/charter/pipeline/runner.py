"""Proceso completo de una cancion: de un archivo de audio a una carpeta lista para YARG / Clone Hero.

Cada paso guarda su resultado en la carpeta de trabajo, asi una pausa (que cierra el proceso
para liberar toda la VRAM) puede continuar donde se quedo.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import paths
from ..audio import SR, decode_to_wav, export_ogg, load, save
from ..chart.midi_io import write_midi
from ..chart.model import DIFFICULTIES, RES, Chart
from ..songini import write_ini

# Segundos de proceso por cada segundo de audio (se recalibran solos con el uso)
DEFAULT_COST = {
    "decodificar": 0.01,
    "metadatos": 0.02,
    "separacion": 3.6,
    "pulso": 0.06,
    "transcripcion": 1.1,
    "voz": 0.5,
    "chart": 0.08,
    "audio": 0.12,
    "video": 0.05,
}
STEP_NAMES = {
    "decodificar": "Preparando audio",
    "metadatos": "Metadatos y portada",
    "separacion": "Separando instrumentos (IA)",
    "pulso": "Detectando tempo y compases",
    "transcripcion": "Transcribiendo notas (IA)",
    "voz": "Voz y letra",
    "chart": "Generando el chart",
    "audio": "Exportando audio",
    "video": "Video de fondo",
}


@dataclass
class JobConfig:
    audio_path: str
    output_dir: str
    cover_path: str = ""
    video_path: str = ""
    video_url: str = ""
    video_offset_ms: int = 0
    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""
    genre: str = ""
    lyrics_text: str = ""
    online: bool = True


class Paused(Exception):
    pass


ALL_INSTRUMENTS = ("guitar", "bass", "drums", "keys", "vocals")
# Guitarra, bajo, voz y teclado (y bateria si su modelo esta entrenado). Para cambiarlos: "instrumentos" en config/ajustes.json, p. ej. ["guitar", "vocals"]
DEFAULT_INSTRUMENTS = ("guitar", "bass", "vocals", "keys")


def _drum_model_ready() -> bool:
    """Con el modelo neuronal de bateria no hace falta separar la bateria por piezas (ahorra tiempo)."""
    from ..calibration.neural_drums import MODEL_PATH

    return MODEL_PATH.exists()


def active_instruments() -> set[str]:
    try:
        v = json.loads((paths.CONFIG_DIR / "ajustes.json").read_text(encoding="utf-8")).get("instrumentos")
        if isinstance(v, list) and v:
            return {i for i in v if i in ALL_INSTRUMENTS}
    except Exception:  # noqa: BLE001
        pass
    # la bateria entra por defecto cuando su modelo neuronal esta entrenado
    return set(DEFAULT_INSTRUMENTS) | ({"drums"} if _drum_model_ready() else set())


def slug(text: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "", text).strip().rstrip(".")
    return s or "cancion"


def _costs_file() -> Path:
    return paths.CONFIG_DIR / "tiempos.json"


def load_costs() -> dict:
    try:
        c = json.loads(_costs_file().read_text(encoding="utf-8"))
        return {**DEFAULT_COST, **c}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_COST)


def save_cost(step: str, secs_per_audio_sec: float) -> None:
    c = load_costs()
    old = c.get(step, DEFAULT_COST.get(step, 0.1))
    c[step] = old * 0.6 + secs_per_audio_sec * 0.4
    try:
        _costs_file().write_text(json.dumps(c, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


class Reporter:
    """Envia progreso y mensajes (al GUI por stdout en JSON, o a la consola)."""

    def __init__(self, emit):
        self.emit = emit
        self.step = ""
        self.duration = 240.0
        self.costs = load_costs()
        self.done_steps: list[str] = []
        self.step_start = time.time()
        self.step_frac = 0.0

    def log(self, msg: str) -> None:
        self.emit({"type": "log", "msg": msg})

    def start(self, step: str) -> None:
        self.step = step
        self.step_start = time.time()
        self.step_frac = 0.0
        self.log(f"--- {STEP_NAMES.get(step, step)} ---")
        self._send()

    def progress(self, frac: float) -> None:
        self.step_frac = float(np.clip(frac, 0, 1))
        self._send()

    def finish(self, measure: bool = True) -> None:
        el = time.time() - self.step_start
        per_sec = el / self.duration if self.duration > 0 else 0
        # si el paso se ha saltado casi entero (resultado ya guardado tras una pausa) no cuenta para la media
        if measure and el > 2 and per_sec > 0.35 * self.costs.get(self.step, 0.1):
            save_cost(self.step, per_sec)
        self.done_steps.append(self.step)
        self.step_frac = 1.0
        self._send()

    def _send(self) -> None:
        total = sum(self.costs.get(s, 0.1) for s in STEP_NAMES) * self.duration
        done = sum(self.costs.get(s, 0.1) for s in self.done_steps) * self.duration
        cur = self.costs.get(self.step, 0.1) * self.duration
        elapsed_cur = time.time() - self.step_start
        # si el paso actual va mas lento de lo previsto, la estimacion crece
        if self.step_frac > 0.05:
            cur_est = max(cur, elapsed_cur / self.step_frac)
        else:
            cur_est = max(cur, elapsed_cur * 1.1)
        remaining_cur = max(cur_est - elapsed_cur, 0) if self.step not in self.done_steps else 0
        future = total - done - (cur if self.step not in self.done_steps else 0)
        eta = remaining_cur + max(future, 0)
        overall = (done + (cur * self.step_frac if self.step not in self.done_steps else 0)) / max(total, 1)
        self.emit({"type": "progress", "step": self.step, "step_name": STEP_NAMES.get(self.step, self.step),
                   "frac": self.step_frac, "overall": min(overall, 0.999), "eta": eta})


def _cached(path: Path, fn):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = fn()
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def process_song(cfg: JobConfig, work: Path, emit, pause_flag: Path | None = None) -> Path:
    from .metadata import complete_online, from_file, has_internet, save_cover

    rep = Reporter(emit)
    work.mkdir(parents=True, exist_ok=True)
    active = active_instruments()

    def check_pause():
        if pause_flag is not None and pause_flag.exists():
            raise Paused()

    log = rep.log
    names = {"guitar": "guitarra", "bass": "bajo", "drums": "bateria", "keys": "teclado", "vocals": "voz"}
    log("Instrumentos: " + ", ".join(names[i] for i in ALL_INSTRUMENTS if i in active))
    online = cfg.online and has_internet()
    if cfg.online and not online:
        emit({"type": "warning", "msg": "No hay conexion a internet: no se podra buscar la letra, la portada "
                                        "ni los datos que falten, ni descargar el video de YouTube. "
                                        "La letra se sacara con IA (Whisper)."})

    # ---------------------------------------------------------------- 1. decodificar
    rep.start("decodificar")
    mix_wav = work / "mix.wav"
    if not mix_wav.exists():
        decode_to_wav(cfg.audio_path, mix_wav)
    mix, sr = load(mix_wav)
    rep.duration = mix.shape[1] / sr
    log(f"Duracion: {int(rep.duration // 60)}:{int(rep.duration % 60):02d}")
    rep.finish(measure=False)

    # ---------------------------------------------------------------- 2. metadatos
    rep.start("metadatos")
    meta_p = work / "meta.pkl"

    def get_meta():
        m = from_file(cfg.audio_path)
        for k in ("title", "artist", "album", "year", "genre"):
            v = getattr(cfg, k)
            if v:
                setattr(m, k, v)
        if cfg.cover_path:
            m.cover = Path(cfg.cover_path).read_bytes()
            m.cover_source = "usuario"
        if online:
            m = complete_online(m, log)
        return m

    meta = _cached(meta_p, get_meta)
    log(f"Cancion: {meta.artist} - {meta.title} | album: {meta.album or '?'} | ano: {meta.year or '?'} | "
        f"genero: {meta.genre or '?'} | portada: {meta.cover_source or 'no encontrada'}")
    if not meta.cover:
        emit({"type": "warning", "msg": "La cancion no tiene portada y no se ha encontrado en internet. "
                                        "Puedes anadirla tu desde la lista de canciones."})
    rep.finish(measure=False)
    check_pause()

    # ---------------------------------------------------------------- 3. separacion
    rep.start("separacion")
    from .separation import separate

    stems = separate(mix_wav, work, log=log, progress=rep.progress, check_pause=check_pause,
                     full=bool(active & {"drums", "vocals"}), drum_parts="drums" in active and not _drum_model_ready())
    rep.finish()
    check_pause()

    # ---------------------------------------------------------------- 4. pulso
    rep.start("pulso")
    from .beats import refine_grid, track_beats

    grid = refine_grid(_cached(work / "beats.pkl", lambda: track_beats(mix.mean(axis=0), sr, log)))
    log(f"Tempo: {'fijo ' + format(grid.constant_bpm, '.1f') + ' BPM' if grid.constant_bpm else 'variable'}"
        f" ({60 / np.median(np.diff(grid.beats)):.0f} BPM aprox.)")
    rep.finish()
    check_pause()

    # ---------------------------------------------------------------- 5. transcripcion
    rep.start("transcripcion")
    if "keys" in active:
        _keys_audio(stems, sr, work)
    trans = _transcribe_all(stems, mix, sr, rep, check_pause, work, active)
    rep.finish()
    check_pause()

    # ---------------------------------------------------------------- 6. voz
    rep.start("voz")
    voc = (_cached(work / "voz.pkl", lambda: _vocals_analysis(stems, meta, cfg, online, rep, check_pause))
           if "vocals" in active else {"present": False})
    rep.finish()
    check_pause()

    # ---------------------------------------------------------------- 7. chart
    rep.start("chart")
    chart, report, used_audio, lead_in = _build_chart(stems, mix, sr, grid, trans, voc, meta, rep)
    rep.finish()

    # ---------------------------------------------------------------- 8. exportar
    rep.start("audio")
    out_dir = Path(cfg.output_dir) / slug(f"{meta.artist} - {meta.title}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_midi(chart, out_dir / "notes.mid")
    _export_audio(stems, mix, sr, used_audio, out_dir, lead_in, rep)
    if meta.cover:
        try:
            save_cover(meta.cover, out_dir / "album.png")
        except Exception as e:  # noqa: BLE001
            log(f"Aviso: no se pudo guardar la portada ({e})")
    rep.finish()

    # ---------------------------------------------------------------- 9. video
    rep.start("video")
    video_name = ""
    try:
        from .video import download_youtube, prepare_for_game

        for old in out_dir.glob("video.*"):
            old.unlink()
        if cfg.video_path:
            video_name = prepare_for_game(cfg.video_path, out_dir, log).name
        elif cfg.video_url and online:
            video_name = prepare_for_game(download_youtube(cfg.video_url, out_dir, log, rep.progress), out_dir, log).name
    except Exception as e:  # noqa: BLE001
        emit({"type": "warning", "msg": f"No se pudo preparar el video: {e}"})
    rep.finish(measure=False)

    _write_song_ini(out_dir, chart, meta, report, rep.duration + lead_in, cfg, lead_in, bool(video_name))
    (out_dir / "charter_info.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    log(f"Terminado: {out_dir}")
    return out_dir


# ===================================================================== transcripcion
def _stem_presence(stem: np.ndarray, mix: np.ndarray) -> tuple[float, float]:
    """(nivel relativo a la mezcla en dB cuando suena, fraccion del tiempo que suena)."""
    from ..gen.structure import activity

    m = stem.mean(axis=0)
    t, act = activity(m, SR)
    if not np.any(act):
        return -99.0, 0.0
    hop = int(SR * 0.1)
    n = len(act)
    s_db = 20 * np.log10(np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    mm = mix.mean(axis=0)
    x_db = 20 * np.log10(np.sqrt((mm[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    rel = float(np.median(s_db[act] - x_db[act]))
    return rel, float(act.mean())


# dos guitarras distintas (izquierda / derecha): antes se separaban en guitarra principal y ritmica, pero el
# modelo neuronal aprende de la pista de guitarra entera (como la chartean los oficiales) y asi acierta mas
# (R U Mine: 0,90 -> 0,91). Se genera una sola guitarra
SPLIT_TWO_GUITARS = False


def _transcribe_all(stems: dict, mix: np.ndarray, sr: int, rep: Reporter, check_pause, work: Path,
                    active: set[str] = frozenset(ALL_INSTRUMENTS)) -> dict:
    """Transcribe cada instrumento. Lo que sale del modelo (pasadas en bruto) se guarda por instrumento,
    asi una pausa no pierde lo ya hecho; la votacion y validacion se rehacen siempre (son rapidas)."""
    from . import transcription as tr

    log = rep.log
    out: dict = {"present": {}, "notes": {}, "analysis": {}, "analysis_src": {}, "two_guitars": False}
    plan = [(s, i) for s, i in (("guitar", "guitar"), ("bass", "bass"), ("piano", "keys")) if i in active]
    total_steps = 2 * len(plan) + 2 * ("drums" in active)
    k = 0
    for stem_name, inst in plan:
        audio, _ = load(stems[stem_name])
        rel, frac = _stem_presence(audio, mix)
        if inst == "keys" and _keys_with_model(audio, sr, rel, frac, out, stem_name, tr, log):
            k += 2
            rep.progress(k / max(total_steps, 1))
            check_pause()
            continue
        # umbrales de presencia: un instrumento muy flojo o casi ausente no se chartea
        min_rel = {"guitar": -20, "bass": -22, "keys": -16}[inst]
        present = rel > min_rel and frac > 0.12
        # teclado muy flojo en la mezcla: casi siempre son restos de otros instrumentos -> no se chartea
        doubtful = inst == "keys" and present and rel < -11
        present = present and not doubtful
        out["present"][inst] = {"present": bool(present), "rel_db": rel, "active": frac, "doubtful": bool(doubtful)}
        log(f"{inst}: nivel {rel:.0f} dB, suena el {frac * 100:.0f}% del tiempo -> "
            f"{'se chartea' if present else 'no hay (no se crea pista)'}{' (demasiado flojo: seguramente restos de otros instrumentos)' if doubtful else ''}")
        if not present:
            k += 2
            rep.progress(k / max(total_steps, 1))
            continue
        res = _cached(work / f"pasadas_{inst}.pkl",
                      lambda: tr.transcribe_pitched(audio, sr, log=log, check_pause=check_pause, passes="full"))
        sa = tr.analyze_stem(audio.mean(axis=0), sr)
        runs = res["runs"]
        notes = tr.ensemble_notes(runs, min_votes=0.5)
        prog_ok = _program_filter(inst)
        notes = [n for n in notes if prog_ok(n.program) or n.votes >= 0.75]
        before = len(notes)
        notes = tr.validate_and_measure(notes, sa)
        log(f"  {inst}: {before} notas votadas -> {len(notes)} con energia real en la pista")
        out["notes"][inst] = notes
        out["analysis"][inst] = sa
        out["analysis_src"][inst] = (stem_name, None)
        if inst == "guitar" and SPLIT_TWO_GUITARS and res["left"] is not None:
            two = _two_guitar_parts(res["left"], res["right"], log)
            if two:
                out["two_guitars"] = True
                lead_ch, rhythm_ch = two
                ch_audio = {0: audio[0], 1: audio[1]}
                for role, ch in (("guitar_lead", lead_ch), ("rhythm", rhythm_ch)):
                    run = res["left"] if ch == 0 else res["right"]
                    sa_c = tr.analyze_stem(ch_audio[ch], sr)
                    shifted = res["runs"][-1]
                    ns = tr.ensemble_notes([run, shifted if role == "guitar_lead" else run], min_votes=0.5)
                    ns = tr.validate_and_measure([n for n in ns if prog_ok(n.program)], sa_c)
                    out["notes"][role] = ns
                    out["analysis"][role] = sa_c
                    out["analysis_src"][role] = (stem_name, ch)
        k += 2
        rep.progress(k / max(total_steps, 1))
        check_pause()
    if "drums" not in active:
        tr.unload()
        rep.progress(1.0)
        return out
    # bateria: piezas separadas + YourMT3 como segunda opinion
    audio, _ = load(stems["drums"])
    rel, frac = _stem_presence(audio, mix)
    present = rel > -22 and frac > 0.15
    out["present"]["drums"] = {"present": bool(present), "rel_db": rel, "active": frac}
    log(f"drums: nivel {rel:.0f} dB, suena el {frac * 100:.0f}% -> {'se chartea' if present else 'no hay'}")
    if present:
        log("  transcripcion de bateria (YourMT3)")
        out["drum_mt3"] = _cached(work / "pasadas_drums.pkl", lambda: tr.transcribe_drums(audio, sr, log))
    tr.unload()
    rep.progress(1.0)
    return out


def _anchor_times(trans: dict) -> np.ndarray:
    """Momentos en los que un charter pondria nota, para ajustar los pulsos: los de la guitarra o, si no hay, el teclado."""
    from ..calibration import neural_keys, neural_onset
    from ..calibration.onset_model import compact_spectrum

    for key, mod in (("guitar_lead", neural_onset), ("guitar", neural_onset), ("keys", neural_keys)):
        sa = trans["analysis"].get(key)
        nets = mod.load_models() if sa is not None else None
        if nets:
            spec = compact_spectrum(sa)
            prob = mod.predict_all(nets, mod._inputs(spec["cqt"], spec["onset"]))[0][:, 0]
            return neural_onset.peak_times(prob, neural_onset.peaks(prob, nets[0][1]["thr"]))
    return np.zeros(0)


def _keys_audio(stems: dict, sr: int, work: Path) -> None:
    """Muchos teclados son sintetizadores y la separacion los deja en "other": lo que oye el modelo de teclado (y lo
    que suena como keys.ogg) es entonces piano + resto. Sin modelo entrenado se queda solo el piano."""
    from ..calibration.neural_keys import MODEL_PATH, USE_MIX

    if not (USE_MIX and MODEL_PATH.exists() and "other" in stems):
        return
    path = work / "teclado_y_resto.wav"
    if not path.exists():
        a, b = load(stems["piano"])[0], load(stems["other"])[0]
        n = min(a.shape[1], b.shape[1])
        save(path, a[:, :n] + b[:, :n], sr)
    stems["piano"] = path


def _keys_with_model(audio, sr, rel, frac, out, stem_name, tr, log) -> bool:
    """Teclado con su modelo: dice las notas y si de verdad hay teclado o solo restos de otros instrumentos.
    Devuelve False si no hay modelo entrenado (entonces se usa la transcripcion normal)."""
    from ..calibration.neural_keys import load_model, notes_from_model

    net = load_model()
    if net is None:
        return False
    sa = tr.analyze_stem(audio.mean(axis=0), sr)
    notes, rate = notes_from_model(sa, log)
    present = rate >= net[1].get("presencia", 15.0)
    out["present"]["keys"] = {"present": bool(present), "rel_db": rel, "active": frac, "notas_por_min": rate}
    log(f"keys: nivel {rel:.0f} dB, {rate:.0f} notas por minuto -> {'se chartea' if present else 'no hay (no se crea pista)'}")
    if present:
        out["notes"]["keys"] = notes
        out["analysis"]["keys"] = sa
        out["analysis_src"]["keys"] = (stem_name, None)
    return True


def _program_filter(inst: str):
    """En una pista ya separada la clase de instrumento que da el modelo no es fiable (una guitarra limpia
    sale como piano u organo), asi que solo se descartan las voces (fugas de la voz en la pista)."""
    voice = set(range(52, 55)) | {65, 85}
    return lambda p: p not in voice


def _two_guitar_parts(left, right, log):
    """Decide si izquierda y derecha tocan cosas distintas (dos guitarras). Devuelve (canal_lead, canal_ritmica)."""
    def onsets(ns):
        return np.array(sorted(n.onset for n in ns))

    L, R = onsets(left), onsets(right)
    if len(L) < 50 or len(R) < 50:
        return None
    end = max(L[-1], R[-1])
    differ, active = 0, 0
    for a in np.arange(0, end, 10.0):
        l = L[(L >= a) & (L < a + 10)]
        r = R[(R >= a) & (R < a + 10)]
        if len(l) < 5 or len(r) < 5:
            continue
        active += 1
        hits = sum(np.min(np.abs(r - x)) < 0.05 for x in l)
        f1 = 2 * hits / (len(l) + len(r))
        if f1 < 0.45:
            differ += 1
    if active < 6 or differ / active < 0.35:
        return None

    def leadness(ns):
        ps = np.array([n.pitch for n in ns])
        return np.mean(ps) + np.std(ps) * 0.5 + len(set(ps)) * 0.05

    lead = 0 if leadness(left) >= leadness(right) else 1
    log(f"  dos guitarras distintas detectadas ({differ}/{active} tramos diferentes): "
        f"principal = canal {'izquierdo' if lead == 0 else 'derecho'}")
    return lead, 1 - lead


# ===================================================================== voz
def _vocals_analysis(stems, meta, cfg: JobConfig, online: bool, rep: Reporter, check_pause) -> dict:
    import librosa

    from ..gen.vocals import track_pitch
    from .lyrics import Line, Lyrics, align_lyrics, detect_language, fetch_lrclib, trim_lyrics, whisper_transcribe, whisper_words

    log = rep.log
    lead, _ = load(stems["lead_vocals"])
    rel = 20 * np.log10(np.sqrt(np.mean(lead ** 2)) + 1e-9)
    if rel < -45:
        log("No se detecta voz: la cancion es instrumental")
        return {"present": False}
    lead_m = lead.mean(axis=0)
    v16 = librosa.resample(lead_m, orig_sr=SR, target_sr=16000).astype(np.float32)
    lyr = None
    if cfg.lyrics_text.strip():
        lyr = Lyrics([Line(l.strip()) for l in cfg.lyrics_text.splitlines() if l.strip()], "usuario")
    elif online:
        log("Buscando la letra en internet (LRCLIB)...")
        lyr = fetch_lrclib(meta.artist, meta.title, meta.album, rep.duration, log)
        if lyr:
            log(f"  letra encontrada ({'sincronizada' if lyr.source == 'lrclib-synced' else 'sin tiempos'}, "
                f"{len(lyr.lines)} lineas)")
    rep.progress(0.15)
    check_pause()
    lang = detect_language(v16)
    log(f"  idioma de la voz: {lang}")
    if lyr is None:
        log("  letra no encontrada: se transcribe con Whisper")
        lyr = whisper_transcribe(v16, lang, log)
    lyr.language = lang
    if lyr.source.startswith("lrclib"):
        # la letra de internet puede ser de otra version (videoclip cortado, otra intro): se escucha la voz y se
        # quitan las lineas del principio o del final que no se cantan en este audio
        try:
            lyr = trim_lyrics(lyr, whisper_words(v16, lang, log), v16, log)
        except Exception as e:  # noqa: BLE001
            log(f"  no se pudo comparar la letra con lo que se oye ({e}); se usa entera")
    rep.progress(0.45)
    check_pause()
    lyr = align_lyrics(lyr, v16, log)
    rep.progress(0.75)
    check_pause()
    log("  analizando el tono de la voz (CREPE)...")
    times, midi, per = track_pitch(lead_m, SR, log)
    rep.progress(1.0)
    return {"present": True, "lyrics": lyr, "pitch": (times, midi, per)}


# ===================================================================== chart
def _first_sound(trans, voc) -> float:
    firsts = []
    for inst, ns in trans["notes"].items():
        if ns:
            firsts.append(min(n.onset for n in ns))
    for n in trans.get("drum_mt3", [])[:1]:
        firsts.append(n.onset)
    if voc.get("present"):
        for l in voc["lyrics"].lines:
            if l.words:
                firsts.append(l.words[0].start)
                break
    return min(firsts) if firsts else 0.0


def _build_chart(stems, mix, sr, grid, trans, voc, meta, rep: Reporter):
    from ..gen import drums as gdrums
    from ..gen.frets import generate_fret_track
    from ..gen.prokeys import generate_pro_keys, generate_pro_keys_neural
    from ..gen.structure import activity, add_star_power, find_sections, find_solos
    from ..gen.vocals import generate_vocal_track
    from .beats import build_tempo_map, fit_grid_to_notes

    log = rep.log
    # el analisis de cada pista se recalcula aqui (es rapido) para no depender de la cache
    from .transcription import analyze_stem

    for key, (stem_name, ch) in trans.get("analysis_src", {}).items():
        a = load(stems[stem_name])[0]
        trans["analysis"][key] = analyze_stem(a.mean(axis=0) if ch is None else a[ch], sr)
    grid = fit_grid_to_notes(grid, _anchor_times(trans), log)
    first = _first_sound(trans, voc)
    lead_in = float(max(0.0, 3.0 - first))
    if lead_in > 0:
        log(f"Se anaden {lead_in:.1f} s de silencio al principio para que de tiempo a ver las primeras notas")
    tm, lead_in = build_tempo_map(grid, lead_in)
    chart = Chart(tempo=tm, name=meta.title)
    duration = mix.shape[1] / sr + lead_in
    chart.end_tick = int(tm.sec_to_tick(duration)) + RES
    report: dict = {"instrumentos": {}, "lead_in": lead_in, "present": trans["present"]}
    used_audio = []

    # el bajo, aunque no se chartee: sirve para reconocer sus golpes cuando se cuelan en la guitarra
    bass_sa = trans["analysis"].get("bass")
    if bass_sa is None and "bass" in stems and trans["notes"].get("guitar"):
        bass_sa = analyze_stem(load(stems["bass"])[0].mean(axis=0), sr)

    voc_mono = load(stems.get("lead_vocals") or stems["vocals"])[0].mean(axis=0)
    vocal_act = activity(voc_mono, sr, rel_db=-22)
    guitar_audio = load(stems["guitar"])[0].mean(axis=0)
    guitar_act = activity(guitar_audio, sr)

    # secciones
    try:
        chart.sections = find_sections(mix.mean(axis=0), sr, tm, lead_in, vocal_act, chart.end_tick, log)
    except Exception as e:  # noqa: BLE001
        log(f"Aviso: no se pudieron detectar las secciones ({e})")

    # guitarra / rhythm / bajo / teclado
    solos = []
    gnotes = trans["notes"].get("guitar_lead") or trans["notes"].get("guitar")
    if gnotes:
        solos = find_solos([n.onset for n in gnotes], guitar_act, vocal_act)
        if solos:
            log(f"  solos de guitarra: {', '.join(f'{a:.0f}-{b:.0f}s' for a, b in solos)}")
    plan = []
    if trans["notes"].get("guitar_lead"):
        plan += [("guitar", "guitar_lead", "guitar"), ("rhythm", "rhythm", "guitar")]
    elif trans["notes"].get("guitar"):
        plan.append(("guitar", "guitar", "guitar"))
    if trans["notes"].get("bass"):
        plan.append(("bass", "bass", "bass"))
    if trans["notes"].get("keys"):
        plan.append(("keys", "keys", "keys"))
    for track, src, inst in plan:
        log(f"Generando {track}...")
        ft, info = generate_fret_track(trans["notes"][src], trans["analysis"][src], tm, inst, lead_in,
                                       solos_hint=solos if track == "guitar" else None, log=log,
                                       bass_sa=bass_sa if inst == "guitar" else None)
        if not ft.has_notes():
            log(f"  {track}: muy pocas notas, se omite")
            continue
        key = "bass" if inst == "bass" else ("keys" if inst == "keys" else "guitar")
        add_star_power(ft, tm, key, avoid=ft.solos)
        setattr(chart, track, ft)
        pitches = info.pop("event_pitches", {})
        canon, beats = info.pop("canon", None), info.pop("beats", None)
        report["instrumentos"][track] = info
        log(f"  {track}: {len(ft.notes['expert'])} notas en experto ({info['nps']:.1f}/s, "
            f"{info['chord_ratio'] * 100:.0f}% acordes)")
        if track == "keys":
            from ..calibration.neural_keys import load_model as load_keys_model

            if load_keys_model() is not None:
                chart.pro_keys = generate_pro_keys_neural(ft, trans["analysis"][src], tm, lead_in, canon, beats)
                log(f"  pro keys: {len({n.tick for n in chart.pro_keys.notes['expert']})} golpes en experto")
            else:
                chart.pro_keys = generate_pro_keys(ft, pitches)
            used_audio.append("piano")
        if track == "guitar":
            used_audio.append("guitar")
        if track == "bass":
            used_audio.append("bass")

    # bateria
    if trans["present"].get("drums", {}).get("present"):
        log("Generando bateria...")
        from ..calibration import neural_drums

        # modelo neuronal (aprendido de los charts de bateria de la biblioteca) si esta entrenado; si no, por piezas
        nh = None
        if neural_drums.MODEL_PATH.exists():
            a_d, sr_d = load(stems["drums"])
            nh = neural_drums.hits_from_audio(a_d.mean(axis=0), sr_d)
        if nh is not None:
            cls_of = (gdrums.KICK, gdrums.SNARE, gdrums.TOM_HI, gdrums.HIHAT, gdrums.TOM_MID, gdrums.RIDE,
                      gdrums.TOM_FLOOR, gdrums.CRASH)
            hits = [gdrums.Hit(t, cls_of[k], p, 1.0) for t, k, p in nh]
            log(f"  bateria (modelo neuronal): {len(hits)} golpes")
        else:
            pieces = {}
            import librosa

            for p in ("kick", "snare", "toms", "hh", "ride", "crash"):
                a = load(stems[f"drum_{p}"])[0].mean(axis=0)
                pieces[p] = librosa.resample(a, orig_sr=sr, target_sr=gdrums.SR)
            hits = gdrums.detect_hits(pieces, trans.get("drum_mt3", []), log)
        dt, dinfo = gdrums.generate_drum_track(hits, tm, lead_in, log)
        if dt.has_notes():
            fake = _drums_as_fret(dt)
            add_star_power(fake, tm, "drums")
            dt.star_power = fake.star_power
            chart.drums = dt
            used_audio.append("drums")
            report["instrumentos"]["drums"] = dinfo
            log(f"  bateria: {len(dt.notes['expert'])} golpes en experto, {dinfo.get('double_kick', 0)} de doble bombo")

    # voz
    if voc.get("present"):
        log("Generando voz...")
        times, midi, per = voc["pitch"]
        vt, vinfo = generate_vocal_track(voc["lyrics"], times, midi, per, tm, lead_in, log)
        if vt.has_notes():
            chart.vocals = vt
            used_audio.append("vocals")
            report["instrumentos"]["vocals"] = vinfo
            log(f"  voz: {vinfo['notes']} notas en {vinfo['phrases']} frases (letra: {vinfo['source']}), "
                f"{vinfo['unpitched']} sin tono")

    return chart, report, used_audio, lead_in


def _drums_as_fret(dt):
    """Adaptador para reutilizar el colocador de Star Power con la bateria."""
    from ..chart.model import FretNote, FretTrack

    ft = FretTrack()
    for d in DIFFICULTIES:
        seen = set()
        lst = []
        for n in dt.notes[d]:
            if n.tick not in seen and not n.double_kick:
                seen.add(n.tick)
                lst.append(FretNote(n.tick, (n.pad,)))
        ft.notes[d] = lst
    return ft


# ===================================================================== audio
def _export_audio(stems, mix, sr, used, out_dir: Path, lead_in: float, rep: Reporter) -> None:
    """Exporta las pistas de los instrumentos charteados y el resto como song.ogg (suman la mezcla exacta)."""
    names = {"guitar": "guitar.ogg", "bass": "bass.ogg", "drums": "drums.ogg", "vocals": "vocals.ogg",
             "piano": "keys.ogg"}
    for f in names.values():
        p = out_dir / f
        if p.exists():
            p.unlink()
    rest = mix.copy()
    tmp = out_dir / "_tmp.wav"
    for i, s in enumerate(used):
        a, _ = load(stems[s])
        n = min(a.shape[1], rest.shape[1])
        rest[:, :n] -= a[:, :n]
        save(tmp, a, sr)
        export_ogg(tmp, out_dir / names[s], pad_seconds=lead_in)
        rep.progress((i + 1) / (len(used) + 1))
    save(tmp, rest, sr)
    export_ogg(tmp, out_dir / "song.ogg", pad_seconds=lead_in)
    tmp.unlink()
    rep.progress(1.0)


def _tier(nps: float, lo: float, hi: float) -> int:
    return int(np.clip(round((nps - lo) / (hi - lo) * 6), 0, 6))


def _write_song_ini(out_dir, chart: Chart, meta, report, length_s, cfg: JobConfig, lead_in, has_video) -> None:
    inst = report["instrumentos"]

    def diff(track, lo, hi):
        if track not in inst:
            return -1
        return _tier(inst[track].get("nps", 0) if track != "drums" else
                     len(chart.drums.notes["expert"]) / max(length_s, 1), lo, hi)

    preview = 30000
    for tick, name in chart.sections:
        if name.startswith("Estribillo"):
            preview = int(chart.tempo.tick_to_sec(tick) * 1000)
            break
    vals = {
        "name": meta.title,
        "artist": meta.artist,
        "album": meta.album,
        "genre": meta.genre,
        "year": meta.year,
        "album_track": meta.track or None,
        "charter": "Charter (IA)",
        "song_length": int(length_s * 1000),
        "preview_start_time": preview,
        "delay": 0,
        "hopo_frequency": 170,
        "pro_drums": True,
        "five_lane_drums": False,
        "diff_band": -1,
        "diff_guitar": diff("guitar", 1.5, 7.5),
        "diff_rhythm": diff("rhythm", 1.5, 7.5),
        "diff_bass": diff("bass", 1.0, 6.5),
        "diff_drums": diff("drums", 2.0, 11.0),
        "diff_drums_real": diff("drums", 2.0, 11.0),
        "diff_keys": diff("keys", 1.0, 6.0),
        "diff_keys_real": diff("keys", 1.0, 6.0) if chart.pro_keys.has_notes() else -1,
        "diff_vocals": 3 if "vocals" in inst else -1,
        "diff_vocals_harm": -1,
        "diff_guitar_real": -1,
        "diff_bass_real": -1,
        "loading_phrase": "Chart generado automaticamente con Charter",
        "icon": "",
    }
    if has_video:
        # el audio lleva `lead_in` segundos de silencio al principio: el video tiene que esperar lo mismo
        vals["video_start_time"] = int(round(cfg.video_offset_ms - lead_in * 1000))
    bands = [v for k, v in vals.items() if k.startswith("diff_") and k not in ("diff_band",) and isinstance(v, int) and v >= 0]
    vals["diff_band"] = int(round(np.mean(bands))) if bands else -1
    write_ini(out_dir / "song.ini", vals)
