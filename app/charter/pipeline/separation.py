"""Separacion de pistas con varios modelos RoFormer (ensemble) para maxima calidad.

Resultado (WAV 44.1 kHz estereo en la carpeta de trabajo):
  vocals.wav        voz completa (ensemble de 3 modelos)
  lead_vocals.wav   voz principal (sin coros)
  guitar.wav, bass.wav, drums.wav, piano.wav, other.wav  (BS-RoFormer SW, 2 pasadas)
  drum_kick.wav, drum_snare.wav, drum_toms.wav, drum_hh.wav, drum_ride.wav, drum_crash.wav
Las pistas suman exactamente la mezcla original (el resto va a other.wav).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

from .. import paths
from ..audio import SR, load, save

VOCAL_MODELS = ("vocals_mel_band_roformer.ckpt", "melband_roformer_big_beta4.ckpt")
SIX_STEM_MODEL = "BS-Roformer-SW.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_becruily.ckpt"
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
ALL_MODELS = VOCAL_MODELS + (SIX_STEM_MODEL, KARAOKE_MODEL, DRUMSEP_MODEL)
SIX_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
DRUM_PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")


def model_dir() -> Path:
    d = paths.MODELS / "separator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _separator(out_dir: Path, log, overlap: int = 8, autocast: bool = False):
    import logging

    from audio_separator.separator import Separator

    return Separator(
        log_level=logging.WARNING,
        model_file_dir=str(model_dir()),
        output_dir=str(out_dir),
        output_format="WAV",
        sample_rate=SR,
        use_autocast=autocast,
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1,
                     "overlap": overlap, "pitch_shift": 0},
    )


def _run_model(model: str, src: Path, out_dir: Path, log) -> dict[str, Path]:
    """Ejecuta un modelo y devuelve {nombre_de_pista: ruta}. Si ya se ejecuto (pausa), reutiliza el resultado."""
    import json

    tmp = out_dir / f"_tmp_{Path(model).stem}_{Path(src).stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    done = tmp / "done.json"
    if done.exists():
        cached = {k: Path(v) for k, v in json.loads(done.read_text(encoding="utf-8")).items()}
        if all(p.exists() for p in cached.values()):
            log(f"  (reutilizando resultado guardado de {model})")
            return cached
    sep = _separator(tmp, log)
    sep.load_model(model_filename=model)
    files = sep.separate(str(src))
    out = {}
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = tmp / p
        name = p.stem
        # nombre tipo "mix_(Vocals)_modelo"
        if "_(" in name:
            stem = name.split("_(")[-1].split(")")[0].lower()
        else:
            stem = name.lower()
        out[stem] = p
    del sep
    _free_gpu()
    done.write_text(json.dumps({k: str(v) for k, v in out.items()}), encoding="utf-8")
    return out


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _load_fix(path: Path, n: int) -> np.ndarray:
    a, sr = load(path)
    if a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    if a.shape[1] < n:
        a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
    return a[:, :n]


def separate(mix_wav: Path, work: Path, log=print, progress=lambda f: None, check_pause=lambda: None,
             full: bool = True, drum_parts: bool = True) -> dict[str, Path]:
    """Separa la mezcla. Con full=False solo hace lo necesario para guitarra/bajo/teclado (sin voz principal
    ni bateria por piezas); con drum_parts=False se salta la bateria por piezas; lo hecho se reutiliza si luego
    se pide la separacion completa."""
    work = Path(work)
    stems_dir = work / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    core_names = ("vocals", "guitar", "bass", "drums", "piano", "other")
    final = {name: stems_dir / f"{name}.wav" for name in core_names + ("lead_vocals", "backing_vocals")}
    final.update({f"drum_{p}": stems_dir / f"drum_{p}.wav" for p in DRUM_PARTS})
    core = {k: final[k] for k in core_names}
    need_core = not all(p.exists() for p in core.values())
    need_lead = full and not (final["lead_vocals"].exists() and final["backing_vocals"].exists())
    need_drums = full and drum_parts and not all(final[f"drum_{p}"].exists() for p in DRUM_PARTS)
    done_stems = lambda: {k: p for k, p in final.items() if p.exists()} if full else core  # noqa: E731
    if not (need_core or need_lead or need_drums):
        log("Separacion ya hecha (reutilizando)")
        return done_stems()
    mix, sr = load(mix_wav)
    n = mix.shape[1]
    steps = (len(VOCAL_MODELS) + 2) * need_core + need_lead + need_drums
    done = 0
    if need_core:
        _separate_core(mix, sr, n, mix_wav, work, final, log, progress, check_pause, steps)
        done = len(VOCAL_MODELS) + 2
    if need_lead:
        _separate_lead(load(final["vocals"])[0], n, work, final, log, check_pause)
        done += 1
        progress(done / steps)
    if need_drums:
        _separate_drums(n, sr, work, final, log, check_pause)
        done += 1
    progress(1.0)
    _cleanup(work)
    return done_stems()


def _separate_core(mix, sr, n, mix_wav, work, final, log, progress, check_pause, steps) -> None:
    done = 0

    # 1) Voz: ensemble de modelos especializados + la voz del modelo de 6 pistas
    vocal_estimates = []
    for m in VOCAL_MODELS:
        check_pause()
        log(f"Separando voz con {m}...")
        res = _run_model(m, mix_wav, work, log)
        key = "vocals" if "vocals" in res else next(k for k in res if "voc" in k)
        vocal_estimates.append(_load_fix(res[key], n))
        done += 1
        progress(done / steps)

    # 2) Seis pistas sobre la mezcla y sobre la mezcla sin voz (dos opiniones)
    check_pause()
    log("Separando instrumentos (BS-RoFormer SW, pasada 1: mezcla completa)...")
    res_mix = _run_model(SIX_STEM_MODEL, mix_wav, work, log)
    done += 1
    progress(done / steps)
    six_mix = {s: _load_fix(res_mix[s], n) for s in SIX_STEMS if s in res_mix}
    vocal_estimates.append(six_mix.get("vocals", np.zeros_like(mix)))
    vocals = np.mean(vocal_estimates, axis=0)
    inst = mix - vocals
    inst_wav = work / "instrumental.wav"
    save(inst_wav, inst, sr)
    check_pause()
    log("Separando instrumentos (pasada 2: mezcla sin voz)...")
    res_inst = _run_model(SIX_STEM_MODEL, inst_wav, work, log)
    done += 1
    progress(done / steps)
    six_inst = {s: _load_fix(res_inst[s], n) for s in SIX_STEMS if s in res_inst}
    inst_stems = {}
    for s in ("drums", "bass", "guitar", "piano"):
        a = six_mix.get(s)
        b = six_inst.get(s)
        inst_stems[s] = (a + b) / 2 if a is not None and b is not None else (a if a is not None else b)
    # el resto (lo que no es voz ni los 4 instrumentos) va a "other" para que todo sume la mezcla
    other = inst - sum(inst_stems.values())
    save(final["vocals"], vocals, sr)
    for s, a in inst_stems.items():
        save(final[s], a, sr)
    save(final["other"], other, sr)
    del six_mix, six_inst, vocal_estimates
    gc.collect()
    try:
        os.remove(inst_wav)
    except OSError:
        pass


def _separate_lead(vocals, n, work, final, log, check_pause) -> None:
    sr = SR
    # 3) Voz principal vs coros
    check_pause()
    log("Separando voz principal de los coros...")
    try:
        res = _run_model(KARAOKE_MODEL, final["vocals"], work, log)
        lead_key = next((k for k in res if k in ("vocals", "lead", "lead vocals", "karaoke")), None)
        back_key = next((k for k in res if k in ("instrumental", "backing", "other", "no vocals")), None)
        if lead_key is None:
            lead_key = sorted(res)[0]
        lead = _load_fix(res[lead_key], n)
        save(final["lead_vocals"], lead, sr)
        save(final["backing_vocals"], vocals - lead, sr)
    except Exception as e:  # noqa: BLE001
        log(f"Aviso: no se pudo separar la voz principal ({e}); se usa la voz completa")
        save(final["lead_vocals"], vocals, sr)
        save(final["backing_vocals"], np.zeros_like(vocals), sr)


def _separate_drums(n, sr, work, final, log, check_pause) -> None:
    # 4) Bateria por piezas
    check_pause()
    log("Separando la bateria por piezas (bombo, caja, toms, charles, ride, crash)...")
    res = _run_model(DRUMSEP_MODEL, final["drums"], work, log)
    for p in DRUM_PARTS:
        key = next((k for k in res if k == p or k.startswith(p)), None)
        if key is None:
            save(final[f"drum_{p}"], np.zeros((2, n), np.float32), sr)
        else:
            save(final[f"drum_{p}"], _load_fix(res[key], n), sr)


def _cleanup(work: Path) -> None:
    # limpiar temporales
    for d in work.glob("_tmp_*"):
        for f in d.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass
