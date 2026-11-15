"""Utilidades de audio: decodificar con ffmpeg, cargar, guardar y exportar a OGG."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import paths

SR = 44100
_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW, **kw)


def decode_to_wav(src: str | Path, dst: str | Path, sr: int = SR) -> Path:
    """Decodifica cualquier formato (mp3, flac, m4a, wav...) a WAV estereo float."""
    dst = Path(dst)
    r = run([paths.ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", str(sr),
             "-c:a", "pcm_f32le", str(dst)])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo leer el audio: {r.stderr.decode(errors='ignore')[-500:]}")
    return dst


def load(path: str | Path, mono: bool = False) -> tuple[np.ndarray, int]:
    """Devuelve (audio [canales, muestras] float32, sr)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T
    if mono:
        data = data.mean(axis=0)
    return data, sr


def save(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    a = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), a, sr, subtype="FLOAT" if str(path).endswith(".wav") else None)


def export_ogg(src_wav: str | Path, dst_ogg: str | Path, quality: int = 7, pad_seconds: float = 0.0) -> None:
    """Exporta a Ogg Vorbis (formato habitual de YARG / Clone Hero)."""
    cmd = [paths.ffmpeg(), "-y", "-v", "error", "-i", str(src_wav)]
    if pad_seconds > 0:
        ms = int(round(pad_seconds * 1000))
        cmd += ["-af", f"adelay={ms}|{ms}"]
    cmd += ["-c:a", "libvorbis", "-q:a", str(quality), str(dst_ogg)]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo crear {dst_ogg}: {r.stderr.decode(errors='ignore')[-500:]}")


def duration(path: str | Path) -> float:
    r = run([paths.ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(path)])
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return 0.0


def rms_db(x: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    """Envolvente RMS en dB (mono)."""
    if x.ndim == 2:
        x = x.mean(axis=0)
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)
