"""Rutas del programa. Todo vive dentro de la carpeta raiz de Charter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CHARTER_ROOT") or Path(__file__).resolve().parents[2])
APP = ROOT / "app"
RUNTIME = ROOT / "runtime"
MODELS = ROOT / "models"
WORK = ROOT / "trabajo"
CONFIG_DIR = ROOT / "config"
DATA = APP / "charter" / "data"
FFMPEG_DIR = RUNTIME / "ffmpeg"


def ffmpeg() -> str:
    exe = FFMPEG_DIR / "ffmpeg.exe"
    return str(exe) if exe.exists() else "ffmpeg"


def ffprobe() -> str:
    exe = FFMPEG_DIR / "ffprobe.exe"
    return str(exe) if exe.exists() else "ffprobe"


def setup_env() -> None:
    """Redirige todas las caches de modelos a la carpeta del programa."""
    for d in (MODELS, WORK, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CHARTER_ROOT", str(ROOT))
    os.environ["HF_HOME"] = str(MODELS / "huggingface")
    os.environ["TORCH_HOME"] = str(MODELS / "torch")
    os.environ["XDG_CACHE_HOME"] = str(MODELS / "cache")
    os.environ["MT3_CHECKPOINT_DIR"] = str(MODELS / "mt3")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if FFMPEG_DIR.exists():
        os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
