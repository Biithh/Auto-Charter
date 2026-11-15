"""Descarga de todos los modelos de IA a la carpeta models/ (la usa el instalador)."""

from __future__ import annotations

import os

from . import paths


def download_all(log=print) -> bool:
    ok = True
    paths.setup_env()

    # 1) separacion de pistas
    try:
        import logging

        from audio_separator.separator import Separator

        from .pipeline.separation import ALL_MODELS, model_dir

        sep = Separator(log_level=logging.WARNING, model_file_dir=str(model_dir()), info_only=False)
        for m in ALL_MODELS:
            log(f"  separacion: {m}")
            try:
                if hasattr(sep, "download_model_files"):
                    sep.download_model_files(m)
                else:
                    sep.load_model(model_filename=m)
            except Exception as e:  # noqa: BLE001
                log(f"    fallo: {e}")
                ok = False
    except Exception as e:  # noqa: BLE001
        log(f"  separacion: no disponible ({e})")
        ok = False

    # 2) pulso / compas
    try:
        from .pipeline.beats import ensure_models

        log("  pulso: Beat This (3 modelos)")
        ensure_models(log)
    except Exception as e:  # noqa: BLE001
        log(f"    fallo: {e}")
        ok = False

    # 3) transcripcion
    try:
        os.environ.setdefault("MT3_CHECKPOINT_DIR", str(paths.MODELS / "mt3"))
        from mt3_infer import download_model

        log("  transcripcion: YourMT3+")
        download_model("yourmt3")
    except Exception as e:  # noqa: BLE001
        log(f"    fallo: {e}")
        ok = False

    # 4) letra: Whisper large-v3 y alineadores (espanol e ingles; el resto se baja si hace falta)
    try:
        from huggingface_hub import snapshot_download

        from .pipeline.lyrics import ALIGN_MODELS, WHISPER_MODEL

        log("  letra: Whisper large-v3")
        snapshot_download(WHISPER_MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                          ignore_patterns=["*fp32*", "*.msgpack", "*.h5", "*.bin"])
        for lang in ("es", "en"):
            log(f"  alineador de letra ({lang})")
            snapshot_download(ALIGN_MODELS[lang], allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
    except Exception as e:  # noqa: BLE001
        log(f"    fallo: {e}")
        ok = False
    return ok
