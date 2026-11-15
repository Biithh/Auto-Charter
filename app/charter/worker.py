"""Proceso de trabajo (se lanza aparte para poder pausar liberando toda la VRAM).

Uso: python -m charter.worker <job.json>
Escribe en stdout una linea JSON por evento: log, progress, warning, done, error, paused.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path


def main() -> int:
    from charter import paths

    paths.setup_env()
    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.WARNING)
    job_file = Path(sys.argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    out = sys.stdout
    # las librerias escriben cosas por stdout: las mandamos a stderr para no mezclarlas con los eventos
    sys.stdout = sys.stderr

    def emit(ev: dict) -> None:
        out.write(json.dumps(ev, ensure_ascii=False) + "\n")
        out.flush()

    from charter.pipeline.runner import JobConfig, Paused, process_song

    cfg = JobConfig(**job["config"])
    work = Path(job["work_dir"])
    pause_flag = work / "PAUSA"
    try:
        result = process_song(cfg, work, emit, pause_flag)
        emit({"type": "done", "output": str(result)})
        return 0
    except Paused:
        emit({"type": "paused"})
        return 3
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "msg": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    import io
    import os

    # con pythonw no siempre hay stdout: lo abrimos sobre el descriptor que nos pasa el GUI
    if sys.stdout is None:
        try:
            sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", buffering=0), encoding="utf-8", line_buffering=True)
        except OSError:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        try:
            sys.stderr = io.TextIOWrapper(os.fdopen(2, "wb", buffering=0), encoding="utf-8", line_buffering=True)
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
    # stdout sin buffer y en UTF-8 aunque Windows use otra codificacion
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
