"""Cola de canciones: lanza el proceso de trabajo, lee su progreso y gestiona pausas."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .. import paths
from ..pipeline.runner import JobConfig
from ..i18n import t as tr

AUDIO_EXT = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".wma")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# avisos de onnxruntime al no encontrar CUDA 13 (solo lo usa el modelo de pulso, que entonces va por CPU): inofensivos
_NOISE = re.compile(r"Failed to load (cu|nv)|onnxruntime\.ai/docs|install CUDA|installed PyTorch .* uses CUDA")
_BAR = re.compile(r"(\d{1,3})%\|")


@dataclass
class Job:
    config: JobConfig
    status: str = "pendiente"  # pendiente, procesando, pausada, terminada, error, cancelada
    progress: float = 0.0
    step_name: str = ""
    eta: float = 0.0
    output: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def work_dir(self) -> Path:
        p = Path(self.config.audio_path)
        try:
            st = p.stat()
            key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            key = str(p)
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return paths.WORK / f"{p.stem[:40]}_{h}"


class JobQueue(QObject):
    changed = Signal(int)  # indice del trabajo que ha cambiado
    log = Signal(str, str)  # (nivel, mensaje)
    finished_all = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: list[Job] = []
        self.proc: QProcess | None = None
        self.current: int | None = None
        self.running = False
        self.paused = False
        self._buf = b""
        self._bar_last = 0
        self._bar_shown = -1

    # ------------------------------------------------------------ cola
    def add(self, cfg: JobConfig) -> int:
        self.jobs.append(Job(cfg))
        return len(self.jobs) - 1

    def remove(self, idx: int) -> None:
        if idx == self.current and self.proc is not None:
            return
        job = self.jobs.pop(idx)
        self._cleanup(job)
        if self.current is not None and idx < self.current:
            self.current -= 1

    def start(self) -> None:
        self.running = True
        self.paused = False
        self._next()

    def _next(self) -> None:
        if not self.running or self.paused or self.proc is not None:
            return
        for i, j in enumerate(self.jobs):
            if j.status in ("pendiente", "pausada"):
                self._launch(i)
                return
        self.running = False
        self.current = None
        self.finished_all.emit()

    def _launch(self, idx: int) -> None:
        try:
            self._launch_inner(idx)
        except Exception as e:  # noqa: BLE001
            # si algo falla al lanzar, la cola no se queda bloqueada
            job = self.jobs[idx]
            job.status = "error"
            job.error = tr(f"No se pudo lanzar el proceso: {e}")
            self.proc = None
            self.current = None
            self.log.emit("error", job.error)
            self.changed.emit(idx)

    def _launch_inner(self, idx: int) -> None:
        job = self.jobs[idx]
        job.status = "procesando"
        job.error = ""
        work = job.work_dir
        work.mkdir(parents=True, exist_ok=True)
        flag = work / "PAUSA"
        if flag.exists():
            flag.unlink()
        job_file = work / "job.json"
        job_file.write_text(json.dumps({"config": asdict(job.config), "work_dir": str(work)}, ensure_ascii=False),
                            encoding="utf-8")
        self.current = idx
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(paths.APP))
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("CHARTER_ROOT", str(paths.ROOT))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._read_stdout)
        self.proc.readyReadStandardError.connect(self._read_stderr)
        self.proc.finished.connect(self._finished)
        # pythonw: sin ventana de consola; la salida llega por las tuberias de QProcess
        python = sys.executable
        if python.lower().endswith("python.exe") and Path(python[:-10] + "pythonw.exe").exists():
            python = python[:-10] + "pythonw.exe"
        self.proc.start(python, ["-m", "charter.worker", str(job_file)])
        name = Path(job.config.audio_path).name
        self.log.emit("info", tr(f"===== Empezando: {name} =====") if job.progress == 0 else tr(f"===== Reanudando: {name} ====="))
        self.changed.emit(idx)

    # ------------------------------------------------------------ pausa
    def pause(self) -> None:
        """Pausa inmediata: se cierra el proceso (libera toda la VRAM); lo hecho queda guardado."""
        self.paused = True
        if self.proc is not None and self.current is not None:
            job = self.jobs[self.current]
            job.status = "pausada"
            job.step_name = tr("En pausa")
            self.log.emit("warn", tr("Pausa: cerrando el proceso para liberar la tarjeta grafica..."))
            self._kill_proc()
            self.changed.emit(self.current)

    def resume(self) -> None:
        self.paused = False
        self.running = True
        self.log.emit("info", tr("Reanudando (se recargan los modelos, puede tardar unos segundos)..."))
        self._next()

    def cancel_current(self) -> None:
        if self.proc is not None and self.current is not None:
            job = self.jobs[self.current]
            job.status = "cancelada"
            self._kill_proc()
            self._cleanup(job)
            self.changed.emit(self.current)

    def stop_all(self) -> None:
        """Al cerrar el programa: se cancela todo y se borran los temporales."""
        self.running = False
        self._kill_proc()
        for j in self.jobs:
            self._cleanup(j)
        shutil.rmtree(paths.WORK / "youtube", ignore_errors=True)

    def regenerate(self, idx: int) -> None:
        """Vuelve a generar el chart reutilizando lo ya calculado (separacion, transcripcion...)."""
        job = self.jobs[idx]
        if job.status == "procesando":
            return
        job.status = "pendiente"
        job.progress = 0.0
        job.error = ""
        self.changed.emit(idx)
        if not self.running:
            self.start()
        else:
            self._next()

    def _kill_proc(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            proc.finished.disconnect(self._finished)
        except Exception:  # noqa: BLE001
            pass
        pid = proc.processId()
        try:
            import psutil

            p = psutil.Process(pid)
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except Exception:  # noqa: BLE001
            proc.kill()
        proc.waitForFinished(5000)

    def _cleanup(self, job: Job) -> None:
        try:
            if job.work_dir.exists():
                shutil.rmtree(job.work_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ salida del proceso
    def _read_stdout(self) -> None:
        if self.proc is None:
            return
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                self.log.emit("raw", text)
                continue
            self._handle(ev)

    def _read_stderr(self) -> None:
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in re.split(r"[\r\n]+", data):
            line = _ANSI.sub("", line).rstrip()
            if not line or "warn" in line.lower() or line.strip().startswith("warnings.") or _NOISE.search(line):
                continue
            # barras de progreso de las librerias: solo al empezar, cada 25 % y al terminar
            m = _BAR.search(line)
            if m:
                pct = int(m.group(1))
                if pct < self._bar_last:
                    self._bar_shown = -1  # empieza una barra nueva
                self._bar_last = pct
                bucket = 100 if pct == 100 else pct // 25 * 25
                if bucket <= self._bar_shown:
                    continue
                self._bar_shown = bucket
                line = "  " + line.strip()
            self.log.emit("raw", line[-300:])

    def _handle(self, ev: dict) -> None:
        if self.current is None:
            return
        job = self.jobs[self.current]
        t = ev.get("type")
        if t == "log":
            self.log.emit("info", ev["msg"])
        elif t == "warning":
            job.warnings.append(ev["msg"])
            self.log.emit("warn", "AVISO: " + ev["msg"])
        elif t == "progress":
            job.progress = ev.get("overall", 0)
            job.step_name = tr(ev.get("step_name", ""))
            job.eta = ev.get("eta", 0)
            self.changed.emit(self.current)
        elif t == "done":
            job.status = "terminada"
            job.output = ev.get("output", "")
            job.progress = 1.0
            job.eta = 0
            job.step_name = tr("Terminada")
            self.log.emit("good", tr(f"Listo: {job.output}"))
        elif t == "error":
            job.status = "error"
            job.error = tr(ev.get("msg", ""))
            self.log.emit("error", "ERROR: " + job.error)
            for l in (ev.get("trace") or "").splitlines()[-12:]:
                self.log.emit("raw", l)
        elif t == "paused":
            job.status = "pausada"

    def _finished(self, code, status) -> None:
        idx = self.current
        self._read_stdout()
        self.proc = None
        if idx is not None:
            job = self.jobs[idx]
            if job.status == "procesando":
                job.status = "error"
                job.error = job.error or tr(f"El proceso termino de forma inesperada (codigo {code})")
                self.log.emit("error", job.error)
            # al terminar se conserva lo pesado (pistas separadas, transcripcion) para poder regenerar el
            # chart en un momento; se borra al cerrar el programa o al quitar la cancion de la lista
            self.changed.emit(idx)
        self.current = None
        self._next()
