"""Ventana "Entrenar con mi biblioteca": lanza el entrenamiento en un proceso aparte y muestra su avance."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .. import paths
from ..i18n import t as tr


class TrainDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Entrenar con mi biblioteca de canciones"))
        self.resize(780, 620)
        self.settings = settings
        self.proc: QProcess | None = None
        self._buf = b""
        self.on_finished = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            tr("Charter aprende de tus charts cómo tienen que ser las canciones: cuántas notas, cuántos acordes, "
            "HOPOs, Star Power y cómo se simplifican en Difícil, Medio y Fácil (vale para todas las canciones, "
            "aunque solo tengan chart). Con las que traen las pistas separadas (guitar.ogg, bass.ogg...) "
            "aprende además qué golpes del audio se convierten en nota y cuándo poner acordes.\n\n"
            "Solo se leen tus archivos, no se modifica nada. Lo analizado se guarda: si lo paras, la próxima "
            "vez sigue donde lo dejó."))
        intro.setWordWrap(True)
        intro.setObjectName("subtitle")
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.folder = QLineEdit(settings.get("library_dir", ""))
        self.folder.setPlaceholderText(tr("Carpeta de tus canciones (por ejemplo la de YARG / Clone Hero)"))
        pick = QPushButton(tr("Elegir..."))
        pick.clicked.connect(self._pick)
        row.addWidget(self.folder, 1)
        row.addWidget(pick)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(tr("Tiempo máximo (horas):")))
        self.hours = QDoubleSpinBox()
        self.hours.setRange(0.25, 24)
        self.hours.setSingleStep(0.5)
        self.hours.setValue(float(settings.get("train_hours", 4)))
        row2.addWidget(self.hours)
        self.unlimited = QCheckBox(tr("Sin límite (hasta que lo analice todo; puede tardar días)"))
        self.unlimited.setChecked(bool(settings.get("train_unlimited", False)))
        self.unlimited.toggled.connect(lambda on: self.hours.setEnabled(not on))
        self.hours.setEnabled(not self.unlimited.isChecked())
        row2.addWidget(self.unlimited)
        row2.addStretch()
        lay.addLayout(row2)
        self.sep = QCheckBox(tr("Aprovechar todo el tiempo con la IA de separación: separa las canciones sin pistas y "
                             "vuelve a separar las que sí las traen, para aprender con el mismo audio que verá al "
                             "generar (usa la tarjeta gráfica, ~1-2 min por canción)"))
        self.sep.setToolTip(tr("Recomendado. Sin esta opción, cuando acaba con las canciones que traen pistas pasa "
                            "directamente a entrenar aunque quede tiempo."))
        self.sep.setChecked(bool(settings.get("train_separate", False)))
        lay.addWidget(self.sep)
        self.phase = QLabel(tr("Listo"))
        lay.addWidget(self.phase)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        lay.addWidget(self.console, 1)
        btns = QHBoxLayout()
        self.reset_btn = QPushButton(tr("Volver a lo de serie"))
        self.reset_btn.setToolTip(tr("Borra lo aprendido de tu biblioteca y vuelve a los modelos que trae el programa"))
        self.reset_btn.clicked.connect(self._reset)
        self.start_btn = QPushButton(tr("Empezar a entrenar"))
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        self.finish_btn = QPushButton(tr("Terminar ya y entrenar"))
        self.finish_btn.setToolTip(tr("Deja de analizar canciones y entrena los modelos con todo lo analizado hasta ahora"))
        self.finish_btn.setEnabled(False)
        self.finish_btn.clicked.connect(self._finish_now)
        self.stop_btn = QPushButton(tr("Parar"))
        self.stop_btn.setToolTip(tr("Para del todo sin entrenar (lo analizado se guarda para la próxima vez)"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.reset_btn)
        btns.addStretch()
        btns.addWidget(self.finish_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.start_btn)
        lay.addLayout(btns)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, tr("Carpeta de tus canciones"), self.folder.text())
        if d:
            self.folder.setText(d)

    def _log(self, msg: str):
        self.console.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {tr(msg)}")

    def _start(self):
        folder = self.folder.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, tr("Entrenar"), tr("Elige la carpeta de tus canciones."))
            return
        self.settings["library_dir"] = folder
        self.settings["train_hours"] = self.hours.value()
        self.settings["train_separate"] = self.sep.isChecked()
        self.settings["train_unlimited"] = self.unlimited.isChecked()
        hours = 0 if self.unlimited.isChecked() else self.hours.value()
        job = paths.CONFIG_DIR / "entrenamiento" / "trabajo.json"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(json.dumps({"folder": folder, "hours": hours, "separate": self.sep.isChecked()}),
                       encoding="utf-8")
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(paths.APP))
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("CHARTER_ROOT", str(paths.ROOT))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.readyReadStandardError.connect(lambda: self.proc and self.proc.readAllStandardError())
        self.proc.finished.connect(self._finished)
        python = sys.executable
        if python.lower().endswith("python.exe") and Path(python[:-10] + "pythonw.exe").exists():
            python = python[:-10] + "pythonw.exe"
        self.proc.start(python, ["-m", "charter.calibration.library_train", str(job)])
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.finish_btn.setEnabled(True)
        self._deadline = None if hours == 0 else time.time() + hours * 3600
        self._t0 = time.time()
        self._phase_name = tr("Empezando")
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._update_clock)
        self._clock.start(5000)
        self._log(tr("Entrenamiento iniciado"))

    def _update_clock(self):
        if self.proc is None:
            self._clock.stop()
            return
        if self._deadline is None:
            el = int(time.time() - self._t0)
            self.phase.setText(tr(f"{self._phase_name}  ·  sin límite · lleva {el // 3600} h {(el % 3600) // 60:02d} min"))
            return
        rem = max(int(self._deadline - time.time()), 0)
        self.phase.setText(tr(f"{self._phase_name}  ·  tiempo máximo restante: {rem // 3600} h {(rem % 3600) // 60:02d} min"))

    def _read(self):
        if self.proc is None:
            return
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            try:
                ev = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "log":
                self._log(ev["msg"])
            elif t == "progress":
                self.bar.setValue(int(ev.get("frac", 0) * 1000))
                self._phase_name = ev.get("phase", "")
                self._update_clock()
            elif t == "done":
                self.bar.setValue(1000)
                self.phase.setText(tr("Terminado"))
            elif t == "error":
                self._log("ERROR: " + ev.get("msg", ""))
                for l in (ev.get("trace") or "").splitlines()[-8:]:
                    self._log(l)

    def _finished(self, *_):
        self._read()
        self.proc = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
        self._reload_models()
        if self.on_finished:
            self.on_finished()

    def _finish_now(self):
        from ..calibration.library_train import FINISH_FLAG

        FINISH_FLAG.parent.mkdir(parents=True, exist_ok=True)
        FINISH_FLAG.write_text("1")
        self.finish_btn.setEnabled(False)
        self._log(tr("Terminando: acaba la canción que está analizando y entrena con todo lo analizado..."))

    def _stop(self):
        if self.proc is None:
            return
        try:
            import psutil

            p = psutil.Process(self.proc.processId())
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self._log(tr("Parado. Lo analizado queda guardado: la próxima vez seguirá desde aquí."))

    def _reset(self):
        r = QMessageBox.question(self, tr("Volver a lo de serie"), tr("¿Borrar lo aprendido de tu biblioteca?"))
        if r != QMessageBox.Yes:
            return
        for name in ("reference_stats.json", "reducer.pkl", "drum_reducer.pkl", "onset_model.pkl", "notas_neuronal.pt",
                     "notas_neuronal_anterior.pt", "notas_neuronal_anterior2.pt"):
            p = paths.CONFIG_DIR / name
            if p.exists():
                p.unlink()
        self._reload_models()
        self._log(tr("Se han restaurado los modelos de serie (lo analizado de tu biblioteca sigue guardado)."))

    def _reload_models(self):
        from ..calibration import drum_reducer_model, onset_model, reducer_model
        from ..gen.common import calibration

        calibration.cache_clear()
        reducer_model._CACHE = None
        drum_reducer_model._CACHE = None
        onset_model._CACHE = None

    def closeEvent(self, ev):  # noqa: N802
        if self.proc is not None:
            r = QMessageBox.question(self, tr("Entrenando"), tr("El entrenamiento sigue en marcha. ¿Pararlo y cerrar?"))
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            self._stop()
        super().closeEvent(ev)
