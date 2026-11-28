"""Ventana principal de Charter."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, i18n, paths
from ..pipeline.runner import JobConfig
from .jobs import AUDIO_EXT, JobQueue
from .theme import ACCENT, BAD, GOOD, MUTED, STYLE, WARN
from ..i18n import t as tr

SETTINGS = paths.CONFIG_DIR / "ajustes.json"
STATUS_ES = {"pendiente": tr("En cola"), "procesando": tr("Procesando"), "pausada": tr("En pausa"), "terminada": tr("Terminada"),
             "error": "Error", "cancelada": tr("Cancelada")}
STATUS_COLOR = {"terminada": GOOD, "error": BAD, "pausada": WARN, "procesando": ACCENT}


def fmt_eta(s: float) -> str:
    if s <= 0:
        return "—"
    s = int(s)
    if s >= 3600:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    if s >= 60:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s} s"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_settings(d: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


class DropLabel(QLabel):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(tr("Arrastra aquí tus canciones (mp3, flac, wav, m4a...)\no pulsa «Añadir canciones» o «Añadir de YouTube»"))
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("hover", True)
            self.style().polish(self)

    def dragLeaveEvent(self, e):  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)

    def dropEvent(self, e):  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)
        files = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.is_dir():
                files += [f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT]
            elif p.suffix.lower() in AUDIO_EXT:
                files.append(p)
        self.files_dropped.emit(files)


class _CalibrationThread(QThread):
    log = Signal(str)
    done = Signal(int)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        from ..calibration import drum_reducer_model, reducer_model
        from ..calibration.reference import analyze_folder

        res = analyze_folder(self.folder, paths.CONFIG_DIR / "reference_stats.json", log=self.log.emit)
        n = res["aggregate"].get("n_songs", 0)
        if n >= 20:
            # con suficientes canciones, se reentrenan tambien los modelos de reduccion de dificultad
            self.log.emit(tr("Entrenando el modelo de dificultades con tus charts (puede tardar)..."))
            try:
                reducer_model.save(reducer_model.train(self.folder, log=lambda *_: None),
                                   paths.CONFIG_DIR / "reducer.pkl")
                drum_reducer_model.save(drum_reducer_model.train(self.folder, log=lambda *_: None),
                                        paths.CONFIG_DIR / "drum_reducer.pkl")
                reducer_model._CACHE = None
                drum_reducer_model._CACHE = None
                self.log.emit(tr("Modelo de dificultades entrenado con tu biblioteca."))
            except Exception as e:  # noqa: BLE001
                self.log.emit(tr(f"No se pudo entrenar el modelo de dificultades: {e}"))
        self.done.emit(n)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr(f"Charter {__version__} — charts para YARG y Clone Hero"))
        self.resize(1280, 860)
        self.settings = load_settings()
        self.queue = JobQueue(self)
        self.queue.changed.connect(self._job_changed)
        self.queue.log.connect(self._log)
        self.queue.finished_all.connect(self._all_done)
        self.visualizers = []
        self._build()
        self._update_buttons()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_total)
        self.timer.start(1000)

    def _change_lang(self):
        self.settings["idioma"] = self.lang_cb.currentData()
        save_settings(self.settings)
        # texto en los dos idiomas: todavía no se ha reiniciado
        QMessageBox.information(self, "Charter", "Reinicia Charter para cambiar el idioma.\n"
                                                 "Restart Charter to change the language.")

    # ------------------------------------------------------------ interfaz
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 12, 16, 12)
        head = QHBoxLayout()
        title = QLabel("Charter")
        title.setObjectName("title")
        sub = QLabel(tr("Convierte tus canciones en charts jugables para YARG / Clone Hero: guitarra, rítmica, bajo, "
                     "batería, teclado, Pro Keys y voz, en Fácil, Medio, Difícil y Experto."))
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        tbox = QVBoxLayout()
        tbox.addWidget(title)
        tbox.addWidget(sub)
        head.addLayout(tbox, 1)
        self.lang_cb = QComboBox()
        for code, name in i18n.LANGS.items():
            self.lang_cb.addItem(name, code)
        self.lang_cb.setCurrentIndex(list(i18n.LANGS).index(i18n.LANG))
        self.lang_cb.setToolTip("Idioma / Language")
        self.lang_cb.currentIndexChanged.connect(self._change_lang)
        head.addWidget(self.lang_cb, 0, Qt.AlignTop)
        lay.addLayout(head)

        split = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        # --- izquierda: cola
        left = QVBoxLayout()
        self.drop = DropLabel()
        self.drop.files_dropped.connect(self._add_files)
        left.addWidget(self.drop)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", tr("Canción"), tr("Estado"), tr("Progreso"), tr("Tiempo restante")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setIconSize(QSize(44, 44))
        self.table.verticalHeader().setDefaultSectionSize(52)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 56)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c, w in ((2, 170), (3, 170), (4, 120)):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            self.table.setColumnWidth(c, w)
        self.table.itemSelectionChanged.connect(self._select)
        self.table.doubleClicked.connect(lambda _: self._open_visualizer())
        left.addWidget(self.table, 1)
        row = QHBoxLayout()
        add = QPushButton(tr("Añadir canciones..."))
        add.clicked.connect(self._pick_files)
        yt = QPushButton(tr("Añadir de YouTube..."))
        yt.setToolTip(tr("Descarga la música de un enlace de YouTube y, si quieres, también su vídeo para el fondo."))
        yt.clicked.connect(self._add_youtube)
        self.remove_btn = QPushButton(tr("Quitar"))
        self.remove_btn.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(yt)
        row.addWidget(self.remove_btn)
        row.addStretch()
        left.addLayout(row)
        tl.addLayout(left, 3)
        # --- derecha: detalles + salida + controles
        right = QVBoxLayout()
        det = QGroupBox(tr("Canción seleccionada"))
        dl = QVBoxLayout(det)
        crow = QHBoxLayout()
        self.cover_lbl = QLabel(tr("Sin\nportada"))
        self.cover_lbl.setFixedSize(120, 120)
        self.cover_lbl.setAlignment(Qt.AlignCenter)
        self.cover_lbl.setStyleSheet("border: 1px solid #3a3a48; border-radius: 8px; color: #777;")
        cbtns = QVBoxLayout()
        self.cover_btn = QPushButton(tr("Cambiar portada..."))
        self.cover_btn.clicked.connect(self._pick_cover)
        self.video_btn = QPushButton(tr("Video de fondo..."))
        self.video_btn.clicked.connect(self._video)
        self.video_lbl = QLabel(tr("Sin video"))
        self.video_lbl.setObjectName("subtitle")
        self.video_lbl.setWordWrap(True)
        cbtns.addWidget(self.cover_btn)
        cbtns.addWidget(self.video_btn)
        cbtns.addWidget(self.video_lbl)
        cbtns.addStretch()
        crow.addWidget(self.cover_lbl)
        crow.addLayout(cbtns, 1)
        dl.addLayout(crow)
        form = QFormLayout()
        self.f_title = QLineEdit()
        self.f_artist = QLineEdit()
        self.f_album = QLineEdit()
        self.f_year = QLineEdit()
        self.f_genre = QLineEdit()
        for lbl, w, key in ((tr("Título"), self.f_title, "title"), (tr("Artista"), self.f_artist, "artist"),
                            (tr("Álbum"), self.f_album, "album"), (tr("Año"), self.f_year, "year"),
                            (tr("Género"), self.f_genre, "genre")):
            w.setPlaceholderText(tr("automático"))
            w.editingFinished.connect(lambda w=w, key=key: self._edit_field(key, w.text()))
            form.addRow(lbl + ":", w)
        dl.addLayout(form)
        brow = QHBoxLayout()
        self.vis_btn = QPushButton(tr("Abrir visualizador"))
        self.vis_btn.clicked.connect(self._open_visualizer)
        self.folder_btn = QPushButton(tr("Abrir carpeta"))
        self.folder_btn.clicked.connect(self._open_folder)
        brow.addWidget(self.vis_btn)
        brow.addWidget(self.folder_btn)
        self.regen_btn = QPushButton(tr("Regenerar"))
        self.regen_btn.setToolTip(tr("Vuelve a crear el chart con lo último aprendido. Reutiliza la separación y la "
                                  "transcripción ya hechas, así que tarda poco."))
        self.regen_btn.clicked.connect(self._regenerate)
        brow.addWidget(self.regen_btn)
        dl.addLayout(brow)
        right.addWidget(det)
        out = QGroupBox(tr("Carpeta de salida"))
        ol = QHBoxLayout(out)
        self.out_edit = QLineEdit(self.settings.get("output_dir", str(paths.ROOT / "Canciones generadas")))
        self.out_edit.editingFinished.connect(self._save_out)
        ob = QPushButton(tr("Elegir..."))
        ob.clicked.connect(self._pick_out)
        ol.addWidget(self.out_edit, 1)
        ol.addWidget(ob)
        right.addWidget(out)
        ctl = QGroupBox(tr("Proceso"))
        cl = QVBoxLayout(ctl)
        self.step_lbl = QLabel(tr("Listo para empezar"))
        self.total_bar = QProgressBar()
        self.total_bar.setRange(0, 1000)
        self.eta_lbl = QLabel("")
        self.eta_lbl.setObjectName("subtitle")
        cl.addWidget(self.step_lbl)
        cl.addWidget(self.total_bar)
        cl.addWidget(self.eta_lbl)
        bl = QHBoxLayout()
        self.start_btn = QPushButton(tr("Generar charts"))
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        self.pause_btn = QPushButton(tr("Pausar"))
        self.pause_btn.clicked.connect(self._pause)
        self.cancel_btn = QPushButton(tr("Cancelar canción"))
        self.cancel_btn.clicked.connect(self._cancel)
        bl.addWidget(self.start_btn, 1)
        bl.addWidget(self.pause_btn)
        bl.addWidget(self.cancel_btn)
        cl.addLayout(bl)
        right.addWidget(ctl)
        cal = QPushButton(tr("Entrenar con mi biblioteca de canciones..."))
        cal.setToolTip(tr("Aprende de tus charts (por ejemplo tu carpeta de Clone Hero / YARG) cómo tienen que ser "
                       "las canciones. Puedes dejarlo varias horas; solo lee tus archivos."))
        cal.clicked.connect(self._calibrate)
        right.addWidget(cal)
        openv = QPushButton(tr("Abrir en el visualizador un chart ya generado..."))
        openv.clicked.connect(self._open_existing)
        right.addWidget(openv)
        right.addStretch()
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(420)
        tl.addWidget(rw)
        split.addWidget(top)
        # --- consola
        cons = QWidget()
        cv = QVBoxLayout(cons)
        cv.setContentsMargins(0, 6, 0, 0)
        ch = QHBoxLayout()
        cl_ = QLabel(tr("Consola (lo que está haciendo el programa en cada momento)"))
        cl_.setObjectName("subtitle")
        clr = QPushButton(tr("Limpiar"))
        clr.clicked.connect(lambda: self.console.clear())
        ch.addWidget(cl_)
        ch.addStretch()
        ch.addWidget(clr)
        cv.addLayout(ch)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        cv.addWidget(self.console)
        split.addWidget(cons)
        split.setSizes([560, 280])
        lay.addWidget(split, 1)
        self._log("info", tr(f"Charter {__version__} listo. Carpeta del programa: {paths.ROOT}"))
        self._check_gpu()

    def _check_gpu(self):
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                self._log("good", tr(f"Tarjeta grafica: {name} ({vram:.0f} GB). Se usara para la IA."))
            else:
                self._log("warn", tr("No se ha encontrado una tarjeta NVIDIA: todo ira por CPU (mucho mas lento)."))
        except Exception as e:  # noqa: BLE001
            self._log("warn", tr(f"No se pudo comprobar la GPU: {e}"))

    # ------------------------------------------------------------ cola
    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, tr("Añadir canciones"), self.settings.get("last_dir", ""),
                                                "Audio (*.mp3 *.flac *.wav *.m4a *.ogg *.opus *.aac)")
        if files:
            self.settings["last_dir"] = str(Path(files[0]).parent)
            save_settings(self.settings)
            self._add_files([Path(f) for f in files])

    def _add_files(self, files):
        from ..pipeline.metadata import from_file

        for f in files:
            if any(Path(j.config.audio_path) == Path(f) for j in self.queue.jobs):
                continue
            cfg = JobConfig(audio_path=str(f), output_dir=self.out_edit.text())
            idx = self.queue.add(cfg)
            job = self.queue.jobs[idx]
            try:
                m = from_file(f)
                job.meta = m  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                job.meta = None  # type: ignore[attr-defined]
            self.table.insertRow(idx)
            self._fill_row(idx)
            self._log("info", tr(f"Añadida: {Path(f).name}"))
        if self.table.currentRow() < 0 and self.table.rowCount():
            self.table.selectRow(0)
        self._update_buttons()

    def _add_youtube(self):
        from .youtube_dialog import YoutubeDialog

        dlg = YoutubeDialog(self)
        dlg.song_ready.connect(self._add_youtube_song)
        dlg.exec()
        dlg.deleteLater()

    def _add_youtube_song(self, song, url):
        self._add_files([Path(song.audio_path)])
        idx = next(i for i, j in enumerate(self.queue.jobs) if Path(j.config.audio_path) == Path(song.audio_path))
        job = self.queue.jobs[idx]
        if job.status != "procesando":
            cfg = job.config
            cfg.title, cfg.artist, cfg.album, cfg.year = song.title, song.artist, song.album, song.year
            cfg.video_path, cfg.video_url = song.video_path, url if song.video_path else ""
            cfg.video_offset_ms = 0
        self._fill_row(idx)
        self.table.selectRow(idx)
        self._select()

    def _fill_row(self, idx):
        job = self.queue.jobs[idx]
        m = getattr(job, "meta", None)
        cover = None
        if job.config.cover_path:
            cover = QPixmap(job.config.cover_path)
        elif m is not None and m.cover:
            img = QImage.fromData(m.cover)
            if not img.isNull():
                cover = QPixmap.fromImage(img)
        it0 = QTableWidgetItem()
        if cover is not None and not cover.isNull():
            it0.setIcon(QIcon(cover.scaled(44, 44, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)))
        self.table.setItem(idx, 0, it0)
        artist = job.config.artist or (m.artist if m else "")
        title = job.config.title or (m.title if m else Path(job.config.audio_path).stem)
        self.table.setItem(idx, 1, QTableWidgetItem(f"{artist} - {title}" if artist else title))
        st = QTableWidgetItem(STATUS_ES.get(job.status, job.status) + (f" · {job.step_name}" if job.status == "procesando" and job.step_name else ""))
        if job.status in STATUS_COLOR:
            st.setForeground(QColor(STATUS_COLOR[job.status]))
        if job.error:
            st.setToolTip(job.error)
        self.table.setItem(idx, 2, st)
        bar = self.table.cellWidget(idx, 3)
        if bar is None:
            bar = QProgressBar()
            bar.setRange(0, 1000)
            self.table.setCellWidget(idx, 3, bar)
        bar.setValue(int(job.progress * 1000))
        self.table.setItem(idx, 4, QTableWidgetItem(fmt_eta(job.eta) if job.status == "procesando" else ""))

    def _job_changed(self, idx):
        if 0 <= idx < self.table.rowCount():
            self._fill_row(idx)
        job = self.queue.jobs[idx] if 0 <= idx < len(self.queue.jobs) else None
        if job and job.status == "procesando":
            self.step_lbl.setText(f"{Path(job.config.audio_path).stem}: {job.step_name}")
        self._update_buttons()
        if self.table.currentRow() == idx:
            self._select()

    def _remove(self):
        r = self.table.currentRow()
        if r < 0:
            return
        if r == self.queue.current:
            QMessageBox.information(self, tr("Quitar"), tr("No se puede quitar la canción que se está procesando. "
                                                     "Cancélala primero."))
            return
        self.queue.remove(r)
        self.table.removeRow(r)
        self._update_buttons()

    def _selected_job(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.queue.jobs):
            return self.queue.jobs[r]
        return None

    def _select(self):
        job = self._selected_job()
        enabled = job is not None
        editable = enabled and job.status in ("pendiente", "error", "cancelada", "terminada")
        for w in (self.f_title, self.f_artist, self.f_album, self.f_year, self.f_genre, self.cover_btn, self.video_btn):
            w.setEnabled(bool(editable))
        if not job:
            return
        m = getattr(job, "meta", None)
        for w, key in ((self.f_title, "title"), (self.f_artist, "artist"), (self.f_album, "album"),
                       (self.f_year, "year"), (self.f_genre, "genre")):
            w.setText(getattr(job.config, key) or (getattr(m, key, "") if m else ""))
        pm = None
        if job.config.cover_path:
            pm = QPixmap(job.config.cover_path)
        elif m is not None and m.cover:
            pm = QPixmap.fromImage(QImage.fromData(m.cover))
        if pm is not None and not pm.isNull():
            self.cover_lbl.setPixmap(pm.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.cover_lbl.setText(tr("Sin portada\n(se buscará\nen internet)"))
        if job.config.video_path:
            self.video_lbl.setText(tr(f"Video: {Path(job.config.video_path).name}\nDesfase: {job.config.video_offset_ms} ms"))
        else:
            self.video_lbl.setText(tr("Sin video"))
        done = job.status == "terminada" and job.output
        self.vis_btn.setEnabled(bool(done))
        self.folder_btn.setEnabled(bool(done))
        self.regen_btn.setEnabled(job.status in ("terminada", "error"))

    def _edit_field(self, key, value):
        job = self._selected_job()
        if job:
            setattr(job.config, key, value.strip())
            self._fill_row(self.table.currentRow())

    def _pick_cover(self):
        job = self._selected_job()
        if not job:
            return
        f, _ = QFileDialog.getOpenFileName(self, tr("Elegir portada"), "", tr("Imagen (*.png *.jpg *.jpeg *.webp *.bmp)"))
        if f:
            job.config.cover_path = f
            self._fill_row(self.table.currentRow())
            self._select()

    def _video(self):
        job = self._selected_job()
        if not job:
            return
        from .video_dialog import VideoDialog

        dlg = VideoDialog(job.config.audio_path, job.config.video_path, job.config.video_url, job.config.video_offset_ms, self)
        if dlg.exec():
            job.config.video_path, job.config.video_url, job.config.video_offset_ms = dlg.result_values()
            self._select()

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, tr("Carpeta de salida"), self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self._save_out()

    def _save_out(self):
        self.settings["output_dir"] = self.out_edit.text()
        save_settings(self.settings)
        for j in self.queue.jobs:
            if j.status == "pendiente":
                j.config.output_dir = self.out_edit.text()

    # ------------------------------------------------------------ control
    def _start(self):
        if not self.queue.jobs:
            QMessageBox.information(self, "Charter", tr("Añade primero alguna canción."))
            return
        Path(self.out_edit.text()).mkdir(parents=True, exist_ok=True)
        for j in self.queue.jobs:
            if j.status in ("pendiente", "error", "cancelada"):
                j.config.output_dir = self.out_edit.text()
                if j.status != "pendiente":
                    j.status = "pendiente"
                    j.progress = 0
        if self.queue.paused:
            self.queue.resume()
        else:
            self.queue.start()
        self._update_buttons()

    def _pause(self):
        if self.queue.paused:
            self.queue.resume()
        else:
            self.queue.pause()
            self.step_lbl.setText(tr("En pausa — la tarjeta gráfica está libre. Pulsa «Reanudar» para seguir."))
        self._update_buttons()

    def _cancel(self):
        if self.queue.current is None:
            return
        r = QMessageBox.question(self, tr("Cancelar"), tr("¿Cancelar la canción que se está procesando?"))
        if r == QMessageBox.Yes:
            self.queue.cancel_current()
            self.queue._next()

    def _update_buttons(self):
        busy = self.queue.proc is not None
        self.start_btn.setEnabled(not busy and not self.queue.paused and bool(self.queue.jobs))
        self.pause_btn.setEnabled(busy or self.queue.paused)
        self.pause_btn.setText(tr("Reanudar") if self.queue.paused else tr("Pausar"))
        self.cancel_btn.setEnabled(busy)
        self.remove_btn.setEnabled(bool(self.queue.jobs))

    def _refresh_total(self):
        jobs = self.queue.jobs
        if not jobs:
            self.total_bar.setValue(0)
            return
        cur = self.queue.current
        if cur is not None:
            job = jobs[cur]
            self.total_bar.setValue(int(job.progress * 1000))
            remaining = job.eta
            pending = [j for j in jobs if j.status in ("pendiente", "pausada") and j is not job]
            # estimacion de las que faltan: lo que tardo de media por segundo de audio
            from ..pipeline.runner import load_costs

            per_sec = sum(load_costs().values())
            for j in pending:
                try:
                    from ..audio import duration

                    if not j.duration:
                        j.duration = duration(j.config.audio_path)
                    remaining += per_sec * j.duration
                except Exception:  # noqa: BLE001
                    remaining += per_sec * 240
            self.eta_lbl.setText(tr(f"Esta canción: {fmt_eta(job.eta)}   ·   Toda la cola: {fmt_eta(remaining)}"))
            self._fill_row(cur)

    def _all_done(self):
        n_ok = sum(j.status == "terminada" for j in self.queue.jobs)
        n_err = sum(j.status == "error" for j in self.queue.jobs)
        self.step_lbl.setText(tr(f"Cola terminada: {n_ok} listas") + (tr(f", {n_err} con error") if n_err else ""))
        self.total_bar.setValue(1000)
        self.eta_lbl.setText("")
        self._update_buttons()
        QApplication.alert(self)

    # ------------------------------------------------------------ resultados
    def _open_visualizer(self):
        job = self._selected_job()
        if not job or job.status != "terminada":
            return
        from .visualizer import VisualizerWindow

        w = VisualizerWindow(job.output)
        w.show()
        self.visualizers.append(w)

    def _regenerate(self):
        r = self.table.currentRow()
        if r < 0:
            return
        self._log("info", tr(f"Regenerando {Path(self.queue.jobs[r].config.audio_path).name} con lo último aprendido..."))
        self.queue.regenerate(r)
        self._update_buttons()

    def _open_existing(self):
        d = QFileDialog.getExistingDirectory(self, tr("Carpeta de la canción (con notes.mid)"), self.out_edit.text())
        if not d:
            return
        if not (Path(d) / "notes.mid").exists():
            QMessageBox.warning(self, tr("Visualizador"), tr("En esa carpeta no hay un notes.mid."))
            return
        from .visualizer import VisualizerWindow

        w = VisualizerWindow(d)
        w.show()
        self.visualizers.append(w)

    def _open_folder(self):
        job = self._selected_job()
        if job and job.output:
            os.startfile(job.output)  # noqa: S606

    def _calibrate(self):
        from .train_dialog import TrainDialog

        if self.queue.proc is not None:
            r = QMessageBox.question(self, tr("Entrenar"), tr("Se están generando canciones. Entrenar a la vez hará que "
                                                       "las dos cosas vayan más lentas. ¿Abrir igualmente?"))
            if r != QMessageBox.Yes:
                return
        dlg = TrainDialog(self.settings, self)
        dlg.on_finished = lambda: self._log("good", tr("Entrenamiento terminado: las próximas canciones usarán lo aprendido."))
        dlg.exec()
        save_settings(self.settings)

    def _reset_calibration(self):
        from ..gen.common import calibration

        calibration.cache_clear()

    # ------------------------------------------------------------ consola
    def _log(self, level: str, msg: str):
        colors = {"info": "#d8d8e0", "warn": WARN, "error": BAD, "good": GOOD, "raw": MUTED}
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colors.get(level, "#d8d8e0")))
        cur = self.console.textCursor()
        cur.movePosition(QTextCursor.End)
        stamp = time.strftime("%H:%M:%S")
        cur.insertText(f"[{stamp}] {tr(msg)}\n", fmt)
        self.console.setTextCursor(cur)
        self.console.ensureCursorVisible()

    def closeEvent(self, ev):  # noqa: N802
        if self.queue.proc is not None or self.queue.paused:
            r = QMessageBox.question(self, tr("Salir"), tr("Hay canciones en proceso. Si cierras se cancelará todo lo que "
                                                    "no haya terminado. ¿Salir?"))
            if r != QMessageBox.Yes:
                ev.ignore()
                return
        self.queue.stop_all()
        for w in self.visualizers:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(ev)


def _install_excepthook() -> None:
    """Cualquier error inesperado se guarda en config/errores.log y se muestra (pythonw no tiene consola)."""
    import traceback

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(paths.CONFIG_DIR / "errores.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{text}")
        except Exception:  # noqa: BLE001
            pass
        try:
            QMessageBox.critical(None, tr("Error inesperado"), tr(f"{value}\n\nDetalles guardados en config/errores.log"))
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def _collect_garbage_on_main_thread(app: QApplication) -> None:
    """Python libera la memoria en el hilo que este corriendo en ese momento. Si es un hilo de descarga o el del audio
    y entre lo liberado hay objetos de Qt, el programa se cierra de golpe: por eso se libera solo desde aqui."""
    gc.disable()

    def collect():
        counts, limits = gc.get_count(), gc.get_threshold()
        gen = 2 if counts[2] > limits[2] else 1 if counts[1] > limits[1] else 0 if counts[0] > limits[0] else -1
        if gen >= 0:
            gc.collect(gen)

    timer = QTimer(app)
    timer.timeout.connect(collect)
    timer.start(500)


def main() -> int:
    paths.setup_env()
    _install_excepthook()
    app = QApplication(sys.argv)
    _collect_garbage_on_main_thread(app)
    app.setApplicationName("Charter")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    return app.exec()
