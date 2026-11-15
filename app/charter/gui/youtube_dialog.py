"""Dialogo para anadir canciones desde enlaces de YouTube: se descargan de una en una, con su video de fondo o sin el."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class _Cancelled(Exception):
    pass


class _Downloader(QThread):
    status = Signal(str)
    progress = Signal(float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, url: str, with_video: bool):
        super().__init__()
        self.url, self.with_video = url, with_video
        self.stop = False

    def _progress(self, f: float) -> None:
        # yt-dlp llama a esto mientras descarga: el error corta la descarga
        if self.stop:
            raise _Cancelled()
        self.progress.emit(f)

    def run(self):
        try:
            from ..pipeline.youtube import download_song

            self.done.emit(download_song(self.url, self.with_video, log=self.status.emit, progress=self._progress))
        except _Cancelled:
            self.failed.emit("")
        except Exception as e:  # noqa: BLE001
            self.failed.emit("" if self.stop else str(e))


@dataclass
class _Item:
    url: str
    with_video: bool
    row: int


class YoutubeDialog(QDialog):
    song_ready = Signal(object, str)  # (cancion descargada, enlace), en cuanto termina cada una

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Añadir de YouTube")
        self.resize(780, 520)
        self.pending: list[_Item] = []
        self.current: _Item | None = None
        self.total = 0
        self.dl: _Downloader | None = None
        lay = QVBoxLayout(self)
        self.urls = QPlainTextEdit()
        self.urls.setPlaceholderText("Pega aquí uno o varios enlaces de YouTube, uno por línea...")
        self.urls.setFixedHeight(90)
        lay.addWidget(self.urls)
        row = QHBoxLayout()
        self.with_video = QRadioButton("Música y vídeo (de fondo en el juego)")
        self.with_video.setChecked(True)
        self.only_audio = QRadioButton("Solo la música")
        self.add_btn = QPushButton("Añadir a las descargas")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self._add_links)
        row.addWidget(self.with_video)
        row.addWidget(self.only_audio)
        row.addStretch()
        row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Enlace", "Descargar", "Estado"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c, w in ((1, 130), (2, 240)):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            self.table.setColumnWidth(c, w)
        lay.addWidget(self.table, 1)
        self.status = QLabel("Se descargan de una en una y cada canción pasa a la lista al terminar. Puedes añadir "
                             "más enlaces mientras tanto, y luego cambiar o quitar el vídeo con «Video de fondo...».")
        self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        btns = QHBoxLayout()
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self.close_btn)
        lay.addLayout(btns)

    def _add_links(self):
        links = self.urls.toPlainText().split()
        if not links:
            return
        video = self.with_video.isChecked()
        for url in links:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(url))
            self.table.setItem(r, 1, QTableWidgetItem("Música y vídeo" if video else "Solo la música"))
            self._set_state(r, "En espera")
            self.pending.append(_Item(url, video, r))
        self.total += len(links)
        self.urls.clear()
        self._next()

    def _set_state(self, row: int, text: str, tip: str = ""):
        it = QTableWidgetItem(text)
        it.setToolTip(tip or text)
        self.table.setItem(row, 2, it)

    def _next(self):
        if self.dl is not None:
            return
        if not self.pending:
            self.current = None
            self.bar.setVisible(False)
            self.close_btn.setText("Cerrar")
            self.status.setText(f"Descargas terminadas: {self._count('Añadida')} añadidas a la lista"
                                + (f", {self._count('Error')} con error (pon el ratón encima para ver el motivo)."
                                   if self._count("Error") else "."))
            return
        self.current = self.pending.pop(0)
        self.close_btn.setText("Cancelar descargas")
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.table.scrollToItem(self.table.item(self.current.row, 0))
        self.status.setText(f"Descargando {self.total - len(self.pending)} de {self.total}...")
        self.dl = _Downloader(self.current.url, self.current.with_video)
        self.dl.status.connect(self._on_status)
        self.dl.progress.connect(self._on_progress)
        self.dl.done.connect(self._on_done)
        self.dl.failed.connect(self._on_failed)
        self.dl.start()

    def _count(self, state: str) -> int:
        return sum(self.table.item(r, 2).text().startswith(state) for r in range(self.table.rowCount()))

    def _on_status(self, text: str):
        if self.current is not None:
            self._set_state(self.current.row, text)
        self.bar.setValue(0)

    def _on_progress(self, f: float):
        self.bar.setValue(int(f * 100))

    def _on_done(self, song):
        item = self._finish_thread()
        self._set_state(item.row, "Añadida", f"{song.artist} - {song.title}" if song.artist else song.title)
        self.song_ready.emit(song, item.url)
        self._after_download()

    def _on_failed(self, msg: str):
        stopped = self.dl.stop
        item = self._finish_thread()
        if stopped:
            self._set_state(item.row, "Cancelada")
        else:
            short = re.sub(r"^ERROR:\s*(\[[^\]]*\]\s*)?", "", msg)
            self._set_state(item.row, f"Error: {short}", msg)
        self._after_download()

    def _finish_thread(self) -> _Item:
        self.dl.wait()
        self.dl = None
        return self.current

    def _after_download(self):
        if self._closing:
            super().reject()
        else:
            self._next()

    _closing = False

    def reject(self):
        # el hilo de descarga no puede quedarse suelto: primero se corta y al terminar se cierra la ventana
        if self.dl is not None:
            self._closing = True
            self.dl.stop = True
            for it in self.pending:
                self._set_state(it.row, "Cancelada")
            self.pending.clear()
            self.status.setText("Cancelando...")
            self.close_btn.setEnabled(False)
            self.add_btn.setEnabled(False)
            return
        super().reject()
