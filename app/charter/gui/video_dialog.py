"""Dialogo para elegir el video de fondo y ajustar su desfase respecto a la cancion."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Signal, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from .. import paths
from .player import MultiTrackPlayer
from ..i18n import t as tr


class _Downloader(QThread):
    progress = Signal(float)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, out_dir: Path):
        super().__init__()
        self.url, self.out_dir = url, out_dir

    def run(self):
        try:
            from ..pipeline.video import download_youtube

            p = download_youtube(self.url, self.out_dir, log=lambda *_: None, progress=self.progress.emit)
            self.done.emit(str(p))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class VideoDialog(QDialog):
    def __init__(self, audio_path: str, video_path: str = "", video_url: str = "", offset_ms: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Video de fondo"))
        self.resize(900, 640)
        self.audio_path = audio_path
        self.video_path = video_path
        self.video_url = video_url
        self.player = MultiTrackPlayer()
        self._audio_loaded = False
        lay = QVBoxLayout(self)
        src = QHBoxLayout()
        self.url_edit = QLineEdit(video_url)
        self.url_edit.setPlaceholderText(tr("Pega aqui un enlace de YouTube..."))
        dl = QPushButton(tr("Descargar y ver"))
        dl.clicked.connect(self._download)
        pick = QPushButton(tr("Elegir archivo de video..."))
        pick.clicked.connect(self._pick)
        src.addWidget(self.url_edit, 1)
        src.addWidget(dl)
        src.addWidget(pick)
        lay.addLayout(src)
        self.status = QLabel(tr("Elige un video o pega un enlace de YouTube (se descarga sin audio, como maximo 1080p)."))
        self.status.setObjectName("subtitle")
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(360)
        lay.addWidget(self.video, 1)
        self.media = QMediaPlayer(self)
        self.media.setVideoOutput(self.video)
        ctr = QHBoxLayout()
        self.play_btn = QPushButton(tr("▶  Reproducir juntos"))
        self.play_btn.clicked.connect(self._toggle)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._seek)
        ctr.addWidget(self.play_btn)
        ctr.addWidget(self.slider, 1)
        lay.addLayout(ctr)
        off = QHBoxLayout()
        off.addWidget(QLabel(tr("Desfase del video (ms):")))
        self.offset = QSpinBox()
        self.offset.setRange(-600000, 600000)
        self.offset.setSingleStep(50)
        self.offset.setValue(offset_ms)
        self.offset.valueChanged.connect(lambda _: self._resync(force=True))
        off.addWidget(self.offset)
        for d in (-1000, -100, -10, 10, 100, 1000):
            b = QPushButton(f"{d:+d}")
            b.clicked.connect(lambda _, d=d: self.offset.setValue(self.offset.value() + d))
            off.addWidget(b)
        off.addStretch()
        lay.addLayout(off)
        help_ = QLabel(tr("Positivo = el video va por delante (se salta el principio del video). "
                       "Negativo = el video empieza mas tarde. Ajusta hasta que lo que ves coincida con lo que oyes."))
        help_.setWordWrap(True)
        help_.setObjectName("subtitle")
        lay.addWidget(help_)
        btns = QHBoxLayout()
        none = QPushButton(tr("Sin video"))
        none.clicked.connect(self._no_video)
        ok = QPushButton(tr("Aceptar"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(none)
        btns.addWidget(ok)
        lay.addLayout(btns)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._resync)
        self.timer.start(200)
        if video_path:
            self._set_video(video_path)

    def _pick(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("Elegir video"), "", "Video (*.mp4 *.webm *.mkv *.avi *.mov)")
        if f:
            self.video_url = ""
            self._set_video(f)

    def _download(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        self.status.setText(tr("Descargando video..."))
        self.bar.setVisible(True)
        self.bar.setValue(0)
        out = paths.WORK / "videos" / str(abs(hash(url)))
        self.dl = _Downloader(url, out)
        self.dl.progress.connect(lambda f: self.bar.setValue(int(f * 100)))
        self.dl.done.connect(lambda p: (self.bar.setVisible(False), setattr(self, "video_url", url), self._set_video(p)))
        self.dl.failed.connect(lambda e: (self.bar.setVisible(False), QMessageBox.warning(self, "Video", tr(f"No se pudo descargar: {e}"))))
        self.dl.start()

    def _set_video(self, path: str):
        self.video_path = path
        self.media.setSource(QUrl.fromLocalFile(path))
        self.media.setAudioOutput(None)
        self.status.setText(f"Video: {Path(path).name}")
        if not self._audio_loaded:
            self.player.load_files({"song": Path(self.audio_path)})
            self._audio_loaded = True

    def _toggle(self):
        if not self._audio_loaded:
            return
        if self.player.playing:
            self.player.pause()
            self.media.pause()
            self.play_btn.setText(tr("▶  Reproducir juntos"))
        else:
            self.player.play()
            self._resync(force=True)
            self.media.play()
            self.play_btn.setText(tr("⏸  Pausa"))

    def _seek(self, v):
        if not self._audio_loaded:
            return
        self.player.seek(v / 1000 * self.player.duration())
        self._resync(force=True)

    def _resync(self, force=False):
        if not self._audio_loaded or not self.video_path:
            return
        t = self.player.position()
        target = int(t * 1000 + self.offset.value())
        if not self.slider.isSliderDown() and self.player.duration() > 0:
            self.slider.setValue(int(t / self.player.duration() * 1000))
        if target < 0:
            if self.media.playbackState() == QMediaPlayer.PlayingState:
                self.media.pause()
            self.media.setPosition(0)
            return
        if self.player.playing and self.media.playbackState() != QMediaPlayer.PlayingState:
            self.media.play()
        if force or abs(self.media.position() - target) > 120:
            self.media.setPosition(target)

    def _no_video(self):
        self.video_path = ""
        self.video_url = ""
        self.accept()

    def result_values(self) -> tuple[str, str, int]:
        return self.video_path, self.video_url, self.offset.value()

    def done(self, r):  # noqa: D401
        self.timer.stop()
        self.player.stop()
        self.media.stop()
        super().done(r)
