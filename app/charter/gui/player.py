"""Reproductor de audio multipista (para el visualizador y el ajuste del video)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


class MultiTrackPlayer:
    """Mezcla varias pistas en tiempo real; cada una se puede silenciar."""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.tracks: dict[str, np.ndarray] = {}
        self.muted: set[str] = set()
        self.volume = 0.9
        self.pos = 0  # en muestras
        self.playing = False
        self.length = 0
        self._stream = None
        self._lock = threading.Lock()
        self._t_ref = (0.0, 0)  # (time.perf_counter, pos) para posicion suave

    def load_files(self, files: dict[str, Path]) -> None:
        import soundfile as sf

        from ..audio import decode_to_wav

        self.stop()
        tracks = {}
        for name, p in files.items():
            p = Path(p)
            if not p.exists():
                continue
            try:
                data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            except Exception:  # noqa: BLE001
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    wav = decode_to_wav(p, Path(td) / "x.wav", self.sr)
                    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            if sr != self.sr:
                import soxr

                data = soxr.resample(data, sr, self.sr).astype(np.float32)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            tracks[name] = data[:, :2]
        with self._lock:
            self.tracks = tracks
            self.length = max((len(t) for t in tracks.values()), default=0)
            self.pos = 0

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            out = np.zeros((frames, 2), np.float32)
            if self.playing:
                a = self.pos
                b = a + frames
                for name, t in self.tracks.items():
                    if name in self.muted or a >= len(t):
                        continue
                    seg = t[a:min(b, len(t))]
                    out[: len(seg)] += seg
                self.pos = b
                self._t_ref = (time.perf_counter(), a)
                if self.pos >= self.length:
                    self.playing = False
            np.clip(out * self.volume, -1, 1, out=out)
            outdata[:] = out

    def _ensure_stream(self) -> None:
        if self._stream is None:
            import sounddevice as sd

            self._stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                           callback=self._callback, blocksize=1024, latency="low")
            self._stream.start()

    def play(self) -> None:
        self._ensure_stream()
        with self._lock:
            if self.pos >= self.length:
                self.pos = 0
            self.playing = True
            self._t_ref = (time.perf_counter(), self.pos)

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self.pos = int(np.clip(seconds, 0, self.length / self.sr) * self.sr)
            self._t_ref = (time.perf_counter(), self.pos)

    def position(self) -> float:
        """Posicion actual en segundos (interpolada para que la animacion sea suave)."""
        with self._lock:
            if not self.playing:
                return self.pos / self.sr
            t0, p0 = self._t_ref
            latency = self._stream.latency if self._stream is not None else 0.0
            return max(0.0, (p0 / self.sr) + (time.perf_counter() - t0) - latency)

    def duration(self) -> float:
        return self.length / self.sr

    def stop(self) -> None:
        with self._lock:
            self.playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
