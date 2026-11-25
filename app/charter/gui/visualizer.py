"""Visualizador: reproduce la cancion y muestra el chart como en el juego."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QPolygonF,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..chart.midi_io import read_midi, resolve_hopos, write_midi
from ..chart.model import DIFFICULTIES, DIFF_NAMES_ES, OPEN, RES, Chart
from .player import MultiTrackPlayer
from .theme import LANE_COLORS, OPEN_COLOR, PANEL, REVIEW_COLOR, SP_COLOR
from ..i18n import t as tr

TRACKS = [("guitar", tr("Guitarra")), ("rhythm", tr("Rítmica")), ("bass", tr("Bajo")), ("drums", tr("Batería")),
          ("keys", tr("Teclado")), ("pro_keys", "Pro Keys"), ("vocals", tr("Voz"))]
STEMS = [("song", tr("Resto")), ("guitar", tr("Guitarra")), ("bass", tr("Bajo")), ("drums", tr("Batería")),
         ("keys", tr("Teclado")), ("vocals", tr("Voz")), ("rhythm", tr("Rítmica")), ("crowd", tr("Público"))]
DRUM_COLORS = ["#ff8c1a", "#e8383d", "#f2d22e", "#3d8ff2", "#2ecc52"]


class Highway(QWidget):
    """Dibuja la autopista de notas."""

    seekRequested = Signal(float)
    noteClicked = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chart: Chart | None = None
        self.track = "guitar"
        self.diff = "expert"
        self.time = 0.0
        self.window_s = 1.6  # segundos visibles por delante (como la velocidad normal de YARG)
        self.review_ticks: set = set()
        self.selected = None
        self.setMinimumSize(520, 420)
        self._note_rects = []

    def set_chart(self, chart: Chart | None, review_ticks: set) -> None:
        self.chart = chart
        self.review_ticks = review_ticks
        self.update()

    # ------------------------------------------------------------ dibujo
    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0d0d12"))
        self._note_rects = []
        if self.chart is None:
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignCenter, tr("Sin chart cargado"))
            return
        if self.track == "vocals":
            self._paint_vocals(p)
        else:
            self._paint_highway(p)
        p.end()

    # ------------------------------------------------------------ perspectiva 3D
    PERSP = 2.6  # cuanto se estrecha la autopista hacia el fondo

    def _geom(self, lanes: int):
        w, h = self.width(), self.height()
        strike = h * 0.86
        horizon = h * 0.10
        base_w = min(w * 0.82, 130.0 * lanes if lanes <= 5 else w * 0.9)
        return w / 2.0, strike, horizon, base_w

    def _proj(self, t: float, lane_pos: float, lanes: int):
        """(tiempo, posicion en carriles 0..lanes) -> (x, y, escala)."""
        cx, strike, horizon, base_w = self._geom(lanes)
        z = (t - self.time) / self.window_s
        s = 1.0 / (1.0 + self.PERSP * max(z, -0.2))
        s1 = 1.0 / (1.0 + self.PERSP)
        y = strike - (strike - horizon) * (1.0 - s) / (1.0 - s1)
        x = cx + (lane_pos - lanes / 2.0) * (base_w / lanes) * s
        return x, y, s

    def _fade(self, t: float) -> int:
        z = (t - self.time) / self.window_s
        return int(255 * max(0.0, min(1.0, 1.0 - max(z - 0.75, 0) / 0.25)))

    def _y_of(self, t: float, top: float = 0, strike_y: float = 0) -> float:
        return self._proj(t, 0, 5)[1]

    def _paint_highway(self, p: QPainter):
        tm = self.chart.tempo
        w, h = self.width(), self.height()
        drums = self.track == "drums"
        pro = self.track == "pro_keys"
        lanes = 4 if drums else (25 if pro else 5)
        cx, strike, horizon, base_w = self._geom(lanes)
        # fondo tipo escenario
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor("#05050a"))
        bg.setColorAt(0.5, QColor("#11101c"))
        bg.setColorAt(1, QColor("#07070c"))
        p.fillRect(self.rect(), bg)
        glow = QRadialGradient(QPointF(cx, horizon), w * 0.45)
        glow.setColorAt(0, QColor(255, 140, 60, 45))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), glow)
        t_far = self.time + self.window_s
        t_near = self.time - 0.12
        ft = self._track_obj()
        sp = getattr(ft, "star_power", [])
        sp_now = any(tm.tick_to_sec(s.start) <= self.time < tm.tick_to_sec(s.end) for s in sp)

        def quad(t0, t1, a, b):
            x0, y0, _ = self._proj(t0, a, lanes)
            x1, _, _ = self._proj(t0, b, lanes)
            x2, y2, _ = self._proj(t1, b, lanes)
            x3, _, _ = self._proj(t1, a, lanes)
            return QPolygonF([QPointF(x0, y0), QPointF(x1, y0), QPointF(x2, y2), QPointF(x3, y2)])

        # tablero
        board = quad(t_near, t_far, -0.12, lanes + 0.12)
        g = QLinearGradient(0, horizon, 0, strike)
        g.setColorAt(0, QColor(20, 18, 30, 0))
        g.setColorAt(0.25, QColor(28, 24, 40, 230))
        g.setColorAt(1, QColor(45, 38, 60, 255) if not sp_now else QColor(30, 70, 110, 255))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawPolygon(board)
        # solos (fondo azulado)
        for s in getattr(ft, "solos", []):
            a, b = tm.tick_to_sec(s.start), tm.tick_to_sec(s.end)
            if b < t_near or a > t_far:
                continue
            p.setBrush(QColor(90, 150, 255, 45))
            p.drawPolygon(quad(max(a, t_near), min(b, t_far), 0, lanes))
        # zonas de revision (franja rosa en el borde)
        for a, b, _ in (self.parent_regions() if hasattr(self, "parent_regions") else []):
            if b < t_near or a > t_far:
                continue
            p.setBrush(QColor(REVIEW_COLOR))
            p.drawPolygon(quad(max(a, t_near), min(b, t_far), -0.32, -0.16))
        # lineas de pulso y compas
        tick = int(tm.sec_to_tick(max(t_near, 0)) // RES * RES)
        measures = set(tm.measure_ticks(int(tm.sec_to_tick(t_far)) + RES * 8))
        while tm.tick_to_sec(tick) < t_far:
            t = tm.tick_to_sec(tick)
            if t >= t_near:
                x0, y, s = self._proj(t, 0, lanes)
                x1, _, _ = self._proj(t, lanes, lanes)
                a = self._fade(t)
                strong = tick in measures
                p.setPen(QPen(QColor(255, 255, 255, int(a * (0.45 if strong else 0.16))), max(1.0, (3 if strong else 1.5) * s)))
                p.drawLine(QPointF(x0, y), QPointF(x1, y))
            tick += RES // 2 if pro else RES
        # carriles y bordes
        for i in range(0, lanes + 1):
            xa, ya, _ = self._proj(t_near, i, lanes)
            xb, yb, _ = self._proj(t_far, i, lanes)
            edge = i in (0, lanes)
            col = QColor(SP_COLOR if sp_now else "#ff8a1f") if edge else QColor(255, 255, 255, 22)
            if pro and not edge:
                col = QColor(255, 255, 255, 12)
            p.setPen(QPen(col, 4 if edge else 1))
            p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
        if drums:
            self._paint_drum_notes(p, lanes, sp)
        elif pro:
            self._paint_pro_keys(p, lanes, sp)
        else:
            self._paint_fret_notes(p, lanes, sp)
        self._paint_hud(p, lanes)

    def _track_obj(self):
        return getattr(self.chart, self.track)

    # ------------------------------------------------------------ piezas
    def _gem(self, p: QPainter, x: float, y: float, s: float, base: QColor, kind: str = "strum", alpha: int = 255,
             width: float = 1.0):
        """Gema con volumen. kind: strum | hopo | tap | cymbal | tom."""
        rw = 46 * s * width
        rh = 20 * s if kind != "cymbal" else 14 * s
        col = QColor(base)
        col.setAlpha(alpha)
        # base oscura (grosor)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, int(alpha * 0.6)))
        p.drawEllipse(QPointF(x, y + 5 * s), rw, rh)
        dark = col.darker(170)
        dark.setAlpha(alpha)
        p.setBrush(dark)
        p.drawEllipse(QPointF(x, y + 3 * s), rw, rh)
        # cara superior
        if kind == "tap":
            p.setBrush(QColor(20, 20, 25, alpha))
            p.setPen(QPen(col, max(2.0, 5 * s)))
            p.drawEllipse(QPointF(x, y), rw * 0.92, rh * 0.92)
            return
        grad = QRadialGradient(QPointF(x - rw * 0.3, y - rh * 0.4), rw * 1.2)
        light = col.lighter(150)
        light.setAlpha(alpha)
        grad.setColorAt(0, light)
        grad.setColorAt(0.6, col)
        grad.setColorAt(1, dark)
        p.setBrush(grad)
        rim = QColor(255, 255, 255, alpha) if kind == "hopo" else QColor(15, 15, 18, alpha)
        p.setPen(QPen(rim, max(1.5, (4 if kind == "hopo" else 2.5) * s)))
        p.drawEllipse(QPointF(x, y), rw, rh)
        if kind in ("strum", "tom"):
            # centro blanco tipo "boton"
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(alpha * 0.85)))
            p.drawEllipse(QPointF(x, y), rw * 0.32, rh * 0.32)
        if kind == "hopo":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(alpha * 0.95)))
            p.drawEllipse(QPointF(x, y), rw * 0.45, rh * 0.45)

    def _sustain(self, p, t0, t1, lane_pos, lanes, col: QColor, width: float = 0.18):
        t0 = max(t0, self.time)
        t1 = min(t1, self.time + self.window_s)
        if t1 <= t0:
            return
        steps = 10
        left, right = [], []
        for k in range(steps + 1):
            t = t0 + (t1 - t0) * k / steps
            xa, y, s = self._proj(t, lane_pos - width / 2, lanes)
            xb, _, _ = self._proj(t, lane_pos + width / 2, lanes)
            left.append(QPointF(xa, y))
            right.append(QPointF(xb, y))
        poly = QPolygonF(left + right[::-1])
        c = QColor(col)
        c.setAlpha(200)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(poly)

    def _paint_fret_notes(self, p, lanes, sp):
        tm = self.chart.tempo
        notes = self._track_obj().notes[self.diff]
        hopos = resolve_hopos(notes, 170)
        t_far = self.time + self.window_s

        def in_sp(tick):
            return any(s.start <= tick < s.end for s in sp)

        self._hits = []
        visible = []
        for n, hopo in zip(notes, hopos):
            t = tm.tick_to_sec(n.tick)
            if t > t_far:
                break
            t_end = tm.tick_to_sec(n.tick + n.sustain) if n.sustain else t
            if t_end < self.time - 0.15:
                continue
            visible.append((t, t_end, n, hopo))
        # primero las notas largas (debajo), luego las gemas de lejos a cerca
        for t, t_end, n, hopo in visible:
            if n.sustain and t_end > self.time:
                col = QColor(SP_COLOR) if in_sp(n.tick) else None
                for f in n.frets:
                    if f == OPEN:
                        c = col or QColor(OPEN_COLOR)
                        self._sustain(p, t, t_end, lanes / 2, lanes, c, width=lanes * 0.9)
                    else:
                        self._sustain(p, t, t_end, f + 0.5, lanes, col or QColor(LANE_COLORS[f]))
        for t, t_end, n, hopo in reversed(visible):
            if t < self.time - 0.02:
                if t >= self.time - 0.15:
                    self._hits.append((t, n.frets, n.sustain > 0))
                continue
            star = in_sp(n.tick)
            review = (self.track, self.diff, n.tick) in self.review_ticks
            a = self._fade(t)
            kind = "tap" if n.tap else ("hopo" if hopo else "strum")
            for f in n.frets:
                if f == OPEN:
                    x0, y, s = self._proj(t, 0.15, lanes)
                    x1, _, _ = self._proj(t, lanes - 0.15, lanes)
                    c = QColor(SP_COLOR if star else OPEN_COLOR)
                    c.setAlpha(a)
                    p.setPen(QPen(QColor(255, 255, 255, a) if hopo else QColor(10, 10, 10, a), max(1.5, 3 * s)))
                    p.setBrush(c)
                    r = QRectF(x0, y - 9 * s, x1 - x0, 18 * s)
                    p.drawRoundedRect(r, 9 * s, 9 * s)
                    self._note_rects.append((r, n))
                    continue
                x, y, s = self._proj(t, f + 0.5, lanes)
                base = QColor(SP_COLOR) if star else QColor(LANE_COLORS[f])
                self._gem(p, x, y, s, base, kind, a)
                r = QRectF(x - 46 * s, y - 20 * s, 92 * s, 40 * s)
                if review:
                    p.setPen(QPen(QColor(REVIEW_COLOR), max(2.0, 3 * s), Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(r.adjusted(-6 * s, -6 * s, 6 * s, 6 * s))
                if self.selected is n:
                    p.setPen(QPen(QColor("white"), 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(r.adjusted(-8, -8, 8, 8))
                self._note_rects.append((r, n))
        # notas largas que se estan tocando ahora mismo
        for t, t_end, n, hopo in visible:
            if t <= self.time <= t_end and n.sustain:
                self._hits.append((self.time, n.frets, True))

    def _paint_drum_notes(self, p, lanes, sp):
        tm = self.chart.tempo
        notes = self.chart.drums.notes[self.diff]
        t_far = self.time + self.window_s

        def in_sp(tick):
            return any(s.start <= tick < s.end for s in sp)

        self._hits = []
        visible = []
        for n in notes:
            t = tm.tick_to_sec(n.tick)
            if t > t_far:
                break
            if t < self.time - 0.15:
                continue
            visible.append((t, n))
        for t, n in reversed(visible):
            if t < self.time - 0.02:
                self._hits.append((t, (n.pad,), False))
                continue
            a = self._fade(t)
            star = in_sp(n.tick)
            review = ("drums", self.diff, n.tick) in self.review_ticks
            if n.pad == 0:
                x0, y, s = self._proj(t, 0.05, lanes)
                x1, _, _ = self._proj(t, lanes - 0.05, lanes)
                c = QColor(SP_COLOR if star else DRUM_COLORS[0])
                if n.double_kick:
                    c = c.lighter(150)
                c.setAlpha(a)
                p.setPen(QPen(QColor(10, 10, 10, a), max(1.0, 2 * s)))
                p.setBrush(c)
                r = QRectF(x0, y - 6 * s, x1 - x0, 12 * s)
                p.drawRoundedRect(r, 5 * s, 5 * s)
                self._note_rects.append((r, n))
                continue
            x, y, s = self._proj(t, n.pad - 0.5, lanes)
            base = QColor(SP_COLOR if star else DRUM_COLORS[n.pad])
            self._gem(p, x, y, s, base, "cymbal" if n.cymbal else "tom", a, width=1.15 if n.cymbal else 1.0)
            r = QRectF(x - 50 * s, y - 20 * s, 100 * s, 40 * s)
            if review:
                p.setPen(QPen(QColor(REVIEW_COLOR), max(2.0, 3 * s), Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(r)
            self._note_rects.append((r, n))

    def _paint_pro_keys(self, p, lanes, sp):
        tm = self.chart.tempo
        notes = self.chart.pro_keys.notes[self.diff]
        t_far = self.time + self.window_s
        black = {1, 3, 6, 8, 10}
        self._hits = []
        for n in notes:
            t = tm.tick_to_sec(n.tick)
            if t > t_far:
                break
            t2 = tm.tick_to_sec(n.tick + n.sustain) if n.sustain else t
            if t2 < self.time - 0.15:
                continue
            k = n.pitch - 48 + 0.5
            star = any(s.start <= n.tick < s.end for s in sp)
            col = QColor(SP_COLOR) if star else (QColor("#e6e6f0") if (n.pitch % 12) not in black else QColor("#8a7aff"))
            if n.sustain:
                self._sustain(p, t, t2, k, lanes, col, width=0.5)
            if t < self.time - 0.02:
                continue
            x0, y, s = self._proj(t, k - 0.42, lanes)
            x1, _, _ = self._proj(t, k + 0.42, lanes)
            c = QColor(col)
            c.setAlpha(self._fade(t))
            p.setPen(QPen(QColor(10, 10, 10), 1))
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x0, y - 8 * s, x1 - x0, 16 * s), 4 * s, 4 * s)

    def _paint_hud(self, p: QPainter, lanes: int):
        """Botones de la linea de golpeo (se iluminan al "tocar" la nota), destellos y racha."""
        drums = self.track == "drums"
        pro = self.track == "pro_keys"
        hits = getattr(self, "_hits", [])
        active: dict[int, float] = {}
        sustaining: set[int] = set()
        for t, frets, sus in hits:
            for f in frets:
                age = self.time - t
                active[f] = min(active.get(f, 9), age)
                if sus:
                    sustaining.add(f)
        if pro:
            x0, y, s = self._proj(self.time, 0, lanes)
            x1, _, _ = self._proj(self.time, lanes, lanes)
            p.setPen(QPen(QColor(255, 255, 255, 200), 3))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            return
        n_btn = 4 if drums else 5
        for i in range(n_btn):
            lane_pos = i + 0.5
            x, y, s = self._proj(self.time, lane_pos, lanes)
            col = QColor(DRUM_COLORS[i + 1] if drums else LANE_COLORS[i])
            key = i + 1 if drums else i
            hit_age = active.get(key)
            lit = hit_age is not None and (hit_age < 0.12 or key in sustaining)
            # boton
            p.setPen(QPen(col, 5))
            p.setBrush(QColor(15, 15, 20) if not lit else col.lighter(130))
            p.drawEllipse(QPointF(x, y), 44, 18)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 40 if not lit else 200))
            p.drawEllipse(QPointF(x, y), 16, 6)
            if hit_age is not None and hit_age < 0.18:
                # destello que se expande
                k = hit_age / 0.18
                fl = QRadialGradient(QPointF(x, y - 10), 70 + 60 * k)
                c1 = col.lighter(170)
                c1.setAlpha(int(220 * (1 - k)))
                fl.setColorAt(0, c1)
                fl.setColorAt(1, QColor(0, 0, 0, 0))
                p.setBrush(fl)
                p.drawEllipse(QPointF(x, y - 10), 70 + 60 * k, 45 + 35 * k)
                p.setPen(QPen(QColor(255, 255, 255, int(200 * (1 - k))), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x, y), 44 + 30 * k, 18 + 12 * k)
        if drums and 0 in active and active[0] < 0.12:
            x0, y, _ = self._proj(self.time, 0, lanes)
            x1, _, _ = self._proj(self.time, lanes, lanes)
            p.setPen(QPen(QColor(DRUM_COLORS[0]).lighter(140), 6))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
        # racha y multiplicador (como si lo tocara un jugador perfecto)
        streak = self._streak()
        mult = 1 + min(streak // 10, 3)
        cx, strike, _, base_w = self._geom(lanes)
        bx = cx - base_w / 2 - 120
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(bx, strike - 70, 100, 90), 12, 12)
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Segoe UI", 22, QFont.Bold))
        p.drawText(QRectF(bx, strike - 68, 100, 40), Qt.AlignCenter, f"{streak}")
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor("#b0b0c0"))
        p.drawText(QRectF(bx, strike - 34, 100, 18), Qt.AlignCenter, tr("RACHA"))
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        p.setPen(QColor(["#ffffff", "#ffd23f", "#4ec27a", "#b45cff"][mult - 1]))
        p.drawText(QRectF(bx, strike - 18, 100, 30), Qt.AlignCenter, f"x{mult}")

    def _streak(self) -> int:
        tm = self.chart.tempo
        obj = self._track_obj()
        notes = obj.notes[self.diff] if self.track != "vocals" else []
        if self.track == "drums":
            ticks = sorted(set(n.tick for n in notes))
        else:
            ticks = [n.tick for n in notes]
        now = tm.sec_to_tick(self.time)
        import bisect

        return bisect.bisect_right(ticks, now)

    def _paint_vocals(self, p: QPainter):
        tm = self.chart.tempo
        vt = self.chart.vocals
        w, h = self.width(), self.height()
        p.fillRect(QRectF(0, 40, w, h - 120), QColor(28, 28, 40))
        strike_x = w * 0.2
        span = 5.0
        lo, hi = 36, 84
        ps = [n.pitch for n in vt.notes if not n.unpitched]
        if ps:
            lo, hi = max(36, min(ps) - 2), min(84, max(ps) + 2)

        def x_of(t):
            return strike_x + (t - self.time) / span * (w - strike_x)

        def y_of(pitch):
            return 50 + (hi - pitch) / max(hi - lo, 1) * (h - 140)

        for ph in vt.phrases:
            xa, xb = x_of(tm.tick_to_sec(ph.start)), x_of(tm.tick_to_sec(ph.end))
            if xb < 0 or xa > w:
                continue
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawLine(QPointF(xa, 40), QPointF(xa, h - 80))
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(QPointF(strike_x, 40), QPointF(strike_x, h - 80))
        font = QFont("Segoe UI", 12)
        p.setFont(font)
        prev = None
        last_right = [-1e9, 0]
        sp = vt.star_power
        for n in vt.notes:
            ta, tb = tm.tick_to_sec(n.tick), tm.tick_to_sec(n.tick + n.length)
            if tb < self.time - span * 0.25 or ta > self.time + span:
                prev = n
                continue
            xa, xb = x_of(ta), x_of(tb)
            star = any(s.start <= n.tick < s.end for s in sp)
            if n.unpitched:
                y = y_of((lo + hi) / 2)
                p.setPen(QPen(QColor("#aaaaaa"), 2, Qt.DashLine))
                p.drawLine(QPointF(xa, y), QPointF(xb, y))
            else:
                y = y_of(n.pitch)
                col = QColor(SP_COLOR if star else "#38d9a9")
                if (("vocals", "", n.tick) in self.review_ticks) or n.review:
                    col = QColor(REVIEW_COLOR)
                if n.lyric == "+" and prev is not None and not prev.unpitched:
                    py = y_of(prev.pitch)
                    p.setPen(QPen(col, 8, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(QPointF(x_of(tm.tick_to_sec(prev.tick + prev.length)), py), QPointF(xa, y))
                p.setPen(QPen(col, 10, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(xa, y), QPointF(max(xb, xa + 2), y))
            if n.lyric and n.lyric != "+":
                txt = n.lyric.rstrip("#^").replace("-", "").replace("=", "-")
                active = ta <= self.time <= tb
                p.setPen(QColor("#ffffff") if active else QColor("#9a9aab"))
                tw = p.fontMetrics().horizontalAdvance(txt)
                row = 0 if xa >= last_right[0] + 3 else (1 - last_right[1])
                p.drawText(QPointF(xa, h - 50 + row * 20), txt)
                if row == 0 or xa + tw > last_right[0]:
                    last_right[0], last_right[1] = xa + tw, row
            prev = n

    # ------------------------------------------------------------ interaccion
    def mousePressEvent(self, ev):  # noqa: N802
        pos = ev.position()
        for r, n in self._note_rects:
            if r.adjusted(-6, -6, 6, 6).contains(pos):
                self.selected = n
                self.noteClicked.emit(n)
                self.update()
                return
        self.selected = None
        self.update()

    def wheelEvent(self, ev):  # noqa: N802
        delta = ev.angleDelta().y() / 120.0
        self.seekRequested.emit(max(0.0, self.time + delta * 0.5))


class VisualizerWindow(QWidget):
    def __init__(self, song_dir: str | Path, parent=None):
        super().__init__(parent, Qt.Window)
        self.song_dir = Path(song_dir)
        self.setWindowTitle(tr(f"Visualizador - {self.song_dir.name}"))
        self.resize(1180, 820)
        self.setObjectName("root")
        self.player = MultiTrackPlayer()
        self.chart = read_midi(self.song_dir / "notes.mid")
        info_p = self.song_dir / "charter_info.json"
        self.info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
        self.dirty = False
        self._build()
        self._load_audio()
        self._refresh_stats()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    # ------------------------------------------------------------ interfaz
    def _build(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        top = QHBoxLayout()
        self.track_cb = QComboBox()
        for key, name in TRACKS:
            obj = getattr(self.chart, key)
            if obj.has_notes():
                self.track_cb.addItem(name, key)
        self.diff_cb = QComboBox()
        for d in reversed(DIFFICULTIES):
            self.diff_cb.addItem(tr(DIFF_NAMES_ES[d]), d)
        self.track_cb.currentIndexChanged.connect(self._change_view)
        self.diff_cb.currentIndexChanged.connect(self._change_view)
        top.addWidget(QLabel(tr("Instrumento:")))
        top.addWidget(self.track_cb)
        top.addWidget(QLabel(tr("Dificultad:")))
        top.addWidget(self.diff_cb)
        top.addStretch()
        self.speed_cb = QComboBox()
        for s in (tr("Rápida"), "Normal", tr("Lenta")):
            self.speed_cb.addItem(s)
        self.speed_cb.setCurrentIndex(1)
        self.speed_cb.currentIndexChanged.connect(self._change_speed)
        top.addWidget(QLabel(tr("Velocidad de notas:")))
        top.addWidget(self.speed_cb)
        left.addLayout(top)
        self.highway = Highway()
        self.highway.parent_regions = lambda: []
        self.highway.seekRequested.connect(self._seek)
        self.highway.noteClicked.connect(self._note_clicked)
        left.addWidget(self.highway, 1)
        ctr = QHBoxLayout()
        self.play_btn = QPushButton(tr("▶  Reproducir"))
        self.play_btn.setObjectName("primary")
        self.play_btn.clicked.connect(self._toggle)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(lambda v: self._seek(v / 1000 * self.player.duration()))
        self.time_lbl = QLabel("0:00 / 0:00")
        ctr.addWidget(self.play_btn)
        ctr.addWidget(self.slider, 1)
        ctr.addWidget(self.time_lbl)
        left.addLayout(ctr)
        mute = QHBoxLayout()
        mute.addWidget(QLabel(tr("Pistas:")))
        self.mute_boxes = {}
        names = dict(STEMS)
        for key in self._audio_files():
            base = key.split("_")[0]
            label = names.get(key, names.get(base, key.capitalize()))
            if key != base and base in names:
                label += key[len(base):].replace("_", " ")
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, k=key: self._mute(k, not on))
            mute.addWidget(cb)
            self.mute_boxes[key] = cb
        mute.addStretch()
        left.addLayout(mute)
        hint = QLabel(tr("Espacio: reproducir/pausar · Rueda del ratón: avanzar/retroceder · Clic en una nota y Supr: quitarla"))
        hint.setObjectName("subtitle")
        left.addWidget(hint)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        t = QLabel(tr("Resumen del chart"))
        t.setStyleSheet("font-size: 13pt; font-weight: 700;")
        right.addWidget(t)
        self.save_btn = QPushButton(tr("Guardar cambios"))
        self.save_btn.setObjectName("primary")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        right.addWidget(self.save_btn)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setObjectName("subtitle")
        right.addWidget(self.stats)
        right.addStretch()
        panel = QWidget()
        panel.setLayout(right)
        panel.setStyleSheet(f"background: {PANEL}; border-radius: 10px;")
        panel.setMinimumWidth(260)
        root.addWidget(panel, 1)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_note)
        self._change_view()

    def _audio_files(self) -> dict:
        """Todas las pistas de audio de la carpeta (tambien charts de otros: drums_1, rhythm, .opus...)."""
        out = {}
        for p in sorted(self.song_dir.iterdir()):
            if p.suffix.lower() in (".ogg", ".opus", ".mp3", ".wav", ".flac") and "preview" not in p.stem.lower():
                out[p.stem.lower()] = p
        return out

    def _load_audio(self):
        self.player.load_files(self._audio_files())

    # ------------------------------------------------------------ resumen
    def _refresh_stats(self):
        self.highway.set_chart(self.chart, set())
        L = max(self.chart.tempo.tick_to_sec(self.chart.end_tick), 1.0)
        lines = []
        for k in ("guitar", "rhythm", "bass", "keys"):
            obj = getattr(self.chart, k)
            if obj.has_notes():
                n = len(obj.notes["expert"])
                lines.append(tr(f"{dict(TRACKS)[k]}: {n} notas en Experto ({n / L:.1f}/s)"))
        if self.chart.drums.has_notes():
            lines.append(tr(f"Batería: {len(self.chart.drums.notes['expert'])} golpes en Experto"))
        if self.chart.vocals.has_notes():
            lines.append(tr(f"Voz: {len(self.chart.vocals.notes)} notas"))
        self.stats.setText("\n\n".join(lines))

    def _fmt(self, s: float) -> str:
        return f"{int(s // 60)}:{int(s % 60):02d}"

    def _delete_selected_note(self):
        n = self.highway.selected
        if n is None:
            return
        track = self.highway.track
        if track in ("guitar", "rhythm", "bass", "keys", "drums"):
            obj = getattr(self.chart, track)
            d = self.highway.diff
            obj.notes[d] = [x for x in obj.notes[d] if x is not n]
            self.highway.selected = None
            self._mark_dirty()
            self.highway.update()

    def _note_clicked(self, n):
        pass

    def _mark_dirty(self):
        self.dirty = True
        self.save_btn.setEnabled(True)
        self.highway.update()

    def _save(self):
        try:
            write_midi(self.chart, self.song_dir / "notes.mid")
            (self.song_dir / "charter_info.json").write_text(json.dumps(self.info, indent=1, ensure_ascii=False),
                                                             encoding="utf-8")
            self.dirty = False
            self.save_btn.setEnabled(False)
            QMessageBox.information(self, tr("Guardado"), tr("Los cambios se han guardado en notes.mid"))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", tr(f"No se pudo guardar: {e}"))

    # ------------------------------------------------------------ reproduccion
    def _change_view(self):
        self.highway.track = self.track_cb.currentData() or "guitar"
        self.highway.diff = self.diff_cb.currentData() or "expert"
        self.diff_cb.setEnabled(self.highway.track != "vocals")
        self.highway.update()

    def _change_speed(self, i):
        self.highway.window_s = [1.1, 1.6, 2.4][i]

    def _toggle(self):
        self.player.toggle()
        self.play_btn.setText(tr("⏸  Pausa") if self.player.playing else tr("▶  Reproducir"))

    def _seek(self, t):
        self.player.seek(t)
        self.highway.time = t
        self.highway.update()

    def _mute(self, key, muted):
        if muted:
            self.player.muted.add(key)
        else:
            self.player.muted.discard(key)

    def _tick(self):
        t = self.player.position()
        self.highway.time = t
        dur = self.player.duration()
        if dur > 0 and not self.slider.isSliderDown():
            self.slider.setValue(int(t / dur * 1000))
        self.time_lbl.setText(f"{self._fmt(t)} / {self._fmt(dur)}")
        if not self.player.playing:
            self.play_btn.setText(tr("▶  Reproducir"))
        self.highway.update()

    def closeEvent(self, ev):  # noqa: N802
        if self.dirty:
            r = QMessageBox.question(self, tr("Cambios sin guardar"), tr("¿Guardar los cambios antes de cerrar?"),
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                ev.ignore()
                return
            if r == QMessageBox.Save:
                self._save()
        self.timer.stop()
        self.player.stop()
        super().closeEvent(ev)
