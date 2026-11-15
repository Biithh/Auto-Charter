"""Representacion interna de un chart (independiente del formato de archivo).

Todas las posiciones van en ticks (resolucion RES por negra), con un TempoMap
para convertir a segundos.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

RES = 480  # ticks por negra (estandar Rock Band / YARG)

DIFFICULTIES = ("easy", "medium", "hard", "expert")
DIFF_NAMES_ES = {"easy": "Fácil", "medium": "Medio", "hard": "Difícil", "expert": "Experto"}

OPEN = 7  # carril especial para notas abiertas


# --------------------------------------------------------------------- tempo
@dataclass
class TempoMap:
    """Lista de cambios de tempo (tick, microsegundos por negra) y compases."""

    tempos: list[tuple[int, int]] = field(default_factory=lambda: [(0, 500000)])
    time_sigs: list[tuple[int, int, int]] = field(default_factory=lambda: [(0, 4, 4)])
    res: int = RES

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        self.tempos.sort()
        self._ticks = [t for t, _ in self.tempos]
        self._secs = []
        acc = 0.0
        prev_tick, prev_us = self.tempos[0]
        for tick, us in self.tempos:
            acc += (tick - prev_tick) * prev_us / 1e6 / self.res
            self._secs.append(acc)
            prev_tick, prev_us = tick, us

    def set(self, tempos: list[tuple[int, int]], time_sigs: list[tuple[int, int, int]] | None = None) -> None:
        self.tempos = list(tempos)
        if time_sigs is not None:
            self.time_sigs = list(time_sigs)
        self._rebuild()

    def tick_to_sec(self, tick: float) -> float:
        i = bisect.bisect_right(self._ticks, tick) - 1
        i = max(i, 0)
        t0, us = self.tempos[i]
        return self._secs[i] + (tick - t0) * us / 1e6 / self.res

    def sec_to_tick(self, sec: float) -> float:
        i = bisect.bisect_right(self._secs, sec) - 1
        i = max(i, 0)
        t0, us = self.tempos[i]
        return t0 + (sec - self._secs[i]) * 1e6 * self.res / us

    def bpm_at(self, tick: float) -> float:
        i = max(bisect.bisect_right(self._ticks, tick) - 1, 0)
        return 60e6 / self.tempos[i][1]

    def ts_at(self, tick: float) -> tuple[int, int, int]:
        best = self.time_sigs[0]
        for ts in self.time_sigs:
            if ts[0] <= tick:
                best = ts
            else:
                break
        return best

    def measure_ticks(self, end_tick: int) -> list[int]:
        """Ticks de inicio de cada compas hasta end_tick."""
        out = []
        sigs = sorted(self.time_sigs)
        for i, (tick, num, den) in enumerate(sigs):
            stop = sigs[i + 1][0] if i + 1 < len(sigs) else end_tick
            length = int(num * self.res * 4 / den)
            t = tick
            while t < stop:
                out.append(t)
                t += length
        return out


# --------------------------------------------------------------------- notas
@dataclass
class FretNote:
    """Nota de guitarra / bajo / teclado de 5 carriles."""

    tick: int
    frets: tuple[int, ...]  # 0..4 = verde..naranja, OPEN = abierta
    sustain: int = 0
    hopo: bool | None = None  # None = natural, True = forzar HOPO, False = forzar rasgueo
    tap: bool = False
    review: bool = False  # marcada como "revisar"
    conf: float = 1.0

    @property
    def is_chord(self) -> bool:
        return len(self.frets) > 1


@dataclass
class DrumNote:
    """Golpe de bateria. pad: 0=bombo 1=rojo 2=amarillo 3=azul 4=verde."""

    tick: int
    pad: int
    cymbal: bool = False  # solo para 2,3,4
    double_kick: bool = False  # bombo de Expert+
    review: bool = False
    conf: float = 1.0


@dataclass
class ProKeyNote:
    tick: int
    pitch: int  # 48..72
    sustain: int = 0
    review: bool = False


@dataclass
class VocalNote:
    tick: int
    length: int
    pitch: int  # 36..84 ; ignorado si unpitched
    lyric: str
    unpitched: bool = False
    review: bool = False


@dataclass
class Phrase:
    start: int
    end: int


@dataclass
class FretTrack:
    notes: dict[str, list[FretNote]] = field(default_factory=lambda: {d: [] for d in DIFFICULTIES})
    star_power: list[Phrase] = field(default_factory=list)
    solos: list[Phrase] = field(default_factory=list)
    review_regions: list[tuple[int, int, str]] = field(default_factory=list)

    def has_notes(self) -> bool:
        return any(self.notes[d] for d in DIFFICULTIES)


@dataclass
class DrumTrack:
    notes: dict[str, list[DrumNote]] = field(default_factory=lambda: {d: [] for d in DIFFICULTIES})
    star_power: list[Phrase] = field(default_factory=list)
    solos: list[Phrase] = field(default_factory=list)
    fills: list[Phrase] = field(default_factory=list)
    review_regions: list[tuple[int, int, str]] = field(default_factory=list)

    def has_notes(self) -> bool:
        return any(self.notes[d] for d in DIFFICULTIES)


@dataclass
class ProKeysTrack:
    notes: dict[str, list[ProKeyNote]] = field(default_factory=lambda: {d: [] for d in DIFFICULTIES})
    ranges: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: {d: [] for d in DIFFICULTIES})
    star_power: list[Phrase] = field(default_factory=list)
    solos: list[Phrase] = field(default_factory=list)

    def has_notes(self) -> bool:
        return any(self.notes[d] for d in DIFFICULTIES)


@dataclass
class VocalTrack:
    notes: list[VocalNote] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    star_power: list[Phrase] = field(default_factory=list)
    review_regions: list[tuple[int, int, str]] = field(default_factory=list)

    def has_notes(self) -> bool:
        return bool(self.notes)


@dataclass
class Chart:
    tempo: TempoMap = field(default_factory=TempoMap)
    name: str = ""
    guitar: FretTrack = field(default_factory=FretTrack)
    rhythm: FretTrack = field(default_factory=FretTrack)
    bass: FretTrack = field(default_factory=FretTrack)
    keys: FretTrack = field(default_factory=FretTrack)
    drums: DrumTrack = field(default_factory=DrumTrack)
    pro_keys: ProKeysTrack = field(default_factory=ProKeysTrack)
    vocals: VocalTrack = field(default_factory=VocalTrack)
    sections: list[tuple[int, str]] = field(default_factory=list)
    end_tick: int = 0

    FRET_TRACKS = ("guitar", "rhythm", "bass", "keys")
