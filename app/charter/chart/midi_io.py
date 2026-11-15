"""Lectura y escritura de notes.mid en formato Rock Band / YARG / Clone Hero."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import mido

from .model import (
    DIFFICULTIES,
    OPEN,
    RES,
    Chart,
    DrumNote,
    DrumTrack,
    FretNote,
    FretTrack,
    Phrase,
    ProKeyNote,
    ProKeysTrack,
    TempoMap,
    VocalNote,
    VocalTrack,
)

DIFF_BASE = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
PROKEYS_SUFFIX = {"easy": "E", "medium": "M", "hard": "H", "expert": "X"}
FRET_TRACK_NAMES = {
    "guitar": "PART GUITAR",
    "rhythm": "PART RHYTHM",
    "bass": "PART BASS",
    "keys": "PART KEYS",
}
SP_NOTE = 116
SOLO_NOTE = 103
TAP_NOTE = 104
PHRASE_NOTES = (105, 106)
TOM_MARKERS = {110: 2, 111: 3, 112: 4}
FILL_NOTES = range(120, 125)
PROKEYS_RANGES = (0, 2, 4, 5, 7, 9)
PROKEYS_SOLO = 115
NOSUSTAIN_LEN = RES // 8  # 1/32: por debajo del umbral de sustain de YARG / CH


# =========================================================== utilidades
def _abs_events(track: mido.MidiTrack, scale: float):
    t = 0
    for msg in track:
        t += msg.time
        yield int(round(t * scale)), msg


def _note_spans(track: mido.MidiTrack, scale: float):
    """Devuelve (inicio, fin, nota, velocidad) de todas las notas del track."""
    open_notes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    spans = []
    texts = []
    for tick, msg in _abs_events(track, scale):
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes[msg.note].append((tick, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            if open_notes[msg.note]:
                start, vel = open_notes[msg.note].pop(0)
                spans.append((start, tick, msg.note, vel))
        elif msg.type in ("text", "lyrics"):
            texts.append((tick, msg.text, msg.type))
        elif msg.type == "sysex":
            texts.append((tick, bytes(msg.data), "sysex"))
    return spans, texts


def _phrases(spans, note: int) -> list[Phrase]:
    return [Phrase(s, e) for s, e, n, _ in spans if n == note]


def _in_any(tick: int, phrases: list[Phrase]) -> bool:
    return any(p.start <= tick < p.end for p in phrases)


# =========================================================== lectura
def read_midi(path: str | Path, hopo_frequency: int | None = None) -> Chart:
    mid = mido.MidiFile(str(path), clip=True)
    scale = RES / mid.ticks_per_beat
    chart = Chart()
    tempos, sigs = [], []
    end_tick = 0
    tracks_by_name = {}
    for i, tr in enumerate(mid.tracks):
        name = None
        for tick, msg in _abs_events(tr, scale):
            end_tick = max(end_tick, tick)
            if msg.type == "track_name" and name is None:
                name = msg.name.strip()
            if i == 0 or msg.type in ("set_tempo", "time_signature"):
                if msg.type == "set_tempo":
                    tempos.append((tick, msg.tempo))
                elif msg.type == "time_signature":
                    sigs.append((tick, msg.numerator, msg.denominator))
        if i == 0:
            chart.name = name or ""
        elif name:
            tracks_by_name[name] = tr
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))
    if not sigs or sigs[0][0] != 0:
        sigs.insert(0, (0, 4, 4))
    chart.tempo = TempoMap(tempos=tempos, time_sigs=sigs)
    chart.end_tick = end_tick
    hopo_thr = int(round((hopo_frequency or 170) * RES / 480))

    for key, tname in FRET_TRACK_NAMES.items():
        if tname in tracks_by_name:
            setattr(chart, key, _read_fret_track(tracks_by_name[tname], scale, hopo_thr))
    if "PART DRUMS" in tracks_by_name:
        chart.drums = _read_drums(tracks_by_name["PART DRUMS"], scale)
    if "PART VOCALS" in tracks_by_name:
        chart.vocals = _read_vocals(tracks_by_name["PART VOCALS"], scale)
    pk = ProKeysTrack()
    for d in DIFFICULTIES:
        tname = f"PART REAL_KEYS_{PROKEYS_SUFFIX[d]}"
        if tname in tracks_by_name:
            spans, _ = _note_spans(tracks_by_name[tname], scale)
            pk.notes[d] = sorted(
                (ProKeyNote(s, n, e - s if e - s > NOSUSTAIN_LEN * 2 else 0) for s, e, n, _ in spans if 48 <= n <= 72),
                key=lambda x: (x.tick, x.pitch),
            )
            pk.ranges[d] = sorted((s, n) for s, e, n, _ in spans if n in PROKEYS_RANGES)
            if d == "expert":
                pk.star_power = _phrases(spans, SP_NOTE)
                pk.solos = _phrases(spans, PROKEYS_SOLO)
    chart.pro_keys = pk
    if "EVENTS" in tracks_by_name:
        for tick, msg in _abs_events(tracks_by_name["EVENTS"], scale):
            if msg.type == "text":
                m = re.match(r"\[(?:section|prc)[ _](.+)\]", msg.text.strip())
                if m:
                    chart.sections.append((tick, m.group(1).replace("_", " ")))
    return chart


def _read_fret_track(track, scale, hopo_thr) -> FretTrack:
    spans, texts = _note_spans(track, scale)
    ft = FretTrack()
    ft.star_power = _phrases(spans, SP_NOTE)
    ft.solos = _phrases(spans, SOLO_NOTE)
    taps = _phrases(spans, TAP_NOTE)
    enhanced_opens = any(isinstance(t, str) and "ENHANCED_OPENS" in t for _, t, _ in texts)
    # Notas abiertas via sysex de Phase Shift
    sysex_open = defaultdict(list)
    for tick, data, kind in texts:
        if kind == "sysex" and len(data) >= 7 and data[:3] == b"PS\x00":
            diff_idx, typ, on = data[4], data[5], data[6]
            if typ == 1:
                sysex_open[diff_idx].append((tick, on))
    for di, d in enumerate(DIFFICULTIES):
        base = DIFF_BASE[d]
        by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)
        forced_hopo = _phrases(spans, base + 5)
        forced_strum = _phrases(spans, base + 6)
        open_ranges = []
        evs = sorted(sysex_open.get(di, []))
        cur = None
        for tick, on in evs:
            if on:
                cur = tick
            elif cur is not None:
                open_ranges.append(Phrase(cur, tick + 1))
                cur = None
        for s, e, n, _ in spans:
            if base <= n <= base + 4:
                by_tick[s].append((n - base, e - s))
            elif enhanced_opens and n == base - 1:
                by_tick[s].append((OPEN, e - s))
        notes = []
        for tick in sorted(by_tick):
            frets = sorted(set(f for f, _ in by_tick[tick]))
            length = max(l for _, l in by_tick[tick])
            if open_ranges and _in_any(tick, open_ranges) and frets == [0]:
                frets = [OPEN]
            sus = length if length > RES // 3 else 0
            fn = FretNote(tick, tuple(frets), sus)
            if _in_any(tick, forced_hopo):
                fn.hopo = True
            elif _in_any(tick, forced_strum):
                fn.hopo = False
            fn.tap = _in_any(tick, taps)
            notes.append(fn)
        ft.notes[d] = notes
    return ft


def resolve_hopos(notes: list[FretNote], hopo_thr: int = 170) -> list[bool]:
    """Devuelve si cada nota se juega como HOPO (natural o forzado)."""
    out = []
    prev = None
    for n in notes:
        if n.tap:
            out.append(True)
        elif n.hopo is not None:
            out.append(n.hopo)
        elif prev is None or n.is_chord:
            out.append(False)
        else:
            close = n.tick - prev.tick <= hopo_thr
            same = set(n.frets) == set(prev.frets) or (prev.is_chord and set(n.frets) <= set(prev.frets))
            out.append(close and not same)
        prev = n
    return out


def _read_drums(track, scale) -> DrumTrack:
    spans, _ = _note_spans(track, scale)
    dt = DrumTrack()
    dt.star_power = _phrases(spans, SP_NOTE)
    dt.solos = _phrases(spans, SOLO_NOTE)
    fill_starts = defaultdict(list)
    for s, e, n, _ in spans:
        if n in FILL_NOTES:
            fill_starts[(s, e)].append(n)
    dt.fills = [Phrase(s, e) for (s, e) in sorted(fill_starts)]
    toms = {pad: _phrases(spans, note) for note, pad in TOM_MARKERS.items()}
    for d in DIFFICULTIES:
        base = DIFF_BASE[d]
        notes = []
        for s, e, n, _ in spans:
            if base <= n <= base + 4:
                pad = n - base
                cym = pad >= 2 and not _in_any(s, toms[pad])
                notes.append(DrumNote(s, pad, cymbal=cym))
            elif d == "expert" and n == 95:
                notes.append(DrumNote(s, 0, double_kick=True))
        notes.sort(key=lambda x: (x.tick, x.pad))
        dt.notes[d] = notes
    return dt


def _read_vocals(track, scale) -> VocalTrack:
    spans, texts = _note_spans(track, scale)
    vt = VocalTrack()
    vt.phrases = sorted(_phrases(spans, 105) + _phrases(spans, 106), key=lambda p: p.start)
    vt.star_power = _phrases(spans, SP_NOTE)
    lyrics = {}
    for tick, text, kind in texts:
        if kind in ("lyrics", "text") and isinstance(text, str) and not text.startswith("["):
            lyrics[tick] = text
    for s, e, n, _ in sorted(spans):
        if 36 <= n <= 84:
            lyr = lyrics.get(s, "")
            unp = lyr.endswith("#") or lyr.endswith("^")
            vt.notes.append(VocalNote(s, e - s, n, lyr, unpitched=unp))
    return vt


# =========================================================== escritura
class _TrackBuilder:
    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []

    def note(self, tick: int, length: int, note: int, vel: int = 100) -> None:
        length = max(int(length), 1)
        self.events.append((int(tick), 1, mido.Message("note_on", note=note, velocity=vel)))
        self.events.append((int(tick) + length, 0, mido.Message("note_off", note=note, velocity=0)))

    def text(self, tick: int, text: str, kind: str = "text") -> None:
        self.events.append((int(tick), 2, mido.MetaMessage(kind, text=text)))

    def meta(self, tick: int, msg) -> None:
        self.events.append((int(tick), -1, msg))

    def build(self) -> mido.MidiTrack:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=self.name, time=0))
        # orden: metas, note_off, note_on, textos dentro del mismo tick
        self.events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _, msg in self.events:
            tr.append(msg.copy(time=tick - last))
            last = tick
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr


def _note_len(sustain: int) -> int:
    return sustain if sustain > 0 else NOSUSTAIN_LEN


def _write_fret_track(name: str, ft: FretTrack) -> mido.MidiTrack | None:
    if not ft.has_notes():
        return None
    tb = _TrackBuilder(name)
    uses_open = any(OPEN in n.frets for d in DIFFICULTIES for n in ft.notes[d])
    if uses_open:
        tb.text(0, "[ENHANCED_OPENS]")
    for d in DIFFICULTIES:
        base = DIFF_BASE[d]
        for n in ft.notes[d]:
            length = _note_len(n.sustain)
            for f in n.frets:
                note = base - 1 if f == OPEN else base + f
                tb.note(n.tick, length, note)
            if n.tap:
                if d == "expert" or d == "hard":
                    tb.note(n.tick, NOSUSTAIN_LEN, TAP_NOTE)
            elif n.hopo is True:
                tb.note(n.tick, NOSUSTAIN_LEN, base + 5)
            elif n.hopo is False:
                tb.note(n.tick, NOSUSTAIN_LEN, base + 6)
    _merge_marker_notes(tb, ft.notes, TAP_NOTE)
    for p in ft.star_power:
        tb.note(p.start, p.end - p.start, SP_NOTE)
    for p in ft.solos:
        tb.note(p.start, p.end - p.start, SOLO_NOTE)
    return tb.build()


def _merge_marker_notes(tb: _TrackBuilder, notes, marker: int) -> None:
    """Evita marcadores de tap solapados (si dos dificultades comparten tick)."""
    seen = set()
    filtered = []
    pending_off = {}
    for ev in tb.events:
        tick, kind, msg = ev
        if getattr(msg, "type", None) == "note_on" and msg.note == marker:
            if tick in seen:
                pending_off[tick] = True
                continue
            seen.add(tick)
        filtered.append(ev)
    if pending_off:
        # quitar los note_off sobrantes del mismo marcador
        out = []
        counts = defaultdict(int)
        for tick, kind, msg in filtered:
            if getattr(msg, "type", None) == "note_on" and msg.note == marker:
                counts[msg.note] += 1
            if getattr(msg, "type", None) == "note_off" and msg.note == marker:
                if counts[msg.note] <= 0:
                    continue
                counts[msg.note] -= 1
            out.append((tick, kind, msg))
        filtered = out
    tb.events = filtered


def _write_drums(dt: DrumTrack) -> mido.MidiTrack | None:
    if not dt.has_notes():
        return None
    tb = _TrackBuilder("PART DRUMS")
    tom_ticks = {2: set(), 3: set(), 4: set()}
    for d in DIFFICULTIES:
        base = DIFF_BASE[d]
        for n in dt.notes[d]:
            if n.double_kick:
                tb.note(n.tick, NOSUSTAIN_LEN, 95)
                continue
            tb.note(n.tick, NOSUSTAIN_LEN, base + n.pad)
            if n.pad >= 2 and not n.cymbal:
                tom_ticks[n.pad].add(n.tick)
    for note, pad in TOM_MARKERS.items():
        for t in sorted(tom_ticks[pad]):
            tb.note(t, NOSUSTAIN_LEN, note)
    for p in dt.star_power:
        tb.note(p.start, p.end - p.start, SP_NOTE)
    for p in dt.solos:
        tb.note(p.start, p.end - p.start, SOLO_NOTE)
    for p in dt.fills:
        for n in FILL_NOTES:
            tb.note(p.start, p.end - p.start, n)
    return tb.build()


def _write_vocals(vt: VocalTrack) -> mido.MidiTrack | None:
    if not vt.has_notes():
        return None
    tb = _TrackBuilder("PART VOCALS")
    for n in vt.notes:
        tb.note(n.tick, max(n.length, RES // 16), max(36, min(84, n.pitch)))
        tb.text(n.tick, n.lyric, "lyrics")
    for p in vt.phrases:
        tb.note(p.start, p.end - p.start, 105)
    for p in vt.star_power:
        tb.note(p.start, p.end - p.start, SP_NOTE)
    return tb.build()


def _write_pro_keys(pk: ProKeysTrack) -> list[mido.MidiTrack]:
    out = []
    if not pk.has_notes():
        return out
    for d in DIFFICULTIES:
        tb = _TrackBuilder(f"PART REAL_KEYS_{PROKEYS_SUFFIX[d]}")
        for tick, rng in pk.ranges[d]:
            tb.note(tick, NOSUSTAIN_LEN, rng)
        for n in pk.notes[d]:
            tb.note(n.tick, _note_len(n.sustain), n.pitch)
        for p in pk.star_power:
            tb.note(p.start, p.end - p.start, SP_NOTE)
        for p in pk.solos:
            tb.note(p.start, p.end - p.start, PROKEYS_SOLO)
        out.append(tb.build())
    return out


def write_midi(chart: Chart, path: str | Path) -> None:
    mid = mido.MidiFile(type=1, ticks_per_beat=RES)
    t0 = _TrackBuilder(chart.name or "Charter")
    for tick, us in chart.tempo.tempos:
        t0.meta(tick, mido.MetaMessage("set_tempo", tempo=int(us)))
    for tick, num, den in chart.tempo.time_sigs:
        t0.meta(tick, mido.MetaMessage("time_signature", numerator=num, denominator=den,
                                       clocks_per_click=24, notated_32nd_notes_per_beat=8))
    mid.tracks.append(t0.build())

    for key, tname in FRET_TRACK_NAMES.items():
        tr = _write_fret_track(tname, getattr(chart, key))
        if tr is not None:
            mid.tracks.append(tr)
    tr = _write_drums(chart.drums)
    if tr is not None:
        mid.tracks.append(tr)
    tr = _write_vocals(chart.vocals)
    if tr is not None:
        mid.tracks.append(tr)
    mid.tracks.extend(_write_pro_keys(chart.pro_keys))

    end = chart.end_tick
    ev = _TrackBuilder("EVENTS")
    first_note = _first_note_tick(chart)
    ev.text(max(0, min(first_note - RES * 2, RES * 2)), "[music_start]")
    for tick, name in chart.sections:
        ev.text(tick, f"[section {name}]")
    ev.text(end, "[music_end]")
    ev.text(end + RES, "[end]")
    mid.tracks.append(ev.build())

    beat = _TrackBuilder("BEAT")
    measures = set(chart.tempo.measure_ticks(end))
    for tick, num, den in [(0, 0, 0)]:
        pass
    t = 0
    while t < end:
        _, num, den = chart.tempo.ts_at(t)
        step = int(RES * 4 / den)
        beat.note(t, RES // 4, 12 if t in measures else 13)
        t += step
    mid.tracks.append(beat.build())
    mid.save(str(path))


def _first_note_tick(chart: Chart) -> int:
    ticks = []
    for key in Chart.FRET_TRACKS:
        ft = getattr(chart, key)
        for d in DIFFICULTIES:
            if ft.notes[d]:
                ticks.append(ft.notes[d][0].tick)
    for d in DIFFICULTIES:
        if chart.drums.notes[d]:
            ticks.append(chart.drums.notes[d][0].tick)
    if chart.vocals.notes:
        ticks.append(chart.vocals.notes[0].tick)
    return min(ticks) if ticks else RES * 4
