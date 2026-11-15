"""Evalua la reduccion de dificultad: parte del Experto oficial y compara con Dificil/Medio/Facil oficiales."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import paths


def main(folder: str, only_official: bool = True) -> None:
    from ..chart.chart_io import load_chart
    from ..chart.model import RES
    from ..gen.frets import Event, _reduce
    from ..songini import read_ini
    from .reference import OFFICIAL_CHARTERS, find_songs

    tot = {d: {"tp": 0, "ours": 0, "ref": 0, "lane_ok": 0, "chord_ours": 0, "chord_ref": 0} for d in ("hard", "medium", "easy")}
    n_songs = 0
    for f in find_songs(folder):
        ini = read_ini(f / "song.ini")
        official = any(o in ini.get("charter", "").lower() for o in OFFICIAL_CHARTERS)
        if only_official and not official:
            continue
        try:
            chart = load_chart(f)
        except Exception:  # noqa: BLE001
            continue
        for key in ("guitar", "bass"):
            ft = getattr(chart, key)
            exp = ft.notes["expert"]
            if len(exp) < 50:
                continue
            n_songs += 1
            events = [Event(chart.tempo.tick_to_sec(n.tick), [60], 0, 0.5, 1.0, tick=n.tick) for n in exp]
            for d in ("hard", "medium", "easy"):
                ours = _reduce(exp, events, chart.tempo, d, key)
                ref = ft.notes[d]
                rt = {n.tick: n for n in ref}
                tp = 0
                lane_ok = 0
                for n in ours:
                    if n.tick in rt:
                        tp += 1
                        lane_ok += int(tuple(n.frets) == tuple(rt[n.tick].frets))
                t = tot[d]
                t["tp"] += tp
                t["ours"] += len(ours)
                t["ref"] += len(ref)
                t["lane_ok"] += lane_ok
                t["chord_ours"] += sum(n.is_chord for n in ours)
                t["chord_ref"] += sum(n.is_chord for n in ref)
    print("pistas evaluadas:", n_songs)
    for d, t in tot.items():
        p = t["tp"] / max(t["ours"], 1)
        r = t["tp"] / max(t["ref"], 1)
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"{d:7s} F1 notas={f1:.3f} (P={p:.3f} R={r:.3f})  carriles iguales={t['lane_ok'] / max(t['tp'], 1):.3f}  "
              f"n ours/ref={t['ours'] / max(t['ref'], 1):.2f}  acordes ours={t['chord_ours'] / max(t['ours'], 1):.2f} "
              f"ref={t['chord_ref'] / max(t['ref'], 1):.2f}")


if __name__ == "__main__":
    paths.setup_env()
    main(sys.argv[1], only_official="--todas" not in sys.argv)
