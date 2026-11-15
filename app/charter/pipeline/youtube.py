"""Canciones desde YouTube: la musica de un enlace y, si se quiere, tambien su video para el fondo.

Se guardan en trabajo/youtube/<id del video> hasta que se cierra el programa: lo que se queda es la cancion generada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from .metadata import _norm
from .runner import slug
from .video import download_youtube, fetch

FOLDER = paths.WORK / "youtube"

# lo que se anade al titulo del video y no es parte del nombre de la cancion
_TAG = re.compile(r"\s*[(\[][^)\]]*\b(official|oficial|video|v[ií]deo|videoclip|clip|lyrics?|letra|audio|visuali[sz]er|"
                  r"hd|hq|4k)\b[^)\]]*[)\]]", re.I)


@dataclass
class YoutubeSong:
    audio_path: str
    video_path: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""


def describe(info: dict) -> tuple[str, str, str, str]:
    """(titulo, artista, album, ano) de un video: los datos de YouTube Music si los tiene, y si no, su titulo."""
    artists = info.get("artists") or ([info["artist"]] if info.get("artist") else [])
    if info.get("track") and artists:
        return info["track"], artists[0], info.get("album") or "", str(info.get("release_year") or "")
    title = _TAG.sub("", info.get("title") or "").split(" | ")[0].strip()
    channel = re.sub(r"\s*(- Topic|VEVO)$", "", info.get("channel") or info.get("uploader") or "").strip()
    parts = re.split(r"\s+[-–—]\s+", title, maxsplit=1)
    if len(parts) < 2:
        return title.strip(' "'), channel, "", ""
    first, second = (p.strip(' "') for p in parts)
    # "Cancion - Grupo" subido por el propio grupo
    if _norm(channel) and _norm(channel) in _norm(second) and _norm(channel) not in _norm(first):
        first, second = second, first
    return second, first, "", ""


def _downloaded(folder: Path, stem: str) -> Path | None:
    """El archivo ya completo con ese nombre (los que estan a medias acaban en .part y su nombre es otro)."""
    return next((p for p in folder.glob("*") if p.stem == stem), None)


def download_song(url: str, with_video: bool = True, log=print, progress=lambda f: None) -> YoutubeSong:
    """Descarga la musica y, si with_video, el video del mismo enlace: al ser la misma grabacion van sincronizados."""
    import yt_dlp

    log("Buscando el video...")
    # extract_flat: de una lista no hay que mirar cada video para saber que es una lista
    with yt_dlp.YoutubeDL({"noplaylist": True, "quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}) as ydl:
        info = ydl.extract_info(url, download=False)
    if "entries" in info:
        raise RuntimeError("Ese enlace es una lista de reproduccion: pega el enlace de una sola cancion.")
    if info.get("is_live"):
        raise RuntimeError("Ese video es un directo que sigue en marcha: no se puede descargar entero.")
    title, artist, album, year = describe(info)
    out = FOLDER / slug(str(info.get("id") or ""))
    stem = slug(f"{artist} - {title}" if artist else title)[:120].rstrip(" .") or "cancion"
    audio = _downloaded(out / "musica", stem)
    if audio is None:
        log("Descargando la musica...")
        fn, _ = fetch(url, out / "musica", "bestaudio", "audio", progress)
        audio = fn.replace(fn.with_name(stem + fn.suffix))
    video = None
    if with_video:
        video = _downloaded(out / "video", "video")
        if video is None:
            log("Descargando el video...")
            video = download_youtube(url, out / "video", log, progress)
    return YoutubeSong(str(audio), str(video) if video else "", title, artist, album, year)
