"""Metadatos y portada: etiquetas del archivo y, si faltan, busqueda en internet (iTunes / MusicBrainz)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

UA = "Charter/0.1 ( generador de charts para YARG )"


@dataclass
class SongMeta:
    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""
    genre: str = ""
    track: str = ""
    cover: bytes | None = None
    cover_source: str = ""
    sources: list[str] = field(default_factory=list)


def from_file(path: str | Path) -> SongMeta:
    import mutagen

    path = Path(path)
    meta = SongMeta()
    try:
        f = mutagen.File(str(path))
    except Exception:  # noqa: BLE001
        f = None
    if f is not None and f.tags is not None:
        tags = f.tags

        def get(*keys):
            for k in keys:
                try:
                    v = tags.get(k)
                except Exception:  # noqa: BLE001
                    v = None
                if v:
                    if isinstance(v, list):
                        v = v[0]
                    if hasattr(v, "text"):
                        v = v.text[0] if v.text else ""
                    return str(v).strip()
            return ""

        meta.title = get("TIT2", "\xa9nam", "title", "TITLE")
        meta.artist = get("TPE1", "\xa9ART", "artist", "ARTIST", "aART", "TPE2")
        meta.album = get("TALB", "\xa9alb", "album", "ALBUM")
        meta.year = get("TDRC", "\xa9day", "date", "DATE", "year", "TYER")[:4]
        meta.genre = get("TCON", "\xa9gen", "genre", "GENRE")
        meta.track = get("TRCK", "trkn", "tracknumber", "TRACKNUMBER").split("/")[0]
        if meta.track.startswith("("):
            meta.track = re.sub(r"\D", " ", meta.track).split()[0] if re.sub(r"\D", " ", meta.track).split() else ""
        meta.cover = _embedded_cover(f)
        if meta.cover:
            meta.cover_source = "archivo"
        meta.sources.append("etiquetas del archivo")
    # completar con el nombre del archivo "Artista - Titulo"
    stem = path.stem
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        meta.artist = meta.artist or a.strip()
        meta.title = meta.title or t.strip()
    meta.title = meta.title or stem
    return meta


def _embedded_cover(f) -> bytes | None:
    try:
        tags = f.tags
        # MP3 (ID3)
        for k in list(tags.keys()):
            if str(k).startswith("APIC"):
                return tags[k].data
        # M4A
        if "covr" in tags and tags["covr"]:
            return bytes(tags["covr"][0])
    except Exception:  # noqa: BLE001
        pass
    # FLAC
    try:
        if getattr(f, "pictures", None):
            return f.pictures[0].data
    except Exception:  # noqa: BLE001
        pass
    # OGG/Opus con METADATA_BLOCK_PICTURE
    try:
        import base64

        from mutagen.flac import Picture

        vals = f.tags.get("metadata_block_picture") or []
        if vals:
            return Picture(base64.b64decode(vals[0])).data
    except Exception:  # noqa: BLE001
        pass
    return None


def has_internet(timeout: float = 4.0) -> bool:
    import requests

    for url in ("https://lrclib.net", "https://itunes.apple.com", "https://huggingface.co"):
        try:
            requests.head(url, timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def complete_online(meta: SongMeta, log=print) -> SongMeta:
    """Rellena lo que falte (album, ano, genero, portada) buscando en iTunes y MusicBrainz."""
    import requests

    need = not (meta.album and meta.year and meta.genre and meta.cover)
    if not need:
        return meta
    try:
        term = f"{meta.artist} {meta.title}"
        r = requests.get("https://itunes.apple.com/search", params={"term": term, "entity": "song", "limit": 10},
                         headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            best = None
            for it in r.json().get("results", []):
                if _similar(it.get("artistName", ""), meta.artist) and _similar(it.get("trackName", ""), meta.title):
                    best = it
                    break
            if best:
                meta.album = meta.album or best.get("collectionName", "")
                meta.year = meta.year or (best.get("releaseDate", "") or "")[:4]
                meta.genre = meta.genre or best.get("primaryGenreName", "")
                if not meta.track and best.get("trackNumber"):
                    meta.track = str(best["trackNumber"])
                if not meta.cover and best.get("artworkUrl100"):
                    url = best["artworkUrl100"].replace("100x100bb", "1000x1000bb")
                    img = requests.get(url, timeout=20)
                    if img.status_code == 200 and len(img.content) > 5000:
                        meta.cover = img.content
                        meta.cover_source = "iTunes"
                meta.sources.append("iTunes")
    except Exception as e:  # noqa: BLE001
        log(f"  iTunes no disponible: {e}")
    if meta.album and meta.year and meta.genre and meta.cover:
        return meta
    try:
        q = f'recording:"{meta.title}" AND artist:"{meta.artist}"'
        r = requests.get("https://musicbrainz.org/ws/2/recording", params={"query": q, "fmt": "json", "limit": 5},
                         headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            recs = r.json().get("recordings", [])
            for rec in recs:
                rels = rec.get("releases", [])
                if not rels:
                    continue
                rel = rels[0]
                meta.album = meta.album or rel.get("title", "")
                meta.year = meta.year or (rel.get("date", "") or "")[:4]
                tags = rec.get("tags") or []
                if not meta.genre and tags:
                    meta.genre = sorted(tags, key=lambda t: -t.get("count", 0))[0]["name"].title()
                if not meta.cover:
                    img = requests.get(f"https://coverartarchive.org/release/{rel['id']}/front-500", timeout=20,
                                       headers={"User-Agent": UA})
                    if img.status_code == 200 and len(img.content) > 5000:
                        meta.cover = img.content
                        meta.cover_source = "Cover Art Archive"
                meta.sources.append("MusicBrainz")
                break
    except Exception as e:  # noqa: BLE001
        log(f"  MusicBrainz no disponible: {e}")
    return meta


def _norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _similar(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na and nb) and (na in nb or nb in na)


def save_cover(data: bytes, dst: str | Path, size: int = 512) -> None:
    """Guarda la portada como album.png cuadrada (512x512, lo que usan YARG / CH)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.LANCZOS)
    img.save(str(dst), "PNG")
