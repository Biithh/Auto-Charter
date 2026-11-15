"""Video de fondo: descarga de YouTube (sin audio, maximo 1080p) o copia de un archivo local.

El resultado siempre se deja como video.mp4 normal en H.264: YouTube entrega MP4 fragmentados que el decodificador
de Windows (el que usa YARG) da por terminados sin leer ni un fotograma, y el juego se queda con el fondo en negro.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..audio import run


def fetch(url: str, out_dir: Path, fmt: str, name: str, progress=lambda f: None) -> tuple[Path, dict]:
    """Descarga el formato elegido como out_dir/<name>.<ext>. Si progress lanza un error, la descarga se corta."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": fmt,
        "outtmpl": str(out_dir).replace("%", "%%") + f"/{name}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "ffmpeg_location": str(paths.FFMPEG_DIR) if paths.FFMPEG_DIR.exists() else None,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fn = Path(ydl.prepare_filename(info))
    if not fn.exists():
        cands = sorted(out_dir.glob(f"{name}.*"))
        if not cands:
            raise RuntimeError(f"No se pudo descargar: {url}")
        fn = cands[0]
    return fn, info


def download_youtube(url: str, out_dir: Path, log=print, progress=lambda f: None) -> Path:
    log(f"Descargando video de YouTube: {url}")
    # mejor video sin audio hasta 1080p; preferimos H.264 (mp4) por compatibilidad con YARG y Clone Hero
    fn, info = fetch(url, out_dir, "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "video",
                     progress)
    log(f"Video descargado: {fn.name} ({info.get('height', '?')}p)")
    return fn


def _is_h264(src: Path) -> bool:
    r = run([paths.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "json", str(src)])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe no pudo leer el video: {r.stderr.decode(errors='ignore')[-500:]}")
    streams = json.loads(r.stdout).get("streams") or []
    if not streams:
        raise RuntimeError("El archivo no tiene imagen de video")
    return streams[0].get("codec_name") == "h264" and streams[0].get("pix_fmt") == "yuv420p"


def prepare_for_game(src: str | Path, out_dir: Path, log=print) -> Path:
    """Crea out_dir/video.mp4 sin audio: reempaqueta si ya es H.264 (no pierde calidad) y si no, lo convierte."""
    src = Path(src)
    dst = out_dir / "video.mp4"
    tmp = out_dir / "video_tmp.mp4"
    if _is_h264(src):
        codec = ["-c:v", "copy"]
    else:
        log("Convirtiendo el video a H.264 para que el juego pueda reproducirlo...")
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-vf", "scale=-2:'2*trunc(min(1080,ih)/2)'"]
    r = run([paths.ffmpeg(), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", *codec,
             "-movflags", "+faststart", str(tmp)])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg no pudo preparar el video: {r.stderr.decode(errors='ignore')[-500:]}")
    tmp.replace(dst)
    if src.parent == out_dir and src != dst:
        src.unlink()
    return dst
