"""Letras: busqueda en internet (LRCLIB), transcripcion con Whisper y alineacion palabra a palabra."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

ALIGN_MODELS = {
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}
WHISPER_MODEL = "openai/whisper-large-v3"
UA = "Charter/0.1 (generador de charts para YARG)"


@dataclass
class Line:
    text: str
    start: float | None = None  # segundos (si la letra viene sincronizada)
    end: float | None = None
    words: list["Word"] = field(default_factory=list)


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class Lyrics:
    lines: list[Line]
    source: str  # "lrclib-synced", "lrclib-plain", "whisper"
    language: str = "es"


# ------------------------------------------------------------------ LRCLIB
def fetch_lrclib(artist: str, title: str, album: str = "", duration: float = 0.0, log=print) -> Lyrics | None:
    import requests

    def parse(item) -> Lyrics | None:
        synced = item.get("syncedLyrics")
        plain = item.get("plainLyrics")
        if synced:
            lines = []
            for raw in synced.splitlines():
                m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", raw.strip())
                if not m:
                    continue
                t = int(m.group(1)) * 60 + float(m.group(2))
                txt = m.group(3).strip()
                lines.append(Line(txt, t))
            for a, b in zip(lines, lines[1:]):
                a.end = b.start
            lines = [l for l in lines if l.text and not re.fullmatch(r"[♪\s\W]*", l.text)]
            if lines:
                return Lyrics(lines, "lrclib-synced")
        if plain:
            lines = [Line(l.strip()) for l in plain.splitlines() if l.strip()]
            if lines:
                return Lyrics(lines, "lrclib-plain")
        return None

    import time

    def call(url, params):
        """LRCLIB a veces responde 503 (ocupado): se reintenta con espera."""
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
            except Exception as e:  # noqa: BLE001
                log(f"  LRCLIB no responde ({e}), reintentando...")
                time.sleep(2 + attempt * 2)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 + attempt * 3)
                continue
            return r
        return None

    def pick(items):
        if not isinstance(items, list):
            return None
        from .metadata import _similar

        items = [it for it in items if _similar(it.get("artistName", ""), artist)
                 and _similar(it.get("trackName", ""), title)] or []
        if duration:
            items = sorted(items, key=lambda it: abs((it.get("duration") or 0) - duration))
        synced_first = sorted(items[:8], key=lambda it: not it.get("syncedLyrics"))
        for it in synced_first:
            if duration and it.get("duration") and abs(it["duration"] - duration) > 8:
                continue
            res = parse(it)
            if res:
                return res
        return None

    attempts = []
    base = {"artist_name": artist, "track_name": title}
    if duration:
        attempts.append(("get", {**base, "album_name": album, "duration": int(round(duration))} if album else
                         {**base, "duration": int(round(duration))}))
    attempts.append(("search", base))
    attempts.append(("search", {"q": f"{artist} {title}"}))
    for kind, params in attempts:
        url = "https://lrclib.net/api/get" if kind == "get" else "https://lrclib.net/api/search"
        r = call(url, params)
        if r is None or r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        res = parse(data) if kind == "get" else pick(data)
        if res:
            return res
    return None


# ------------------------------------------------------------------ Whisper
def whisper_transcribe(audio16: np.ndarray, language: str | None, log=print) -> Lyrics:
    import torch
    from transformers import pipeline

    dev = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log("  transcribiendo la letra con Whisper large-v3...")
    asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, torch_dtype=dtype, device=dev,
                   chunk_length_s=30, batch_size=4)
    kw = {"task": "transcribe"}
    if language:
        kw["language"] = language
    res = asr({"raw": audio16, "sampling_rate": 16000}, return_timestamps=True, generate_kwargs=kw)
    lines = []
    for ch in res.get("chunks", []):
        txt = ch["text"].strip()
        ts = ch.get("timestamp") or (None, None)
        if not txt:
            continue
        lines.append(Line(txt, ts[0], ts[1]))
    del asr
    torch.cuda.empty_cache()
    lines = _drop_hallucinations(lines)
    return Lyrics(lines, "whisper", language or "es")


def _drop_hallucinations(lines: list[Line]) -> list[Line]:
    """Whisper a veces repite frases o inventa creditos en los silencios."""
    out = []
    bad = ("subtítulos", "subtitulos", "amara.org", "gracias por ver", "suscríbete", "thanks for watching")
    for l in lines:
        low = l.text.lower()
        if any(b in low for b in bad):
            continue
        if out and out[-1].text == l.text and len(out) >= 2 and out[-2].text == l.text:
            continue
        out.append(l)
    return out


def detect_language(audio16: np.ndarray) -> str:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, torch_dtype=torch.float16
                                                            if torch.cuda.is_available() else torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    # usamos el tramo con mas voz
    seg = audio16[: 16000 * 30]
    energy = [np.sqrt(np.mean(audio16[i:i + 16000 * 30] ** 2)) for i in range(0, max(len(audio16) - 16000 * 30, 1), 16000 * 10)]
    if energy:
        i = int(np.argmax(energy)) * 16000 * 10
        seg = audio16[i:i + 16000 * 30]
    feats = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to(dev, model.dtype)
    with torch.no_grad():
        out = model.generate(feats, max_new_tokens=1)
    tok = proc.tokenizer.convert_ids_to_tokens(out[0][:3])
    del model
    torch.cuda.empty_cache()
    for t in tok:
        m = re.fullmatch(r"<\|(\w\w)\|>", t)
        if m:
            return m.group(1)
    return "es"


# ------------------------------------------------------------------ alineacion CTC
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    return text


def shown_words(line: Line) -> list[str]:
    """Palabras alineadas de la linea tal como se escriben en la letra (con sus mayusculas)."""
    orig = iter(re.findall(r"[\wáéíóúüñçàèìòùâêîôûäëïöÿ']+", re.sub(r"\(.*?\)|\[.*?\]", " ", line.text), re.I))
    return [next((o for o in orig if o.lower() == w.text), w.text) for w in line.words]


def _fix_pos_conv(model, name: str) -> None:
    """Con PyTorch reciente, transformers no carga los pesos weight_g/weight_v de la convolucion
    posicional de wav2vec2 (quedan aleatorios y la alineacion sale mal). Se copian a mano."""
    import torch
    from huggingface_hub import hf_hub_download

    conv = model.wav2vec2.encoder.pos_conv_embed.conv
    if not hasattr(conv, "parametrizations"):
        return
    sd = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(name, fname)
        except Exception:  # noqa: BLE001
            continue
        if fname.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        break
    if not sd:
        return
    g = sd.get("wav2vec2.encoder.pos_conv_embed.conv.weight_g")
    v = sd.get("wav2vec2.encoder.pos_conv_embed.conv.weight_v")
    if g is None or v is None:
        return
    with torch.no_grad():
        conv.parametrizations.weight.original0.copy_(g)
        conv.parametrizations.weight.original1.copy_(v)


class Aligner:
    def __init__(self, language: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        name = ALIGN_MODELS.get(language, ALIGN_MODELS["es"])
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name)
        _fix_pos_conv(self.model, name)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.vocab.get("<pad>", 0)
        self.space = self.vocab.get("|", None)

    def close(self):
        import torch

        del self.model
        torch.cuda.empty_cache()

    def emissions(self, audio16: np.ndarray) -> np.ndarray:
        import torch

        out = []
        step = 16000 * 20
        for i in range(0, len(audio16), step):
            chunk = audio16[i:i + step]
            if len(chunk) < 400:
                break
            inp = self.proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
            with torch.no_grad():
                lp = torch.log_softmax(self.model(inp).logits[0], dim=-1).cpu().numpy()
            out.append(lp)
        return np.concatenate(out) if out else np.zeros((0, len(self.vocab)))

    def tokens(self, word: str) -> list[int]:
        toks = []
        w = word
        for ch in w:
            if ch in self.vocab:
                toks.append(self.vocab[ch])
            else:
                base = unicodedata.normalize("NFD", ch)[0]
                if base in self.vocab:
                    toks.append(self.vocab[base])
        return toks

    def align_words(self, audio16: np.ndarray, t0: float, words: list[str]) -> list[Word]:
        """Alinea una lista de palabras dentro de un tramo de audio que empieza en t0."""
        em = self.emissions(audio16)
        if len(em) == 0:
            return []
        frame_dur = len(audio16) / 16000 / len(em)
        # donde no se canta (silencio en la pista de voz) solo puede haber "blank": asi el alineador
        # no mete palabras en los huecos y no las amontona
        hop = max(1, int(round(frame_dur * 16000)))
        n = len(em)
        seg = audio16[: n * hop]
        if len(seg) < n * hop:
            seg = np.pad(seg, (0, n * hop - len(seg)))
        rms = np.sqrt((seg.reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)
        silent = db < max(np.percentile(db, 95) - 32, np.percentile(db, 10) + 3)
        from scipy.ndimage import binary_dilation, binary_erosion

        # 24-11: al final de un videoclip cortado sobraban letras y el alineador las ponia en el silencio de los
        # creditos ("do-" 14 s, "lor" 18 s): una letra fuera de la voz (+-0,3 s) no cuenta para el tiempo de su palabra
        voiced = binary_dilation(~silent, np.ones(2 * int(0.3 / frame_dur) + 1))
        silent = binary_erosion(silent, np.ones(5))  # margen en los bordes de cada frase
        em = em.copy()
        mask = np.ones(em.shape[1], bool)
        mask[self.blank] = False
        em[np.ix_(silent, mask)] -= 8.0
        seq, owner = [], []
        for wi, w in enumerate(words):
            tk = self.tokens(w)
            if not tk:
                continue
            if seq and self.space is not None:
                seq.append(self.space)
                owner.append(-1)
            seq += tk
            owner += [wi] * len(tk)
        if not seq or len(seq) > len(em):
            return []
        path = _ctc_viterbi(em, seq, self.blank)
        if path is None:
            return []
        spans: dict[int, list[int]] = {}
        scores: dict[int, list[float]] = {}
        for tok_i, frame in path:
            w = owner[tok_i]
            if w < 0:
                continue
            spans.setdefault(w, []).append(frame)
            scores.setdefault(w, []).append(float(np.exp(em[frame, seq[tok_i]])))
        out = []
        for wi, w in enumerate(words):
            if wi not in spans:
                continue
            fr = [f for f in spans[wi] if voiced[f]]
            if not fr:
                continue  # la palabra entera cae donde no se canta
            out.append(Word(w, t0 + min(fr) * frame_dur, t0 + (max(fr) + 1) * frame_dur, float(np.mean(scores[wi]))))
        return out


def _ctc_viterbi(em: np.ndarray, seq: list[int], blank: int):
    """Alineacion forzada CTC (trellis clasico). Devuelve [(indice_token, frame)]."""
    T, N = len(em), len(seq)
    em = em.astype(np.float32)
    trellis = np.full((T + 1, N + 1), -np.inf, dtype=np.float32)
    trellis[0, 0] = 0
    trellis[1:, 0] = np.cumsum(em[:, blank])
    seq_arr = np.array(seq)
    for t in range(T):
        stay = trellis[t, 1:] + em[t, blank]
        change = trellis[t, :-1] + em[t, seq_arr]
        trellis[t + 1, 1:] = np.maximum(stay, change)
    if not np.isfinite(trellis[T, N]):
        return None
    # backtrack
    path = []
    j = N
    t = T
    while j > 0 and t > 0:
        stay = trellis[t - 1, j] + em[t - 1, blank]
        change = trellis[t - 1, j - 1] + em[t - 1, seq[j - 1]]
        if change >= stay:
            path.append((j - 1, t - 1))
            j -= 1
        t -= 1
    return path[::-1]


def split_words(text: str) -> list[str]:
    text = _normalize(text)
    words = re.findall(r"[\wáéíóúüñçàèìòùâêîôûäëïöÿ']+", text)
    return words


LRC_ANCHOR = True  # corregir con los tiempos de la letra sincronizada las lineas que se van a otra repeticion
LRC_WINDOW = 6  # lineas a cada lado para saber el desfase local entre la letra de internet y este audio
LRC_MAX_DEV = 1.0  # segundos: si una linea se aparta mas que esto del desfase local, se vuelve a colocar


def _fix_with_lrc(lyr: Lyrics, lrc: list, vocals16: np.ndarray, aligner, log=print) -> None:
    """En estribillos repetidos el alineador global a veces pone una linea en otra repeticion (segundos antes o
    despues). Con letra sincronizada se mide el desfase local (mediana de las lineas vecinas: asi vale tambien
    para videoclips con otra intro o una parte de mas) y cada linea que se aparta se alinea sola en su hueco."""
    idx = [i for i, l in enumerate(lyr.lines) if l.words and lrc[i] is not None]
    if len(idx) < 5:
        return
    dev = {i: lyr.lines[i].words[0].start - lrc[i] for i in idx}
    local = {}
    for k, i in enumerate(idx):
        near = [dev[j] for j in idx[max(0, k - LRC_WINDOW):k + LRC_WINDOW + 1]]
        local[i] = float(np.median(near))
    n_audio = len(vocals16) / 16000
    fixed = 0
    for k, i in enumerate(idx):
        if abs(dev[i] - local[i]) <= LRC_MAX_DEV:
            continue
        t0 = max(0.0, lrc[i] + local[i] - 0.6)
        nxt = next((j for j in range(i + 1, len(lyr.lines)) if lrc[j] is not None), None)
        t1 = lrc[nxt] + local.get(nxt, local[i]) + 0.2 if nxt is not None else lrc[i] + local[i] + 8.0
        t1 = min(t1, n_audio)
        if t1 - t0 < 0.8:
            continue
        seg = vocals16[int(t0 * 16000):int(t1 * 16000)]
        words = split_words(lyr.lines[i].text)
        ws = aligner.align_words(seg, t0, words)
        # solo si ahi se oye mejor la linea que donde la puso el alineador global
        if len(ws) == len(words) and np.mean([w.score for w in ws]) >= np.mean([w.score for w in lyr.lines[i].words]):
            lyr.lines[i].words = ws
            fixed += 1
    if fixed:
        log(f"  {fixed} lineas recolocadas con los tiempos de la letra sincronizada")


def align_lyrics(lyr: Lyrics, vocals16: np.ndarray, log=print) -> Lyrics:
    """Rellena las palabras de cada linea con su tiempo real."""
    aligner = Aligner(lyr.language)
    lrc = [l.start for l in lyr.lines] if lyr.source == "lrclib-synced" else None
    try:
        # Se alinea TODA la letra contra TODA la cancion (alineacion forzada monotona). Los tiempos de internet
        # no se usan tal cual (suelen ser de otra version: otra intro, radio edit...), solo para corregir lineas
        # sueltas que el alineador pone en otra repeticion del estribillo (ver _fix_with_lrc).
        all_words = [(li, w) for li, l in enumerate(lyr.lines) for w in split_words(l.text)]
        ws = aligner.align_words(vocals16, 0.0, [w for _, w in all_words])
        by_text = iter(ws)
        cur = next(by_text, None)
        for l in lyr.lines:
            l.words = []
        for li, w in all_words:
            if cur is not None and cur.text == w:
                lyr.lines[li].words.append(cur)
                cur = next(by_text, None)
        if LRC_ANCHOR and lrc is not None:
            _fix_with_lrc(lyr, lrc, vocals16, aligner, log)
        for l in lyr.lines:
            if l.words:
                l.start, l.end = l.words[0].start, l.words[-1].end
    finally:
        aligner.close()
    n_words = sum(len(l.words) for l in lyr.lines)
    log(f"  letra alineada: {n_words} palabras en {len(lyr.lines)} lineas ({lyr.source})")
    return lyr


# ------------------------------------------------------------------ letra de internet + lo que se oye
# La letra de internet suele ser de la version de estudio: si el audio es otra version (videoclip con creditos o
# intro hablada, radio edit, directo), encajarla entera a la fuerza en toda la cancion la descuadra. Primero se
# transcribe lo que se oye (Whisper, con tiempo por palabra) y luego cada linea de la letra de internet se coloca
# donde se oyen sus palabras; las lineas que no se oyen en este audio se quitan.
SYNC_MIN_MATCH = 0.25  # si se oye menos de esta parte de la letra de internet, se usa el metodo anterior
SYNC_LINE_MIN = 0.3  # una linea se coloca si se oye al menos esta parte de sus palabras
SYNC_MARGIN = 0.8  # segundos de margen alrededor de cada linea para el alineador fino


def whisper_words(audio16: np.ndarray, language: str | None, log=print) -> list[Word]:
    """Lo que se oye en la pista de voz, palabra a palabra, con su tiempo."""
    import torch
    from transformers import pipeline

    dev = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log("  escuchando la voz con Whisper (palabra a palabra)...")
    asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, torch_dtype=dtype, device=dev,
                   chunk_length_s=30, batch_size=4)
    kw = {"task": "transcribe"}
    if language:
        kw["language"] = language
    try:
        res = asr({"raw": audio16, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs=kw)
    finally:
        del asr
        torch.cuda.empty_cache()
    # nivel de la voz: Whisper se inventa palabras (creditos, "gracias por ver") donde no se canta nada
    hop = 1600
    n = len(audio16) // hop
    rms = np.sqrt((audio16[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12) if n else np.zeros(0)
    db = 20 * np.log10(rms + 1e-9)
    loud = np.percentile(db, 95) if n else 0.0
    out = []
    for ch in res.get("chunks", []):
        ts = ch.get("timestamp") or (None, None)
        if ts[0] is None:
            continue
        a, b = float(ts[0]), float(ts[1] if ts[1] is not None else ts[0] + 0.3)
        for w in split_words(ch.get("text", "")):
            fa, fb = int(a * 10), max(int(b * 10) + 1, int(a * 10) + 1)
            if n and (fa >= n or np.max(db[fa:min(fb, n)]) < loud - 35):
                continue
            out.append(Word(w, a, b))
    log(f"  se oyen {len(out)} palabras")
    return out


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    import difflib

    a2 = unicodedata.normalize("NFD", a).encode("ascii", "ignore").decode()
    b2 = unicodedata.normalize("NFD", b).encode("ascii", "ignore").decode()
    if a2 == b2:
        return 0.95
    return difflib.SequenceMatcher(None, a2, b2).ratio()


def _match_words(ref: list[str], heard: list[str]) -> list[int]:
    """Para cada palabra de la letra, la palabra oida con la que se empareja (-1 si ninguna). Alineacion monotona
    en la que saltarse palabras (de la letra o de lo oido) cuesta poco, para no forzar emparejamientos."""
    n, m = len(ref), len(heard)
    gap = -0.15
    score = np.zeros((n + 1, m + 1), np.float32)
    score[1:, 0] = gap * np.arange(1, n + 1)
    score[0, 1:] = gap * np.arange(1, m + 1)
    move = np.zeros((n + 1, m + 1), np.int8)  # 0 emparejar, 1 saltar letra, 2 saltar oido
    move[1:, 0] = 1
    move[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _similar(ref[i - 1], heard[j - 1])
            diag = score[i - 1, j - 1] + (2 * s - 1 if s >= 0.6 else -1.0)
            up, left = score[i - 1, j] + gap, score[i, j - 1] + gap
            if diag >= up and diag >= left:
                score[i, j], move[i, j] = diag, 0
            elif up >= left:
                score[i, j], move[i, j] = up, 1
            else:
                score[i, j], move[i, j] = left, 2
    out = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        mv = move[i, j]
        if mv == 0:
            if _similar(ref[i - 1], heard[j - 1]) >= 0.6:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif mv == 1:
            i -= 1
        else:
            j -= 1
    return out


def sync_lyrics(lyr: Lyrics, heard: list[Word], vocals16: np.ndarray, log=print) -> Lyrics | None:
    """Coloca la letra de internet donde se oye cada linea y alinea sus palabras ahi. None si no encaja."""
    words = [(li, w) for li, l in enumerate(lyr.lines) for w in split_words(l.text)]
    if not words or not heard:
        return None
    match = _match_words([w for _, w in words], [h.text for h in heard])
    frac = float(np.mean([m >= 0 for m in match]))
    if frac < SYNC_MIN_MATCH:
        log(f"  solo se oye el {frac:.0%} de la letra de internet: se encaja en toda la cancion")
        return None
    # ventana de cada linea segun las palabras que se le oyen
    spans: dict[int, tuple[float, float, int, int]] = {}
    for (li, _), m in zip(words, match):
        if m < 0:
            continue
        a, b, k, _n = spans.get(li, (1e9, -1e9, 0, 0))
        spans[li] = (min(a, heard[m].start), max(b, heard[m].end), k + 1, 0)
    n_words = {li: sum(1 for l2, _ in words if l2 == li) for li in range(len(lyr.lines))}
    anch = {li: (a, b) for li, (a, b, k, _) in spans.items() if k >= max(1, SYNC_LINE_MIN * n_words[li])}
    total = len(vocals16) / 16000
    order = sorted(anch)
    if not order:
        return None
    # voz activa (para decidir si las lineas de antes de la primera / despues de la ultima reconocida se cantan)
    hop = 1600
    nfr = len(vocals16) // hop
    rms = np.sqrt((vocals16[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    voiced = db > np.percentile(db, 95) - 25

    def voiced_secs(a: float, b: float) -> float:
        return float(voiced[int(max(a, 0) * 10):int(max(b, 0) * 10)].sum()) / 10

    # bloques: cada linea reconocida es un bloque; las lineas sin reconocer entre dos reconocidas van juntas en
    # el hueco entre ellas (se cantan, solo que Whisper no las ha entendido)
    blocks: list[tuple[list[int], float, float]] = []
    first, last = order[0], order[-1]
    pre = list(range(0, first))
    if pre:
        lo, hi = 0.0, anch[first][0]
        need = 0.25 * sum(n_words[li] for li in pre)
        if voiced_secs(lo, hi) >= need:
            blocks.append((pre, lo, hi))
    for idx, li in enumerate(order):
        a, b = anch[li]
        prev_end = anch[order[idx - 1]][1] if idx else 0.0
        next_start = anch[order[idx + 1]][0] if idx + 1 < len(order) else total
        blocks.append(([li], max(prev_end, a - SYNC_MARGIN), min(next_start, b + SYNC_MARGIN * 2)))
        if idx + 1 < len(order):
            mid = list(range(li + 1, order[idx + 1]))
            if mid:
                blocks.append((mid, b, anch[order[idx + 1]][0]))
    post = list(range(last + 1, len(lyr.lines)))
    if post:
        lo, hi = anch[last][1], total
        need = 0.25 * sum(n_words[li] for li in post)
        if voiced_secs(lo, hi) >= need:
            blocks.append((post, lo, hi))
    blocks.sort(key=lambda x: x[1])
    lines_out: list[Line] = []
    aligner = Aligner(lyr.language)
    try:
        for lis, a, b in blocks:
            if b - a < 0.3:
                continue
            ws_all = [(li, w) for li in lis for w in split_words(lyr.lines[li].text)]
            seg = vocals16[int(a * 16000):int(b * 16000)]
            got = aligner.align_words(seg, a, [w for _, w in ws_all]) if len(seg) > 1600 else []
            it = iter(got)
            cur = next(it, None)
            per: dict[int, list[Word]] = {li: [] for li in lis}
            for li, w in ws_all:
                if cur is not None and cur.text == w:
                    per[li].append(cur)
                    cur = next(it, None)
            for li in lis:
                if per[li]:
                    line = Line(lyr.lines[li].text, per[li][0].start, per[li][-1].end, per[li])
                    lines_out.append(line)
    finally:
        aligner.close()
    lines_out.sort(key=lambda l: l.start)
    dropped = len(lyr.lines) - len(lines_out)
    log(f"  letra sincronizada con lo que se oye: {len(lines_out)} lineas ({frac:.0%} de sus palabras se oyen"
        f"{f', {dropped} lineas no se oyen en este audio y se quitan' if dropped else ''})")
    if not lines_out:
        return None
    return Lyrics(lines_out, lyr.source + "+oido", lyr.language)


def trim_lyrics(lyr: Lyrics, heard: list[Word], vocals16: np.ndarray, log=print) -> Lyrics:
    """Quita de la letra de internet las lineas del principio o del final que no se cantan en este audio (videoclip
    con otra intro o cortado antes, radio edit...). Lo demas se alinea como siempre con toda la cancion."""
    words = [(li, w) for li, l in enumerate(lyr.lines) for w in split_words(l.text)]
    if not words or not heard:
        return lyr
    match = _match_words([w for _, w in words], [h.text for h in heard])
    if float(np.mean([m >= 0 for m in match])) < SYNC_MIN_MATCH:
        return lyr
    n_words = {li: sum(1 for l2, _ in words if l2 == li) for li in range(len(lyr.lines))}
    hits: dict[int, list[float]] = {}
    for (li, _), m in zip(words, match):
        if m >= 0:
            hits.setdefault(li, []).append(heard[m].start)
            hits[li].append(heard[m].end)
    anch = sorted(li for li, t in hits.items() if len(t) // 2 >= max(1, SYNC_LINE_MIN * n_words[li]))
    if not anch:
        return lyr
    hop = 1600
    nfr = len(vocals16) // hop
    rms = np.sqrt((vocals16[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    voiced = db > np.percentile(db, 95) - 25
    total = len(vocals16) / 16000

    def voiced_secs(a: float, b: float) -> float:
        return float(voiced[int(max(a, 0) * 10):int(max(b, 0) * 10)].sum()) / 10

    first, last = anch[0], anch[-1]
    lo = first
    # lineas antes de la primera reconocida: se quedan las que caben en la voz que suena antes (de la ultima hacia atras)
    room = voiced_secs(0.0, min(hits[first]))
    for li in range(first - 1, -1, -1):
        room -= 0.25 * n_words[li]
        if room < 0:
            break
        lo = li
    hi = last
    room = voiced_secs(max(hits[last]), total)
    for li in range(last + 1, len(lyr.lines)):
        room -= 0.25 * n_words[li]
        if room < 0:
            break
        hi = li
    if lo == 0 and hi == len(lyr.lines) - 1:
        return lyr
    log(f"  {lo + len(lyr.lines) - 1 - hi} lineas de la letra de internet no se cantan en este audio: se quitan")
    return Lyrics(lyr.lines[lo:hi + 1], lyr.source, lyr.language)
