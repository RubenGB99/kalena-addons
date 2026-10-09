"""Letras: lo que Jellyfin devuelve, cómo se divide en palabras y cómo se escribe el LRC.

El LRC es el mismo que escribe el editor de Kalena: `[mm:ss.xx]` al principio de cada línea y
`<mm:ss.xx>` delante de cada palabra (LRC mejorado), con el final de la última palabra al final.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

TICKS_PER_MS = 10_000

# Letras que entiende el modelo de alineación (MMS): a-z y el apóstrofo.
ALIGN_CHARS = set("abcdefghijklmnopqrstuvwxyz'")


@dataclass
class Word:
    text: str            # tal cual aparece en la letra («¿Qué»)
    start_char: int      # posición en el texto de la línea
    end_char: int
    backing: bool        # va entre paréntesis: lo suelen cantar los coros
    tokens: str          # letras para el alineador («que»); puede quedar vacío («2», «—»)
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float = 0.0


@dataclass
class Line:
    text: str
    start_ms: int | None          # tiempo de la línea en la letra original (si la tiene)
    words: list[Word] = field(default_factory=list)
    had_word_times: bool = False  # la letra original ya traía tiempos por palabra
    original: str = ""            # la línea tal como estaba, en LRC (con sus tiempos por palabra si los tenía)
    confidence: float = 0.0
    timed: bool = False           # la IA le ha puesto tiempos por palabra

    @property
    def lead_words(self) -> list[Word]:
        return [w for w in self.words if not w.backing and w.tokens]

    @property
    def backing_words(self) -> list[Word]:
        return [w for w in self.words if w.backing and w.tokens]


def to_tokens(word: str) -> str:
    """«Canción» -> «cancion», «¿Qué?» -> «que», «I'm» -> «i'm». Cualquier alfabeto se pasa a latino."""
    text = unicodedata.normalize("NFKC", word).replace("’", "'").replace("`", "'")
    try:
        from unidecode import unidecode
        text = unidecode(text)
    except ImportError:  # pragma: no cover - solo en pruebas sin la dependencia
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    out = "".join(c for c in text if c in ALIGN_CHARS)
    return out.strip("'")


def split_words(text: str) -> list[Word]:
    """Palabras de una línea, marcando las que van entre paréntesis (los coros)."""
    words: list[Word] = []
    depth = 0
    for m in re.finditer(r"\S+", text):
        raw = m.group(0)
        opens = raw.count("(") + raw.count("[")
        closes = raw.count(")") + raw.count("]")
        backing = depth > 0 or opens > 0
        depth = max(0, depth + opens - closes)
        words.append(Word(raw, m.start(), m.end(), backing, to_tokens(raw)))
    return words


def from_jellyfin(payload: dict) -> list[Line]:
    """Las líneas de `GET /Audio/{id}/Lyrics`."""
    lines: list[Line] = []
    for item in payload.get("Lyrics") or []:
        text = (item.get("Text") or "").rstrip()
        start = item.get("Start")
        start_ms = int(start) // TICKS_PER_MS if start is not None else None
        line = Line(text=text, start_ms=start_ms, words=split_words(text))
        line.had_word_times = bool(item.get("Cues"))
        line.original = _item_lrc(item)
        lines.append(line)
    return lines


def is_synced(lines: list[Line]) -> bool:
    return any(l.start_ms is not None for l in lines if l.text.strip())


def has_word_times(lines: list[Line]) -> bool:
    return any(l.had_word_times for l in lines)


def stamp(ms: int) -> str:
    """`mm:ss.xx` (los minutos pueden pasar de 99), igual que Kalena."""
    t = max(0, int(ms))
    return "%02d:%02d.%02d" % (t // 60_000, (t // 1000) % 60, (t % 1000) // 10)


def write_lrc(lines: list[Line]) -> str:
    """El LRC final: las líneas con tiempos por palabra donde la IA está segura, y el resto como estaban."""
    out = []
    for line in lines:
        text = line.text
        if line.timed and line.words:
            timed = [w for w in line.words if w.start_ms is not None]
            start = min(w.start_ms for w in timed)
            parts = ["[%s]" % stamp(start), text[: line.words[0].start_char]]
            for i, w in enumerate(line.words):
                end_char = line.words[i + 1].start_char if i + 1 < len(line.words) else len(text)
                if w.start_ms is not None:
                    parts.append("<%s>" % stamp(w.start_ms))
                parts.append(text[w.start_char:end_char])
            last = max((w for w in timed), key=lambda w: w.start_ms)
            if last.end_ms is not None:
                parts.append("<%s>" % stamp(last.end_ms))
            out.append("".join(parts))
        elif line.had_word_times and line.original:
            # La IA no la tiene clara: se queda exactamente como estaba, con sus tiempos por palabra.
            out.append(line.original)
        elif line.start_ms is not None:
            out.append("[%s]%s" % (stamp(line.start_ms), text))
        else:
            out.append(text)
    return "\n".join(out) + "\n"


def _item_lrc(item: dict) -> str:
    """Una línea de `GET /Audio/{id}/Lyrics` como LRC, con sus tiempos por palabra si los tiene."""
    text = item.get("Text") or ""
    start = item.get("Start")
    cues = item.get("Cues") or []
    if start is None:
        return text
    if not cues:
        return "[%s]%s" % (stamp(int(start) // TICKS_PER_MS), text)
    cues = sorted(cues, key=lambda c: c.get("Position", 0))
    parts = ["[%s]" % stamp(int(start) // TICKS_PER_MS), text[: cues[0].get("Position", 0)]]
    for i, c in enumerate(cues):
        pos = c.get("Position", 0)
        end = cues[i + 1].get("Position", len(text)) if i + 1 < len(cues) else len(text)
        parts.append("<%s>" % stamp(int(c.get("Start", 0)) // TICKS_PER_MS))
        parts.append(text[pos:end])
    last_end = cues[-1].get("End")
    if last_end:
        parts.append("<%s>" % stamp(int(last_end) // TICKS_PER_MS))
    return "".join(parts)


def original_lrc(payload: dict) -> str:
    """La letra que había antes, como LRC (para que «Recuperar original» de Kalena funcione)."""
    return "\n".join(_item_lrc(item) for item in payload.get("Lyrics") or []) + "\n"


def fingerprint(text: str) -> str:
    """El texto cantado, sin tiempos, para comparar dos versiones de una letra."""
    plain = re.sub(r"\[[0-9:.]+\]|<[0-9:.]+>", "", text)
    lines = [re.sub(r"\s+", " ", l).strip() for l in plain.splitlines()]
    return "\n".join(l for l in lines if l)
