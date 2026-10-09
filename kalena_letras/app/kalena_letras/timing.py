"""Pone tiempos por palabra a las líneas de una letra a partir de las emisiones de cada voz.

- Voz principal: lo que no va entre paréntesis, alineado sobre la voz principal aislada.
- Coros: lo que va entre paréntesis, alineado sobre los coros aislados.
- Si una línea no queda clara sobre la voz principal (la separación no fue limpia en ese trozo),
  se prueba sobre la voz completa (principal + coros) y se queda la mejor.
- Una línea solo recibe tiempos por palabra si la confianza llega al mínimo; si no, se queda como
  estaba (con su tiempo de línea), sin inventar nada.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .align import ENERGY_HOP_MS, refine_start, viterbi, word_timings
from .lyrics import Line, Word

# Confianza mínima (media de las letras) para dar por buenas las palabras de una línea. Se ajusta
# con los informes de la fase de prueba.
MIN_LINE_CONFIDENCE = 0.30
MIN_BACKING_CONFIDENCE = 0.35
# Una palabra de coro solo se da por buena si la IA la reconoce con esta seguridad…
MIN_BACKING_WORD_CONFIDENCE = 0.5
# …y si en la pista de coros hay voz de verdad en ese momento: no más de esto por debajo de lo
# más fuerte de los coros (lo demás son restos de la voz principal).
BACKING_ENERGY_RANGE_DB = 25.0
# Margen alrededor de cada línea al buscarla (la voz puede entrar algo antes o terminar después).
LINE_MARGIN_MS = 400
# Si la primera palabra cae a más de esto del tiempo de línea original, algo no cuadra.
MAX_LINE_SHIFT_MS = 1_500
MIN_WORD_MS = 60


@dataclass
class Voice:
    """Una voz aislada: sus emisiones (T, C), lo que dura cada fotograma y su energía."""
    name: str
    emissions: np.ndarray
    frame_ms: float
    energy: np.ndarray

    @property
    def duration_ms(self) -> float:
        return self.emissions.shape[0] * self.frame_ms


def _align_words(voice: Voice, words: list[Word], token_ids: Callable[[str], list[int]],
                 start_ms: float, end_ms: float, before: list[Word] = (), after: list[Word] = ()):
    """Alinea `words` dentro de [start_ms, end_ms) de `voice`. Devuelve (timings, confianza) o None.

    `before` y `after` son las palabras de la línea anterior y la siguiente: se alinean también
    (y se descartan) para que ningún sonido de las líneas vecinas se confunda con esta.
    """
    def usable(ws):
        return [w for w in ws if token_ids(w.tokens)]

    before, words, after = usable(before), usable(words), usable(after)
    if not words:
        return None
    every = list(before) + words + list(after)
    ids_per_word = [token_ids(w.tokens) for w in every]
    a = max(0, int(start_ms / voice.frame_ms))
    b = min(voice.emissions.shape[0], int(end_ms / voice.frame_ms) + 1)
    if b - a < 2:
        return None
    targets = [i for ids in ids_per_word for i in ids]
    al = viterbi(voice.emissions[a:b], targets)
    if al is None:
        return None
    timings = word_timings(al, [len(ids) for ids in ids_per_word], voice.frame_ms, offset_ms=a * voice.frame_ms)
    mine = timings[len(before):len(before) + len(words)]
    n0 = sum(len(ids) for ids in ids_per_word[:len(before)])
    n1 = n0 + sum(len(ids) for ids in ids_per_word[len(before):len(before) + len(words)])
    return list(zip(words, mine)), float(al.token_conf[n0:n1].mean())


def _apply(pairs, voice: Voice, earliest_ms: int):
    """Copia los tiempos a las palabras con el ajuste fino y sin solaparse."""
    prev = earliest_ms
    for word, t in pairs:
        start = refine_start(t.start_ms, voice.energy, earliest_ms=prev, hop_ms=ENERGY_HOP_MS)
        start = max(start, prev)
        end = max(t.end_ms, start + MIN_WORD_MS)
        word.start_ms, word.end_ms, word.confidence = start, end, t.confidence
        prev = start + 1


def _apply_backing(pairs, voice: Voice, earliest_ms: int) -> int:
    """Como `_apply`, pero solo con las palabras de coros que de verdad suenan en su pista."""
    loud = float(np.percentile(voice.energy, 95)) if voice.energy.size else 0.0
    keep = []
    for word, t in pairs:
        a = int(t.start_ms / ENERGY_HOP_MS)
        b = max(a + 1, int(t.end_ms / ENERGY_HOP_MS))
        level = float(voice.energy[a:b].mean()) if a < voice.energy.size else -120.0
        if t.confidence >= MIN_BACKING_WORD_CONFIDENCE and level >= loud - BACKING_ENERGY_RANGE_DB and t.start_ms >= earliest_ms:
            keep.append((word, t))
    _apply(keep, voice, earliest_ms)
    return len(keep)


def _windows(lines: list[Line], total_ms: float, global_starts: dict[int, int] | None) -> dict[int, tuple[float, float]]:
    """Dónde buscar cada línea: desde su tiempo hasta el de la siguiente, con margen."""
    starts = {}
    for i, l in enumerate(lines):
        if not (l.lead_words or l.backing_words):
            continue
        s = l.start_ms if l.start_ms is not None else (global_starts or {}).get(i)
        if s is not None:
            starts[i] = s
    order = sorted(starts)
    out = {}
    for n, i in enumerate(order):
        s = starts[i]
        nxt = starts[order[n + 1]] if n + 1 < len(order) else total_ms
        out[i] = (max(0.0, s - LINE_MARGIN_MS), min(total_ms, max(nxt, s + 1_000) + LINE_MARGIN_MS))
    return out


def global_line_starts(lines: list[Line], voice: Voice, token_ids) -> dict[int, int] | None:
    """Letra sin tiempos de línea: alinea toda la voz principal de una vez para saber dónde va cada línea."""
    words, owner = [], []
    for i, l in enumerate(lines):
        for w in l.lead_words:
            if token_ids(w.tokens):
                words.append(w)
                owner.append(i)
    if not words:
        return None
    res = _align_words(voice, words, token_ids, 0, voice.duration_ms)
    if res is None:
        return None
    pairs, _ = res
    starts: dict[int, int] = {}
    for (w, t), i in zip(pairs, owner):
        starts.setdefault(i, t.start_ms)
    return starts


def line_windows(lines: list[Line], lead: Voice, token_ids, synced: bool) -> dict[int, tuple[float, float]]:
    """Dónde va cada línea en la canción (con margen): por su tiempo o, sin tiempos, alineando toda la letra."""
    global_starts = None if synced else global_line_starts(lines, lead, token_ids)
    return _windows(lines, lead.duration_ms, global_starts)


def time_lines(lines: list[Line], lead: Voice, backing: Voice | None, full: Callable[[], Voice | None],
               token_ids: Callable[[str], list[int]], synced: bool) -> dict:
    """Pone tiempos por palabra a `lines` (las modifica). Devuelve un resumen para el informe."""
    windows = line_windows(lines, lead, token_ids, synced)
    full_voice: Voice | None = None
    report = {"lines": [], "fallback_full": 0, "backing_from_full": 0, "timed": 0, "total": 0}
    order = [i for i in sorted(windows) if lines[i].lead_words]
    for n, i in enumerate(order):
        line = lines[i]
        report["total"] += 1
        prev = order[n - 1] if n > 0 else None
        nxt = order[n + 1] if n + 1 < len(order) else None
        # La ventana abarca también las líneas vecinas, que van de contexto.
        a = windows[prev][0] if prev is not None else windows[i][0]
        b = windows[nxt][1] if nxt is not None else windows[i][1]
        before = lines[prev].lead_words if prev is not None else []
        after = lines[nxt].lead_words if nxt is not None else []
        used = lead.name
        res = _align_words(lead, line.lead_words, token_ids, a, b, before, after)
        conf = res[1] if res else 0.0
        if conf < MIN_LINE_CONFIDENCE + 0.15:
            if full_voice is None:
                full_voice = full()
            if full_voice is not None:
                alt = _align_words(full_voice, line.lead_words, token_ids, a, b, before, after)
                if alt and alt[1] > conf:
                    res, conf, used = alt, alt[1], full_voice.name
                    report["fallback_full"] += 1
        voice = lead if used == lead.name else full_voice
        ok = res is not None and conf >= MIN_LINE_CONFIDENCE
        if ok and synced and line.start_ms is not None:
            first = res[0][0][1].start_ms
            if abs(first - line.start_ms) > MAX_LINE_SHIFT_MS:
                ok = False
        if ok:
            wa, wb = windows[i]
            _apply(res[0], voice, earliest_ms=int(max(0, wa)))
            # Coros: sobre su propia voz, dentro de la ventana de la línea.
            if line.backing_words:
                first_lead = min(w.start_ms for w in line.lead_words if w.start_ms is not None)
                earliest = max(int(wa), first_lead - 300)
                if backing is not None:
                    bres = _align_words(backing, line.backing_words, token_ids, wa, wb)
                    if bres and bres[1] >= MIN_BACKING_CONFIDENCE:
                        _apply_backing(bres[0], backing, earliest_ms=earliest)
                # Si en la pista de coros no estaban (el modelo los dejó con la voz principal), se busca
                # la línea entera, en su orden, sobre la voz completa y se toman de ahí los coros.
                missing = [w for w in line.backing_words if w.start_ms is None]
                if missing:
                    if full_voice is None:
                        full_voice = full()
                    if full_voice is not None:
                        fres = _align_words(full_voice, line.words, token_ids, wa, wb)
                        if fres:
                            pairs = [(w, t) for w, t in fres[0] if any(w is m for m in missing)]
                            report["backing_from_full"] += _apply_backing(pairs, full_voice, earliest_ms=earliest)
            line.timed = True
            line.confidence = conf
            report["timed"] += 1
        report["lines"].append({"index": i, "line": line.text, "confidence": round(conf, 3), "voice": used, "timed": ok})
    return report
