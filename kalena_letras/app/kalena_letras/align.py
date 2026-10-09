"""Alineación: en qué instante se canta cada palabra.

1. Un modelo de reconocimiento de voz multilingüe (MMS de Meta, preparado para alinear) escucha la
   voz ya aislada y da, cada 20 ms, la probabilidad de cada letra (las «emisiones»).
2. Con la letra conocida, el camino más probable (Viterbi sobre CTC) coloca cada letra en su
   instante. No adivina palabras: solo busca dónde suenan las que ya están escritas.
3. Ajuste fino: el comienzo de cada palabra se lleva al arranque real de la voz (subida de
   energía) si está a menos de ~0,1 s, para que no vaya adelantado ni retrasado.

Todo lo que no es el modelo es numpy puro, para poder probarlo sin descargar nada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
BLANK = 0


# ---------------------------------------------------------------- Viterbi (CTC)

@dataclass
class Alignment:
    """Para cada letra del texto: primer y último fotograma y su confianza (0-1)."""
    token_first: np.ndarray
    token_last: np.ndarray
    token_conf: np.ndarray
    score: float  # confianza media


def viterbi(log_probs: np.ndarray, targets: list[int]) -> Alignment | None:
    """Camino CTC más probable que pasa por todas las letras de `targets`, en orden.

    `log_probs`: (T, C) log-probabilidades por fotograma. Devuelve None si no caben (audio más
    corto que el texto).
    """
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0 or T == 0:
        return None
    # Secuencia extendida: blank, t1, blank, t2, ..., tL, blank
    ext = np.full(2 * L + 1, BLANK, dtype=np.int64)
    ext[1::2] = targets
    S = ext.shape[0]
    repeats = sum(1 for i in range(1, L) if targets[i] == targets[i - 1])
    if T < L + repeats:
        return None
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])

    neg = -1e30
    alpha = np.full(S, neg)
    alpha[0] = log_probs[0, ext[0]]
    alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([neg], alpha[:-1]))
        skip = np.concatenate(([neg, neg], alpha[:-2]))
        skip = np.where(can_skip, skip, neg)
        best = np.maximum(stay, np.maximum(step, skip))
        choice = np.where(best == stay, 0, np.where(best == step, 1, 2)).astype(np.int8)
        alpha = best + log_probs[t, ext]
        back[t] = choice
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[end] <= neg / 2:
        return None
    states = np.empty(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= back[t, s]
    if states[0] > 1:
        return None

    first = np.full(L, -1, dtype=np.int64)
    last = np.full(L, -1, dtype=np.int64)
    conf = np.zeros(L)
    for t, st in enumerate(states):
        if st % 2 == 1:
            k = st // 2
            if first[k] < 0:
                first[k] = t
            last[k] = t
            conf[k] = max(conf[k], math.exp(log_probs[t, ext[st]]))
    if (first < 0).any():
        return None
    return Alignment(first, last, conf, float(conf.mean()))


# ---------------------------------------------------------------- words

@dataclass
class WordTiming:
    start_ms: int
    end_ms: int
    confidence: float


def word_timings(alignment: Alignment, word_lengths: list[int], frame_ms: float, offset_ms: float = 0.0) -> list[WordTiming]:
    """Agrupa las letras alineadas en palabras (`word_lengths` = letras de cada palabra)."""
    out = []
    k = 0
    for n in word_lengths:
        a, b = k, k + n
        start = offset_ms + alignment.token_first[a] * frame_ms
        end = offset_ms + (alignment.token_last[b - 1] + 1) * frame_ms
        out.append(WordTiming(int(round(start)), int(round(end)), float(alignment.token_conf[a:b].mean())))
        k = b
    return out


# ---------------------------------------------------------------- fine adjustment

ENERGY_HOP_MS = 10


def energy_db(samples: np.ndarray, sample_rate: int = SAMPLE_RATE, hop_ms: int = ENERGY_HOP_MS) -> np.ndarray:
    """Energía de la voz cada `hop_ms` (dB), para encontrar dónde arranca cada palabra."""
    hop = max(1, int(sample_rate * hop_ms / 1000))
    win = hop * 2
    if samples.size < win:
        return np.full(1, -120.0)
    n = 1 + (samples.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = samples[idx].astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-9)


def refine_start(start_ms: int, energy: np.ndarray, earliest_ms: int, hop_ms: int = ENERGY_HOP_MS,
                 before_ms: int = 120, after_ms: int = 80, min_rise_db: float = 9.0) -> int:
    """Lleva el comienzo de una palabra al arranque real de la voz, si hay uno claro cerca.

    Busca en [start-120 ms, start+80 ms] el punto donde la energía cruza la mitad entre el
    silencio previo y el pico de la palabra. Si no hay una subida clara (≥ 9 dB, palabra ligada
    a la anterior), deja el tiempo como estaba.
    """
    n = energy.size
    i0 = int(start_ms / hop_ms)
    lo = max(0, int((start_ms - before_ms) / hop_ms), int(math.ceil(earliest_ms / hop_ms)))
    hi = min(n - 1, int((start_ms + after_ms) / hop_ms))
    if hi <= lo or i0 >= n:
        return start_ms
    floor_lo = max(0, lo - int(150 / hop_ms))
    floor = energy[floor_lo:lo + 1].min()
    peak = energy[i0:min(n, i0 + int(200 / hop_ms))].max() if i0 < n else energy[lo:hi + 1].max()
    if peak - floor < min_rise_db:
        return start_ms
    half = floor + (peak - floor) / 2
    for i in range(lo, hi + 1):
        if energy[i] >= half and (i == 0 or energy[i - 1] < half):
            return int(i * hop_ms)
    return start_ms


# ---------------------------------------------------------------- the model

class EmissionModel:
    """MMS (Meta) para alineación forzada, vía torchaudio. Se descarga una vez a `model_dir`."""

    def __init__(self, model_dir: str, threads: int = 4):
        import os
        os.environ.setdefault("TORCH_HOME", model_dir)
        import torch
        import torchaudio
        torch.set_num_threads(threads)
        self.torch = torch
        bundle = torchaudio.pipelines.MMS_FA
        # Sin el comodín (*) del modelo: probado con coros y «adlibs» sin escribir, bajaba la
        # confianza de todas las palabras a la mitad y se perdían líneas enteras (y sin él solo
        # una palabra de 28 se desviaba).
        self.model = bundle.get_model(with_star=False)
        self.model.eval()
        self.dictionary = bundle.get_dict(star=None)

    def token_ids(self, tokens: str) -> list[int]:
        return [self.dictionary[c] for c in tokens if c in self.dictionary]

    def emissions(self, samples: np.ndarray, window_s: int = 30, context_s: int = 2) -> tuple[np.ndarray, float]:
        """Log-probabilidades (T, C) del audio entero, por ventanas de 30 s con 2 s de contexto."""
        torch = self.torch
        audio = samples.astype(np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        window = window_s * SAMPLE_RATE
        context = context_s * SAMPLE_RATE
        total = audio.shape[0]
        expected = max(1, total // 320)
        parts = []
        pos = 0
        with torch.inference_mode():
            while pos < total:
                a = max(0, pos - context)
                b = min(total, pos + window + context)
                chunk = torch.from_numpy(audio[a:b]).unsqueeze(0)
                out, _ = self.model(chunk)
                logp = torch.log_softmax(out[0], dim=-1).numpy()
                per_sample = logp.shape[0] / max(1, b - a)
                skip = int(round((pos - a) * per_sample))
                keep = int(round((min(total, pos + window) - pos) * per_sample))
                parts.append(logp[skip:skip + keep])
                pos += window
        em = np.concatenate(parts, axis=0)
        if em.shape[0] > expected:
            em = em[:expected]
        frame_ms = total / SAMPLE_RATE * 1000.0 / em.shape[0]
        return em.astype(np.float32), frame_ms
