#!/usr/bin/env python3
"""Lo que hará Kalena en el móvil, escrito en numpy + ONNX Runtime: la especificación del motor.

Usa solo los modelos de `exportar.py` (nada de PyTorch ni audio-separator) y el mismo alineador
del complemento, así que la autoprueba mide exactamente la precisión que tendrá el móvil:

    python3 movil/referencia.py <carpeta de modelos ONNX> fp32|int8 cancion.wav letra.json resultado.json

Pasos que el móvil tiene que copiar tal cual:
1. Separación por trozos de `trozo_muestras` (estéreo, 44,1 kHz), uno cada `paso_segundos`; el
   último, pegado al final. Cada trozo: STFT (ventana de Hann periódica, centrada con reflejo),
   máscara de la red, multiplicación compleja, STFT inversa; los trozos se suman con una ventana
   de Hamming (simétrica) y se divide por la suma de ventanas.
2. Alineación: audio a 16 kHz mono normalizado al 0,9 del pico, en ventanas de 30 s con 2 s de
   contexto a cada lado.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalena_letras", "app"))

from kalena_letras import lyrics as L  # noqa: E402
from kalena_letras.__main__ import Aligner, align_audio  # noqa: E402
from kalena_letras.audio import read_stereo, to_wav, write_wav  # noqa: E402


def session(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def hann(n: int) -> np.ndarray:
    """torch.hann_window(n) (periódica)."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float64)


def hamming(n: int) -> np.ndarray:
    """scipy.signal.windows.hamming(n) (simétrica)."""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def stft(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """torch.stft(center=True, pad_mode="reflect"): (frecuencias, fotogramas) complejo."""
    pad = n_fft // 2
    xp = np.pad(x, pad, mode="reflect")
    frames = 1 + (xp.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(xp[idx] * window, axis=1).T


def istft(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """torch.istft(center=True) sin `length`: hop·(fotogramas-1) muestras."""
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * window
    count = frames.shape[0]
    total = n_fft + hop * (count - 1)
    out = np.zeros(total)
    env = np.zeros(total)
    w2 = window ** 2
    for t in range(count):
        out[t * hop:t * hop + n_fft] += frames[t]
        env[t * hop:t * hop + n_fft] += w2
    out /= np.where(env > 1e-11, env, 1.0)
    pad = n_fft // 2
    return out[pad:pad + hop * (count - 1)]


class OnnxSeparator:
    def __init__(self, folder: str, variant: str):
        self.folder = folder
        self.variant = variant
        self.manifest = json.load(open(os.path.join(folder, "manifiesto.json"), encoding="utf-8"))
        self.sessions = {}
        self.seconds = 0.0

    def _model(self, name: str):
        if name not in self.sessions:
            spec = self.manifest["separacion"][name]
            self.sessions[name] = (session(os.path.join(self.folder, spec["archivos"][self.variant])), spec)
        return self.sessions[name]

    def demix(self, name: str, mix: np.ndarray) -> list[np.ndarray]:
        """`mix` (canales, muestras) -> una señal (canales, muestras) por voz del modelo."""
        sess, spec = self._model(name)
        n_fft, hop, win, chunk = spec["n_fft"], spec["salto"], spec["ventana"], spec["trozo_muestras"]
        channels = spec["canales"]
        window = hann(win)
        weights = hamming(chunk)
        step = int(spec["paso_segundos"] * spec["frecuencia_muestreo"])
        length = mix.shape[1]
        if length < chunk:  # el móvil rellena con silencio las canciones de menos de un trozo
            mix = np.pad(mix, ((0, 0), (0, chunk - length)))
        total = mix.shape[1]
        stems = len(spec["voces"])
        result = np.zeros((stems, channels, total))
        counter = np.zeros(total)
        for i in range(0, total, step):
            start = i if i + chunk <= total else total - chunk
            part = mix[:, start:start + chunk]
            specs = [stft(part[c], n_fft, hop, window) for c in range(channels)]  # (F, T) cada uno
            ri = np.stack([np.stack([s.real, s.imag], axis=-1) for s in specs], axis=1)  # (F, canales, T, 2)
            ri = ri.reshape(-1, ri.shape[2], 2)[None].astype(np.float32)  # (1, F·canales, T, 2)
            t = time.time()
            mask = sess.run(None, {"stft": ri})[0][0]  # (voces, F·canales, T, 2)
            self.seconds += time.time() - t
            for k in range(stems):
                m = mask[k].reshape(-1, channels, mask.shape[2], 2)
                for c in range(channels):
                    mc = m[:, c, :, 0] + 1j * m[:, c, :, 1]
                    y = istft(specs[c] * mc, n_fft, hop, window)
                    result[k, c, start:start + chunk] += y[:chunk] * weights
            counter[start:start + chunk] += weights
        out = result / np.maximum(counter, 1e-10)
        return [out[k, :, :length] for k in range(stems)]

    def _run(self, name: str, mix_wav: str, out_dir: str, prefix: str):
        sess, spec = self._model(name)
        mix = read_stereo(mix_wav, spec["frecuencia_muestreo"]).T.astype(np.float64)
        peak = np.abs(mix).max()
        if peak > 0.9:
            mix *= 0.9 / peak
        voices = self.demix(name, mix)
        if spec["secundaria_es_resto"]:
            voices.append(mix - voices[0])
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for k, v in enumerate(voices[:2]):
            peak = np.abs(v).max()
            if peak > 0.9:
                v = v * (0.9 / peak)
            path = os.path.join(out_dir, f"{prefix}_{k + 1}.wav")
            write_wav(path, v.T.astype(np.float32), spec["frecuencia_muestreo"])
            paths.append(path)
        return paths

    def karaoke(self, mix_wav: str, out_dir: str):
        a, b = self._run("karaoke", mix_wav, out_dir, "karaoke")
        return a, b

    def vocals(self, mix_wav: str, out_dir: str) -> str:
        spec = self.manifest["separacion"]["voces"]
        paths = self._run("voces", mix_wav, out_dir, "voz")
        names = spec["voces"] + (["resto"] if spec["secundaria_es_resto"] else [])
        for name, path in zip(names, paths):
            if "vocal" in name.lower():
                return path
        return paths[0]


class OnnxEmissionModel:
    def __init__(self, folder: str, variant: str):
        manifest = json.load(open(os.path.join(folder, "manifiesto.json"), encoding="utf-8"))
        spec = manifest["alineacion"]
        self.dictionary = spec["letras"]
        self.window_s, self.context_s = spec["ventana_s"], spec["contexto_s"]
        self.sess = session(os.path.join(folder, spec["archivos"][variant]))
        self.seconds = 0.0

    def token_ids(self, tokens: str) -> list[int]:
        return [self.dictionary[c] for c in tokens if c in self.dictionary]

    def emissions(self, samples: np.ndarray):
        """Igual que EmissionModel.emissions del complemento."""
        rate = 16_000
        audio = samples.astype(np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        window, context = self.window_s * rate, self.context_s * rate
        total = audio.shape[0]
        expected = max(1, total // 320)
        parts = []
        pos = 0
        while pos < total:
            a = max(0, pos - context)
            b = min(total, pos + window + context)
            t = time.time()
            logp = self.sess.run(None, {"audio": audio[None, a:b]})[0][0]
            self.seconds += time.time() - t
            per_sample = logp.shape[0] / max(1, b - a)
            skip = int(round((pos - a) * per_sample))
            keep = int(round((min(total, pos + window) - pos) * per_sample))
            parts.append(logp[skip:skip + keep])
            pos += window
        em = np.concatenate(parts, axis=0)
        if em.shape[0] > expected:
            em = em[:expected]
        frame_ms = total / rate * 1000.0 / em.shape[0]
        return em.astype(np.float32), frame_ms


class OnnxAligner(Aligner):
    def __init__(self, folder: str, variant: str):
        self.separator = OnnxSeparator(folder, variant)
        self.model = OnnxEmissionModel(folder, variant)

    def emission_model(self):
        return self.model


def main(argv: list[str]) -> int:
    import tempfile
    folder, variant, song, lyrics_json, out = argv[:5]
    spec = json.load(open(lyrics_json, encoding="utf-8"))
    lines = [L.Line(x["text"], x.get("start_ms"), L.split_words(x["text"])) for x in spec]
    synced = L.is_synced(lines)
    aligner = OnnxAligner(folder, variant)
    with tempfile.TemporaryDirectory() as work:
        mix = os.path.join(work, "mezcla.wav")
        to_wav(song, mix, 44_100, 2)
        t = time.time()
        report, score, t_sep, t_align = align_audio(aligner, mix, lines, synced, work)
    words = [{"text": w.text, "backing": w.backing, "start_ms": w.start_ms, "end_ms": w.end_ms,
              "confidence": round(w.confidence, 3), "line": i}
             for i, l in enumerate(lines) for w in l.words]
    json.dump({"words": words, "report": report, "voices": score, "separation_s": round(t_sep),
               "alignment_s": round(t_align), "total_s": round(time.time() - t),
               "red_separacion_s": round(aligner.separator.seconds), "red_alineacion_s": round(aligner.model.seconds),
               "lrc": L.write_lrc(lines)},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{variant}: redes de separación {aligner.separator.seconds:.0f} s, de alineación {aligner.model.seconds:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
