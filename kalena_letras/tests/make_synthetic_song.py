#!/usr/bin/env python3
"""Canción sintética para la autoprueba: palabras habladas en instantes conocidos sobre música y coros.

Necesita espeak-ng y ffmpeg. Escribe en `out_dir`: cancion.wav, letra_sincronizada.json,
letra_sin_tiempos.json y verdad.json (cuándo empieza de verdad cada palabra).
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

SR = 44_100
LINES = [
    (2.0, "Hoy camino por la ciudad"),
    (6.0, "y el viento me lleva contigo (oh oh)"),
    (11.0, "cuando cae la noche"),
    (15.0, "todo vuelve a empezar (otra vez)"),
    (20.0, "canto bajito tu nombre"),
    (24.0, "y la luna me responde"),
]


def speak(text, voice, pitch, path):
    subprocess.run(["espeak-ng", "-v", voice, "-p", str(pitch), "-s", "150", "-w", path, text], check=True)
    with wave.open(path) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    idx = np.linspace(0, len(data) - 1, int(len(data) * SR / sr))
    data = np.interp(idx, np.arange(len(data)), data)
    nz = np.nonzero(np.abs(data) > 0.02 * np.abs(data).max())[0]
    onset = nz[0] / SR if nz.size else 0.0
    return data, onset


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = 36.0
    lead = np.zeros(int(total * SR))
    back = np.zeros_like(lead)
    truth, synced, plain = [], [], []
    tmp = os.path.join(out_dir, "w.wav")
    prev_end = 0.0
    for start, text in LINES:
        # Como en una canción: cada línea empieza cuando ha terminado la anterior (coros incluidos).
        t = max(start, prev_end + 0.8)
        first = None
        in_paren = False
        for raw in text.split():
            opens = "(" in raw
            word = raw.strip("()")
            is_back = in_paren or opens
            in_paren = (in_paren or opens) and ")" not in raw
            audio, onset = speak(word, "es", 70 if is_back else 40, tmp)
            target = back if is_back else lead
            pos = int((t - onset) * SR)
            if pos < 0:
                pos = 0
            end = min(len(target), pos + len(audio))
            target[pos:end] += audio[: end - pos] * (0.5 if is_back else 1.0)
            truth.append({"text": raw, "start_ms": int(round(t * 1000)), "backing": is_back})
            if first is None:
                first = t
            t += len(audio) / SR - onset + 0.12
            prev_end = t
        synced.append({"text": text, "start_ms": int(round(first * 1000))})
        plain.append({"text": text, "start_ms": None})
    # Música: acordes que cambian cada 2 s, un bajo y un ruido suave tipo platillos.
    tt = np.arange(len(lead)) / SR
    chords = [(220, 277, 330), (196, 247, 294), (174, 220, 262), (196, 247, 294)]
    music = np.zeros_like(lead)
    for k in range(int(total / 2)):
        a, b = int(k * 2 * SR), int((k + 1) * 2 * SR)
        for f in chords[k % 4]:
            music[a:b] += 0.08 * np.sin(2 * np.pi * f * tt[a:b])
        music[a:b] += 0.12 * np.sin(2 * np.pi * chords[k % 4][0] / 2 * tt[a:b])
    rng = np.random.default_rng(1)
    music += 0.02 * rng.standard_normal(len(music))
    mix = lead + back + music
    mix = mix / np.abs(mix).max() * 0.9
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(os.path.join(out_dir, "cancion.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    os.remove(tmp)
    json.dump(synced, open(os.path.join(out_dir, "letra_sincronizada.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(plain, open(os.path.join(out_dir, "letra_sin_tiempos.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(truth, open(os.path.join(out_dir, "verdad.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
