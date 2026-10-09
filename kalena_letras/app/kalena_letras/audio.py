"""Conversión de audio con ffmpeg (cualquier formato, también E-AC-3 / Dolby Atmos)."""
from __future__ import annotations

import subprocess

import numpy as np


def to_wav(src: str, dest: str, sample_rate: int, channels: int) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-map", "0:a:0",
         "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", dest],
        check=True,
    )


def read_mono(path: str, sample_rate: int = 16_000) -> np.ndarray:
    """El audio como float32 mono a `sample_rate`, sea cual sea el archivo."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sample_rate),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def read_stereo(path: str, sample_rate: int = 44_100) -> np.ndarray:
    """El audio como float32 (n, 2) a `sample_rate`."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", str(sample_rate),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).copy()


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """WAV de 16 bits (mono o estéreo)."""
    import wave
    data = np.clip(samples, -1.0, 1.0)
    channels = 1 if data.ndim == 1 else data.shape[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((data * 32767).astype("<i2").tobytes())


def duration_s(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
