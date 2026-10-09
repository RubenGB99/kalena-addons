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


def duration_s(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
