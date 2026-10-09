"""Autoprueba sin Jellyfin: alinea un audio con una letra dada y escribe los tiempos de cada palabra.

    python3 -m kalena_letras.selftest cancion.wav letra.json resultado.json [maxima|rapida]

`letra.json`: [{"text": "Hola mundo", "start_ms": 1000 | null}, ...]
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

from . import lyrics as L
from .__main__ import Aligner, align_audio
from .audio import to_wav


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    song, lyrics_json, out = argv[:3]
    quality = argv[3] if len(argv) > 3 else "maxima"
    spec = json.load(open(lyrics_json, encoding="utf-8"))
    lines = [L.Line(x["text"], x.get("start_ms"), L.split_words(x["text"])) for x in spec]
    synced = L.is_synced(lines)
    aligner = Aligner(quality)
    with tempfile.TemporaryDirectory() as work:
        mix = os.path.join(work, "mezcla.wav")
        to_wav(song, mix, 44_100, 2)
        t = time.time()
        report, score, t_sep, t_align = align_audio(aligner, mix, lines, synced, work)
    words = [{"text": w.text, "backing": w.backing, "start_ms": w.start_ms, "end_ms": w.end_ms,
              "confidence": round(w.confidence, 3), "line": i}
             for i, l in enumerate(lines) for w in l.words]
    json.dump({"words": words, "report": report, "voices": score, "separation_s": round(t_sep),
               "alignment_s": round(t_align), "total_s": round(time.time() - t), "lrc": L.write_lrc(lines)},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(L.write_lrc(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
