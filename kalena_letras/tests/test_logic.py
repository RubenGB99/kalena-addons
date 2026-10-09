"""Pruebas de la lógica sin modelos: letras, LRC, Viterbi, ajuste fino y tiempos por línea."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from kalena_letras import lyrics  # noqa: E402
from kalena_letras.align import energy_db, refine_start, viterbi, word_timings  # noqa: E402
from kalena_letras.timing import Voice, time_lines  # noqa: E402

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
DICT = {c: i + 1 for i, c in enumerate(LETTERS)}  # 0 = blank
C = len(DICT) + 1


def ids(tokens):
    return [DICT[c] for c in tokens if c in DICT]


def synthetic(script, total_frames, noise=0.02, seed=0):
    """Emisiones falsas: cada letra «suena» en los fotogramas indicados y el resto es silencio (blank).

    `script` = [(primer_fotograma, palabra), ...]; cada letra ocupa 2 fotogramas.
    """
    rng = np.random.default_rng(seed)
    p = np.full((total_frames, C), noise / C)
    p[:, 0] = 1.0
    truth = []
    for start, word in script:
        f = start
        truth.append((word, start))
        for ch in word:
            for k in range(2):
                p[f + k] = noise / C
                p[f + k, DICT[ch]] = 1.0
            f += 3
    p += rng.uniform(0, noise / C, p.shape)
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p).astype(np.float32), truth


def test_tokens_and_backing_words():
    words = lyrics.split_words("¿Qué pasa? (oh, oh) Canción")
    assert [w.tokens for w in words] == ["que", "pasa", "oh", "oh", "cancion"]
    assert [w.backing for w in words] == [False, False, True, True, False]
    assert lyrics.to_tokens("I'm") == "i'm"
    assert lyrics.to_tokens("2") == ""


def test_lrc_round_trip_matches_kalena_format():
    line = lyrics.Line("Hola mundo", 1000, lyrics.split_words("Hola mundo"))
    line.words[0].start_ms, line.words[0].end_ms = 1000, 1500
    line.words[1].start_ms, line.words[1].end_ms = 1600, 2300
    line.timed = True
    untimed = lyrics.Line("Segunda", 5000, lyrics.split_words("Segunda"))
    out = lyrics.write_lrc([line, untimed])
    assert out == "[00:01.00]<00:01.00>Hola <00:01.60>mundo<00:02.30>\n[00:05.00]Segunda\n"
    assert lyrics.fingerprint(out) == "Hola mundo\nSegunda"


def test_original_lrc_keeps_existing_cues():
    payload = {"Lyrics": [
        {"Text": "Hola mundo", "Start": 10_000_000, "Cues": [
            {"Position": 0, "EndPosition": 4, "Start": 10_000_000, "End": 15_000_000},
            {"Position": 5, "EndPosition": 10, "Start": 16_000_000, "End": 23_000_000}]},
        {"Text": "Sin cues", "Start": 50_000_000},
    ]}
    assert lyrics.original_lrc(payload) == "[00:01.00]<00:01.00>Hola <00:01.60>mundo<00:02.30>\n[00:05.00]Sin cues\n"


def test_viterbi_finds_each_word():
    em, truth = synthetic([(10, "hola"), (40, "mundo")], 80)
    al = viterbi(em, ids("hola") + ids("mundo"))
    assert al is not None and al.score > 0.8
    t = word_timings(al, [4, 5], frame_ms=20.0)
    assert [x.start_ms for x in t] == [200, 800]


def test_viterbi_handles_repeated_letters():
    em, _ = synthetic([(5, "allí"[:2] + "l")], 40)
    assert viterbi(em, ids("all")) is not None
    assert viterbi(em[:2], ids("all")) is None


def test_refine_moves_start_to_the_onset():
    sr = 16_000
    x = np.zeros(sr * 2)
    x[int(1.05 * sr):int(1.5 * sr)] = 0.5 * np.sin(np.arange(int(0.45 * sr)) * 0.1)
    e = energy_db(x, sr)
    assert abs(refine_start(1000, e, earliest_ms=0) - 1050) <= 20
    # Sin subida clara (voz continua), no se toca.
    flat = energy_db(np.full(sr * 2, 0.3), sr)
    assert refine_start(1000, flat, earliest_ms=0) == 1000


def test_time_lines_synced_lead_and_backing():
    script = [(50, "hola"), (75, "mundo"), (200, "otra"), (225, "vez")]
    em_lead, _ = synthetic(script, 320)
    em_back, _ = synthetic([(260, "oh")], 320)
    silent = np.full(320, -60.0)
    lead = Voice("lead", em_lead, 20.0, silent)
    back = Voice("backing", em_back, 20.0, silent)
    lines = [
        lyrics.Line("Hola mundo", 1000, lyrics.split_words("Hola mundo")),
        lyrics.Line("Otra vez (oh)", 4000, lyrics.split_words("Otra vez (oh)")),
    ]
    rep = time_lines(lines, lead, back, lambda: None, ids, synced=True)
    assert rep["timed"] == 2
    assert [w.start_ms for w in lines[0].words] == [1000, 1500]
    assert [w.start_ms for w in lines[1].words] == [4000, 4500, 5200]
    out = lyrics.write_lrc(lines)
    assert out.splitlines()[1].startswith("[00:04.00]<00:04.00>Otra <00:04.50>vez <00:05.20>(oh)")


def test_time_lines_unsynced_uses_global_alignment():
    script = [(30, "uno"), (60, "dos"), (150, "tres")]
    em, _ = synthetic(script, 220)
    lead = Voice("lead", em, 20.0, np.full(220, -60.0))
    lines = [lyrics.Line("Uno dos", None, lyrics.split_words("Uno dos")),
             lyrics.Line("Tres", None, lyrics.split_words("Tres"))]
    rep = time_lines(lines, lead, None, lambda: None, ids, synced=False)
    assert rep["timed"] == 2
    assert lines[1].words[0].start_ms == 3000


def test_unclear_line_is_left_as_it_was():
    em = np.log(np.full((200, C), 1.0 / C)).astype(np.float32)  # nada claro
    lead = Voice("lead", em, 20.0, np.full(200, -60.0))
    lines = [lyrics.Line("Nada claro", 500, lyrics.split_words("Nada claro"))]
    rep = time_lines(lines, lead, None, lambda: None, ids, synced=True)
    assert rep["timed"] == 0 and not lines[0].timed
    assert lyrics.write_lrc(lines) == "[00:00.50]Nada claro\n"


def test_a_line_does_not_steal_sounds_from_the_next_one():
    # «dos» acaba en «s» y la línea siguiente («tres») también: la «s» no debe saltar a «tres».
    script = [(30, "uno"), (60, "dos"), (85, "tres")]
    em, _ = synthetic(script, 160)
    lead = Voice("lead", em, 20.0, np.full(160, -60.0))
    lines = [lyrics.Line("Uno dos", 600, lyrics.split_words("Uno dos")),
             lyrics.Line("Tres", 1700, lyrics.split_words("Tres"))]
    time_lines(lines, lead, None, lambda: None, ids, synced=True)
    assert lines[0].words[1].end_ms <= 1400
    assert lines[1].words[0].start_ms == 1700
