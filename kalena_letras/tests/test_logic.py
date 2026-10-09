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
    level = np.full(640, -60.0)  # energía cada 10 ms durante los 6,4 s
    lead = Voice("lead", em_lead, 20.0, level)
    back = Voice("backing", em_back, 20.0, level)
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


def test_backing_word_without_voice_in_its_track_gets_no_time():
    em_lead, _ = synthetic([(50, "hola"), (75, "mundo")], 200)
    em_back, _ = synthetic([(110, "oh")], 200)
    back_energy = np.full(400, -90.0)
    back_energy[:100] = -20.0  # los coros solo suenan al principio, no donde estaría «oh»
    lead = Voice("lead", em_lead, 20.0, np.full(400, -60.0))
    back = Voice("backing", em_back, 20.0, back_energy)
    lines = [lyrics.Line("Hola mundo (oh)", 1000, lyrics.split_words("Hola mundo (oh)"))]
    time_lines(lines, lead, back, lambda: None, ids, synced=True)
    assert lines[0].timed and lines[0].words[2].start_ms is None
    assert lyrics.write_lrc(lines) == "[00:01.00]<00:01.00>Hola <00:01.50>mundo (oh)<00:01.78>\n"



def test_search_song_falls_back_and_lists_similar():
    from kalena_letras.jellyfin import Jellyfin
    library = [
        {"Id": "a1", "Name": "SUENO MOJADITO", "Artists": ["DANNA"]},
        {"Id": "a2", "Name": "Otra cancion", "Artists": ["DANNA"]},
    ]
    asked = []

    class Fake(Jellyfin):
        def __init__(self):
            pass

        def _json(self, method, path, params=None, body=None):
            term = params["searchTerm"]
            asked.append(term)
            # Jellyfin busca el texto tal cual: con la tilde no sale.
            return {"Items": [it for it in library if term.lower() in it["Name"].lower() or term in it["Artists"]]}

    jf = Fake()
    song, _ = jf.search_song("u", "DANNA - SUEÑO MOJADITO")
    assert song["Id"] == "a1"
    assert asked[0] == "SUEÑO MOJADITO" and asked[1] == "sueno mojadito"
    asked.clear()
    song, similar = jf.search_song("u", "DANNA - Cancion inventada")
    assert song is None
    assert {it["Id"] for it in similar} == {"a1", "a2"}


def test_search_song_ignores_the_kind_of_apostrophe():
    from kalena_letras.jellyfin import Jellyfin

    class Fake(Jellyfin):
        def __init__(self):
            pass

        def _json(self, method, path, params=None, body=None):
            item = {"Id": "b1", "Name": "Don’t Stop", "Artists": ["Artista"]}
            return {"Items": [item] if params["searchTerm"] in item["Name"] else []}

    assert Fake().search_song("u", "Don't Stop")[0]["Id"] == "b1"
    assert Fake().search_song("u", "Artista - Don't Stop")[0]["Id"] == "b1"


def test_lines_the_ai_does_not_time_keep_their_word_times():
    payload = {"Lyrics": [
        {"Text": "Hola mundo", "Start": 10_000_000,
         "Cues": [{"Position": 0, "Start": 10_000_000}, {"Position": 5, "Start": 15_000_000, "End": 18_000_000}]},
        {"Text": "Otra vez", "Start": 40_000_000},
    ]}
    lines = lyrics.from_jellyfin(payload)
    # La IA solo alinea la segunda línea.
    second = lines[1]
    second.timed = True
    second.words[0].start_ms, second.words[1].start_ms, second.words[1].end_ms = 4000, 4500, 5000
    out = lyrics.write_lrc(lines).splitlines()
    assert out[0] == "[00:01.00]<00:01.00>Hola <00:01.50>mundo<00:01.80>"
    assert out[1] == "[00:04.00]<00:04.00>Otra <00:04.50>vez<00:05.00>"
    assert lyrics.original_lrc(payload).splitlines()[0] == out[0]


def test_lrclib_payload_and_pick():
    from kalena_letras import lrclib
    text = "[ar:Camila]\n[00:12.34]Don't go yet\n[00:15.00][01:02.5]Estribillo\n[00:20.00]\n"
    payload = lrclib.to_payload(text, True)
    starts = [(l["Text"], l["Start"] // lyrics.TICKS_PER_MS) for l in payload["Lyrics"]]
    assert starts == [("Don't go yet", 12340), ("Estribillo", 15000), ("", 20000), ("Estribillo", 62500)]
    lines = lyrics.from_jellyfin(payload)
    assert lyrics.is_synced(lines) and [w.tokens for w in lines[0].words] == ["don't", "go", "yet"]
    assert lrclib.to_payload("Una\n\nDos\n", False) == {"Lyrics": [{"Text": "Una"}, {"Text": "Dos"}]}
    results = [
        {"duration": 170, "plainLyrics": "a"},
        {"duration": 181, "plainLyrics": "b"},
        {"duration": 182, "syncedLyrics": "[00:01.00]c", "plainLyrics": "c"},
        {"duration": 180, "instrumental": True},
    ]
    assert lrclib.pick(results, 181.5)["syncedLyrics"] == "[00:01.00]c"
    assert lrclib.pick(results, 300) is None
