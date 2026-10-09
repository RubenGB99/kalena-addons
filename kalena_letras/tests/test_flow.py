"""Recorrido completo contra un Jellyfin simulado, con la separación y el modelo sustituidos."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(__file__))

import kalena_letras.__main__ as M  # noqa: E402
from kalena_letras import lyrics as L  # noqa: E402
from kalena_letras.timing import Voice  # noqa: E402
from test_logic import ids, synthetic  # noqa: E402

SONG_ID = "a" * 32
USER_ID = "b" * 32


class FakeJellyfin(BaseHTTPRequestHandler):
    state = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        assert 'Token="clave-secreta"' in self.headers.get("Authorization", "")
        if u.path == "/System/Info":
            return self._json({"Version": "12.2.0", "ServerName": "Prueba"})
        if u.path == "/Users":
            return self._json([{"Name": "RubenGB", "Id": USER_ID}])
        if u.path == "/Items":
            assert q["searchTerm"] == ["Hola mundo"]
            return self._json({"Items": [{"Id": SONG_ID, "Name": "Hola mundo", "Artists": ["Artista"], "Path": "/media/music/01 Hola mundo.flac"}]})
        if u.path == f"/Items/{SONG_ID}":
            return self._json(self.state.setdefault("item", {"Id": SONG_ID, "Name": "Hola mundo", "LockData": False}))
        if u.path == f"/Audio/{SONG_ID}/Lyrics":
            if "uploaded" in self.state:
                lines = [{"Text": l, "Cues": [{}]} for l in L.fingerprint(self.state["uploaded"]).splitlines()]
                return self._json({"Lyrics": lines})
            return self._json({"Lyrics": [{"Text": "Hola mundo", "Start": 10_000_000}, {"Text": "Otra vez (oh)", "Start": 40_000_000}]})
        if u.path == f"/Audio/{SONG_ID}/stream":
            return self._send(200, open(self.state["audio"], "rb").read(), "audio/wav")
        if u.path == "/DisplayPreferences/kalena-edits":
            return self._json(self.state.get("prefs", {"Id": "kalena-edits", "CustomPrefs": {"otra:1": "x"}}))
        self._send(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == f"/Audio/{SONG_ID}/Lyrics":
            assert q["fileName"] == ["01 Hola mundo.lrc"]
            self.state["uploaded"] = body.decode()
            return self._send(200)
        if u.path == f"/Items/{SONG_ID}":
            self.state["item"] = json.loads(body)
            return self._send(204)
        if u.path == "/DisplayPreferences/kalena-edits":
            assert q["userId"] == [USER_ID] and q["client"] == ["kalena"]
            self.state["prefs"] = json.loads(body)
            return self._send(204)
        self._send(404)


def test_whole_flow(tmp_path, monkeypatch):
    audio = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=7", str(audio)], check=True)
    FakeJellyfin.state = {"audio": str(audio)}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeJellyfin)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    em_lead, _ = synthetic([(50, "hola"), (75, "mundo"), (200, "otra"), (225, "vez")], 350)
    em_back, _ = synthetic([(260, "oh")], 350)

    class FakeAligner:
        def __init__(self, quality):
            self.separator = self

        def karaoke(self, mix, out):
            return "a", "b"

        def vocals(self, mix, out):
            return "v"

        def voice(self, name, path):
            em = {"a": em_back, "b": em_lead}[path]  # la voz principal llega como «b»
            return Voice(name, em, 20.0, np.full(700, -60.0))

        def voice_on_parts(self, name, samples, parts, layout, length):
            assert name in ("coros", "voz_completa")
            return Voice(name, em_back if name == "coros" else em_lead, 20.0, np.full(700, -60.0))

        def emission_model(self):
            return type("Mdl", (), {"token_ids": staticmethod(ids)})()

    monkeypatch.setattr(M, "Aligner", FakeAligner)
    monkeypatch.setattr(M, "read_mono", lambda path, sr=16_000: np.zeros(16_000 * 7, dtype=np.float32))
    monkeypatch.setattr(M, "DATA", str(tmp_path / "data"))
    monkeypatch.setattr(M, "MODELS", str(tmp_path / "data" / "modelos"))
    monkeypatch.setattr(M, "WORK", str(tmp_path / "data" / "trabajo"))
    monkeypatch.setattr(M, "STATE", str(tmp_path / "data" / "estado.json"))
    monkeypatch.setattr(M, "SHARE", str(tmp_path / "share"))
    opts = tmp_path / "options.json"
    opts.write_text(json.dumps({"jellyfin_url": f"http://127.0.0.1:{server.server_port}", "clave_api": "clave-secreta",
                                "usuario": "rubengb", "canciones": ["Artista - Hola mundo"], "separacion": "maxima"}))
    monkeypatch.setattr(M, "OPTIONS", str(opts))
    monkeypatch.setenv("KALENA_EXIT", "1")
    # Versiones anteriores apuntaban también las canciones saltadas: esta no cuenta como hecha.
    os.makedirs(tmp_path / "data", exist_ok=True)
    (tmp_path / "data" / "estado.json").write_text(json.dumps({SONG_ID: {"resultado": "sin letra en Jellyfin"}}))
    assert M.main() == 0

    st = FakeJellyfin.state
    assert st["uploaded"].splitlines()[0] == "[00:01.00]<00:01.00>Hola <00:01.50>mundo<00:01.78>"
    assert "<00:05.20>(oh)" in st["uploaded"]
    assert st["item"]["LockData"] is True
    prefs = st["prefs"]["CustomPrefs"]
    assert prefs["otra:1"] == "x"
    assert prefs["lyrics:" + SONG_ID] == st["uploaded"]
    assert prefs["lyrics-original:" + SONG_ID] == "[00:01.00]Hola mundo\n[00:04.00]Otra vez (oh)\n"
    summary = json.load(open(tmp_path / "share" / "resumen.json"))
    assert summary[0]["resultado"] == "guardada en Jellyfin", summary
    assert summary[0]["confianza_voz_principal"]["voz_b"] > summary[0]["confianza_voz_principal"]["voz_a"]
    assert json.load(open(tmp_path / "data" / "estado.json"))[SONG_ID]["resultado"] == "guardada en Jellyfin"
    log_text = json.dumps(summary)
    assert "clave-secreta" not in log_text
    server.shutdown()


def test_parts_layout_cut_and_scatter():
    layout = M._layout([(1000, 2000), (1500, 3000), (8000, 9000)], 10_000)
    assert layout == [(1000, 3000, 0.0), (8000, 9000, 3000.0)]
    rate = 100
    x = np.arange(1000, dtype=np.float32)  # 10 s a 100 Hz
    parts = M._cut(x, rate, layout)
    assert parts.size == (200 + 100) + (100 + 100)
    back = M._scatter(parts, rate, layout, x.size)
    assert np.array_equal(back[100:300], x[100:300]) and np.array_equal(back[800:900], x[800:900])
    assert back[500] == 0 and back[950] == 0

