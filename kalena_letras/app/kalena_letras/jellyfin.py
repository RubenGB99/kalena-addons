"""Lo que el complemento necesita de Jellyfin, con la clave de API (nunca se escribe en el registro)."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

CLIENT = "Kalena Letras"
DEVICE_ID = "kalena-letras-addon"
BACKUP_ID = "kalena-edits"   # la misma copia que usa el editor de Kalena
BACKUP_CLIENT = "kalena"
LYRICS_KEY = "lyrics:"
LYRICS_ORIGINAL_KEY = "lyrics-original:"
NO_LYRICS = "\u0000"


class JellyfinError(Exception):
    pass


def _simple(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class Jellyfin:
    def __init__(self, url: str, api_key: str, version: str, timeout: int = 60):
        self.base = url.rstrip("/")
        self.timeout = timeout
        self._auth = (f'MediaBrowser Client="{CLIENT}", Device="Home Assistant", '
                      f'DeviceId="{DEVICE_ID}", Version="{version}", Token="{api_key}"')

    # ------------------------------------------------------------------ http

    def _request(self, method: str, path: str, params: dict | None = None, body: bytes | None = None,
                 content_type: str | None = None, timeout: int | None = None):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, data=body, method=method, headers={"Authorization": self._auth})
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise JellyfinError(f"{method} {path}: HTTP {e.code}") from None
        except urllib.error.URLError as e:
            raise JellyfinError(f"No se puede conectar con Jellyfin en {self.base} ({e.reason})") from None

    def _json(self, method: str, path: str, params: dict | None = None, body=None):
        data = json.dumps(body).encode() if body is not None else None
        with self._request(method, path, params, data, "application/json" if data is not None else None) as r:
            text = r.read().decode("utf-8") or "null"
            return json.loads(text) if text.strip() else None

    # ------------------------------------------------------------------ server

    def check(self) -> dict:
        return self._json("GET", "/System/Info")

    def user_id(self, name: str) -> str:
        users = self._json("GET", "/Users") or []
        for u in users:
            if (u.get("Name") or "").lower() == name.strip().lower():
                return u["Id"]
        names = ", ".join(u.get("Name", "?") for u in users)
        raise JellyfinError(f"No hay ningún usuario «{name}» en Jellyfin (hay: {names})")

    # ------------------------------------------------------------------ songs

    def find_song(self, user_id: str, query: str) -> dict | None:
        """Una canción por su id de Jellyfin o por «Artista - Título» (o solo el título)."""
        return self.search_song(user_id, query)[0]

    def search_song(self, user_id: str, query: str) -> tuple[dict | None, list[dict]]:
        """La canción de `query` (o None) y las más parecidas que ve el usuario, para el registro.

        Si con el título tal cual no aparece, prueba sin tildes ni signos y con el artista: Jellyfin
        busca por el texto exacto, y un título guardado algo distinto (o sin etiquetas, con el
        nombre del archivo) no saldría.
        """
        q = query.strip()
        if re.fullmatch(r"[0-9a-fA-F-]{32,36}", q):
            return self.item(user_id, q.replace("-", "")), []
        artist, title = (q.split(" - ", 1) + [""])[:2] if " - " in q else ("", q)
        if not title:
            artist, title = "", q
        want_t, want_a = _simple(title), _simple(artist)

        def score(it):
            name = _simple(it.get("Name", ""))
            artists = " ".join(_simple(a) for a in (it.get("Artists") or []) + [it.get("AlbumArtist") or ""])
            s = 0
            if name == want_t:
                s += 2
            elif want_t and (want_t in name or want_t.replace(" ", "") in name.replace(" ", "")):
                s += 1
            if want_a and want_a in artists:
                s += 2
            return s

        need = 3 if want_a else 2
        found: dict[str, dict] = {}
        for term in dict.fromkeys(t for t in (title, want_t, artist) if t):
            res = self._json("GET", "/Items", {
                "userId": user_id, "searchTerm": term, "includeItemTypes": "Audio", "recursive": "true",
                "fields": "Path", "limit": "50",
            }) or {}
            for it in res.get("Items") or []:
                found.setdefault(it["Id"], it)
            items = sorted(found.values(), key=score, reverse=True)
            if items and score(items[0]) >= need:
                return items[0], []
        return None, sorted(found.values(), key=score, reverse=True)[:3]

    def item(self, user_id: str, item_id: str) -> dict:
        return self._json("GET", f"/Items/{item_id}", {"userId": user_id})

    def lyrics(self, item_id: str) -> dict | None:
        try:
            return self._json("GET", f"/Audio/{item_id}/Lyrics")
        except JellyfinError as e:
            if "404" in str(e):
                return None
            raise

    def download_audio(self, item_id: str, dest: str) -> None:
        """El archivo original, tal cual (FLAC, M4A/Atmos…)."""
        with self._request("GET", f"/Audio/{item_id}/stream", {"static": "true"}, timeout=600) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    # ------------------------------------------------------------------ writing

    def upload_lyrics(self, item_id: str, file_name: str, lrc: str) -> None:
        with self._request("POST", f"/Audio/{item_id}/Lyrics", {"fileName": file_name}, lrc.encode("utf-8"),
                           "text/plain; charset=utf-8"):
            pass

    def lock(self, user_id: str, item_id: str) -> None:
        """Bloquea la canción, como el editor de Kalena, para que ningún escaneo devuelva la letra vieja."""
        it = self.item(user_id, item_id)
        if it.get("LockData"):
            return
        it["LockData"] = True
        with self._request("POST", f"/Items/{item_id}", body=json.dumps(it).encode(), content_type="application/json"):
            pass

    def remember(self, user_id: str, item_id: str, new_lrc: str, original_lrc: str | None) -> None:
        """Guarda la letra nueva y la original en la copia de Kalena («Recuperar letra original»)."""
        params = {"userId": user_id, "client": BACKUP_CLIENT}
        prefs = self._json("GET", f"/DisplayPreferences/{BACKUP_ID}", params) or {}
        custom = dict(prefs.get("CustomPrefs") or {})
        custom[LYRICS_KEY + item_id] = new_lrc
        custom.setdefault(LYRICS_ORIGINAL_KEY + item_id, original_lrc if original_lrc else NO_LYRICS)
        prefs["CustomPrefs"] = custom
        prefs.setdefault("Id", BACKUP_ID)
        prefs.setdefault("Client", BACKUP_CLIENT)
        self._json("POST", f"/DisplayPreferences/{BACKUP_ID}", params, prefs)
