"""LRCLIB (lrclib.net): letras gratuitas y abiertas, la misma fuente externa que usa Kalena.

Solo se usa cuando Jellyfin no tiene letra para la canción. Se pide primero la coincidencia
exacta (título, artista, álbum y duración) y, si no hay, se busca y se elige la de duración más
parecida (a 3 s como mucho), mejor si trae tiempos por línea.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

from .lyrics import TICKS_PER_MS

log = logging.getLogger("kalena_letras")

BASE = "https://lrclib.net/api"
MAX_DURATION_GAP_S = 3


def _get(path: str, params: dict, version: str):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": f"KalenaLetras/{version} (https://github.com/RubenGB99/kalena-addons)",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log.warning("  · LRCLIB no responde bien (%s).", e.code)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        log.warning("  · No se ha podido consultar LRCLIB: %s", e)
    return None


def pick(results: list[dict], duration_s: float | None) -> dict | None:
    """De una búsqueda, la de duración más parecida (y con tiempos si empatan); nada si ninguna encaja."""
    usable = [r for r in results or [] if not r.get("instrumental") and (r.get("syncedLyrics") or r.get("plainLyrics"))]
    if duration_s:
        usable = [r for r in usable if abs((r.get("duration") or 0) - duration_s) <= MAX_DURATION_GAP_S]
    if not usable:
        return None
    return min(usable, key=lambda r: (abs((r.get("duration") or 0) - (duration_s or 0)), not r.get("syncedLyrics")))


def find(title: str, artist: str, album: str, duration_s: float | None, version: str) -> tuple[str, bool] | None:
    """(letra, con tiempos por línea) de LRCLIB para esa canción, o None."""
    params = {"track_name": title, "artist_name": artist}
    hit = None
    if album and duration_s:
        hit = _get("get", {**params, "album_name": album, "duration": round(duration_s)}, version)
        if hit and (hit.get("instrumental") or not (hit.get("syncedLyrics") or hit.get("plainLyrics"))):
            hit = None
    if hit is None:
        hit = pick(_get("search", params, version) or [], duration_s)
    if not hit:
        return None
    if hit.get("syncedLyrics"):
        return hit["syncedLyrics"], True
    return hit["plainLyrics"], False


_STAMP = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")


def to_payload(text: str, synced: bool) -> dict:
    """La letra de LRCLIB con la misma forma que `GET /Audio/{id}/Lyrics` de Jellyfin."""
    lines = []
    for raw in text.splitlines():
        if not synced:
            if raw.strip():
                lines.append({"Text": raw.strip()})
            continue
        stamps = list(_STAMP.finditer(raw))
        # Una línea puede llevar varios tiempos seguidos («[00:12.00][00:45.00]Estribillo»).
        pos = 0
        times = []
        for m in stamps:
            if m.start() != pos:
                break
            frac = (m.group(3) or "0").ljust(3, "0")[:3]
            times.append((int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(frac))
            pos = m.end()
        if not times:
            continue  # etiquetas como [ar:…] o líneas sin tiempo
        body = raw[pos:].strip()
        for t in times:
            lines.append({"Text": body, "Start": t * TICKS_PER_MS})
    if synced:
        lines.sort(key=lambda l: l["Start"])
    return {"Lyrics": lines}
