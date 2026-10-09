"""Kalena Letras — fase de prueba.

Procesa las canciones de la opción «canciones»: separa la voz, distingue a la cantante de los coros,
pone tiempos a cada palabra y guarda la letra en Jellyfin (con copia de la original para Kalena).
Deja también en /share/kalena_letras el LRC y un informe por canción.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback

from . import lyrics as L
from .align import SAMPLE_RATE, EmissionModel, energy_db
from .audio import duration_s, read_mono, to_wav
from .jellyfin import Jellyfin, JellyfinError
from .separate import Separator
from .timing import Voice, time_lines

VERSION = "0.1.0"
OPTIONS = os.environ.get("KALENA_OPTIONS", "/data/options.json")
DATA = os.environ.get("KALENA_DATA", "/data")
SHARE = os.environ.get("KALENA_SHARE", "/share/kalena_letras")
MODELS = os.path.join(DATA, "modelos")
WORK = os.path.join(DATA, "trabajo")
STATE = os.path.join(DATA, "estado.json")

log = logging.getLogger("kalena_letras")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_.()" else "_" for c in text)
    return keep.strip()[:120] or "cancion"


class Aligner:
    """Carga los modelos una sola vez para todas las canciones."""

    def __init__(self, quality: str):
        self.separator = Separator(os.path.join(MODELS, "separacion"), quality)
        self.model = None

    def emission_model(self) -> EmissionModel:
        if self.model is None:
            log.info("Cargando el modelo de alineación (la primera vez se descarga, ~1,2 GB)…")
            self.model = EmissionModel(os.path.join(MODELS, "alineacion"))
        return self.model

    def voice(self, name: str, path: str) -> Voice:
        samples = read_mono(path, SAMPLE_RATE)
        em, frame_ms = self.emission_model().emissions(samples)
        return Voice(name, em, frame_ms, energy_db(samples))


def align_audio(aligner: "Aligner", mix_wav: str, lines: list, synced: bool, work: str):
    """Separa las voces de `mix_wav` y pone tiempos por palabra a `lines`. Devuelve (informe, confianzas, s separando, s alineando)."""
    log.info("  · Separando la voz de la música y la voz principal de los coros…")
    t = time.time()
    stems = aligner.separator.run(mix_wav, os.path.join(work, "voces"))
    t_sep = time.time() - t

    log.info("  · Alineando la letra…")
    t = time.time()
    a = aligner.voice("voz_a", stems.voice_a)
    b = aligner.voice("voz_b", stems.voice_b)
    model = aligner.emission_model()
    # La voz principal es la que encaja con la letra entera; la otra son los coros.
    score = {v.name: _global_conf(lines, v, model.token_ids) for v in (a, b)}
    lead, backing = (a, b) if score["voz_a"] >= score["voz_b"] else (b, a)
    lead.name, backing.name = "principal", "coros"
    full_cache = {}

    def full():
        if "v" not in full_cache:
            full_cache["v"] = aligner.voice("voz_completa", stems.vocals)
        return full_cache["v"]

    report = time_lines(lines, lead, backing, full, model.token_ids, synced)
    return report, score, t_sep, time.time() - t


def process(song: dict, jf: Jellyfin, user_id: str, aligner: Aligner, opts: dict) -> dict:
    sid, title = song["Id"], song.get("Name", "?")
    artist = ", ".join(song.get("Artists") or []) or song.get("AlbumArtist") or ""
    label = f"{artist} - {title}" if artist else title
    t0 = time.time()
    payload = jf.lyrics(sid)
    if not payload or not payload.get("Lyrics"):
        return {"cancion": label, "resultado": "sin letra en Jellyfin (hace falta la letra para ponerle tiempos)"}
    lines = L.from_jellyfin(payload)
    if L.has_word_times(lines) and not opts.get("repetir"):
        return {"cancion": label, "resultado": "ya tiene tiempos por palabra (activa «repetir» para rehacerla)"}
    synced = L.is_synced(lines)

    work = os.path.join(WORK, sid)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        src = os.path.join(work, "original")
        log.info("  · Descargando el audio…")
        jf.download_audio(sid, src)
        mix = os.path.join(work, "mezcla.wav")
        to_wav(src, mix, 44_100, 2)
        length = duration_s(mix)

        report, score, t_sep, t_align = align_audio(aligner, mix, lines, synced, work)
    finally:
        if not opts.get("conservar_audio"):
            shutil.rmtree(work, ignore_errors=True)

    lrc = L.write_lrc(lines)
    os.makedirs(SHARE, exist_ok=True)
    base = safe_name(label)
    with open(os.path.join(SHARE, base + ".lrc"), "w", encoding="utf-8") as f:
        f.write(lrc)
    result = {
        "cancion": label,
        "id": sid,
        "duracion_cancion_s": round(length),
        "letra_original": "con tiempos por línea" if synced else "sin tiempos",
        "lineas_con_tiempos_por_palabra": f"{report['timed']} de {report['total']}",
        "lineas_reintentadas_con_voz_completa": report["fallback_full"],
        "confianza_voz_principal": {"voz_a": round(score["voz_a"], 3), "voz_b": round(score["voz_b"], 3)},
        "tiempo_separacion_s": round(t_sep),
        "tiempo_alineacion_s": round(t_align),
        "tiempo_total_s": round(time.time() - t0),
        "lineas": report["lines"],
    }
    if report["timed"] == 0:
        result["resultado"] = "no se ha podido alinear ninguna línea con seguridad; no se toca la letra"
    elif opts.get("guardar_en_jellyfin", True):
        base_name = os.path.splitext(os.path.basename(song.get("Path") or "lyrics"))[0] or "lyrics"
        jf.remember(user_id, sid, lrc, L.original_lrc(payload))
        jf.upload_lyrics(sid, base_name + ".lrc", lrc)
        try:
            jf.lock(user_id, sid)
        except JellyfinError as e:
            log.warning("  · No se ha podido bloquear la canción: %s", e)
        back = jf.lyrics(sid) or {}
        shown = "\n".join((x.get("Text") or "").strip() for x in back.get("Lyrics") or [] if (x.get("Text") or "").strip())
        words_back = sum(1 for x in back.get("Lyrics") or [] if x.get("Cues"))
        result["resultado"] = ("guardada en Jellyfin" if shown == L.fingerprint(lrc) and words_back
                               else "subida, pero Jellyfin aún muestra otra versión (revisa «Guardar letras» de la biblioteca)")
    else:
        result["resultado"] = "solo guardada en /share/kalena_letras (guardar_en_jellyfin desactivado)"
    save_json(os.path.join(SHARE, base + ".json"), result)
    return result


def _global_conf(lines, voice, token_ids) -> float:
    from .align import viterbi
    ids = [i for l in lines for w in l.lead_words for i in token_ids(w.tokens)]
    al = viterbi(voice.emissions, ids) if ids else None
    return al.score if al else 0.0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    opts = load_json(OPTIONS, {})
    log.info("Kalena Letras %s (fase de prueba)", VERSION)
    for d in (MODELS, WORK):
        os.makedirs(d, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", os.path.join(MODELS, "alineacion"))

    url, key, user = opts.get("jellyfin_url", ""), opts.get("clave_api", ""), opts.get("usuario", "")
    songs = [s for s in opts.get("canciones") or [] if str(s).strip()]
    if not (url and key and user):
        log.error("Falta configurar la dirección de Jellyfin, la clave de API o el usuario (pestaña Configuración).")
        return idle()
    if not songs:
        log.error("No hay canciones en la lista «canciones». Añade 3 o 4 («Artista - Título») y reinicia.")
        return idle()

    jf = Jellyfin(url, key, VERSION)
    try:
        info = jf.check()
        user_id = jf.user_id(user)
    except JellyfinError as e:
        log.error("%s", e)
        return idle()
    log.info("Conectado a Jellyfin %s («%s»), usuario %s.", info.get("Version"), info.get("ServerName"), user)

    state = load_json(STATE, {})
    aligner = Aligner(opts.get("separacion", "maxima"))
    summary = []
    for query in songs:
        log.info("Canción: %s", query)
        try:
            song = jf.find_song(user_id, query)
            if not song:
                summary.append({"cancion": query, "resultado": "no encontrada en Jellyfin (revisa «Artista - Título»)"})
                log.warning("  · No encontrada en Jellyfin.")
                continue
            if song["Id"] in state and not opts.get("repetir"):
                summary.append({"cancion": query, "resultado": "ya procesada antes (activa «repetir» para rehacerla)"})
                log.info("  · Ya procesada antes.")
                continue
            res = process(song, jf, user_id, aligner, opts)
            state[song["Id"]] = {"cancion": res["cancion"], "resultado": res["resultado"], "cuando": int(time.time())}
            save_json(STATE, state)
            summary.append(res)
            log.info("  · %s. %s líneas con tiempos por palabra. %s s en total.", res["resultado"],
                     res.get("lineas_con_tiempos_por_palabra", "0"), res.get("tiempo_total_s", "?"))
        except Exception as e:  # una canción que falla no para las demás
            log.error("  · Error: %s", e)
            log.debug(traceback.format_exc())
            summary.append({"cancion": query, "resultado": f"error: {e}"})
    os.makedirs(SHARE, exist_ok=True)
    save_json(os.path.join(SHARE, "resumen.json"), summary)
    log.info("Terminado. Resumen en /share/kalena_letras/resumen.json. Ya puedes detener el complemento.")
    return idle()


def idle() -> int:
    if os.environ.get("KALENA_EXIT"):
        return 0
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
