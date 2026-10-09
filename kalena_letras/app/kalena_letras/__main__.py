"""Kalena Letras — fase de prueba.

Procesa las canciones de la opción «canciones»: separa la voz, distingue a la cantante de los coros,
pone tiempos a cada palabra y guarda la letra en Jellyfin (con copia de la original para Kalena).
Deja también en /share/kalena_letras el LRC y un informe por canción.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sys
import time
import traceback

import numpy as np

from . import lyrics as L
from .align import SAMPLE_RATE, EmissionModel, energy_db
from .audio import duration_s, read_mono, read_stereo, to_wav, write_wav
from .jellyfin import Jellyfin, JellyfinError
from .separate import Separator
from .timing import MIN_LINE_CONFIDENCE, Voice, line_windows, time_lines

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
        return self.voice_from_samples(name, read_mono(path, SAMPLE_RATE))

    def voice_from_samples(self, name: str, samples) -> Voice:
        em, frame_ms = self.emission_model().emissions(samples)
        return Voice(name, em, frame_ms, energy_db(samples))

    def voice_on_parts(self, name: str, samples, parts, layout, length: int) -> Voice:
        """Una voz de la que solo hay trozos: el modelo escucha solo los trozos (mucho menos trabajo)
        y fuera de ellos se marca silencio."""
        em_parts, frame_ms = self.emission_model().emissions(parts)
        frames = max(1, int(round(length / SAMPLE_RATE * 1000 / frame_ms)))
        em = np.full((frames, em_parts.shape[1]), -30.0, dtype=np.float32)
        em[:, 0] = 0.0  # silencio («blank»)
        for a, b, off in layout:
            i, j, k = int(a / frame_ms), int(b / frame_ms), int(off / frame_ms)
            n = max(0, min(j - i, em_parts.shape[0] - k, frames - i))
            em[i:i + n] = em_parts[k:k + n]
        return Voice(name, em, frame_ms, energy_db(samples))


GAP_MS = 1_000


def _layout(windows_ms: list[tuple[float, float]], total_ms: float):
    """Junta las ventanas que se solapan y las coloca una tras otra con 1 s de silencio entre medias."""
    merged = []
    for a, b in sorted((max(0.0, a), min(total_ms, b)) for a, b in windows_ms):
        if merged and a <= merged[-1][1] + GAP_MS:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    layout, offset = [], 0.0
    for a, b in merged:
        layout.append((a, b, offset))
        offset += (b - a) + GAP_MS
    return layout


def _cut(samples, rate: int, layout) -> "np.ndarray":
    """Los trozos de `samples` (mono o (n, canales)) según `layout`, uno tras otro con silencio."""
    gap = int(GAP_MS * rate / 1000)
    parts = []
    for a, b, _ in layout:
        parts.append(samples[int(a * rate / 1000):int(b * rate / 1000)])
        parts.append(np.zeros((gap,) + samples.shape[1:], dtype=samples.dtype))
    return np.concatenate(parts) if parts else samples[:0]


def _scatter(parts, rate: int, layout, length: int):
    """Lo contrario de `_cut`: cada trozo vuelve a su sitio en un audio de `length` muestras."""
    out = np.zeros(length, dtype=np.float32)
    for a, b, off in layout:
        i, j = int(a * rate / 1000), int(b * rate / 1000)
        k = int(off * rate / 1000)
        n = min(j - i, max(0, parts.size - k))
        out[i:i + n] = parts[k:k + n]
    return out


def align_audio(aligner: "Aligner", mix_wav: str, lines: list, synced: bool, work: str):
    """Separa las voces de `mix_wav` y pone tiempos por palabra a `lines`. Devuelve (informe, confianzas, s separando, s alineando).

    1. Karaoke sobre la canción entera: la voz principal (imprescindible).
    2. Primera pasada solo con la voz principal.
    3. Solo para las líneas con coros escritos o que no quedaron claras: se separa la voz completa
       de esos trozos (no de toda la canción, que es lo más lento) y se repite con ella.
    """
    t_sep = 0.0
    log.info("  · Separando la voz principal del resto (música y coros)…")
    t = time.time()
    out_dir = os.path.join(work, "voces")
    path_a, path_b = aligner.separator.karaoke(mix_wav, out_dir)
    t_sep += time.time() - t

    log.info("  · Alineando la letra…")
    t_align0 = time.time()
    a = aligner.voice("voz_a", path_a)
    b = aligner.voice("voz_b", path_b)
    model = aligner.emission_model()
    # La voz principal es la que encaja con la letra entera; la otra es la música con los coros.
    score = {v.name: _global_conf(lines, v, model.token_ids) for v in (a, b)}
    lead = a if score["voz_a"] >= score["voz_b"] else b
    lead.name = "principal"
    lead_samples = read_mono(path_a if lead is a else path_b, SAMPLE_RATE)

    first = copy.deepcopy(lines)
    report = time_lines(first, lead, None, lambda: None, model.token_ids, synced)
    unclear = {e["index"] for e in report["lines"] if e["confidence"] < MIN_LINE_CONFIDENCE + 0.15}
    need = unclear | {i for i, l in enumerate(lines) if l.backing_words}
    windows = line_windows(lines, lead, model.token_ids, synced)
    wanted = [windows[i] for i in sorted(need) if i in windows]
    if not wanted:
        lines[:] = first
        return report, score, t_sep, time.time() - t_align0

    layout = _layout(wanted, lead.duration_ms)
    seconds = sum(b - a for a, b, _ in layout) / 1000
    log.info("  · Separando la voz completa de %d trozos (%.0f s) para coros y líneas dudosas…", len(layout), seconds)
    t = time.time()
    stereo = read_stereo(mix_wav, 44_100)
    parts_wav = os.path.join(work, "trozos.wav")
    write_wav(parts_wav, _cut(stereo, 44_100, layout), 44_100)
    vocals_parts = read_mono(aligner.separator.vocals(parts_wav, os.path.join(work, "voces_trozos")), SAMPLE_RATE)
    t_sep += time.time() - t

    length = lead_samples.size
    vocals = _scatter(vocals_parts, SAMPLE_RATE, layout, length)
    full = aligner.voice_on_parts("voz_completa", vocals, vocals_parts, layout, length)
    lead_parts = _cut(lead_samples, SAMPLE_RATE, layout)
    n = min(vocals_parts.size, lead_parts.size)
    backing_parts = vocals_parts[:n] - lead_parts[:n]
    backing = aligner.voice_on_parts("coros", _scatter(backing_parts, SAMPLE_RATE, layout, length), backing_parts, layout, length)
    report = time_lines(lines, lead, backing, lambda: full, model.token_ids, synced)
    report["vocals_seconds"] = round(seconds)
    return report, score, t_sep, time.time() - t_align0


SAVED = "guardada en Jellyfin"


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
        saved = bool(shown == L.fingerprint(lrc) and words_back)
        result["resultado"] = (SAVED if saved
                               else "subida, pero Jellyfin aún muestra otra versión (revisa «Guardar letras» de la biblioteca)")
        result["guardada"] = saved
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

    # Solo cuentan como hechas las canciones cuya letra se guardó de verdad (las saltadas por no tener
    # letra, por tener ya tiempos o por un fallo se vuelven a mirar en el siguiente arranque).
    state = {k: v for k, v in load_json(STATE, {}).items() if v.get("resultado") == SAVED}
    aligner = Aligner(opts.get("separacion", "maxima"))
    summary = []
    for query in songs:
        log.info("Canción: %s", query)
        try:
            song, similar = jf.search_song(user_id, query)
            if not song:
                summary.append({"cancion": query, "resultado": "no encontrada en las bibliotecas de %s" % user})
                if similar:
                    names = "; ".join("«%s» de %s (id %s)" % (it.get("Name", "?"), ", ".join(it.get("Artists") or []) or "?", it["Id"])
                                      for it in similar)
                    log.warning("  · No encontrada en las bibliotecas de %s. Lo más parecido que hay en ellas: %s. "
                                "Escribe el título como sale ahí o pon directamente su id.", user, names)
                else:
                    log.warning("  · No encontrada en las bibliotecas de %s (ni nada parecido). Si está en la biblioteca "
                                "de otra persona, pon su usuario en «Usuario» o dale acceso a %s a esa biblioteca.",
                                user, user)
                continue
            if song["Id"] in state and not opts.get("repetir"):
                summary.append({"cancion": query, "resultado": "ya procesada antes (activa «repetir» para rehacerla)"})
                log.info("  · Ya procesada antes.")
                continue
            res = process(song, jf, user_id, aligner, opts)
            if res.get("guardada"):
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
