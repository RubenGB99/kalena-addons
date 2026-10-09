#!/usr/bin/env python3
"""Compara el resultado de la autoprueba con los tiempos reales y falla si no es lo bastante preciso."""
import json
import statistics
import sys

result = json.load(open(sys.argv[1]))
truth = json.load(open(sys.argv[2]))
label = sys.argv[3] if len(sys.argv) > 3 else ""
words = result["words"]
assert len(words) == len(truth), (len(words), len(truth))
lead_err, back_err, missing = [], [], 0
for w, t in zip(words, truth):
    assert w["text"] == t["text"], (w["text"], t["text"])
    if w["start_ms"] is None:
        if not t["backing"]:
            missing += 1
        continue
    (back_err if t["backing"] else lead_err).append(abs(w["start_ms"] - t["start_ms"]))
lead_total = sum(1 for t in truth if not t["backing"])
for w, t in zip(words, truth):
    err = "—" if w["start_ms"] is None else "%+d ms" % (w["start_ms"] - t["start_ms"])
    print(f'   {"coro " if t["backing"] else "     "}{t["text"]:12} real {t["start_ms"]:6}  IA {str(w["start_ms"]):6}  {err:>9}  conf {w["confidence"]}')
rep = result["report"]
print(f"== {label}: voces {result['voices']}, separación {result['separation_s']} s, alineación {result['alignment_s']} s, "
      f"voz completa de {rep.get('vocals_seconds', 0)} s de trozos, coros sacados de la voz completa: {rep.get('backing_from_full', 0)}")
print(f"   voz principal: {len(lead_err)}/{lead_total} palabras con tiempo")
if lead_err:
    within = sum(1 for e in lead_err if e <= 150) / len(lead_err)
    print(f"   error mediano {statistics.median(lead_err):.0f} ms, máximo {max(lead_err)} ms, {within:.0%} a menos de 150 ms")
print(f"   coros: {len(back_err)} palabras con tiempo, errores {back_err}")
print(result["lrc"])
ok = (len(lead_err) >= 0.8 * lead_total and statistics.median(lead_err) <= 80
      and sum(1 for e in lead_err if e <= 150) >= 0.9 * len(lead_err)
      and all(e <= 300 for e in back_err))  # un coro sin tiempo vale; uno mal puesto, no
sys.exit(0 if ok else 1)
