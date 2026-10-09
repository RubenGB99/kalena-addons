#!/usr/bin/env python3
"""Prepara los modelos de IA para el móvil (Kalena): los convierte a ONNX y comprueba que dan lo mismo.

    python3 movil/exportar.py <carpeta de modelos> <salida> [cancion.wav]

- Separación de voces (MelBand RoFormer, la de máxima calidad): se exporta la red sin la STFT, que
  en el móvil se hace aparte (ONNX no tiene números complejos). Entrada: la STFT de un trozo
  (1, F·canales, T, 2) con las frecuencias de los dos canales intercaladas (f·2 + canal); salida:
  la máscara compleja (1, voces, F·canales, T, 2) por la que se multiplica esa STFT.
- Alineación (MMS de Meta): la red entera, con su normalización, de audio a 16 kHz a
  log-probabilidades (1, fotogramas, letras).

Cada modelo se guarda en fp32 y en int8 (más pequeño y rápido; la referencia mide si pierde
precisión) y `manifiesto.json` describe todo lo que el móvil necesita para usarlos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

SEPARACION = {
    # Voz principal frente a todo lo demás (música y coros): la que siempre se usa.
    "karaoke": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    # Voz completa (principal y coros): solo para los trozos con coros o dudosos.
    "voces": "vocals_mel_band_roformer.ckpt",
}


class RoformerCore(nn.Module):
    """MelBandRoformer.forward entre la STFT y la multiplicación por la máscara (todo en reales)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("idx", model.freq_indices.clone())
        denom = repeat(model.num_bands_per_freq, "f -> (f r)", r=model.audio_channels).float().clamp(min=1e-8)
        self.register_buffer("denom", denom)

    def forward(self, stft):  # (1, F·s, T, 2)
        m = self.m
        x = stft[:, self.idx]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = m.band_split(x)
        for time_transformer, freq_transformer in m.layers:
            b, t, f, d = x.shape
            x = rearrange(x, "b t f d -> b f t d").reshape(b * f, t, d)
            x = time_transformer(x).reshape(b, f, t, d)
            x = rearrange(x, "b f t d -> b t f d").reshape(b * t, f, d)
            x = freq_transformer(x).reshape(b, t, f, d)
        masks = torch.stack([fn(x) for fn in m.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
        b, n, fi, t, c = masks.shape
        out = torch.zeros(b, n, self.denom.shape[0], t, c, dtype=masks.dtype)
        index = self.idx.view(1, 1, -1, 1, 1).expand(b, n, fi, t, c)
        out = out.scatter_add(2, index, masks)
        return out / self.denom.view(1, 1, -1, 1, 1)


class MmsLogProbs(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, audio):  # (1, muestras)
        emissions, _ = self.m(audio)
        return torch.log_softmax(emissions, dim=-1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantize(src: str, dst: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])


def export_roformer(name: str, filename: str, model_dir: str, out: str, song: str | None) -> dict:
    from audio_separator.separator import Separator
    sep = Separator(model_file_dir=model_dir, output_dir=out, log_level=logging.WARNING)
    sep.load_model(model_filename=filename)
    inst = sep.model_instance
    model = inst.model_run.eval()
    for module in model.modules():  # atención «a mano» (einsum): se exporta igual en todas partes
        if module.__class__.__name__ == "Attend":
            module.flash = False
    cfg = inst.model_data_cfgdict
    st = model.stft_kwargs
    n_fft, hop, win = st["n_fft"], st["hop_length"], st["win_length"]
    dim_t = cfg.inference.dim_t
    chunk = cfg.audio.hop_length * (dim_t - 1)
    assert cfg.audio.hop_length == hop, (cfg.audio.hop_length, hop)
    channels = model.audio_channels
    freqs = n_fft // 2 + 1
    frames = 1 + chunk // hop
    instruments = list(cfg.training.instruments)
    target = cfg.training.target_instrument
    stems = [target] if target else instruments
    core = RoformerCore(model).eval()

    # La red exportada tiene que dar lo mismo que el modelo original en un trozo de verdad.
    if song:
        from kalena_letras.audio import read_stereo
        mix = read_stereo(song, cfg.audio.sample_rate).T.copy()
        part = torch.from_numpy(mix[:, :chunk].copy()).float()
        if part.shape[1] < chunk:
            part = torch.nn.functional.pad(part, (0, chunk - part.shape[1]))
    else:
        part = torch.randn(channels, chunk) * 0.1
    window = torch.hann_window(win)
    with torch.no_grad():
        want = model(part.unsqueeze(0))[0]
        spec = torch.stft(part, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        spec_ri = rearrange(torch.view_as_real(spec), "s f t c -> 1 (f s) t c")
        mask = core(spec_ri)
        got = []
        for k in range(mask.shape[1]):
            mk = torch.view_as_complex(rearrange(mask[0, k], "(f s) t c -> s f t c", s=channels).contiguous())
            got.append(torch.istft(spec * mk, n_fft=n_fft, hop_length=hop, win_length=win, window=window))
        got = torch.stack(got)
    want = want.reshape(got.shape)
    err = float((got - want).abs().max())
    print(f"{name}: red sin STFT frente al modelo original, diferencia máxima {err:.2e}", flush=True)
    assert err < 1e-3, err

    fp32 = os.path.join(out, f"separacion_{name}.onnx")
    t = time.time()
    with torch.no_grad():
        torch.onnx.export(core, (spec_ri,), fp32, opset_version=17, input_names=["stft"], output_names=["mascara"],
                          do_constant_folding=True)
    print(f"{name}: exportado en {time.time() - t:.0f} s", flush=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    onnx_mask = sess.run(None, {"stft": spec_ri.numpy()})[0]
    err = float(np.abs(onnx_mask - mask.numpy()).max())
    print(f"{name}: ONNX frente a PyTorch, diferencia máxima en la máscara {err:.2e}", flush=True)
    assert err < 1e-2, err
    rate = cfg.audio.sample_rate
    overlap = inst.overlap
    # Lo que ocupa el modelo de PyTorch se suelta antes de cuantizar (la cuantización carga el ONNX entero).
    del sess, core, model, inst, sep, want, got, mask, spec, spec_ri, onnx_mask
    import gc
    gc.collect()
    int8 = fp32.replace(".onnx", "_int8.onnx")
    quantize(fp32, int8)
    return {
        "archivos": {"fp32": os.path.basename(fp32), "int8": os.path.basename(int8)},
        "frecuencia_muestreo": rate,
        "canales": channels,
        "n_fft": n_fft, "salto": hop, "ventana": win,
        "trozo_muestras": chunk, "fotogramas": frames, "frecuencias": freqs,
        "paso_segundos": overlap,
        "voces": stems,
        "voz_principal": stems[0],
        "secundaria_es_resto": len(stems) == 1,
    }


def export_mms(model_dir: str, out: str) -> dict:
    os.environ.setdefault("TORCH_HOME", model_dir)
    import torchaudio
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).eval()
    wrapper = MmsLogProbs(model).eval()
    dummy = torch.randn(1, 16_000 * 34) * 0.1
    fp32 = os.path.join(out, "alineacion_mms.onnx")
    with torch.no_grad():
        want = wrapper(dummy).numpy()
        torch.onnx.export(wrapper, (dummy,), fp32, opset_version=17, input_names=["audio"], output_names=["logp"],
                          dynamic_axes={"audio": {1: "muestras"}, "logp": {1: "fotogramas"}}, do_constant_folding=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    other = torch.randn(1, 16_000 * 7) * 0.1  # otra duración: los ejes son variables
    got = sess.run(None, {"audio": dummy.numpy()})[0]
    err = float(np.abs(got - want).max())
    print(f"alineación: ONNX frente a PyTorch, diferencia máxima {err:.2e}", flush=True)
    assert err < 1e-2, err
    with torch.no_grad():
        want2 = wrapper(other).numpy()
    err2 = float(np.abs(sess.run(None, {"audio": other.numpy()})[0] - want2).max())
    print(f"alineación: con otra duración, diferencia máxima {err2:.2e}", flush=True)
    assert err2 < 1e-2, err2
    labels = bundle.get_dict(star=None)
    del sess, model, wrapper
    import gc
    gc.collect()
    int8 = fp32.replace(".onnx", "_int8.onnx")
    quantize(fp32, int8)
    return {
        "archivos": {"fp32": os.path.basename(fp32), "int8": os.path.basename(int8)},
        "frecuencia_muestreo": 16_000,
        "letras": labels,  # letra -> índice de la salida (0 = hueco)
        "ventana_s": 30, "contexto_s": 2,
    }


def main(argv: list[str]) -> int:
    """Un paso por proceso (cada modelo ocupa varios GB al exportarlo y cuantizarlo):

        exportar.py <modelos> <salida> karaoke|voces [cancion.wav]
        exportar.py <modelos> <salida> alineacion
        exportar.py <modelos> <salida> manifiesto
    """
    model_dir, out, step = argv[0], argv[1], argv[2]
    song = argv[3] if len(argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    if step in SEPARACION:
        part = export_roformer(step, SEPARACION[step], os.path.join(model_dir, "separacion"), out, song)
    elif step == "alineacion":
        part = export_mms(os.path.join(model_dir, "alineacion"), out)
    elif step == "manifiesto":
        manifest = {"version": 1, "separacion": {}, "alineacion": None}
        for name in SEPARACION:
            manifest["separacion"][name] = json.load(open(os.path.join(out, f"{name}.json"), encoding="utf-8"))
        manifest["alineacion"] = json.load(open(os.path.join(out, "alineacion.json"), encoding="utf-8"))
        sizes = {}
        for f in sorted(os.listdir(out)):
            if f.endswith(".onnx"):
                path = os.path.join(out, f)
                sizes[f] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
                print(f"{f}: {os.path.getsize(path) / 1e6:.0f} MB", flush=True)
        manifest["tamanos"] = sizes
        with open(os.path.join(out, "manifiesto.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        return 0
    else:
        raise SystemExit(f"Paso desconocido: {step}")
    with open(os.path.join(out, f"{step}.json"), "w", encoding="utf-8") as f:
        json.dump(part, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kalena_letras", "app"))
    sys.exit(main(sys.argv[1:]))
