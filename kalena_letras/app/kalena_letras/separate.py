"""Separación de voces con audio-separator (modelos de la comunidad UVR).

1. Voz / música: se queda solo la voz (sin instrumentos).
2. Voz principal / coros: sobre esa voz, un modelo de karaoke separa a la cantante de los coros.

Los nombres de las salidas dependen de cada modelo, así que no se adivina cuál es la voz principal:
lo decide quien llama, viendo cuál de las dos encaja mejor con la letra.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

MODELS = {
    # Máxima calidad (RoFormer): más lento en CPU.
    "maxima": ("vocals_mel_band_roformer.ckpt", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
    # Rápida (MDX-Net, ONNX): bastante peor separando, pero varias veces más rápida.
    "rapida": ("UVR-MDX-NET-Voc_FT.onnx", "UVR_MDXNET_KARA_2.onnx"),
}


@dataclass
class Stems:
    vocals: str
    voice_a: str   # una de las dos voces que salen del modelo de karaoke
    voice_b: str


class Separator:
    def __init__(self, model_dir: str, quality: str = "maxima"):
        from audio_separator.separator import Separator as AudioSeparator
        self._cls = AudioSeparator
        self.model_dir = model_dir
        self.vocals_model, self.karaoke_model = MODELS.get(quality, MODELS["maxima"])
        os.makedirs(model_dir, exist_ok=True)

    def _run(self, model: str, src: str, out_dir: str, names: dict[str, str]) -> list[str]:
        sep = self._cls(model_file_dir=self.model_dir, output_dir=out_dir, output_format="WAV",
                        log_level=logging.WARNING)
        sep.load_model(model_filename=model)
        inst = sep.model_instance
        mapping = {inst.primary_stem_name: names["primary"], inst.secondary_stem_name: names["secondary"]}
        files = sep.separate(src, custom_output_names=mapping)
        paths = [f if os.path.isabs(f) else os.path.join(out_dir, f) for f in files]
        by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        return [by_name.get(names["primary"]), by_name.get(names["secondary"])], (inst.primary_stem_name, inst.secondary_stem_name)

    def run(self, mix_wav: str, out_dir: str) -> Stems:
        os.makedirs(out_dir, exist_ok=True)
        (p, s), (pn, sn) = self._run(self.vocals_model, mix_wav, out_dir, {"primary": "paso1_a", "secondary": "paso1_b"})
        # La voz es la salida llamada «vocals» (o la principal si el modelo no lo dice).
        vocals = s if "vocal" in (sn or "").lower() and "vocal" not in (pn or "").lower() else p
        (a, b), _ = self._run(self.karaoke_model, vocals, out_dir, {"primary": "paso2_a", "secondary": "paso2_b"})
        if not (vocals and a and b):
            raise RuntimeError("La separación de voces no ha generado los archivos esperados")
        return Stems(vocals=vocals, voice_a=a, voice_b=b)
