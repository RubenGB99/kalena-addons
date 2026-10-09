"""Separación de voces con audio-separator (modelos de la comunidad UVR).

- Karaoke (sobre la canción entera): separa la voz principal de todo lo demás (música y coros).
  Es lo único imprescindible: la letra se alinea sobre la voz principal.
- Voz (sobre la canción entera, solo si hace falta): la voz completa, principal y coros. Los coros
  salen de restarle la voz principal.

Los nombres de las salidas dependen de cada modelo, así que no se adivina cuál es la voz principal:
lo decide quien llama, viendo cuál de las dos encaja mejor con la letra.
"""
from __future__ import annotations

import logging
import os

MODELS = {
    # Máxima calidad (RoFormer): más lento en CPU.
    "maxima": ("vocals_mel_band_roformer.ckpt", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
    # Rápida (MDX-Net, ONNX): bastante peor separando, pero varias veces más rápida.
    "rapida": ("UVR-MDX-NET-Voc_FT.onnx", "UVR_MDXNET_KARA_2.onnx"),
}


class Separator:
    def __init__(self, model_dir: str, quality: str = "maxima"):
        from audio_separator.separator import Separator as AudioSeparator
        self._cls = AudioSeparator
        self.model_dir = model_dir
        self.vocals_model, self.karaoke_model = MODELS.get(quality, MODELS["maxima"])
        os.makedirs(model_dir, exist_ok=True)

    def _run(self, model: str, src: str, out_dir: str, prefix: str):
        """Separa `src` con `model`. Devuelve [(nombre de la salida según el modelo, archivo), ...]."""
        os.makedirs(out_dir, exist_ok=True)
        sep = self._cls(model_file_dir=self.model_dir, output_dir=out_dir, output_format="WAV",
                        log_level=logging.WARNING)
        sep.load_model(model_filename=model)
        inst = sep.model_instance
        stems = [inst.primary_stem_name, inst.secondary_stem_name]
        names = {stems[0]: prefix + "_1", stems[1]: prefix + "_2"}
        sep.separate(src, custom_output_names=names)
        out = []
        for stem in stems:
            path = os.path.join(out_dir, names[stem] + ".wav")
            if not os.path.exists(path):
                raise RuntimeError(f"La separación ({model}) no ha generado {os.path.basename(path)}")
            out.append((stem or "", path))
        return out

    def karaoke(self, mix_wav: str, out_dir: str) -> tuple[str, str]:
        """Las dos salidas del modelo de karaoke: una es la voz principal y la otra, todo lo demás."""
        (_, a), (_, b) = self._run(self.karaoke_model, mix_wav, out_dir, "karaoke")
        return a, b

    def vocals(self, mix_wav: str, out_dir: str) -> str:
        """La voz completa (principal y coros), sin música."""
        (n1, p1), (n2, p2) = self._run(self.vocals_model, mix_wav, out_dir, "voz")
        if "vocal" in n2.lower() and "vocal" not in n1.lower():
            return p2
        return p1
