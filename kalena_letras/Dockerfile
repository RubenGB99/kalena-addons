# Imagen fija de Python sobre Debian (con apt-get). No se usa BUILD_FROM: el Supervisor de Home
# Assistant (2026.09) ignora build.yaml y pasaría su imagen base, que es Alpine y no tiene apt-get.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg lee cualquier formato (FLAC, M4A, E-AC-3/Atmos…); build-essential compila diffq.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libsamplerate0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Todo en una sola instalación, con PyTorch para CPU (el mini PC no tiene tarjeta gráfica) fijado:
# así ninguna librería lo cambia de versión y torchaudio sigue casando con él.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && python3 -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)" \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY app /app
WORKDIR /app
CMD ["python3", "-m", "kalena_letras"]
