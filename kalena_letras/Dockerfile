ARG BUILD_FROM=python:3.11-slim-bookworm
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg lee cualquier formato (FLAC, M4A, E-AC-3/Atmos…); build-essential compila diffq.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libsamplerate0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch solo para CPU (el mini PC no tiene tarjeta gráfica): mucho más pequeño.
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY app /app
WORKDIR /app
CMD ["python3", "-m", "kalena_letras"]
