# =============================================================================
# Phase 2: Live Intra-Op AR Pipeline - Docker Image
# Base: NVIDIA Holoscan SDK with CUDA + Vulkan
# =============================================================================

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v4.1.0-dgpu AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/download_checkpoints.sh models/
RUN chmod +x models/download_checkpoints.sh && \
    bash models/download_checkpoints.sh

COPY config/ config/
COPY evaluation/ evaluation/
COPY operators/ operators/
COPY data/ data/
COPY app.py .
COPY run.py .
COPY README.md .

ENTRYPOINT ["python", "run.py", "--config", "config/app_config.yaml"]
