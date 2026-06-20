FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install basic prerequisites needed to set up the Intel GPU apt repository.
# ubuntu:24.04 is minimal and does not include curl, gpg, gnupg, or ca-certificates by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gpg \
        gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Intel GPU apt repository + runtime dependencies + Python 3.12
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
          https://repositories.intel.com/gpu/ubuntu jammy client" \
        > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        libigdgmm12 \
        intel-opencl-icd \
        espeak-ng \
        espeak-ng-data \
        git \
        wget \
        nano \
        g++ \
        cmake \
        libsndfile1 \
        ffmpeg \
        zstd \
        python3.12 \
        python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/

# Create non-root user and set up directories
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install dependencies with XPU extras
RUN uv venv --python 3.12 && \
    uv sync --extra xpu

# Japanese support requires the UniDic dictionary (~526MB on disk) for fugashi/MeCab.
# Enabled by default; set --build-arg INCLUDE_JAPANESE=false to skip and shave the image.
ARG INCLUDE_JAPANESE=true
RUN if [ "$INCLUDE_JAPANESE" = "true" ]; then \
        .venv/bin/python -m unidic download; \
    fi

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser docker/scripts/ ./

RUN chmod +x ./entrypoint.sh

# Pre-create SYCL cache dir so named-volume mounts inherit appuser ownership.
# Without this, Docker creates the mount target as root:root and appuser (uid 1001)
# cannot write to it on first run, causing silent cache misses and 30-60s latency.
RUN mkdir -p /home/appuser/.cache/sycl

# Set all environment variables in one go.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="xpu" \
    SYCL_CACHE_DIR=/home/appuser/.cache/sycl

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
