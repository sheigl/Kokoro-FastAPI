# syntax=docker/dockerfile:1
FROM intel/intel-extension-for-pytorch:2.8.10-xpu

ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        espeak-ng \
        espeak-ng-data \
        git \
        nano \
        wget \
        libsndfile1 \
        ffmpeg \
        zstd \
        g++ \
        cmake \
        make \
        gnupg \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
          https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        libze1 \
        libze-intel-gpu1 \
        intel-opencl-icd \
        libigdgmm12 \
        libigc2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -sf /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ 2>/dev/null || true

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/

RUN useradd -m -u 1001 appuser \
    && groupadd -g 992 render \
    && usermod -aG video,render appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

RUN uv python install 3.11 && \
    uv venv --python 3.11 && \
    uv pip install --python .venv pip setuptools wheel numpy cython && \
    .venv/bin/python -m pip install --no-build-isolation pyopenjtalk unidic && \
    uv sync --extra xpu && \
    uv pip install --python .venv --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us intel-extension-for-pytorch==2.8.10+xpu && \
    echo "import intel_extension_for_pytorch" > .venv/lib/python3.11/site-packages/sitecustomize.py

ARG INCLUDE_JAPANESE=true
RUN if [ "$INCLUDE_JAPANESE" = "true" ]; then \
        .venv/bin/python -m unidic download; \
    fi

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser docker/scripts/ ./

RUN chmod +x ./entrypoint.sh

RUN mkdir -p /home/appuser/.cache/sycl

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE=xpu \
    SYCL_CACHE_DIR=/home/appuser/.cache/sycl

CMD ["./entrypoint.sh"]
