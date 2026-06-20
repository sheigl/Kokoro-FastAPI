# Design: Docker Container for Intel XPU (GPU) Support

## Overview

Add an Intel XPU Docker container to Kokoro-FastAPI, enabling TTS inference on Intel Arc and Data Center GPUs via PyTorch's `torch.xpu` SYCL backend. Follows the ROCm single-stage pattern exactly.

---

## Architecture Decisions

### 1. Single-Stage Build (like ROCm, unlike CPU/GPU CUDA)

**Decision**: Single-stage build on `ubuntu:24.04`. No multi-stage builder.

**Rationale**: Intel does not provide separate devel/runtime images like NVIDIA CUDA. PyTorch XPU wheels are pre-built — no compilation happens inside the container. The ROCm pattern (Ubuntu base + apt-installed GPU packages) maps directly.

### 2. Ubuntu 24.04 Base with Intel Apt Repository

**Decision**: `ubuntu:24.04` + Intel's official GPU apt repository for driver packages.

**Rationale**:
- Matches ROCm approach (Ubuntu base + vendor apt repo)
- Intel does not publish pre-built container images like NVIDIA (`nvcr.io`) or AMD (`rocm/dev-ubuntu-*`)
- `libigdgmm12` and other GPU runtime libraries are only available via Intel's apt repository

**Intel Apt Repository Setup**:
```dockerfile
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
          https://repositories.intel.com/gpu/ubuntu jammy client" \
    > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update
```

### 3. Device String: `"xpu"` (not `"gpu"`)

**Decision**: `DEVICE="xpu"` environment variable.

**Rationale**: The entrypoint.sh runs `uv run --extra $DEVICE`, so `$DEVICE` must match the pyproject.toml extra name (`xpu`). ROCm uses `DEVICE="gpu"` because its extra is named `rocm` and it overrides via compose env — but XPU's extra IS `xpu`.

### 4. Platform: amd64 Only

**Decision**: `linux/amd64` only. No arm64 target.

**Rationale**: Intel Arc and Data Center Max GPUs are x86_64 exclusively.

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `docker/xpu/Dockerfile` | Single-stage container image definition |
| `docker/xpu/docker-compose.yml` | Local development orchestration |
| `docker/xpu/.dockerignore` | Build context exclusions |

### Modified Files

| File | Changes |
|------|---------|
| `docker-bake.hcl` | Add `_xpu_base`, `xpu-amd64`, `xpu-dev` targets; update groups |

---

## Full File Contents

### 1. `docker/xpu/Dockerfile`

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install Intel GPU apt repository + runtime dependencies
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
        curl \
        wget \
        nano \
        g++ \
        cmake \
        libsndfile1 \
        ffmpeg \
        zstd \
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

# Install dependencies with XPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
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
```

### 2. `docker/xpu/docker-compose.yml`

```yaml
name: kokoro-fastapi-xpu
services:
  kokoro-tts:
    # image: ghcr.io/remsky/kokoro-fastapi-xpu:v${VERSION}
    build:
      context: ../..
      dockerfile: docker/xpu/Dockerfile
    devices:
      - /dev/dri
    group_add:
      # NOTE: These groups are the group ids for video and render.
      # Numbers can be found via running: getent group $GROUP_NAME | cut -d: -f3
      # On Ubuntu 24.04: video=44, render=107
      # On other distros render may differ (e.g. 993). Verify with `getent group render`.
      - 44       # video group (DRI devices)
      - 107      # render group (Ubuntu 24.04; verify on your system)
    restart: 'always'
    # Named volume persists the SYCL kernel cache at /home/appuser/.cache/sycl.
    # Survives `docker compose down`; `docker compose down -v` clears it. Inspect
    # contents with `docker volume inspect kokoro-fastapi-xpu_sycl_cache`.
    volumes:
      - sycl_cache:/home/appuser/.cache/sycl
    ports:
      - 8880:8880
    environment:
      - USE_GPU=true
      - DEVICE="xpu"
      - SYCL_CACHE_DIR=/home/appuser/.cache/sycl
      - PYTHONUNBUFFERED=1
      - API_LOG_LEVEL=DEBUG
      - DOWNLOAD_MODEL=true

volumes:
  sycl_cache:
```

### 3. `docker/xpu/.dockerignore`

```gitignore
# Version control
.git

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.py[cod]
*$py.class
.pytest_cache
.coverage
.coveragerc

# Environment
# .env
.venv*
env/
venv/
ENV/

# IDE
.idea
.vscode
*.swp
*.swo

# Project specific
examples/
Kokoro-82M/
ui/
tests/
*.md
*.txt
!requirements.txt

# Docker
Dockerfile*
docker-compose*
```

### 4. `docker-bake.hcl` — Additions and Modifications

#### New targets (insert after the `_rocm_base` / `rocm-amd64` block, around line 184):

```hcl
# Base settings for Intel XPU builds
target "_xpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/xpu/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (XPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. Intel XPU build (amd64 only)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (XPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. Intel XPU build (amd64 only).",
    ]
}

# Intel XPU only supports x86
target "xpu-amd64" {
    inherits = ["_xpu_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-xpu:${VERSION}-amd64"
    ]
}

# Development target for faster local builds
target "xpu-dev" {
    inherits = ["_xpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-xpu:dev"]
}

# Group for all XPU variants
group "xpu-all" {
    targets = ["xpu-amd64"]
}
```

#### Modified groups (replace existing definitions):

```hcl
# Development group — add xpu-dev
group "dev" {
    targets = ["cpu-dev", "gpu-dev", "xpu-dev"]
}

# All platforms — add xpu-amd64
group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "xpu-amd64"]
}

# Individual platforms — add xpu-amd64
group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "xpu-amd64"]
}
```

---

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `USE_GPU` | `true` | Enables GPU processing path in Settings.get_device() |
| `DEVICE` | `"xpu"` | Routes to `torch.xpu` module; used by entrypoint.sh as `--extra xpu` |
| `SYCL_CACHE_DIR` | `/home/appuser/.cache/sycl` | Intel SYCL compiler cache location for kernel persistence |
| `DOWNLOAD_MODEL` | `true` | Triggers model download at container startup via entrypoint.sh |
| `INCLUDE_JAPANESE` | `true` (build arg) | Downloads UniDic dictionary (~526MB) for Japanese TTS support |
| `PYTHONUNBUFFERED` | `1` | Disables Python stdout/stderr buffering for log visibility |
| `PYTHONPATH` | `/app:/app/api` | Python import paths for the application |
| `UV_LINK_MODE` | `copy` | uv package installation mode (avoids symlink issues in containers) |
| `PHONEMIZER_ESPEAK_PATH` | `/usr/bin` | espeak-ng binary location for phonemizer-fork |
| `PHONEMIZER_ESPEAK_DATA` | `/usr/share/espeak-ng-data` | espeak-ng data directory |
| `ESPEAK_DATA_PATH` | `/usr/share/espeak-ng-data` | Alternative env var name some libraries check |

---

## Docker Compose Configuration Reference

### Device Access (required)
```yaml
devices:
  - /dev/dri              # GPU device nodes (renderD128, card0, etc.)
```

### Group Permissions (required for hardware access)
```yaml
group_add:
  - 44                    # video group — DRI devices
  - 107                   # render group — Ubuntu 24.04; verify with `getent group render`
```

### Named Volume (for cache persistence)
```yaml
volumes:
  sycl_cache:/home/appuser/.cache/sycl   # SYCL kernel JIT cache
```

---

## Testing Strategy

### Build Verification
1. **Syntax check**: `docker buildx bake -f docker-bake.hcl --print xpu-amd64`
2. **Full build**: `docker buildx bake -f docker-bake.hcl xpu-amd64`

### Runtime Verification (requires Intel GPU hardware)
3. **Device availability**: 
   ```bash
   docker run --rm --device /dev/dri:/dev/dri \
     ghcr.io/remsky/kokoro-fastapi-xpu:latest \
     python -c "import torch; print('XPU available:', hasattr(torch, 'xpu') and torch.xpu.is_available())"
   ```
4. **Compose startup**: `docker compose -f docker/xpu/docker-compose.yml up --build`
5. **API health check**: `curl http://localhost:8880/docs` — verify Swagger UI loads
6. **TTS generation test**: POST to `/v1/audio/speech` with sample text, verify audio output

### Cache Persistence Verification
7. Run container, generate audio (triggers SYCL compilation)
8. `docker compose down && docker compose up`
9. Second run should be significantly faster (< 5s vs 30-60s first run)

---

## Potential Risks and Mitigations

### Risk 1: Intel Apt Repository Availability
**Risk**: The Intel GPU apt repository (`repositories.intel.com/gpu/ubuntu`) may require registration or have rate limits.

**Mitigation**: 
- Test the repository URL before committing
- If unavailable, fall back to `libigdgmm12` from Ubuntu's main repos (available in 24.04) and skip `intel-opencl-icd` — it is only needed for OpenCL interop, not PyTorch XPU

### Risk 2: Render Group ID Variation
**Risk**: The render group ID varies by distribution (107 on Ubuntu 24.04, 993 on some others).

**Mitigation**: Document in compose file comments; users can verify with `getent group render | cut -d: -f3`.

### Risk 3: First-Run SYCL Kernel Compilation Latency
**Risk**: Initial TTS requests may take 30-60 seconds due to JIT kernel compilation.

**Mitigation**: Named volume for SYCL cache persistence eliminates this after first run. Document in README. Unlike ROCm's MIOpen warmup script, there is no equivalent batch warmup tool for XPU — the cache populates organically on first inference calls.

### Risk 4: Python Version Consistency
**Risk**: ROCm uses Python 3.12 while CPU/GPU use 3.10. 

**Decision**: Use Python 3.10 to match CPU and GPU containers. PyTorch XPU wheels are available for both versions, but consistency reduces maintenance burden.

---

## Handoff to Developer

**Design Document**: This specification
**Estimated Complexity**: Medium
**Key Files**: `docker/xpu/Dockerfile`, `docker-bake.hcl` (modifications), `docker/xpu/docker-compose.yml`
**Start With**: Task 1 — create the Dockerfile, then modify docker-bake.hcl

### Implementation Order
1. Create `docker/xpu/.dockerignore` (trivial copy)
2. Create `docker/xpu/Dockerfile` (main work)
3. Modify `docker-bake.hcl` (add targets + update groups)
4. Create `docker/xpu/docker-compose.yml` (local dev only)
5. Verify build: `docker buildx bake --print xpu-amd64`

### Critical Notes for Developer
- The Intel apt repository setup is the most fragile part — test it early
- Use Python 3.10 (`uv venv --python 3.10`) to match CPU/GPU containers
- `DEVICE="xpu"` must be exact — entrypoint.sh uses it as uv extra name
- Pre-create `/home/appuser/.cache/sycl` in Dockerfile so named volume mounts get correct ownership
