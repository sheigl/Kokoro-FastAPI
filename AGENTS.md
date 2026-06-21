# Kokoro-FastAPI - Agent Coding Standards & Reference

## Project Overview

Kokoro-FastAPI is a text-to-speech (TTS) service built on FastAPI, supporting multiple device backends: CPU, NVIDIA CUDA GPU, AMD ROCm, Apple MPS, and Intel XPU. The project uses PyTorch for model inference with the Kokoro TTS model.

## Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **ML Framework**: PyTorch (with device-specific wheels)
- **Package Manager**: uv (Astral's fast Python package manager)
- **Testing**: pytest with pytest-asyncio, httpx for async HTTP testing
- **Containerization**: Docker with docker buildx bake

## Code Organization

```
api/
├── src/
│   ├── core/           # Configuration, paths, model config
│   │   ├── config.py       # Settings and device detection
│   │   ├── model_config.py # Model-specific settings
│   │   └── paths.py        # File path utilities
│   ├── inference/      # Model backends and managers
│   │   ├── base.py           # Abstract backend interface
│   │   ├── kokoro_v1.py      # Kokoro V1 model implementation
│   │   ├── model_manager.py  # Model lifecycle management
│   │   └── voice_manager.py  # Voice file handling
│   ├── services/       # Business logic
│   │   └── tts_service.py    # TTS orchestration
│   └── structures/     # Data schemas
├── tests/              # Unit and integration tests
web/                    # Web player UI
docker/                 # Container configurations
```

## Backend Code Patterns

### Device Detection Pattern

All device detection flows through `Settings.get_device()`:

```python
def get_device(self) -> str:
    """Get the appropriate device based on settings and availability"""
    if not self.use_gpu:
        return "cpu"

    if self.device_type:
        return self.device_type

    # Auto-detect device (priority: MPS > CUDA > XPU > CPU)
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    return "cpu"
```

### Model Loading Pattern

Device-specific model loading with explicit handling:

```python
async def load_model(self, path: str) -> None:
    # ... validation code ...
    
    self._model = KModel(config=config_path, model=model_path).eval()
    
    if self._device == "mps":
        self._model = self._model.to(torch.device("mps"))
    elif self._device == "cuda":
        self._model = self._model.cuda()
    elif self._device == "xpu":
        self._model = self._model.to("xpu")
    else:
        self._model = self._model.cpu()
```

### Memory Management Pattern

Device-aware memory checking and clearing:

```python
def _check_memory(self) -> bool:
    """Check if memory usage is above threshold."""
    if self._device == "cuda":
        memory_gb = torch.cuda.memory_allocated() / 1e9
        return memory_gb > model_config.pytorch_gpu.memory_threshold
    elif self._device == "xpu":
        memory_gb = torch.xpu.memory_allocated() / 1e9
        return memory_gb > model_config.pytorch_gpu.memory_threshold
    # MPS doesn't provide memory management APIs
    return False

def _clear_memory(self) -> None:
    """Clear device memory."""
    if self._device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif self._device == "xpu":
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    elif self._device == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
```

### OOM Retry Pattern

Memory error handling with retry:

```python
except Exception as e:
    logger.error(f"Generation failed: {e}")
    if (
        self._device in ("cuda", "xpu")  # GPU devices that support memory clearing
        and model_config.pytorch_gpu.retry_on_oom
        and "out of memory" in str(e).lower()
    ):
        self._clear_memory()
        async for chunk in self.generate(text, voice, speed, lang_code):
            yield chunk
    raise
```

### Unload Pattern

Resource cleanup across all device types:

```python
def unload(self) -> None:
    """Unload model and free resources."""
    if self._model is not None:
        del self._model
        self._model = None
    for pipeline in self._pipelines.values():
        del pipeline
    self._pipelines.clear()
    self._voice_cache.clear()
    
    # Clear device-specific caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
```

## Frontend Code Patterns

The web player is a static React application served by FastAPI. Key patterns:

- Static files are in the `web/` directory
- API calls use fetch with streaming response handling
- Audio playback uses Web Audio API

## Database Patterns

No database is used - all data is file-based (models, voices) or in-memory (caches).

## Testing Framework & Patterns

### Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["api/tests", "ui/tests"]
python_files = ["test_*.py"]
addopts = "--cov=api --cov=ui --cov-report=term-missing --cov-config=.coveragerc --full-trace"
asyncio_mode = "auto"
```

### Unit Test Pattern (pytest + async)

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import torch

@pytest.fixture
def kokoro_backend():
    """Create a KokoroV1 instance for testing."""
    from api.src.inference.kokoro_v1 import KokoroV1
    return KokoroV1()

@patch("torch.xpu.is_available", return_value=True)
@patch("torch.xpu.memory_allocated", return_value=5e9)
def test_memory_management_xpu(mock_memory, mock_xpu, kokoro_backend):
    """Test XPU memory management functions."""
    with patch.object(kokoro_backend, "_device", "xpu"):
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.memory_threshold = 4
            assert kokoro_backend._check_memory() == True
            
            mock_config.pytorch_gpu.memory_threshold = 6
            assert kokoro_backend._check_memory() == False

@patch("torch.xpu.empty_cache")
@patch("torch.xpu.synchronize")
def test_clear_memory_xpu(mock_sync, mock_clear, kokoro_backend):
    """Test XPU memory clearing."""
    with patch.object(kokoro_backend, "_device", "xpu"):
        kokoro_backend._clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()

@pytest.mark.asyncio
async def test_generate_validation(kokoro_backend):
    """Test generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate("test", "voice"):
            pass
```

### Integration Test Pattern (httpx + FastAPI)

```python
import httpx
from fastapi.testclient import TestClient

@pytest_asyncio.fixture
async def test_client():
    """Create a test client with mocked services."""
    from api.src.main import app
    
    # Mock the model manager and voice manager
    mock_manager = AsyncMock()
    mock_manager.generate = AsyncMock(return_value=...)
    
    with patch("api.src.services.tts_service.ModelManager", return_value=mock_manager):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client

@pytest.mark.asyncio
async def test_tts_endpoint(test_client):
    """Test TTS generation endpoint."""
    response = await test_client.post("/v1/audio/speech", json={...})
    assert response.status_code == 200
```

### Conftest Fixtures Pattern

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
import torch
import numpy as np

@pytest.fixture
def mock_voice_tensor():
    """Load a real voice tensor for testing."""
    voice_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src/voices/af_bella.pt"
    )
    return torch.load(voice_path, map_location="cpu", weights_only=False)

@pytest_asyncio.fixture
async def mock_model_manager(mock_audio_output):
    """Mock model manager for testing."""
    from api.src.inference.model_manager import ModelManager
    
    manager = AsyncMock(spec=ModelManager)
    manager.get_backend = MagicMock()
    
    async def mock_generate(*args, **kwargs):
        return np.random.rand(24000).astype(np.float32)
    
    manager.generate = AsyncMock(side_effect=mock_generate)
    return manager
```

## Docker Patterns

### Multi-Stage Build Pattern (GPU example)

```dockerfile
# Stage 1: Builder - Use devel image for compilation
FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder

# Install build dependencies and create venv
RUN apt-get update && apt-get install -y python3.10 python3-dev git curl
WORKDIR /app
COPY pyproject.toml ./pyproject.toml
ARG GPU_EXTRA=gpu
RUN uv venv --python 3.10 && \
    uv sync --extra ${GPU_EXTRA} --no-cache --no-install-project

# Stage 2: Runtime - Use smaller runtime image
FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04

# Copy venv from builder and install runtime dependencies
COPY --from=builder /app/.venv /app/.venv
```

### Docker Bake Pattern

```hcl
# Base target with common settings
target "_common" {
    context = "."
    args = {
        DEBIAN_FRONTEND = "noninteractive"
        DOWNLOAD_MODEL = "${DOWNLOAD_MODEL}"
    }
    labels = { ... }
}

# Platform-specific target inheriting from base
target "xpu-amd64" {
    inherits = ["_xpu_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-xpu:${VERSION}-amd64"
    ]
}

# Group for building all variants
group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "rocm-amd64", "xpu-amd64"]
}
```

### Docker Compose Pattern

```yaml
name: kokoro-fastapi-xpu
services:
  kokoro-tts:
    build:
      context: ../..
      dockerfile: docker/xpu/Dockerfile
    devices:
      - /dev/dri          # GPU device nodes
    group_add:
      - 44                # video/render group
    volumes:
      - xpu_cache:/home/appuser/.cache/sycl
    ports:
      - 8880:8880
    environment:
      - USE_GPU=true

volumes:
  xpu_cache:              # Named volume for cache persistence
```

### XPU Docker Container Design (2026-06-19, refined)

**Architecture**: Single-stage build using `ubuntu:24.04` base with Intel GPU packages installed via the official Intel apt repository. No multi-stage needed since Intel does not provide devel/runtime image splits like NVIDIA CUDA.

**Key Design Decisions**:
1. **Base Image**: `ubuntu:24.04` + Intel apt repository for GPU drivers (matches ROCm approach)
2. **Device String**: `DEVICE="xpu"` (not `"gpu"`) to route to `torch.xpu` module and match pyproject.toml extra name
3. **SYCL Cache**: Named volume at `/home/appuser/.cache/sycl` with `SYCL_CACHE_DIR` env var for kernel persistence
4. **Platform**: amd64 only (Intel Arc/Data Center GPUs are x86_64)
5. **Python Version**: 3.10 to match CPU and GPU containers

**Dockerfile Pattern**:
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

# Install dependencies with XPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra xpu

# Japanese support requires the UniDic dictionary (~526MB on disk) for fugashi/MeCab.
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
RUN mkdir -p /home/appuser/.cache/sycl

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="xpu" \
    SYCL_CACHE_DIR=/home/appuser/.cache/sycl

CMD ["./entrypoint.sh"]
```

**Docker Compose Pattern**:
```yaml
name: kokoro-fastapi-xpu
services:
  kokoro-tts:
    build:
      context: ../..
      dockerfile: docker/xpu/Dockerfile
    devices:
      - /dev/dri              # GPU device nodes (renderD128, card0)
    group_add:
      - 44                    # video group (DRI devices)
      - 107                  # render group (Ubuntu 24.04; verify with `getent group render`)
    restart: 'always'
    volumes:
      - sycl_cache:/home/appuser/.cache/sycl  # Persist SYCL kernel cache
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

**Docker Bake Targets**:
```hcl
target "_xpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/xpu/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (XPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. Intel XPU build (amd64 only)."
    }
}

target "xpu-amd64" {
    inherits = ["_xpu_base"]
    platforms = ["linux/amd64"]
    tags = ["${REGISTRY}/${OWNER}/${REPO}-xpu:${VERSION}-amd64"]
}

group "xpu-all" { targets = ["xpu-amd64"] }
```

**Known Risks**:
1. Intel apt repository may need manual setup for `libigdgmm12` package availability
2. Render group ID varies by distro (993 on some, 107 on Ubuntu 24.04) - document in README
3. First-run SYCL kernel compilation causes 30-60s latency; mitigated by named volume cache persistence

## Dependency Management (pyproject.toml)

### Extras Pattern with Conflicts

```toml
[project.optional-dependencies]
cpu = ["torch==2.8.0"]
gpu = [
    "torch==2.8.0+cu126 ; platform_machine == 'x86_64'",
    "torch==2.8.0+cu129 ; platform_machine == 'aarch64'",
]
rocm = ["torch==2.8.0+rocm6.4", "pytorch-triton-rocm>=3.2.0"]
xpu = [
    "torch==2.8.0+xpu",
    "pytorch-triton-xpu>=3.4.0",
]

[tool.uv]
conflicts = [
    [{ extra = "cpu" }, { extra = "gpu" }, { extra = "rocm" }, { extra = "xpu" }],
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu126", extra = "gpu", marker = "platform_machine == 'x86_64'" },
    { index = "pytorch-xpu", extra = "xpu" },
]
pytorch-triton-xpu = [
    { index = "pytorch-xpu", extra = "xpu" },
]

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

## PyTorch XPU API Reference

| CUDA API | XPU Equivalent | Description |
|----------|---------------|-------------|
| `torch.cuda.is_available()` | `torch.xpu.is_available()` | Check device availability |
| `torch.cuda.memory_allocated()` | `torch.xpu.memory_allocated()` | Get allocated memory in bytes |
| `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` | Free cached memory |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` | Wait for all operations to complete |
| `.cuda()` or `.to("cuda")` | `.to("xpu")` | Move model/tensor to device |
| `"cuda"` (device string) | `"xpu"` (device string) | Device identifier |

## Important Notes

1. **Always use `hasattr(torch, 'xpu')`** before calling XPU functions - the module may not exist if PyTorch was installed without XPU support.

2. **Memory management is device-specific**: Each GPU type has its own memory API that must be called independently.

3. **OOM retry only works on devices with cache clearing**: CUDA and XPU support this; MPS does not have reliable memory APIs.

4. **Docker requires proper device access**: XPU needs `/dev/dri` mounted, ROCm needs `/dev/kfd` and `/dev/dri`, CUDA uses NVIDIA runtime.

5. **First-run latency on XPU**: SYCL kernel compilation may cause initial slowdowns - use `SYCL_CACHE_DIR` to persist compiled kernels.

## Changelog

### 2026-06-19: XPU Dependency Resolution Fix (pyproject.toml) — AGENTS.md Updated

Updated `AGENTS.md` dependency pattern to match the corrected `pyproject.toml`: added `pytorch-triton-xpu>=3.4.0` in extras and source mapping.

### 2026-06-19: XPU Dependency Resolution Fix (pyproject.toml)

Fixed `uv sync --extra xpu` resolution failure. The original `xpu = ["torch==2.8.0+xpu"]` extra failed because `torch==2.8.0+xpu` has a hard dependency on `pytorch-triton-xpu==3.4.0`, but uv couldn't find it since the `pytorch-xpu` index is marked as `explicit = true`.

**Changes**:
- Added `pytorch-triton-xpu>=3.4.0` to `[project.optional-dependencies] xpu` (matching ROCm pattern)
- Added source mapping for `pytorch-triton-xpu` in `[tool.uv.sources]` pointing to `pytorch-xpu` index

**Verification**:
- `uv pip install torch==2.8.0+xpu --index-url https://download.pytorch.org/whl/xpu` resolves all 31 packages including full Intel runtime stack (SYCL, MKL, oneCCL, etc.)
- `hasattr(torch, 'xpu')` returns True after install
- Full `uv sync --extra xpu` still blocks on `pyopenjtalk==0.4.1` build failure (needs gcc/g++/make — not an XPU issue)

### 2026-06-19: XPU Docker Container Implementation

Created all files for the Intel XPU Docker container per `docker/xpu/DESIGN.md`:

| File | Action | Lines |
|------|--------|-------|
| `docker/xpu/.dockerignore` | Created (copy of gpu) | 40 |
| `docker/xpu/Dockerfile` | Created (single-stage ubuntu:24.04 + Intel apt repo) | 89 |
| `docker/xpu/docker-compose.yml` | Created (`kokoro-fastapi-xpu` compose config) | 34 |
| `docker-bake.hcl` | Modified (added `_xpu_base`, `xpu-amd64`, `xpu-dev` targets; updated `dev`, `all`, `individual-platforms` groups) | 267 total (+35 lines) |

**Key patterns followed from ROCm**: single-stage build, non-root appuser (1001), uv cache mounts, UniDic download arg, espeak-ng symlink pattern.
**Differences vs ROCm**: Python 3.10 (not 3.12), no kdb_install.sh / rocblas override steps, SYCL cache dir instead of MIOpen cache dirs.

**QA Result**: ✅ Passed — all 5 acceptance criteria verified line-by-line. Code Review: Approved.

### 2026-06-19: XPU Docker Container Design Refined

Updated `docker/xpu/DESIGN.md` with complete, verbatim file contents for developer implementation. Key additions vs prior design:
- Intel apt repository setup (GPG key + sources.list) is now explicit in Dockerfile pattern
- Added `intel-opencl-icd` package alongside `libigdgmm12`
- Full docker-compose.yml with render group ID 107 for Ubuntu 24.04
- Complete docker-bake.hcl target definitions and group modifications
- Environment variables reference table

### 2026-06-18: Intel XPU (GPU) Support Implementation

Added full Intel XPU support across the codebase:

| File | Changes |
|------|---------|
| `api/src/core/config.py` | Added XPU auto-detection in `get_device()` with `hasattr(torch, 'xpu')` guard |
| `api/src/inference/kokoro_v1.py` | Added XPU model placement (`.to("xpu")`), memory check (`torch.xpu.memory_allocated()`), cache clearing (`torch.xpu.empty_cache()/synchronize()`), OOM retry for xpu, pre-check memory management for xpu |
| `api/src/inference/model_manager.py` | Full device auto-detection in `_determine_device()`, XPU cache clearing in `unload()` |
| `api/src/inference/base.py` | XPU cache clearing in `BaseBackend.unload()` |
| `pyproject.toml` | Added `xpu` extra with torch/torchvision/torchaudio XPU wheels, updated conflicts list, added pytorch-xpu index source |

**Note**: Task 6 (entrypoint.sh GPU warmup) was not applicable — the current entrypoint.sh is minimal; warmup logic lives in Python (`model_manager.initialize_with_warmup()`).

**QA Result**: ✅ Passed — 139/139 tests (93 existing + 46 new XPU tests), zero regressions, all 6 acceptance criteria verified. Code Review: Approved.

### 2026-06-21: XPU Docker Container — Working Build (intel/intel-extension-for-pytorch base)

Switched XPU Dockerfile from `ubuntu:24.04` + manual Intel apt repo (which had package conflicts: `libigc2` vs `libigc1`, `libze-intel-gpu1` breaks `intel-level-zero-gpu`) to **`intel/intel-extension-for-pytorch:2.8.10-xpu`** base image.

**Why the base image was necessary**:
- The Intel image bundles PyTorch 2.8.0+xpu + IPEX 2.8.10+xpu + Level-Zero runtime all pre-built and version-matched.
- Manually assembling the GPU stack on `ubuntu:24.04` caused apt conflicts between `libigc1` (old) and `libigc2` (new) and between `intel-level-zero-gpu` and `libze-intel-gpu1`.
- Using `intel/oneapi-runtime:2025.3.1-0-devel-ubuntu24.04` had the same conflicts and shipped outdated GPU packages.

**Key changes to `docker/xpu/Dockerfile`**:
1. Base: `intel/intel-extension-for-pytorch:2.8.10-xpu` (Ubuntu 22.04, Python 3.11, torch 2.8.0+xpu pre-installed)
2. **Intel apt repo (noble)** added to upgrade GPU packages to match Ubuntu 24.04 host driver — the base image ships jammy packages which are too old for a noble host. Uses `https://repositories.intel.com/gpu/ubuntu noble client`.
3. `uv python install 3.11` — downloads a standalone Python from python-build-standalone because the base image's Python 3.11.0rc1 is **headless** (no `Python.h`), which breaks building `pyopenjtalk`.
4. `pyopenjtalk` installed via `.venv/bin/python -m pip install --no-build-isolation` (not `uv pip`) because uv enforces strict metadata validation and rejects pyopenjtalk's `0.0.0` vs `0.4.1` version mismatch.
5. `uv sync --extra xpu` installs torch into the venv from the pytorch-xpu index.

**Key changes to `docker/xpu/docker-compose.yml`**:
- `ipc: host` — required by Intel XPU containers for shared memory access.
- `group_add: [44, 992]` — video=44 (standard), render=992 (host-specific; verify with `stat -c '%g' /dev/dri/renderD128`).
- `/dev/dri/by-path` mount **not needed** (doesn't exist on this host).

**Dockerfile pattern** (`docker/xpu/Dockerfile`):
```dockerfile
FROM intel/intel-extension-for-pytorch:2.8.10-xpu

# Install build deps + upgrade GPU packages from noble repo to match 24.04 host
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl espeak-ng espeak-ng-data git nano wget libsndfile1 ffmpeg zstd \
        g++ cmake make gnupg \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
          https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        libze1 libze-intel-gpu1 intel-opencl-icd libigdgmm12 libigc2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv for package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/

# Non-root user with render group (GID must match host /dev/dri/renderD*)
RUN useradd -m -u 1001 appuser \
    && groupadd -g 992 render \
    && usermod -aG video,render appuser

# Install deps: standalone Python (has Python.h), pyopenjtalk via pip, then uv sync
RUN uv python install 3.11 && \
    uv venv --python 3.11 && \
    uv pip install --python .venv pip setuptools wheel numpy cython && \
    .venv/bin/python -m pip install --no-build-isolation pyopenjtalk unidic && \
    uv sync --extra xpu
```

**Docker Compose pattern** (`docker/xpu/docker-compose.yml`):
```yaml
services:
  kokoro-tts:
    devices:
      - /dev/dri
    ipc: host                          # Required by Intel XPU containers
    group_add:
      - 44                             # video
      - 992                            # render (verify: stat -c '%g' /dev/dri/renderD128)
    volumes:
      - sycl_cache:/home/appuser/.cache/sycl
    environment:
      - USE_GPU=true
      - DEVICE=xpu
      - SYCL_CACHE_DIR=/home/appuser/.cache/sycl
```

**Known Issues & Solutions**:
1. **`Python.h: No such file or directory`** — Base image's Python 3.11.0rc1 has no dev headers. Fixed by `uv python install 3.11` (python-build-standalone includes headers).
2. **`Package metadata version 0.0.0 does not match 0.4.1`** — uv rejects pyopenjtalk's broken metadata. Fixed by using `.venv/bin/python -m pip install --no-build-isolation` (pip doesn't validate metadata versions).
3. **`Level Zero Initialization Error` / `XPU device count is zero`** — Container's GPU packages (jammy) too old for noble host driver. Fixed by adding Intel `noble` apt repo and upgrading `libze1`, `libze-intel-gpu1`, `intel-opencl-icd`, `libigdgmm12`, `libigc2`.
4. **`groups: cannot find name for group ID 992`** — Cosmetic warning. Fixed by `groupadd -g 992 render` in Dockerfile.
5. **`pyproject.toml` Python version** — Widened from `>=3.10,<3.12` to `>=3.10,<3.13` to support Python 3.11 used by the Intel base image.

**Image size**: ~14GB (Intel base image is ~10GB + PyTorch + deps).

**Verification**: `xpu-smi health -l` works inside container; `torch.xpu.is_available()` returns True.
