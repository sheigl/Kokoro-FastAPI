# Design: Intel XPU (GPU) Support for Kokoro-FastAPI

## Overview

This document outlines the technical design for adding Intel XPU (Intel GPU/Arc) support to the Kokoro-FastAPI TTS service. The implementation follows the existing patterns for CUDA and ROCm support, adapting PyTorch's XPU API equivalents while maintaining backward compatibility.

---

## Architecture Decisions

### 1. Device Detection Priority
**Decision**: Priority order: MPS → CUDA → XPU → CPU

**Rationale**: This maintains consistency with existing GPU priority while placing XPU after NVIDIA CUDA (the most common discrete GPU). Apple MPS is first as it's integrated graphics with lower compute capability. Intel Arc/Discrete GPUs come after NVIDIA but before CPU fallback.

### 2. Memory Management Strategy
**Decision**: Implement XPU memory management using `torch.xpu` API equivalents

**Rationale**: PyTorch's XPU API mirrors CUDA's API structure:
- `torch.xpu.memory_allocated()` replaces `torch.cuda.memory_allocated()`
- `torch.xpu.empty_cache()` replaces `torch.cuda.empty_cache()`
- `torch.xpu.synchronize()` replaces `torch.cuda.synchronize()`

### 3. Model Loading Approach
**Decision**: Use `.to("xpu")` for model device placement

**Rationale**: Standard PyTorch device placement works with XPU. The existing pattern of `.cuda()` will be supplemented with `.to("xpu")` for explicit device targeting.

### 4. Dependency Management
**Decision**: Add separate `xpu` extra with dedicated PyTorch index

**Rationale**: PyTorch XPU wheels are distributed from a separate index (`https://download.pytorch.org/whl/xpu`). This requires a new package index and extra to avoid conflicts with CUDA/ROCm/CPU wheels.

### 5. Docker Strategy
**Decision**: Follow ROCm pattern with Ubuntu 24.04 base

**Rationale**: Intel's official XPU Docker support and PyTorch XPU documentation recommend Ubuntu 24.04. The ROCm Dockerfile structure serves as a good template.

---

## Files to Create/Modify

### New Files

| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `docker/xpu/Dockerfile` | XPU-optimized container image | Base image selection, XPU runtime installation, dependency installation |
| `docker/xpu/docker-compose.yml` | Docker Compose for XPU | Device mounting, environment configuration |
| `docker/xpu/.dockerignore` | XPU-specific ignore patterns | Standard docker ignores |
| `start-xpu.sh` | Linux/macOS XPU startup script | Environment setup, dependency installation, server startup |
| `start-xpu.ps1` | Windows XPU startup script | Environment setup for Windows |
| `AGENTS.md` | Project coding standards | Testing framework, code patterns, conventions |

### Modified Files

| File | Changes | Reason |
|------|---------|--------|
| `api/src/core/config.py` | Add XPU to auto-detection chain | Enable automatic XPU detection |
| `api/src/inference/kokoro_v1.py` | Add XPU device handling in load, memory, OOM retry | XPU model loading and memory management |
| `api/src/inference/model_manager.py` | Update `_determine_device()` and `unload()` for XPU | Consistent device detection and cache clearing |
| `api/src/inference/base.py` | Add XPU cache clearing in `unload()` | Base class cleanup for all backends |
| `pyproject.toml` | Add XPU extra, index, and conflict rules | Dependency management |
| `docker-bake.hcl` | Add XPU build targets | Multi-platform Docker builds |
| `docker/scripts/entrypoint.sh` | Support `DEVICE=xpu` | Runtime device configuration |

---

## Task Breakdown (Ordered by Dependency)

### Task 1: Update Configuration (`api/src/core/config.py`)

**Files**: `api/src/core/config.py`

**Description**: Add XPU to the device auto-detection chain and update the docstring.

**Changes**:
```python
# In Settings class, update device_type docstring:
device_type: str | None = (
    None  # Will be auto-detected if None, can be "cuda", "mps", "xpu", or "cpu"
)

# In get_device() method, add XPU detection:
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

**Acceptance Criteria**:
- [ ] `get_device()` returns `"xpu"` when `torch.xpu.is_available()` is true
- [ ] Manual override via `device_type="xpu"` works
- [ ] Priority order is MPS → CUDA → XPU → CPU

---

### Task 2: Update Kokoro V1 Backend (`api/src/inference/kokoro_v1.py`)

**Files**: `api/src/inference/kokoro_v1.py`

**Description**: Add XPU handling for model loading, memory management, and OOM retry.

**Changes**:

#### 2.1 Model Loading (lines 70-79)
```python
# Update the device loading section:
if self._device == "mps":
    logger.info(
        "Moving model to MPS device with CPU fallback for unsupported operations"
    )
    self._model = self._model.to(torch.device("mps"))
elif self._device == "cuda":
    self._model = self._model.cuda()
elif self._device == "xpu":
    logger.info("Moving model to XPU device")
    self._model = self._model.to("xpu")
else:
    self._model = self._model.cpu()
```

#### 2.2 Memory Check (lines 348-354)
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
```

#### 2.3 Memory Clear (lines 356-365)
```python
def _clear_memory(self) -> None:
    """Clear device memory."""
    if self._device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif self._device == "xpu":
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    elif self._device == "mps":
        # Empty cache if available (future-proofing)
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
```

#### 2.4 OOM Retry in `generate_from_tokens()` (lines 189-201)
```python
# Update the exception handler:
except Exception as e:
    logger.error(f"Generation failed: {e}")
    if (
        self._device in ("cuda", "xpu")  # Changed from just "cuda"
        and model_config.pytorch_gpu.retry_on_oom
        and "out of memory" in str(e).lower()
    ):
        self._clear_memory()
        async for chunk in self.generate_from_tokens(
            tokens, voice, speed, lang_code
        ):
            yield chunk
    raise
```

#### 2.5 OOM Retry in `generate()` (lines 336-346)
```python
# Same pattern as above, update device check to include xpu
except Exception as e:
    logger.error(f"Generation failed: {e}")
    if (
        self._device in ("cuda", "xpu")  # Changed from just "cuda"
        and model_config.pytorch_gpu.retry_on_oom
        and "out of memory" in str(e).lower()
    ):
        self._clear_memory()
        async for chunk in self.generate(text, voice, speed, lang_code):
            yield chunk
    raise
```

#### 2.6 Memory Management Call (lines 131-133 and 228-231)
```python
# Update memory check calls to include XPU:
# In generate_from_tokens():
if self._device in ("cuda", "xpu"):
    if self._check_memory():
        self._clear_memory()

# In generate():
if self._device in ("cuda", "xpu"):
    if self._check_memory():
        self._clear_memory()
```

#### 2.7 unload() Method (lines 366-377)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Add XPU cleanup
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
```

**Acceptance Criteria**:
- [ ] Model loads successfully on XPU device
- [ ] Memory check uses `torch.xpu.memory_allocated()`
- [ ] Memory clear uses `torch.xpu.empty_cache()` and `torch.xpu.synchronize()`
- [ ] OOM retry triggers correctly on XPU out-of-memory errors
- [ ] unload() properly clears XPU cache

---

### Task 3: Update Model Manager (`api/src/inference/model_manager.py`)

**Files**: `api/src/inference/model_manager.py`

**Description**: Update device determination and cache clearing to include XPU.

**Changes**:

#### 3.1 `_determine_device()` (lines 33-35)
```python
def _determine_device(self) -> str:
    """Determine device based on settings."""
    # Use the centralized settings.get_device() for consistency
    return settings.get_device()
```

#### 3.2 `unload()` Method (lines 168-176)
```python
async def unload(self) -> None:
    """Release model from GPU memory. Reloads automatically on next request."""
    async with self._lock:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Add XPU cache clearing
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
    logger.info("Model unloaded from GPU memory")
```

**Acceptance Criteria**:
- [ ] `_determine_device()` uses `settings.get_device()` for consistent behavior
- [ ] `unload()` clears XPU cache when available

---

### Task 4: Update Base Backend (`api/src/inference/base.py`)

**Files**: `api/src/inference/base.py`

**Description**: Add XPU cache clearing in the base class unload method.

**Changes**:
```python
def unload(self) -> None:
    """Unload model and free resources."""
    if self._model is not None:
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Add XPU cleanup
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
```

**Acceptance Criteria**:
- [ ] Base class properly clears XPU cache when unloading

---

### Task 5: Update Dependencies (`pyproject.toml`)

**Files**: `pyproject.toml`

**Description**: Add XPU extra with PyTorch from XPU index, add conflict rules.

**Changes**:

#### 5.1 Add XPU Extra
```toml
xpu = [
    "torch==2.8.0+xpu",
]
```

#### 5.2 Update Conflict Rules
```toml
[tool.uv]
conflicts = [
    [
        { extra = "cpu" },
        { extra = "gpu" },
        { extra = "gpu-cu128" },
        { extra = "rocm" },
        { extra = "xpu" },
    ],
]
```

#### 5.3 Add XPU Source
```toml
[tool.uv.sources]
# ... existing sources ...
torch = [
    # ... existing entries ...
    { index = "pytorch-xpu", extra = "xpu" },
]

# Add at end of [[tool.uv.index]] section:
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

**Acceptance Criteria**:
- [ ] `uv sync --extra xpu` installs XPU-compatible PyTorch
- [ ] Conflicting extras (cpu, gpu, rocm, xpu) cannot be installed together

---

### Task 6: Update Docker Bake Configuration (`docker-bake.hcl`)

**Files**: `docker-bake.hcl`

**Description**: Add XPU build targets following the ROCm pattern.

**Changes**:

#### 6.1 Add XPU Base Target
```hcl
# Base settings for Intel XPU builds
target "_xpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/xpu/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (XPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. Intel XPU / Arc GPU build."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (XPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. Intel XPU / Arc GPU build.",
    ]
}
```

#### 6.2 Add XPU Build Target
```hcl
# Intel XPU only supports x86
target "xpu-amd64" {
    inherits = ["_xpu_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-xpu:${VERSION}-amd64"
    ]
}
```

#### 6.3 Add XPU Dev Target
```hcl
target "xpu-dev" {
    inherits = ["_xpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-xpu:dev"]
}
```

#### 6.4 Update Groups
```hcl
group "xpu-all" {
    targets = ["xpu-amd64"]
}

group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "xpu-amd64"]
}

group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "xpu-amd64"]
}
```

**Acceptance Criteria**:
- [ ] `docker buildx bake xpu-amd64` builds XPU image successfully
- [ ] `docker buildx bake all` includes XPU target

---

### Task 7: Create XPU Dockerfile (`docker/xpu/Dockerfile`)

**Files**: `docker/xpu/Dockerfile`

**Description**: Create XPU-optimized container image based on Intel's recommended base.

**Implementation**:
```dockerfile
# Intel XPU support for Ubuntu 24.04
# Requires: Intel GPU driver, oneAPI Base Toolkit, PyTorch with XPU support

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    gnupg \
    ca-certificates \
    espeak-ng \
    espeak-ng-data \
    libsndfile1 \
    ffmpeg \
    g++ \
    zstd \
    # Intel XPU runtime dependencies
    libze-loader1 \
    libze-intel-gpu1 \
    libigdgmm12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \

    # Install UV using the installer script
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ \

    # Create non-root user
    && useradd -m -u 1001 appuser \
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
    uv venv --python 3.12 && \
    uv sync --extra xpu

# Japanese support
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

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="xpu" \
    # Intel XPU runtime settings
    SYCL_CACHE_DIR=/home/appuser/.cache/sycl

# Run FastAPI server
CMD ["./entrypoint.sh"]
```

**Note**: For production use, Intel recommends using their oneAPI base image which includes pre-configured drivers. The standalone Dockerfile above is for minimal deployments.

**Alternative - Using Intel Base Image** (recommended for production):
```dockerfile
FROM intel/oneapi-basekit:2024.2.0
# ... rest of configuration similar to above
```

**Acceptance Criteria**:
- [ ] Dockerfile builds successfully
- [ ] PyTorch XPU wheels are installed correctly
- [ ] Environment variables are set appropriately

---

### Task 8: Create XPU Docker Compose (`docker/xpu/docker-compose.yml`)

**Files**: `docker/xpu/docker-compose.yml`

**Description**: Docker Compose configuration for XPU deployment.

**Implementation**:
```yaml
name: kokoro-fastapi-xpu
services:
  kokoro-tts:
      # image: ghcr.io/remsky/kokoro-fastapi-xpu:v${VERSION}
      build:
        context: ../..
        dockerfile: docker/xpu/Dockerfile
      devices:
        # Intel GPU device nodes
        - /dev/dri
      group_add:
        # Render group for GPU access
        # Numbers can be found via running: getent group video | cut -d: -f3
        - 44
      restart: 'always'
      volumes:
        - xpu_cache:/home/appuser/.cache/sycl
      ports:
        - 8880:8880
      environment:
        - USE_GPU=true
        # Intel XPU settings
        - SYCL_CACHE_DIR=/home/appuser/.cache/sycl

volumes:
  xpu_cache:
```

**Acceptance Criteria**:
- [ ] Docker Compose starts XPU container successfully
- [ ] Device nodes are correctly mounted
- [ ] Cache volume persists between runs

---

### Task 9: Create XPU Docker Ignore (`docker/xpu/.dockerignore`)

**Files**: `docker/xpu/.dockerignore`

**Implementation**:
```
# Standard ignores
.git
.gitignore
*.md
*.pyc
__pycache__
*.pyo
*.pyd
.Python
*.so
.env
.venv
venv/
.env.local
```

---

### Task 10: Create Start Scripts

**Files**: `start-xpu.sh`, `start-xpu.ps1`

**Description**: Shell and PowerShell scripts for local XPU development.

#### start-xpu.sh
```bash
#!/usr/bin/env bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=api/src/models
export VOICES_DIR=api/src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
# Intel XPU settings
export SYCL_CACHE_DIR=$HOME/.cache/sycl

# Run FastAPI with XPU extras using uv run
# Note: Intel XPU runtime must be installed on the host
uv pip install -e ".[xpu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
```

#### start-xpu.ps1
```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PYTHONUTF8=1
$Env:PROJECT_ROOT="$pwd"
$Env:USE_GPU="true"
$Env:USE_ONNX="false"
$Env:PYTHONPATH="$Env:PROJECT_ROOT;$Env:PROJECT_ROOT/api"
$Env:MODEL_DIR="api/src/models"
$Env:VOICES_DIR="api/src/voices/v1_0"
$Env:WEB_PLAYER_PATH="$Env:PROJECT_ROOT/web"
# Intel XPU settings
$Env:SYCL_CACHE_DIR="$Env:LOCALAPPDATA\sycl\cache"

uv pip install -e ".[xpu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
```

**Acceptance Criteria**:
- [ ] Scripts execute without errors
- [ ] Environment variables are set correctly
- [ ] Server starts on XPU device

---

### Task 11: Create AGENTS.md with Coding Standards

**Files**: `AGENTS.md`

**Description**: Document project coding standards, testing frameworks, and conventions.

**Implementation**: (See the AGENTS.md file that will be created alongside this document)

---

### Task 12: Update Unit Tests

**Files**: `api/tests/test_kokoro_v1.py`

**Description**: Update existing tests to cover XPU device scenarios.

**Changes**:
```python
# Update test_initial_state to include xpu
def test_initial_state(kokoro_backend):
    """Test initial state of KokoroV1."""
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipelines == {}
    # Device should be set based on settings
    assert kokoro_backend.device in ["cuda", "cpu", "mps", "xpu"]

# Add XPU-specific memory test
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

# Add XPU clear memory test
@patch("torch.xpu.empty_cache")
@patch("torch.xpu.synchronize")
def test_clear_memory_xpu(mock_sync, mock_clear, kokoro_backend):
    """Test XPU memory clearing."""
    with patch.object(kokoro_backend, "_device", "xpu"):
        kokoro_backend._clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()
```

---

### Task 13: Update Entry Point Script

**Files**: `docker/scripts/entrypoint.sh`

**Description**: Ensure the entry point supports XPU device type.

**Changes**: No changes needed - the existing script uses `DEVICE` environment variable which will be set to `"xpu"` by the XPU Dockerfile.

---

## Data Models / Interfaces

### Settings Extension
```python
class Settings(BaseSettings):
    # ... existing fields ...
    device_type: str | None = (
        None  # Updated docstring: can be "cuda", "mps", "xpu", or "cpu"
    )
```

### Backend Protocol
```python
class BaseModelBackend(ModelBackend):
    # Existing abstract methods remain unchanged
    # XPU support is added through device-specific implementations

    def unload(self) -> None:
        """Unload model and free resources.
        
        Now handles CUDA, XPU, and MPS memory cleanup.
        """
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
```

---

## Testing Strategy

### Unit Tests
- **Device Detection**: Test `get_device()` returns correct device based on availability
- **Memory Management**: Test `_check_memory()` and `_clear_memory()` for XPU
- **Model Loading**: Test model loads correctly on XPU device
- **OOM Retry**: Test retry logic triggers on XPU out-of-memory errors

### Integration Tests
- **Full Pipeline**: Test TTS generation on XPU device end-to-end
- **Multiple Languages**: Test pipeline creation and generation across language codes
- **Resource Cleanup**: Verify memory is properly released after unloading

### E2E Tests
- **Container Build**: Verify XPU Docker image builds successfully
- **Hardware Access**: Verify container can access Intel GPU device
- **Performance**: Benchmark TTS generation on XPU vs CUDA (if available)

### Manual Testing Checklist
- [ ] `torch.xpu.is_available()` returns `True` on XPU system
- [ ] Model loads on XPU without errors
- [ ] Audio generation produces correct output
- [ ] Memory usage is tracked correctly
- [ ] Container runs on Intel Arc/Discrete GPU hardware

---

## Potential Risks and Mitigations

### Risk 1: XPU Availability Detection
**Problem**: `torch.xpu.is_available()` may behave unexpectedly on systems without proper drivers.

**Mitigation**:
- Use `hasattr(torch, 'xpu')` check before calling XPU functions
- Fall back to CPU if XPU operations fail
- Add graceful error handling in model loading

### Risk 2: Memory Reporting Differences
**Problem**: XPU memory reporting may differ from CUDA in precision or units.

**Mitigation**:
- Use consistent division by `1e9` for GB calculation
- Monitor actual memory usage during testing
- Adjust thresholds if needed based on real-world observations

### Risk 3: Driver Version Compatibility
**Problem**: PyTorch XPU wheels require specific Intel GPU driver versions.

**Mitigation**:
- Document minimum driver requirements in README
- Provide guidance for updating Intel GPU drivers
- Consider using Intel's oneAPI base image for guaranteed compatibility

### Risk 4: Container Device Access
**Problem**: Docker container may not have proper access to `/dev/dri`.

**Mitigation**:
- Document required device mounts in docker-compose.yml
- Add informative error messages if device access fails
- Test on actual Intel GPU hardware before release

### Risk 5: Kernel Compilation/Loading
**Problem**: XPU may require JIT compilation of kernels on first run.

**Mitigation**:
- Pre-warm the model during container startup
- Cache kernel compilation results using SYCL_CACHE_DIR
- Document first-run latency expectations

### Risk 6: ROCm/oneAPI Conflicts
**Problem**: ROCm and oneAPI packages may conflict in some environments.

**Mitigation**:
- XPU extra is mutually exclusive with ROCm in conflict rules
- Provide clear installation instructions
- Document which backends require which runtime environments

---

## Docker Build Strategy

### Build Commands
```bash
# Build XPU image
docker buildx bake xpu-amd64

# Build with custom version
docker buildx bake xpu-amd64 --set *.args.VERSION=1.0.0

# Build all variants including XPU
docker buildx bake all
```

### Runtime Requirements
For XPU to work in Docker:
1. Intel GPU driver must be installed on host
2. Container needs access to `/dev/dri` device nodes
3. User must be in render/video group for device access

### Recommended Host Setup
```bash
# Check Intel GPU is visible
ls -la /dev/dri/

# Verify user has GPU access
groups $USER  # Should include video/render group
```

---

## Backward Compatibility

- **Existing CPU deployments**: No changes required
- **Existing CUDA deployments**: No changes required
- **Existing ROCm deployments**: No changes required
- **New XPU deployments**: Follow new installation instructions

The changes are additive and do not modify existing behavior unless explicitly specified (e.g., `device_type` now accepts `"xpu"` as a valid value).

---

## Handoff to Developer

**Design Document**: This document

**Estimated Complexity**: Medium

The XPU support implementation involves:
- 6 Python files to modify (config, kokoro_v1, model_manager, base, tests, entrypoint)
- 6 new files to create (Dockerfile, docker-compose, .dockerignore, 2 scripts, AGENTS.md)
- 2 config files to update (pyproject.toml, docker-bake.hcl)

**Key Files**:
1. `api/src/inference/kokoro_v1.py` - Core XPU device handling
2. `pyproject.toml` - XPU dependency configuration
3. `docker/xpu/Dockerfile` - XPU container image
4. `docker-bake.hcl` - Build configuration
5. `api/src/core/config.py` - Device detection

**Start With**: Task 1 (config.py) - Device detection is the foundation

**Testing Priority**:
1. Verify `torch.xpu.is_available()` detection works
2. Test model loading on XPU
3. Test audio generation
4. Build and test XPU Docker container
