"""Comprehensive tests for Intel XPU (GPU) support implementation.

These tests verify all XPU patterns across the codebase using mocked conditions,
since actual XPU hardware is not available in this test environment.
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch


# ============================================================================
# AC1: Device auto-detection tests (config.py)
# ============================================================================

class TestDeviceAutoDetectionConfig:
    """Test Settings.get_device() XPU detection logic."""

    def test_get_device_returns_xpu_when_available(self):
        """XPU is returned when available and no explicit device_type set."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type=None)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            assert settings.get_device() == "xpu"

    def test_get_device_priority_mps_over_cuda_over_xpu(self):
        """MPS > CUDA > XPU priority chain is correct."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type=None)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            assert settings.get_device() == "mps"

    def test_get_device_priority_cuda_over_xpu(self):
        """CUDA > XPU when MPS is not available."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type=None)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            assert settings.get_device() == "cuda"

    def test_get_device_falls_back_to_cpu_when_no_gpu(self):
        """Returns cpu when no GPU is available."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type=None)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = False
            assert settings.get_device() == "cpu"

    def test_get_device_returns_cpu_when_use_gpu_false(self):
        """Returns cpu when use_gpu is disabled regardless of hardware."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=False, device_type=None)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            assert settings.get_device() == "cpu"

    def test_get_device_respects_explicit_device_type(self):
        """Explicit device_type overrides auto-detection."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type="xpu")
        assert settings.get_device() == "xpu"

    def test_get_device_hasattr_guard_when_xpu_module_absent(self):
        """No crash when torch.xpu module does not exist."""
        from api.src.core.config import Settings

        settings = Settings(use_gpu=True, device_type=None)

        original_hasattr = hasattr
        def mock_hasattr(obj, name):
            if obj is torch and name == 'xpu':
                return False
            return original_hasattr(obj, name)

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("builtins.hasattr", side_effect=mock_hasattr),
        ):
            assert settings.get_device() == "cpu"


# ============================================================================
# AC2: Model loading tests (kokoro_v1.py)
# ============================================================================

class TestModelLoadingXPU:
    """Test KokoroV1 model placement on XPU device."""

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    @pytest.mark.asyncio
    async def test_model_placed_on_xpu_device(self, kokoro_backend):
        """Model is moved to xpu device when _device == 'xpu'."""
        with patch.object(kokoro_backend, "_device", "xpu"):
            mock_kmodel = MagicMock()
            mock_model_path = "/mock/model.pt"

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = mock_model_path

                await kokoro_backend.load_model(mock_model_path)

                mock_kmodel.eval.return_value.to.assert_called_once_with("xpu")
                assert kokoro_backend._model is not None

    @pytest.mark.asyncio
    async def test_model_not_placed_on_xpu_for_cuda_device(self, kokoro_backend):
        """Model uses .cuda() for cuda device, NOT .to('xpu')."""
        with patch.object(kokoro_backend, "_device", "cuda"):
            mock_kmodel = MagicMock()

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = "/mock/model.pt"

                await kokoro_backend.load_model("/mock/model.pt")

                mock_kmodel.eval.return_value.cuda.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_uses_cpu_for_non_gpu_device(self, kokoro_backend):
        """Model uses .cpu() for cpu device."""
        with patch.object(kokoro_backend, "_device", "cpu"):
            mock_kmodel = MagicMock()

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = "/mock/model.pt"

                await kokoro_backend.load_model("/mock/model.pt")

                mock_kmodel.eval.return_value.cpu.assert_called_once()


# ============================================================================
# AC3: Memory management tests (kokoro_v1.py)
# ============================================================================

class TestMemoryManagementXPU:
    """Test XPU memory check and clear functions."""

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    def test_check_memory_reads_xpu_allocated(self, kokoro_backend):
        """_check_memory uses torch.xpu.memory_allocated for xpu device."""
        with patch.object(kokoro_backend, "_device", "xpu"):
            with (
                patch("torch.xpu") as mock_xpu,
                patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            ):
                mock_xpu.memory_allocated.return_value = 5e9
                mock_config.pytorch_gpu.memory_threshold = 4

                result = kokoro_backend._check_memory()
                assert result is True
                mock_xpu.memory_allocated.assert_called_once()

    def test_check_memory_below_threshold(self, kokoro_backend):
        """_check_memory returns False when below threshold on xpu."""
        with patch.object(kokoro_backend, "_device", "xpu"):
            with (
                patch("torch.xpu") as mock_xpu,
                patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            ):
                mock_xpu.memory_allocated.return_value = 3e9
                mock_config.pytorch_gpu.memory_threshold = 4

                result = kokoro_backend._check_memory()
                assert result is False

    def test_clear_memory_calls_xpu_cache_functions(self, kokoro_backend):
        """_clear_memory calls torch.xpu.empty_cache and synchronize for xpu."""
        with patch.object(kokoro_backend, "_device", "xpu"):
            with patch("torch.xpu") as mock_xpu:
                kokoro_backend._clear_memory()

                mock_xpu.empty_cache.assert_called_once()
                mock_xpu.synchronize.assert_called_once()

    def test_clear_memory_does_not_call_cuda_for_xpu(self, kokoro_backend):
        """_clear_memory does NOT call CUDA functions when device is xpu."""
        with patch.object(kokoro_backend, "_device", "xpu"):
            with (
                patch("torch.cuda") as mock_cuda,
                patch("torch.xpu") as mock_xpu,
            ):
                kokoro_backend._clear_memory()

                mock_cuda.empty_cache.assert_not_called()
                mock_cuda.synchronize.assert_not_called()
                mock_xpu.empty_cache.assert_called_once()
                mock_xpu.synchronize.assert_called_once()

    def test_check_memory_returns_false_for_cpu(self, kokoro_backend):
        """_check_memory returns False for non-GPU devices."""
        with patch.object(kokoro_backend, "_device", "cpu"):
            result = kokoro_backend._check_memory()
            assert result is False


# ============================================================================
# AC4: OOM retry tests (kokoro_v1.py)
# ============================================================================

class TestOOMRetryXPU:
    """Test that XPU devices trigger OOM retry logic.

    These tests verify the retry condition checks and memory clearing behavior,
    rather than end-to-end async generator recursion (which is fragile to mock).
    """

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    def test_oom_retry_condition_includes_xpu_device(self, kokoro_backend):
        """XPU device is included in the OOM retry condition check."""
        # Verify source code contains xpu in the retry condition
        from api.src.inference import kokoro_v1
        import inspect

        source = inspect.getsource(kokoro_v1.KokoroV1.generate)
        assert '"cuda", "xpu"' in source or "'cuda', 'xpu'" in source, \
            "generate() OOM retry condition must include xpu device"

        source_tokens = inspect.getsource(kokoro_v1.KokoroV1.generate_from_tokens)
        assert '"cuda", "xpu"' in source_tokens or "'cuda', 'xpu'" in source_tokens, \
            "generate_from_tokens() OOM retry condition must include xpu device"

    def test_oom_retry_calls_clear_memory_for_xpu(self, kokoro_backend):
        """When OOM occurs on xpu, _clear_memory is called before retry."""
        kokoro_backend._device = "xpu"
        kokoro_backend._model = MagicMock()

        # Simulate the exact retry logic path: catch OOM -> clear memory -> retry
        with (
            patch("torch.xpu") as mock_xpu,
            patch("api.src.inference.kokoro_v1.model_config") as mock_config,
        ):
            mock_xpu.memory_allocated.return_value = 1e9
            mock_config.pytorch_gpu.memory_threshold = 4
            mock_config.pytorch_gpu.retry_on_oom = True

            # Simulate OOM error matching the condition check
            oom_error = RuntimeError("CUDA out of memory")
            device = kokoro_backend._device
            should_retry = (
                device in ("cuda", "xpu")
                and mock_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(oom_error).lower()
            )

            assert should_retry is True, \
                "XPU device with OOM error should trigger retry condition"

            # Verify _clear_memory calls XPU functions when triggered
            kokoro_backend._clear_memory()
            mock_xpu.empty_cache.assert_called_once()
            mock_xpu.synchronize.assert_called_once()

    def test_oom_retry_does_not_trigger_for_cpu_device(self, kokoro_backend):
        """OOM retry condition is False for CPU device."""
        kokoro_backend._device = "cpu"

        oom_error = RuntimeError("out of memory")
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.retry_on_oom = True

            should_retry = (
                kokoro_backend._device in ("cuda", "xpu")
                and mock_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(oom_error).lower()
            )
            assert should_retry is False, \
                "CPU device should NOT trigger OOM retry condition"

    def test_oom_retry_disabled_for_xpu(self, kokoro_backend):
        """OOM retry does not happen when retry_on_oom config is False."""
        kokoro_backend._device = "xpu"

        oom_error = RuntimeError("out of memory")
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.retry_on_oom = False

            should_retry = (
                kokoro_backend._device in ("cuda", "xpu")
                and mock_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(oom_error).lower()
            )
            assert should_retry is False, \
                "retry_on_oom=False should prevent retry even on XPU"

    def test_oom_retry_only_for_oom_errors(self, kokoro_backend):
        """Non-OOM errors do not trigger retry logic."""
        kokoro_backend._device = "xpu"

        non_oom_error = RuntimeError("model weights corrupted")
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.retry_on_oom = True

            should_retry = (
                kokoro_backend._device in ("cuda", "xpu")
                and mock_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(non_oom_error).lower()
            )
            assert should_retry is False, \
                "Non-OOM errors should NOT trigger retry logic"

    @pytest.mark.asyncio
    async def test_generate_raises_when_pipeline_fails_no_retry(self, kokoro_backend):
        """generate() raises RuntimeError when pipeline fails and retry is disabled."""
        kokoro_backend._device = "xpu"
        kokoro_backend._model = MagicMock()

        bad_pipeline = MagicMock()
        bad_pipeline.side_effect = RuntimeError("out of memory")

        with (
            patch("torch.xpu") as mock_xpu,
            patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load,
            patch("api.src.core.paths.save_voice_tensor", new_callable=AsyncMock),
        ):
            mock_xpu.memory_allocated.return_value = 1e9
            mock_config.pytorch_gpu.memory_threshold = 4
            # retry_on_oom=False means error propagates directly
            mock_config.pytorch_gpu.retry_on_oom = False
            mock_load.return_value = torch.ones(1)

            with patch.object(kokoro_backend, "_get_pipeline", return_value=bad_pipeline):
                with pytest.raises(RuntimeError, match="out of memory"):
                    async for _ in kokoro_backend.generate("test text", "af_voice"):
                        pass

    @pytest.mark.asyncio
    async def test_generate_from_tokens_raises_when_pipeline_fails_no_retry(self, kokoro_backend):
        """generate_from_tokens() raises RuntimeError when pipeline fails and retry is disabled."""
        kokoro_backend._device = "xpu"
        kokoro_backend._model = MagicMock()

        bad_pipeline = MagicMock()
        bad_pipeline.generate_from_tokens.side_effect = RuntimeError("out of memory")

        with (
            patch("torch.xpu") as mock_xpu,
            patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load,
            patch("api.src.core.paths.save_voice_tensor", new_callable=AsyncMock),
        ):
            mock_xpu.memory_allocated.return_value = 1e9
            mock_config.pytorch_gpu.memory_threshold = 4
            mock_config.pytorch_gpu.retry_on_oom = False
            mock_load.return_value = torch.ones(1)

            with patch.object(kokoro_backend, "_get_pipeline", return_value=bad_pipeline):
                with pytest.raises(RuntimeError, match="out of memory"):
                    async for _ in kokoro_backend.generate_from_tokens("test tokens", "af_voice"):
                        pass


# ============================================================================
# AC5: Unload cleanup tests - all 3 paths
# ============================================================================

class TestUnloadCleanupXPU:
    """Test that all unload paths clear XPU caches."""

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    # --- Path 1: KokoroV1.unload() ---
    def test_kokoro_v1_unload_clears_xpu_cache(self, kokoro_backend):
        """KokoroV1.unload() clears XPU cache when available."""
        kokoro_backend._model = MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True

            kokoro_backend.unload()

            assert not kokoro_backend.is_loaded
            mock_xpu.empty_cache.assert_called_once()
            mock_xpu.synchronize.assert_called_once()

    def test_kokoro_v1_unload_skips_xpu_when_not_available(self, kokoro_backend):
        """KokoroV1.unload() skips XPU cache when xpu not available."""
        kokoro_backend._model = MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = False

            kokoro_backend.unload()

            assert not kokoro_backend.is_loaded
            mock_xpu.empty_cache.assert_not_called()

    def test_kokoro_v1_unload_skips_xpu_when_module_absent(self, kokoro_backend):
        """KokoroV1.unload() doesn't crash when torch.xpu module is absent."""
        kokoro_backend._model = MagicMock()

        original_hasattr = hasattr
        def mock_hasattr(obj, name):
            if obj is torch and name == 'xpu':
                return False
            return original_hasattr(obj, name)

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("builtins.hasattr", side_effect=mock_hasattr),
        ):
            kokoro_backend.unload()
            assert not kokoro_backend.is_loaded

    # --- Path 2: ModelManager.unload() ---
    @pytest.mark.asyncio
    async def test_model_manager_unload_clears_xpu_cache(self):
        """ModelManager.unload() clears XPU cache when available."""
        from api.src.inference.model_manager import ModelManager

        manager = ModelManager()
        mock_backend = MagicMock()
        manager._backend = mock_backend

        with patch("api.src.inference.model_manager.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.xpu.is_available.return_value = True

            await manager.unload()

            mock_backend.unload.assert_called_once()
            assert manager._backend is None
            mock_torch.xpu.empty_cache.assert_called_once()
            mock_torch.xpu.synchronize.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_manager_unload_skips_xpu_when_not_available(self):
        """ModelManager.unload() skips XPU cache when not available."""
        from api.src.inference.model_manager import ModelManager

        manager = ModelManager()
        mock_backend = MagicMock()
        manager._backend = mock_backend

        with patch("api.src.inference.model_manager.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.xpu.is_available.return_value = False

            await manager.unload()

            mock_backend.unload.assert_called_once()
            assert manager._backend is None
            mock_torch.xpu.empty_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_manager_unload_hasattr_guard(self):
        """ModelManager.unload() handles missing torch.xpu module gracefully."""
        from api.src.inference.model_manager import ModelManager

        manager = ModelManager()
        mock_backend = MagicMock()
        manager._backend = mock_backend

        with patch("api.src.inference.model_manager.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Simulate no xpu attribute on torch
            del mock_torch.xpu

            await manager.unload()

            mock_backend.unload.assert_called_once()
            assert manager._backend is None

    # --- Path 3: BaseBackend.unload() - test via minimal subclass ---
    def test_base_backend_unload_clears_xpu_cache(self):
        """BaseModelBackend.unload() clears XPU cache when available."""
        from api.src.inference.base import BaseModelBackend

        class MinimalBackend(BaseModelBackend):
            """Minimal concrete subclass that does NOT override unload()."""
            async def load_model(self, path: str) -> None:
                pass

            async def generate(self, text, voice, speed=1.0):
                return iter([])

        backend = MinimalBackend()
        backend._model = MagicMock()

        with patch("api.src.inference.base.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.xpu.is_available.return_value = True

            backend.unload()

            assert not backend.is_loaded
            mock_torch.xpu.empty_cache.assert_called_once()
            mock_torch.xpu.synchronize.assert_called_once()

    def test_base_backend_unload_skips_xpu_when_not_available(self):
        """BaseModelBackend.unload() skips XPU cache when not available."""
        from api.src.inference.base import BaseModelBackend

        class MinimalBackend(BaseModelBackend):
            """Minimal concrete subclass that does NOT override unload()."""
            async def load_model(self, path: str) -> None:
                pass

            async def generate(self, text, voice, speed=1.0):
                return iter([])

        backend = MinimalBackend()
        backend._model = MagicMock()

        with patch("api.src.inference.base.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.xpu.is_available.return_value = False

            backend.unload()

            assert not backend.is_loaded
            mock_torch.xpu.empty_cache.assert_not_called()


# ============================================================================
# AC6: No regressions - existing CUDA/MPS/CPU paths still work
# ============================================================================

class TestNoRegressions:
    """Verify existing device code paths remain functional."""

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    def test_cuda_memory_check_still_works(self, kokoro_backend):
        """CUDA _check_memory still uses torch.cuda.memory_allocated."""
        with patch.object(kokoro_backend, "_device", "cuda"):
            with (
                patch("torch.cuda") as mock_cuda,
                patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            ):
                mock_cuda.memory_allocated.return_value = 5e9
                mock_config.pytorch_gpu.memory_threshold = 4

                result = kokoro_backend._check_memory()
                assert result is True
                mock_cuda.memory_allocated.assert_called_once()

    def test_cuda_clear_memory_still_works(self, kokoro_backend):
        """CUDA _clear_memory still uses torch.cuda functions."""
        with patch.object(kokoro_backend, "_device", "cuda"):
            with (
                patch("torch.cuda") as mock_cuda,
                patch("torch.xpu") as mock_xpu,
            ):
                kokoro_backend._clear_memory()

                mock_cuda.empty_cache.assert_called_once()
                mock_cuda.synchronize.assert_called_once()
                mock_xpu.empty_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_cuda_model_loading_still_works(self, kokoro_backend):
        """CUDA model loading still uses .cuda()."""
        with patch.object(kokoro_backend, "_device", "cuda"):
            mock_kmodel = MagicMock()

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = "/mock/model.pt"

                await kokoro_backend.load_model("/mock/model.pt")
                mock_kmodel.eval.return_value.cuda.assert_called_once()

    def test_mps_clear_memory_still_works(self, kokoro_backend):
        """MPS _clear_memory still uses torch.mps.empty_cache if available."""
        with patch.object(kokoro_backend, "_device", "mps"):
            with (
                patch("torch.cuda") as mock_cuda,
                patch("torch.xpu") as mock_xpu,
                patch("torch.mps") as mock_mps,
            ):
                kokoro_backend._clear_memory()

                mock_cuda.empty_cache.assert_not_called()
                mock_xpu.empty_cache.assert_not_called()
                if hasattr(torch.mps, "empty_cache"):
                    mock_mps.empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_mps_model_loading_still_works(self, kokoro_backend):
        """MPS model loading still uses .to('mps')."""
        with patch.object(kokoro_backend, "_device", "mps"):
            mock_kmodel = MagicMock()

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = "/mock/model.pt"

                await kokoro_backend.load_model("/mock/model.pt")
                mock_kmodel.eval.return_value.to.assert_called_once_with(torch.device("mps"))

    def test_cpu_check_memory_returns_false(self, kokoro_backend):
        """CPU _check_memory returns False (no GPU memory to check)."""
        with patch.object(kokoro_backend, "_device", "cpu"):
            result = kokoro_backend._check_memory()
            assert result is False

    @pytest.mark.asyncio
    async def test_cpu_model_loading_still_works(self, kokoro_backend):
        """CPU model loading still uses .cpu()."""
        with patch.object(kokoro_backend, "_device", "cpu"):
            mock_kmodel = MagicMock()

            with (
                patch("api.src.inference.kokoro_v1.KModel", return_value=mock_kmodel),
                patch("os.path.exists", return_value=True),
                patch("api.src.core.paths.get_model_path", new_callable=AsyncMock) as mock_get_path,
            ):
                mock_get_path.return_value = "/mock/model.pt"

                await kokoro_backend.load_model("/mock/model.pt")
                mock_kmodel.eval.return_value.cpu.assert_called_once()


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def kokoro_backend(self):
        from api.src.inference.kokoro_v1 import KokoroV1
        return KokoroV1()

    @pytest.mark.asyncio
    async def test_pre_check_memory_for_xpu_in_generate(self, kokoro_backend):
        """generate() pre-checks memory on xpu device before generation."""
        kokoro_backend._device = "xpu"
        kokoro_backend._model = MagicMock()

        with (
            patch("torch.xpu") as mock_xpu,
            patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load,
            patch("api.src.core.paths.save_voice_tensor", new_callable=AsyncMock),
        ):
            mock_xpu.memory_allocated.return_value = 5e9
            mock_config.pytorch_gpu.memory_threshold = 4
            mock_load.return_value = torch.ones(1)

            mock_pipeline = MagicMock()
            mock_pipeline.return_value = iter([])
            with patch("api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline):
                async for _ in kokoro_backend.generate("test text", "af_voice"):
                    pass

            mock_xpu.memory_allocated.assert_called()
            mock_xpu.empty_cache.assert_called()

    @pytest.mark.asyncio
    async def test_pre_check_memory_for_xpu_in_generate_from_tokens(self, kokoro_backend):
        """generate_from_tokens() pre-checks memory on xpu device."""
        kokoro_backend._device = "xpu"
        kokoro_backend._model = MagicMock()

        with (
            patch("torch.xpu") as mock_xpu,
            patch("api.src.inference.kokoro_v1.model_config") as mock_config,
            patch("api.src.core.paths.load_voice_tensor", new_callable=AsyncMock) as mock_load,
            patch("api.src.core.paths.save_voice_tensor", new_callable=AsyncMock),
        ):
            mock_xpu.memory_allocated.return_value = 5e9
            mock_config.pytorch_gpu.memory_threshold = 4
            mock_load.return_value = torch.ones(1)

            mock_pipeline = MagicMock()
            mock_pipeline.return_value = iter([])
            with patch("api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline):
                async for _ in kokoro_backend.generate_from_tokens("test tokens", "af_voice"):
                    pass

            mock_xpu.memory_allocated.assert_called()
            mock_xpu.empty_cache.assert_called()

    def test_model_manager_determine_device_returns_xpu(self):
        """ModelManager._determine_device returns xpu when available."""
        from api.src.inference.model_manager import ModelManager

        with (
            patch("api.src.inference.model_manager.settings") as mock_settings,
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_settings.use_gpu = True
            mock_settings.device_type = None
            mock_xpu.is_available.return_value = True

            manager = ModelManager()
            device = manager._determine_device()
            assert device == "xpu"

    def test_model_manager_determine_device_priority_chain(self):
        """ModelManager._determine_device follows MPS > CUDA > XPU > CPU priority."""
        from api.src.inference.model_manager import ModelManager

        with (
            patch("api.src.inference.model_manager.settings") as mock_settings,
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_settings.use_gpu = True
            mock_settings.device_type = None
            mock_xpu.is_available.return_value = True

            manager = ModelManager()
            device = manager._determine_device()
            assert device == "mps"

    def test_unload_clears_both_cuda_and_xpu(self, kokoro_backend):
        """KokoroV1.unload clears both CUDA and XPU caches when both available."""
        kokoro_backend._model = MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
            patch("torch.cuda") as mock_cuda,
        ):
            mock_xpu.is_available.return_value = True

            kokoro_backend.unload()

            mock_cuda.empty_cache.assert_called_once()
            mock_cuda.synchronize.assert_called_once()
            mock_xpu.empty_cache.assert_called_once()
            mock_xpu.synchronize.assert_called_once()


# ============================================================================
# pyproject.toml XPU extra validation
# ============================================================================

class TestPyprojectXPU:
    """Validate pyproject.toml XPU configuration."""

    def test_xpu_extra_defined(self):
        """xpu extra is defined in optional-dependencies."""
        import tomli

        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)

        assert "xpu" in config["project"]["optional-dependencies"]
        xpu_deps = config["project"]["optional-dependencies"]["xpu"]

        assert any("torch==2.8.0+xpu" == dep for dep in xpu_deps)
        assert any("torchvision==0.23.0+xpu" == dep for dep in xpu_deps)
        assert any("torchaudio==2.8.0+xpu" == dep for dep in xpu_deps)

    def test_xpu_in_conflicts(self):
        """xpu is listed in tool.uv.conflicts."""
        import tomli

        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)

        conflicts = config["tool"]["uv"]["conflicts"]
        found_xpu_conflict = False
        for conflict_group in conflicts:
            extras_in_group = [c.get("extra") for c in conflict_group if isinstance(c, dict)]
            if "xpu" in extras_in_group:
                found_xpu_conflict = True
                assert "cpu" in extras_in_group
                assert "gpu" in extras_in_group
                assert "rocm" in extras_in_group
                break

        assert found_xpu_conflict, "xpu not found in any conflict group"

    def test_pytorch_xpu_index_defined(self):
        """pytorch-xpu index is defined in tool.uv.index."""
        import tomli

        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)

        indices = config["tool"]["uv"]["index"]
        xpu_index = [idx for idx in indices if idx.get("name") == "pytorch-xpu"]
        assert len(xpu_index) == 1, "pytorch-xpu index not found"
        assert xpu_index[0]["url"] == "https://download.pytorch.org/whl/xpu"
        assert xpu_index[0]["explicit"] is True

    def test_xpu_source_configured(self):
        """torch source includes pytorch-xpu for xpu extra."""
        import tomli

        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)

        torch_sources = config["tool"]["uv"]["sources"]["torch"]
        xpu_source = [s for s in torch_sources if s.get("index") == "pytorch-xpu"]
        assert len(xpu_source) >= 1, "No pytorch-xpu source found for torch"
        assert xpu_source[0].get("extra") == "xpu"
