# tests/core/test_gpu_device.py
"""Unit tests for the GPU device abstraction (core.gpu)."""
from __future__ import annotations

import importlib
from unittest.mock import patch, MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_gpu_module():
    """Re-import core.gpu before each test to clear lru_cache."""
    mod = importlib.import_module("core.gpu")
    importlib.reload(mod)
    yield
    importlib.reload(mod)


def _get_device(backend: str) -> torch.device:
    from core.gpu import get_torch_device, reset_device_cache
    reset_device_cache()
    return get_torch_device(backend)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCPUBackend:
    def test_cpu_backend_always_returns_cpu(self):
        device = _get_device("cpu")
        assert str(device) == "cpu"

    def test_cpu_backend_ignores_cuda_availability(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = _get_device("cpu")
            assert str(device) == "cpu"


class TestCUDABackend:
    def test_cuda_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = _get_device("cuda")
            assert "cuda" in str(device)

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = _get_device("cuda")
            assert str(device) == "cpu"


class TestDirectMLBackend:
    def test_dml_returns_dml_device_when_available(self):
        mock_dml = MagicMock()
        mock_dml.device.return_value = torch.device("privateuseone:0")
        with patch.dict("sys.modules", {"torch_directml": mock_dml}):
            # Need to reload so _has_directml picks up the mock
            mod = importlib.import_module("core.gpu")
            importlib.reload(mod)
            mod.reset_device_cache()
            device = mod.get_torch_device("dml")
            assert "cpu" not in str(device)

    def test_dml_falls_back_to_cpu_when_unavailable(self):
        # Ensure torch_directml is NOT importable
        with patch.dict("sys.modules", {"torch_directml": None}):
            mod = importlib.import_module("core.gpu")
            importlib.reload(mod)
            mod.reset_device_cache()
            device = mod.get_torch_device("dml")
            assert str(device) == "cpu"


class TestAutoBackend:
    def test_auto_prefers_cuda_over_dml(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = _get_device("auto")
            assert "cuda" in str(device)

    def test_auto_falls_back_to_cpu_when_nothing_available(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict("sys.modules", {"torch_directml": None}):
                mod = importlib.import_module("core.gpu")
                importlib.reload(mod)
                mod.reset_device_cache()
                device = mod.get_torch_device("auto")
                assert str(device) == "cpu"


class TestInputValidation:
    def test_none_backend_defaults_to_auto(self):
        """None backend should read from Settings, which defaults to 'auto'."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict("sys.modules", {"torch_directml": None}):
                mod = importlib.import_module("core.gpu")
                importlib.reload(mod)
                mod.reset_device_cache()
                device = mod.get_torch_device(None)
                # Without CUDA or DML, auto → cpu
                assert str(device) == "cpu"

    def test_invalid_backend_falls_back_to_auto(self):
        """Garbage input should be normalised to 'auto'."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict("sys.modules", {"torch_directml": None}):
                mod = importlib.import_module("core.gpu")
                importlib.reload(mod)
                mod.reset_device_cache()
                device = mod.get_torch_device("INVALID_BACKEND_XYZ")
                assert str(device) == "cpu"

    def test_whitespace_backend_normalised(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = _get_device("  CPU  ")
            assert str(device) == "cpu"


class TestCaching:
    def test_repeated_calls_return_same_object(self):
        from core.gpu import get_torch_device, reset_device_cache
        reset_device_cache()
        d1 = get_torch_device("cpu")
        d2 = get_torch_device("cpu")
        assert d1 is d2  # same cached object

    def test_reset_clears_cache(self):
        from core.gpu import get_torch_device, reset_device_cache
        reset_device_cache()
        d1 = get_torch_device("cpu")
        reset_device_cache()
        d2 = get_torch_device("cpu")
        # After reset, the function is called fresh (objects may or may not be identical,
        # but the call should succeed without error)
        assert str(d1) == str(d2) == "cpu"
