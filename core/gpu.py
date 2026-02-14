# core/gpu.py
"""
GPU device abstraction for Umaplay.

Centralises PyTorch device selection so that NVIDIA (CUDA), AMD (DirectML),
and CPU backends can all coexist safely.  Every module that needs a
``torch.device`` should call ``get_torch_device()`` instead of hard-coding
``"cuda"`` strings.

Supported backends (selected via ``Settings.GPU_BACKEND``):
  auto  – try CUDA → DirectML → CPU  (default)
  cuda  – NVIDIA CUDA only, fallback to CPU
  dml   – DirectML only (AMD / Intel / any DX12 GPU), fallback to CPU
  cpu   – force CPU
"""
from __future__ import annotations

import functools
from typing import Optional

import torch

from core.utils.logger import logger_uma

# Allowed backend names (whitelist).  Anything else is normalised to "auto".
_VALID_BACKENDS = frozenset({"auto", "cuda", "dml", "cpu"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_directml() -> bool:
    """Return True if ``torch_directml`` is importable and functional."""
    try:
        import torch_directml  # noqa: F811
        # Sanity-check: the package must expose a device constructor.
        _ = torch_directml.device()
        return True
    except Exception:
        return False


def _normalize_backend(raw: Optional[str]) -> str:
    """Coerce an arbitrary string to a valid backend name."""
    if raw is None:
        return "auto"
    cleaned = str(raw).strip().lower()
    if cleaned in _VALID_BACKENDS:
        return cleaned
    logger_uma.warning(
        "[GPU] Unrecognised GPU_BACKEND '%s' — falling back to 'auto'.", raw,
    )
    return "auto"


def _resolve_device(backend: str) -> torch.device:
    """Pick the best device for *backend* (already normalised)."""
    if backend == "cpu":
        return torch.device("cpu")

    if backend == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger_uma.info("[GPU] CUDA requested but unavailable — using CPU.")
        return torch.device("cpu")

    if backend == "dml":
        if _has_directml():
            import torch_directml
            return torch_directml.device()
        logger_uma.info("[GPU] DirectML requested but unavailable — using CPU.")
        return torch.device("cpu")

    # backend == "auto"
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _has_directml():
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_torch_device(backend: Optional[str] = None) -> torch.device:
    """Return the best available ``torch.device``.

    Parameters
    ----------
    backend:
        One of ``"auto"``, ``"cuda"``, ``"dml"``, ``"cpu"``.
        When *None* (the default) the value is read from
        ``Settings.GPU_BACKEND``.

    The result is cached after the first call so that repeated imports
    across modules always receive the same device object.
    """
    if backend is None:
        # Lazy import to avoid circular dependency (Settings → gpu → Settings).
        from core.settings import Settings
        backend = getattr(Settings, "GPU_BACKEND", "auto")

    backend = _normalize_backend(backend)
    device = _resolve_device(backend)

    logger_uma.info(
        "[GPU] Backend=%s → device=%s  (CUDA=%s, DirectML=%s)",
        backend,
        device,
        torch.cuda.is_available(),
        _has_directml(),
    )
    return device


def reset_device_cache() -> None:
    """Clear the cached device (useful after changing Settings.GPU_BACKEND at runtime)."""
    get_torch_device.cache_clear()
