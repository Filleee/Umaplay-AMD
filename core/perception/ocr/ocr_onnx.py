# core/perception/ocr/ocr_onnx.py
"""
ONNX-based OCR engine using RapidOCR + DirectML for AMD GPU acceleration.

This module wraps ``rapidocr_onnxruntime`` (which bundles PP-OCRv4 ONNX models)
and exposes the same ``OCRInterface`` used by the rest of Umaplay.

When PP-OCRv5 models are available in ``models/ocr_v5/``, they are loaded
automatically for improved accuracy on small/stylized game text.  If the v5
models are absent the first time the engine is constructed, they are
**auto-downloaded** from HuggingFace.

When ``onnxruntime-directml`` is installed, RapidOCR automatically prefers
the ``DmlExecutionProvider`` for GPU-accelerated inference.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, cast

import cv2
import numpy as np

from core.perception.ocr.interface import OCRInterface
from core.utils.img import to_bgr
from core.utils.logger import logger_uma

# ---------------------------------------------------------------------------
# PP-OCRv5 model constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/Umaplay
_V5_DIR = _PROJECT_ROOT / "models" / "ocr_v5"
_V5_DET = _V5_DIR / "det_v5.onnx"
_V5_REC = _V5_DIR / "rec_en_v5.onnx"
_V5_DICT = _V5_DIR / "en_dict_v5.txt"

_HF_REPO = "monkt/paddleocr-onnx"
_HF_FILES: Dict[str, str] = {
    # local filename → HuggingFace repo path
    "det_v5.onnx": "detection/v5/det.onnx",
    "rec_en_v5.onnx": "languages/english/rec.onnx",
    "en_dict_v5.txt": "languages/english/dict.txt",
}


def _is_available() -> bool:
    """Return True if RapidOCR + onnxruntime-directml are importable."""
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Auto-download helpers
# ---------------------------------------------------------------------------

def _download_v5_models() -> bool:
    """
    Download PP-OCRv5 ONNX models from HuggingFace into ``models/ocr_v5/``.

    Tries ``huggingface_hub`` first (fast, resumable); falls back to plain
    ``urllib`` if the library is not installed.

    Returns True if **all** files are present after the attempt.
    """
    _V5_DIR.mkdir(parents=True, exist_ok=True)

    missing = {
        name: hf_path
        for name, hf_path in _HF_FILES.items()
        if not (_V5_DIR / name).exists()
    }
    if not missing:
        return True

    logger_uma.info(
        "[OCR-ONNX] Downloading PP-OCRv5 models (%d files) from HuggingFace…",
        len(missing),
    )

    # --- Strategy 1: huggingface_hub (preferred) --------------------------
    try:
        from huggingface_hub import hf_hub_download

        for local_name, hf_path in missing.items():
            logger_uma.info("[OCR-ONNX]   ↓ %s", local_name)
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=hf_path,
                local_dir=str(_V5_DIR),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download puts the file at `local_dir / hf_path`
            # (preserving the repo subfolder).  Move it to the flat name.
            dl_path = Path(downloaded)
            dest = _V5_DIR / local_name
            if dl_path != dest:
                dl_path.rename(dest)

        # Clean up any leftover subdirectories created by hf_hub_download
        _cleanup_hf_subdirs()

        logger_uma.info("[OCR-ONNX] PP-OCRv5 models downloaded successfully.")
        return _v5_models_ready()

    except ImportError:
        logger_uma.debug(
            "[OCR-ONNX] huggingface_hub not installed, falling back to urllib."
        )
    except Exception as exc:
        logger_uma.warning("[OCR-ONNX] huggingface_hub download failed: %s", exc)

    # --- Strategy 2: raw urllib (fallback) --------------------------------
    try:
        import urllib.request

        for local_name, hf_path in missing.items():
            url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{hf_path}"
            dest = _V5_DIR / local_name
            logger_uma.info("[OCR-ONNX]   ↓ %s (urllib)", local_name)
            urllib.request.urlretrieve(url, str(dest))

        logger_uma.info("[OCR-ONNX] PP-OCRv5 models downloaded successfully.")
        return _v5_models_ready()

    except Exception as exc:
        logger_uma.warning("[OCR-ONNX] urllib download failed: %s", exc)
        return False


def _cleanup_hf_subdirs() -> None:
    """Remove empty subdirectories left by hf_hub_download's local_dir layout."""
    for sub in ("detection", "languages", ".huggingface"):
        d = _V5_DIR / sub
        if d.exists() and d.is_dir():
            import shutil
            shutil.rmtree(d, ignore_errors=True)


def _v5_models_ready() -> bool:
    """Return True if all three v5 model files are present."""
    return _V5_DET.exists() and _V5_REC.exists() and _V5_DICT.exists()


# ---------------------------------------------------------------------------
# OnnxOCREngine
# ---------------------------------------------------------------------------

class OnnxOCREngine(OCRInterface):
    """
    OCR engine backed by RapidOCR (ONNX Runtime + DirectML).

    Implements the same ``OCRInterface`` as ``LocalOCREngine`` so the
    rest of Umaplay can swap between them transparently.

    On first construction the engine will attempt to auto-download PP-OCRv5
    models from HuggingFace for better accuracy.  If the download fails it
    falls back to the bundled PP-OCRv4 models that ship with
    ``rapidocr_onnxruntime``.

    Parameters
    ----------
    use_dml : bool
        Whether to use DirectML GPU acceleration. Default ``True``.
    """

    def __init__(self, use_dml: bool = True):
        from rapidocr_onnxruntime import RapidOCR

        # --- Try to use PP-OCRv5 models ---
        use_v5 = _v5_models_ready()
        if not use_v5:
            use_v5 = _download_v5_models()

        kwargs: Dict[str, Any] = dict(
            det_use_cuda=False,
            rec_use_cuda=False,
            cls_use_cuda=False,
            det_use_dml=use_dml,
            rec_use_dml=use_dml,
            cls_use_dml=use_dml,
        )

        if use_v5:
            kwargs["det_model_path"] = str(_V5_DET)
            kwargs["rec_model_path"] = str(_V5_REC)
            kwargs["rec_keys_path"] = str(_V5_DICT)
            logger_uma.info(
                "[OCR-ONNX] Using PP-OCRv5 models from %s", _V5_DIR,
            )
        else:
            logger_uma.warning(
                "[OCR-ONNX] PP-OCRv5 models not available — "
                "falling back to bundled PP-OCRv4.",
            )

        self.engine = RapidOCR(**kwargs)
        self._model_version = "v5" if use_v5 else "v4"

        # Log which provider is active
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        active = (
            "DmlExecutionProvider"
            if "DmlExecutionProvider" in providers
            else "CPUExecutionProvider"
        )
        gpu_accel = "ON" if active == "DmlExecutionProvider" else "OFF"

        # Resolve model file details
        if use_v5:
            det_path = str(_V5_DET)
            rec_path = str(_V5_REC)
            dict_path = str(_V5_DICT)
            det_size = f"{_V5_DET.stat().st_size / 1024 / 1024:.1f} MB" if _V5_DET.exists() else "?"
            rec_size = f"{_V5_REC.stat().st_size / 1024 / 1024:.1f} MB" if _V5_REC.exists() else "?"
        else:
            det_path = "(bundled with rapidocr_onnxruntime)"
            rec_path = "(bundled with rapidocr_onnxruntime)"
            dict_path = "(bundled)"
            det_size = "~4 MB"
            rec_size = "~11 MB"

        ort_ver = getattr(onnxruntime, "__version__", "unknown")

        banner = (
            "\n"
            "┌──────────────────────────────────────────────────────────────┐\n"
            "│                    OCR Engine — ONNX                        │\n"
            "├──────────────────────┬───────────────────────────────────────┤\n"
            f"│ Engine               │ RapidOCR + ONNX Runtime              │\n"
            f"│ Model Version        │ PP-OCR{self._model_version:<34}│\n"
            f"│ ONNX Runtime         │ {ort_ver:<38}│\n"
            "├──────────────────────┼───────────────────────────────────────┤\n"
            f"│ GPU Acceleration     │ DirectML {gpu_accel:<30}│\n"
            f"│ Execution Provider   │ {active:<38}│\n"
            f"│ Available Providers  │ {', '.join(providers):<38}│\n"
            "├──────────────────────┼───────────────────────────────────────┤\n"
            f"│ Det Model            │ {os.path.basename(det_path):<38}│\n"
            f"│ Det Size             │ {det_size:<38}│\n"
            f"│ Rec Model            │ {os.path.basename(rec_path):<38}│\n"
            f"│ Rec Size             │ {rec_size:<38}│\n"
            f"│ Dictionary           │ {os.path.basename(dict_path):<38}│\n"
            "└──────────────────────┴───────────────────────────────────────┘"
        )
        logger_uma.info(banner)

    # ---- Helpers ----

    @staticmethod
    def _ensure_bgr3(img: Any) -> np.ndarray:
        """Return a 3-channel BGR image."""
        if isinstance(img, np.ndarray):
            bgr = img
        else:
            bgr = to_bgr(img)
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        return bgr

    # ---- Core inference ----

    def raw(self, img: Any) -> Dict[str, Any]:
        """
        Run OCR and return a dict matching the PaddleOCR JSON schema:

        ``{"res": {"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}}``
        """
        bgr = self._ensure_bgr3(img)
        result, _elapse = self.engine(bgr)

        rec_texts: List[str] = []
        rec_scores: List[float] = []
        dt_polys: List[Any] = []

        if result:
            for item in result:
                box, text, score = item
                dt_polys.append(box)
                rec_texts.append(text)
                rec_scores.append(float(score))

        return {
            "res": {
                "rec_texts": rec_texts,
                "rec_scores": rec_scores,
                "dt_polys": dt_polys,
            }
        }

    def text(self, img: Any, joiner: str = " ", min_conf: float = 0.2) -> str:
        j = self.raw(img)
        res = j.get("res", {})
        rec_texts = res.get("rec_texts", []) or []
        rec_scores = res.get("rec_scores", []) or []
        kept = []
        for i, t in enumerate(rec_texts):
            if i < len(rec_scores):
                if rec_scores[i] >= min_conf:
                    kept.append(t)
                elif t.strip():
                    logger_uma.debug(f"Low rec score for: {rec_scores[i]:.3f} | {t}")
        return (joiner.join(kept)).strip()

    def digits(self, img: Any) -> int:
        s = self.text(img)
        only = re.sub(r"[^\d]", "", s).strip()
        if not only:
            return -1
        try:
            return int(only)
        except Exception as e:
            logger_uma.warning(f"Couldn't parse digits: {only}. {e}")
            return -1

    # ---- Batch APIs ----

    def batch_text(
        self, imgs: List[Any], *, joiner: str = " ", min_conf: float = 0.2
    ) -> List[str]:
        if not imgs:
            return []
        return [self.text(im, joiner=joiner, min_conf=min_conf) for im in imgs]

    def batch_digits(self, imgs: List[Any]) -> List[str]:
        """Run OCR over a list of images, returning digits-only strings for each."""
        return [re.sub(r"[^\d]", "", self.text(im) or "") for im in imgs]
