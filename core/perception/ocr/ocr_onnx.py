# core/perception/ocr/ocr_onnx.py
"""
ONNX-based OCR engine using RapidOCR + DirectML for AMD GPU acceleration.

This module wraps ``rapidocr_onnxruntime`` (which bundles PP-OCRv4 ONNX models)
and exposes the same ``OCRInterface`` used by the rest of Umaplay.

When ``onnxruntime-directml`` is installed, RapidOCR automatically prefers
the ``DmlExecutionProvider`` for GPU-accelerated inference.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, cast

import cv2
import numpy as np

from core.perception.ocr.interface import OCRInterface
from core.utils.img import to_bgr
from core.utils.logger import logger_uma


def _is_available() -> bool:
    """Return True if RapidOCR + onnxruntime-directml are importable."""
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


class OnnxOCREngine(OCRInterface):
    """
    OCR engine backed by RapidOCR (ONNX Runtime + DirectML).

    Implements the same ``OCRInterface`` as ``LocalOCREngine`` so the
    rest of Umaplay can swap between them transparently.

    Parameters
    ----------
    use_dml : bool
        Whether to use DirectML GPU acceleration. Default ``True``.
    """

    def __init__(self, use_dml: bool = True):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR(
            det_use_cuda=False,
            rec_use_cuda=False,
            cls_use_cuda=False,
            det_use_dml=use_dml,
            rec_use_dml=use_dml,
            cls_use_dml=use_dml,
        )

        # Log which provider is active
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        active = "DmlExecutionProvider" if "DmlExecutionProvider" in providers else "CPUExecutionProvider"
        logger_uma.info(
            "[OCR-ONNX] Initialized | provider=%s | use_dml=%s",
            active,
            use_dml,
        )

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
