"""Smoke test for the ONNX OCR engine (AMD GPU / DirectML only)."""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip the entire module if DirectML is not available
try:
    import onnxruntime

    _has_dml = "DmlExecutionProvider" in onnxruntime.get_available_providers()
except ImportError:
    _has_dml = False

try:
    from rapidocr_onnxruntime import RapidOCR  # noqa: F401

    _has_rapid = True
except ImportError:
    _has_rapid = False

pytestmark = pytest.mark.skipif(
    not (_has_dml and _has_rapid),
    reason="Requires onnxruntime-directml + rapidocr_onnxruntime (AMD GPU only)",
)


import cv2
import numpy as np

from core.perception.ocr.ocr_onnx import OnnxOCREngine, _v5_models_ready


@pytest.fixture(scope="module")
def engine():
    return OnnxOCREngine(use_dml=True)


@pytest.fixture()
def text_img():
    """Realistic-size image with mixed text and digits.

    The PP-OCRv5 server det model is optimized for natural document/game
    images rather than tiny synthetic patches, so we use a larger canvas
    with contextual text for reliable detection.
    """
    img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    cv2.putText(
        img, "Score 847", (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4,
    )
    return img


def test_text(engine, text_img):
    result = engine.text(text_img)
    assert "847" in result


def test_digits(engine, text_img):
    result = engine.digits(text_img)
    assert result == 847, f"Expected 847, got {result}"


def test_raw_schema(engine, text_img):
    result = engine.raw(text_img)
    assert "res" in result
    assert "rec_texts" in result["res"]
    assert "rec_scores" in result["res"]
    assert len(result["res"]["rec_texts"]) > 0


def test_batch_text(engine, text_img):
    results = engine.batch_text([text_img, text_img])
    assert len(results) == 2
    assert all("847" in r for r in results)


def test_batch_digits(engine, text_img):
    results = engine.batch_digits([text_img, text_img])
    assert len(results) == 2
    assert all("847" in r for r in results)


# ---- PP-OCRv5 model tests ----

def test_v5_models_downloaded():
    """Verify that v5 models were auto-downloaded."""
    assert _v5_models_ready(), (
        "PP-OCRv5 models should be present after OnnxOCREngine init"
    )


def test_engine_uses_v5(engine):
    """Verify the engine loaded v5 models."""
    assert engine._model_version == "v5", (
        f"Expected v5, got {engine._model_version}"
    )


def test_text_recognition_accuracy(engine):
    """Verify v5 engine reads mixed text+digits accurately."""
    img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    cv2.putText(
        img, "Hello World 123", (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4,
    )
    result = engine.text(img)
    assert "Hello" in result or "hello" in result.lower()
    # v5 server det may split digits across text regions ("12 3"),
    # so we check via digits() which strips non-digit chars.
    import re
    digits_only = re.sub(r"[^\d]", "", result)
    assert "123" in digits_only
