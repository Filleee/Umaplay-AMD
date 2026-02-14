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

from core.perception.ocr.ocr_onnx import OnnxOCREngine


@pytest.fixture(scope="module")
def engine():
    return OnnxOCREngine(use_dml=True)


@pytest.fixture()
def digit_img():
    """White image with '847' rendered in black."""
    img = np.ones((80, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, "847", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


def test_digits(engine, digit_img):
    result = engine.digits(digit_img)
    assert result == 847, f"Expected 847, got {result}"


def test_text(engine, digit_img):
    result = engine.text(digit_img)
    assert "847" in result


def test_raw_schema(engine, digit_img):
    result = engine.raw(digit_img)
    assert "res" in result
    assert "rec_texts" in result["res"]
    assert "rec_scores" in result["res"]
    assert len(result["res"]["rec_texts"]) > 0


def test_batch_text(engine, digit_img):
    results = engine.batch_text([digit_img, digit_img])
    assert len(results) == 2
    assert all("847" in r for r in results)


def test_batch_digits(engine, digit_img):
    results = engine.batch_digits([digit_img, digit_img])
    assert len(results) == 2
    assert all("847" in r for r in results)
