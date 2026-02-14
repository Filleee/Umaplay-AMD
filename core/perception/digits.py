from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

import torch
import torch.nn.functional as F
# from pytorchcv.model_provider import get_model as ptcv_get_model


def _patch_pytorchcv_utf8() -> None:
    """
    pytorchcv's model_store.load_csv() opens metadata without specifying encoding.
    On Windows this defaults to cp1252 and crashes on UTF-8 bytes.
    We replace it with a UTF-8 version before any get_model() call.
    """
    try:
        import csv
        # from pytorchcv.models.common import model_store as _ms

        def _load_csv_utf8(path: str):
            with open(path, "r", encoding="utf-8", newline="") as f:
                return list(csv.reader(f))

        _ms.load_csv = _load_csv_utf8  # type: ignore[attr-defined]
    except Exception as e:
        # Non-fatal: just log; fallback to user's environment solution
        try:
            from core.utils.logger import logger_uma

            logger_uma.debug("pytorchcv UTF-8 patch not applied: %s", e)
        except Exception:
            pass


# ----------------------------
# Utilities
# ----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_32x32_rgb(pil_img: Image.Image) -> torch.Tensor:
    """
    Resize to 32x32, float32 [0,1], normalize like SVHN-ish (mean=0.5, std=0.5).
    """
    x = pil_img.resize((32, 32), RESAMPLE_BILINEAR)
    x = np.asarray(x).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3x32x32
    return x


def softmax_top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = F.softmax(logits, dim=1)
    p, idx = torch.max(probs, dim=1)
    return int(idx.item()), float(p.item())


def clamp_valid_or_neg1(val: int, allow=(0, 30)) -> int:
    lo, hi = allow
    return val if lo <= val <= hi else -1


# %% [code]
# ----------------------------
# Simple contour splitter (up to 2 digits)
# ----------------------------
def split_digit_boxes(pil_img: Image.Image, max_digits: int = 2) -> List[np.ndarray]:
    """
    Heuristic splitter for UI digits:
      - Adaptive threshold on grayscale
      - External contours
      - Keep big-enough tall blobs
      - Return up to `max_digits` crops (left->right) as BGR np arrays
    Fallback: if no decent contours, return the full image as one crop.
    """
    bgr = pil_to_cv(pil_img)
    H, W = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # size heuristics: tall & not too skinny
        if h >= 0.35 * H and w >= 0.08 * W and h <= H:
            boxes.append((x, y, w, h))

    if not boxes:
        return [bgr]

    # keep two largest by area, then sort left->right
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_digits]
    boxes = sorted(boxes, key=lambda b: b[0])

    crops = []
    for x, y, w, h in boxes:
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        crops.append(bgr[y1:y2, x1:x2])
    return crops


# %% [code]
# ----------------------------
# Torch single-digit classifiers
# ----------------------------
@dataclass
class TorchClassifier:
    name: str
    net: torch.nn.Module
    device: str = ""  # resolved lazily via get_torch_device()

    def __post_init__(self):
        if not self.device:
            from core.gpu import get_torch_device
            self.device = str(get_torch_device())

    @torch.no_grad()
    def predict_number(self, pil_img: Image.Image) -> Tuple[int, float]:
        """
        Predict 0..30:
          - Split image into up to 2 digit crops
          - Classify each crop (0..9) with SVHN model
          - Combine two digits if present; reject if low confidence or >30
        Returns: (value or -1, confidence)
        """
        ALLOW_RANGE = (0, 30)  # valid output range
        CLASSIFIER_THRESHOLD = 0.70  # softmax threshold per digit

        crops = split_digit_boxes(pil_img, max_digits=2)

        digits, confs = [], []
        for crop_bgr in crops:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            x = preprocess_32x32_rgb(crop_pil).to(self.device)
            logits = self.net(x)
            cls, p = softmax_top1(logits)
            if p < CLASSIFIER_THRESHOLD:
                # TODO: debug. print("cls", cls)
                return -1, p
            digits.append(cls)
            confs.append(p)

        if len(digits) == 1:
            val = clamp_valid_or_neg1(digits[0], ALLOW_RANGE)
            return val, confs[0]

        val = digits[0] * 10 + digits[1]
        conf = min(confs)  # conservative
        if val < ALLOW_RANGE[0]:
            return -1, conf

        if val > ALLOW_RANGE[1]:
            val_str = str(val)
            if len(val_str) == 2:
                return int(val_str[-1]), conf
            elif len(val_str) == 3 and set(val_str) == 1:
                return int(val_str[-1]), conf
            else:
                return -1, conf
        return val, conf


# def load_svhn_resnet20() -> TorchClassifier:
#     _patch_pytorchcv_utf8()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     net = ptcv_get_model("resnet20_svhn", pretrained=True)
#     net.eval().to(device)
#     return TorchClassifier("resnet20_svhn", net, device)


# resnet20 = load_svhn_resnet20()
