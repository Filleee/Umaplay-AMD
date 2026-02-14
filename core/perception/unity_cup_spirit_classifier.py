# core/perception/unity_cup_spirit_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from core.settings import Settings


ArrayLike = Union[np.ndarray, "Image.Image"]


# -------------------------
# Model definition (TinyCNN)
# -------------------------
class TinyCNN(nn.Module):
    """
    Compact RGB CNN:
      64x64 -> 32x32 -> 16x16 -> GAP -> Linear(num_classes)
    Matches the training architecture "TinyCNN_SiLU_v1".
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # Global Avg Pool -> (N,64)
        return self.head(x)                          # logits


# -------------------------
# Image utils / preprocessing
# -------------------------
def _ensure_pil(img: ArrayLike) -> Image.Image:
    """Accept PIL.Image or ndarray (HxW or HxWx{3,4}); return PIL RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                # assume [0,1] float or >255 -> clamp/scale to uint8
                arr = np.clip(arr, 0, 1) * 255.0 if arr.max() <= 1.0 else np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)
            if arr.shape[2] == 3:
                return Image.fromarray(arr, mode="RGB")
            if arr.shape[2] == 4:
                return Image.fromarray(arr, mode="RGBA").convert("RGB")
    raise TypeError("img must be a PIL.Image.Image or numpy.ndarray with shape HxW or HxWx{3,4}.")


def _build_val_tfms(size_wh: Tuple[int, int]):
    """Resize + ToTensor (must match training VAL transforms)."""
    w, h = size_wh
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),  # -> float32 [0,1], CHW
    ])


# -------------------------
# Classifier wrapper
# -------------------------
@dataclass
class UnityCupSpiritClassifier:
    """
    Runtime wrapper for Unity Spirit color classifier (CNN).
    Loads a Torch .pt bundle with:
        state_dict, classes (list[str]), img_size (W,H), arch
    """
    device: Any
    model: Any
    classes: List[str]
    img_size: Tuple[int, int]
    tfms: Any

    # ---------- Loading ----------
    @classmethod
    def load_from_settings(cls) -> "UnityCupSpiritClassifier":
        """
        Load CNN bundle from Settings.UNITY_CUP_SPIRIT_COLOR_CLASS_PATH.
        The .pt must contain: state_dict, classes, img_size, arch.
        """
        bundle_path = Settings.UNITY_CUP_SPIRIT_COLOR_CLASS_PATH
        # Use the best available GPU (CUDA / DirectML / CPU)
        from core.gpu import get_torch_device
        device = get_torch_device()

        ckpt = torch.load(bundle_path, map_location=device)
        classes = list(ckpt["classes"])
        img_size = tuple(ckpt.get("img_size", (64, 64)))
        arch = ckpt.get("arch", "TinyCNN_SiLU_v1")

        if arch != "TinyCNN_SiLU_v1":
            # Still try to load; warn in logs if you prefer
            pass

        model = TinyCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        tfms = _build_val_tfms(img_size)
        return cls(device=device, model=model, classes=classes, img_size=img_size, tfms=tfms)

    # ---------- Inference ----------
    def _tensor_from_pil(self, pil_img: Image.Image):
        # Note: self.tfms already resizes to (H,W) and converts to float32 tensor [0,1], CHW.
        x = self.tfms(pil_img).unsqueeze(0)  # -> (1,C,H,W)
        return x.to(self.device, non_blocking=True)

    def predict(self, img: ArrayLike) -> Dict[str, Any]:
        """
        Predict class for a single icon image.

        Returns:
            {
              "pred_id": int,
              "pred_label": str,
              "confidence": float,   # softmax probability of the predicted class
              "raw": List[float]     # full probability vector
            }
        """
        pil = _ensure_pil(img)
        x = self._tensor_from_pil(pil)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return {
            "pred_id": idx,
            "pred_label": self.classes[idx],
            "confidence": conf,
            "raw": probs.tolist(),
        }

    def predict_label(self, img: ArrayLike, threshold: float = 0.0) -> str:
        """
        Return just the label. If threshold > 0 and confidence < threshold, returns 'unknown'.
        """
        out = self.predict(img)
        if threshold > 0.0 and out["confidence"] < threshold:
            return "unknown"
        return out["pred_label"]

    def predict_from_path(self, path: str, threshold: float = 0.0) -> str:
        pil = Image.open(path).convert("RGB")
        return self.predict_label(pil, threshold=threshold)

    def classes_list(self) -> List[str]:
        return list(self.classes)
