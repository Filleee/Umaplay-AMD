# core/perception/yolo/yolo_local.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from core.perception.yolo.interface import IDetector
from core.controllers.base import IController, RegionXYWH
from core.controllers.steam import SteamController
from core.settings import Settings
from core.types import DetectionDict
from core.utils.img import pil_to_bgr
from core.utils.logger import logger_uma


# ---------------------------------------------------------------------------
# Monkey-patch Ultralytics to support DirectML for ONNX inference.
#
# Ultralytics' autobackend.py (the ONNX branch):
#   1. Hardcodes providers = ["CPUExecutionProvider"] and only adds CUDA.
#      It never checks for DmlExecutionProvider (AMD/Intel GPU via DirectML).
#   2. Runs check_requirements("onnxruntime") which pip-installs the plain
#      onnxruntime package, clobbering our onnxruntime-directml.
#   3. For static ONNX models, uses io_binding which is incompatible with DML.
#
# The patch wraps AutoBackend.__init__ to:
#   - Re-create the ONNX session with DmlExecutionProvider
#   - Force dynamic=True so the simpler session.run() path is used
# ---------------------------------------------------------------------------

def _patch_ultralytics_for_directml():
    """Patch Ultralytics to use DirectML when onnxruntime-directml is installed."""
    # Disable Ultralytics auto-install to prevent it from overwriting
    # onnxruntime-directml with plain onnxruntime.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    try:
        import onnxruntime
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            return  # DirectML not available, nothing to patch
    except ImportError:
        return

    from ultralytics.nn.autobackend import AutoBackend
    _original_init = AutoBackend.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        # After Ultralytics creates the ONNX session with CPUExecutionProvider,
        # re-create it with DirectML and force dynamic mode.
        if not getattr(self, "onnx", False) or not hasattr(self, "session"):
            return

        import onnxruntime as ort
        if "DmlExecutionProvider" not in ort.get_available_providers():
            return

        try:
            w = getattr(self, "w", None)
            if not w:
                return

            dml_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(w), providers=dml_providers)

            # Update output names from the new session
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Force dynamic mode so forward() uses session.run() path
            # instead of io_binding (which is incompatible with DirectML).
            self.dynamic = True

            active = self.session.get_providers()
            logger_uma.info(f"[DirectML] ONNX session using {active}")
        except Exception as e:
            logger_uma.warning(f"[DirectML] Failed to create DML session, keeping CPU: {e}")

    AutoBackend.__init__ = _patched_init


_patch_ultralytics_for_directml()

from ultralytics.models import YOLO  # import AFTER patch is applied


def _prefer_onnx(weights_path: str) -> str:
    """If an .onnx sibling exists for a .pt file, return it; else return original."""
    if weights_path.endswith(".pt"):
        onnx_path = weights_path[:-3] + ".onnx"
        if os.path.isfile(onnx_path):
            return onnx_path
    return weights_path


class LocalYOLOEngine(IDetector):
    """
    Ultralytics-backed detector.

    When ONNX weights exist alongside a .pt file, loads the ONNX model so
    that onnxruntime-directml can route inference to an AMD/Intel GPU.
    Falls back gracefully to .pt (PyTorch CPU or CUDA) otherwise.
    """

    def __init__(
        self,
        ctrl: Optional[IController] = None,
        *,
        weights: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ):
        self.ctrl = ctrl
        raw_weights = str(weights or Settings.YOLO_WEIGHTS_URA)
        self.use_gpu = Settings.USE_GPU if use_gpu is None else bool(use_gpu)

        # Prefer ONNX for DirectML GPU acceleration (AMD GPUs).
        if self.use_gpu:
            self.weights_path = _prefer_onnx(raw_weights)
        else:
            self.weights_path = raw_weights

        if self.weights_path.endswith(".onnx"):
            logger_uma.info(f"Loading YOLO ONNX weights (DirectML GPU): {self.weights_path}")
        else:
            logger_uma.info(f"Loading YOLO PyTorch weights: {self.weights_path}")

        self.model = YOLO(self.weights_path)

        # For .pt models only: try CUDA (DirectML crashes Ultralytics' PyTorch path)
        if self.use_gpu and not self.weights_path.endswith(".onnx"):
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to("cuda:0")
            except Exception as e:
                logger_uma.error(f"Couldn't set YOLO to CUDA ({e}). Running on CPU.")

    # ---------- internals ----------
    @staticmethod
    def _extract_dets(res, conf_min: float = 0.25) -> List[DetectionDict]:
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = (
            res.names
            if isinstance(res.names, dict)
            else {i: n for i, n in enumerate(res.names)}
        )
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        out: List[DetectionDict] = []
        for i in range(len(cls)):
            if conf[i] < conf_min:
                continue
            out.append(
                {
                    "idx": i,
                    "name": names.get(int(cls[i]), str(cls[i])),
                    "conf": float(conf[i]),
                    "xyxy": tuple(map(float, xyxy[i])),
                }
            )
        return out

    @staticmethod
    def _maybe_store_debug(
        pil_img: Image.Image,
        dets: List[DetectionDict],
        *,
        tag: str,
        thr: float,
        agent: Optional[str] = None,
    ) -> None:
        import os, time

        if not Settings.STORE_FOR_TRAINING or not dets:
            return
        lows = [d for d in dets if float(d.get("conf", 0.0)) <= float(thr)]
        if not lows:
            return
        try:
            agent_segment = (agent or "").strip()
            base_dir = Settings.DEBUG_DIR / agent_segment if agent_segment else Settings.DEBUG_DIR
            out_dir_raw = base_dir / tag / "raw"
            os.makedirs(out_dir_raw, exist_ok=True)

            ts = (
                time.strftime("%Y%m%d-%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
            )

            lowest = min(lows, key=lambda d: float(d.get("conf", 0.0)))
            conf_line = f"{float(lowest.get('conf', 0.0)):.2f}"
            raw_name = str(lowest.get("name", "unknown")).strip()
            class_segment = "".join(
                ch if ch.isalnum() or ch in "-_" else "-" for ch in raw_name
            ) or "unknown"

            raw_path = out_dir_raw / f"{tag}_{ts}_{class_segment}_{conf_line}.png"
            pil_img.save(raw_path)
            logger_uma.debug("saved low-conf training debug -> %s", raw_path)
        except Exception as e:
            logger_uma.debug("failed saving training debug: %s", e)

    # ---------- public API ----------
    def detect_bgr(
        self,
        bgr: np.ndarray,
        *,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        original_pil_img=None,
        tag="general",
        agent: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[DetectionDict]]:
        imgsz = imgsz if imgsz is not None else Settings.YOLO_IMGSZ
        conf = conf if conf is not None else Settings.YOLO_CONF
        iou = iou if iou is not None else Settings.YOLO_IOU

        res_list = self.model.predict(
            source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False
        )
        result = res_list[0]
        dets = self._extract_dets(result, conf_min=conf)

        if original_pil_img is not None:
            self._maybe_store_debug(
                original_pil_img,
                dets,
                tag=tag,
                thr=Settings.STORE_FOR_TRAINING_THRESHOLD,
                agent=agent,
            )

        meta = {"names": result.names, "imgsz": imgsz, "conf": conf, "iou": iou}
        return meta, dets

    def detect_pil(
        self,
        pil_img: Image.Image,
        *,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        tag="general",
        agent: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[DetectionDict]]:
        bgr = pil_to_bgr(pil_img)

        meta, dets = self.detect_bgr(
            bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            original_pil_img=pil_img,
            tag=tag,
            agent=agent,
        )
        return meta, dets

    def recognize(
        self,
        *,
        region: Optional[RegionXYWH] = None,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        tag: str = "general",
        agent: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any], List[DetectionDict]]:
        if self.ctrl is None:
            raise RuntimeError(
                "LocalYOLOEngine.recognize() requires a controller injected in the constructor."
            )

        if isinstance(self.ctrl, SteamController):
            img = self.ctrl.screenshot_left_half()
        else:
            img = self.ctrl.screenshot(region=region)

        meta, dets = self.detect_pil(img, imgsz=imgsz, conf=conf, iou=iou, agent=agent)
        return img, meta, dets
