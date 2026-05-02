"""
AnomalyDetector — wraps a YOLO-style detector for anomaly localisation.

In production: swap `_mock_inference` for a real ONNX / TorchScript model.
The interface is identical — the rest of the pipeline doesn't care.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Wraps an object-detection backbone to localise surface anomalies.

    The class is backend-agnostic: initialise with ``model_path`` pointing
    to an ONNX file and it will use ``cv2.dnn`` for inference; leave it as
    ``None`` and a fast classical fallback (blob analysis) is used instead.

    Parameters
    ----------
    threshold:
        Minimum detection confidence to keep.
    nms_iou:
        IoU threshold for non-maximum suppression.
    model_path:
        Optional path to an ONNX model file.
    device:
        ``"cpu"``, ``"cuda:0"`` or ``"mps"``.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        nms_iou: float = 0.40,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.device = device
        self._net = None

        if model_path:
            self._net = self._load_onnx(model_path, device)
            logger.info("Loaded ONNX detector from %s", model_path)
        else:
            logger.info("Using classical blob-analysis detector (no model_path)")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Returns a list of raw detections::

            [{"bbox": (x1,y1,x2,y2), "score": float, "area": int}, ...]
        """
        if self._net is not None:
            return self._dnn_inference(image)
        return self._classical_detect(image)

    # ------------------------------------------------------------------
    # Backend: classical OpenCV (no model required)
    # ------------------------------------------------------------------

    def _classical_detect(self, image: np.ndarray) -> list[dict]:
        """
        Robust classical pipeline:
          1. Convert to grayscale
          2. OTSU threshold + morphological cleanup
          3. Find contours → bounding boxes

        Designed to find surface defects (scratches, dents, blobs) without
        a trained neural network — useful for zero-shot deployment.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur reduces sensor noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold catches illumination gradients
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31, C=8,
        )

        # Morphological opening: remove tiny specks; closing: fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours (external only)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Pseudo-score: larger & more circular → more confident
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            score = float(np.clip(area / 2000 + circularity * 0.3, 0, 1))
            if score < self.threshold * 0.5:   # relaxed threshold for classical
                continue
            detections.append({
                "bbox": (x, y, x + w, y + h),
                "score": score,
                "area": int(area),
            })

        return self._nms(detections)

    # ------------------------------------------------------------------
    # Backend: ONNX / cv2.dnn
    # ------------------------------------------------------------------

    @staticmethod
    def _load_onnx(model_path: str, device: str):
        net = cv2.dnn.readNetFromONNX(model_path)
        if device.startswith("cuda"):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _dnn_inference(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255.0, size=(640, 640),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()

        # Parse YOLO-style output tensor: [batch, num_boxes, 85]
        detections = []
        for row in outputs[0]:
            objectness = float(row[4])
            if objectness < self.threshold:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            detections.append({
                "bbox": (
                    max(0, x1), max(0, y1),
                    min(w, x2), min(h, y2),
                ),
                "score": objectness,
                "area": area,
            })

        return self._nms(detections)

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def _nms(self, detections: list[dict]) -> list[dict]:
        if not detections:
            return []
        boxes  = [list(d["bbox"]) for d in detections]
        scores = [d["score"]      for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms_iou)
        if len(indices) == 0:
            return []
        return [detections[i] for i in indices.flatten()]
