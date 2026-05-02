"""
DefectClassifier — classifies cropped ROIs into defect categories.

Backed by a lightweight MobileNetV3 torchvision model by default.
Falls back to a colour/texture heuristic when PyTorch is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard defect taxonomy (MVTec-inspired)
DEFECT_CLASSES = [
    "scratch",
    "dent",
    "crack",
    "contamination",
    "missing_component",
    "discoloration",
    "edge_chip",
    "void",
]


class DefectClassifier:
    """
    Classifies a cropped defect ROI into one of :data:`DEFECT_CLASSES`.

    If PyTorch + torchvision are available and ``model_path`` is provided,
    runs a fine-tuned MobileNetV3-Small.  Otherwise falls back to a fast
    texture/colour heuristic that still produces reasonable labels.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` weights file (torchvision MobileNetV3-Small head).
    device:
        ``"cpu"`` | ``"cuda:0"`` | ``"mps"``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self._model = None

        if model_path:
            self._model = self._load_torch(model_path, device)
            logger.info("Loaded classifier from %s on %s", model_path, device)
        else:
            logger.info("Using heuristic classifier (no model_path provided)")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def classify(self, roi: np.ndarray) -> tuple[str, float]:
        """
        Classify an ROI.

        Returns
        -------
        label : str
            One of :data:`DEFECT_CLASSES`.
        confidence : float
            Softmax confidence in [0, 1].
        """
        if roi.size == 0:
            return "unknown", 0.0
        if self._model is not None:
            return self._torch_classify(roi)
        return self._heuristic_classify(roi)

    # ------------------------------------------------------------------
    # Heuristic classifier (no neural net required)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_classify(roi: np.ndarray) -> tuple[str, float]:
        """
        Fast, rule-based classification using OpenCV features.

        Rules (in priority order):
        1. High saturation variance  → contamination
        2. Many parallel edges (Hough) → scratch
        3. High elongation (aspect)   → crack
        4. Low mean intensity         → void
        5. High Laplacian variance    → edge_chip
        6. Default                    → dent
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Feature 1: saturation variance → contamination
        sat_var = float(np.var(hsv[:, :, 1]))
        if sat_var > 900:
            conf = float(np.clip(sat_var / 2500, 0.55, 0.95))
            return "contamination", conf

        # Feature 2: Hough lines density → scratch
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=10, maxLineGap=5)
        line_count = 0 if lines is None else len(lines)
        if line_count > 4:
            conf = float(np.clip(line_count / 20, 0.55, 0.95))
            return "scratch", conf

        # Feature 3: elongation → crack
        h, w = gray.shape
        aspect = max(h, w) / (min(h, w) + 1e-6)
        if aspect > 3.5:
            conf = float(np.clip(aspect / 8, 0.55, 0.92))
            return "crack", conf

        # Feature 4: dark region → void
        mean_intensity = float(gray.mean())
        if mean_intensity < 60:
            conf = float(np.clip(1 - mean_intensity / 80, 0.55, 0.90))
            return "void", conf

        # Feature 5: high edge sharpness → chip
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var > 500:
            conf = float(np.clip(lap_var / 2000, 0.55, 0.90))
            return "edge_chip", conf

        # Default
        return "dent", 0.62

    # ------------------------------------------------------------------
    # Torch classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _load_torch(model_path: str, device: str):
        try:
            import torch
            import torchvision.models as models
            model = models.mobilenet_v3_small(weights=None)
            # Replace classifier head
            in_features = model.classifier[-1].in_features
            import torch.nn as nn
            model.classifier[-1] = nn.Linear(in_features, len(DEFECT_CLASSES))
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.to(device).eval()
            return model
        except Exception as exc:
            logger.warning("Could not load Torch model (%s); falling back to heuristic.", exc)
            return None

    def _torch_classify(self, roi: np.ndarray) -> tuple[str, float]:
        import torch
        import torch.nn.functional as F

        resized = cv2.resize(roi, (224, 224))
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return DEFECT_CLASSES[idx], float(probs[idx])
