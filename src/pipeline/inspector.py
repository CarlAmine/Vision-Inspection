"""
InspectionPipeline — orchestrates the full inspection workflow.

Flow:
    raw image → preprocess → detect → classify → postprocess → result
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .preprocessor import ImagePreprocessor
from .result import InspectionResult, Defect, DefectSeverity
from ..models.detector import AnomalyDetector
from ..models.classifier import DefectClassifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tuneable knobs — all overridable at init time."""

    # Pre-processing
    target_size: tuple[int, int] = (640, 640)
    normalize_illumination: bool = True
    denoise_strength: int = 7          # 0 = off

    # Detection
    detection_threshold: float = 0.45
    nms_iou_threshold: float = 0.4
    min_defect_area_px: int = 64

    # Misc
    device: str = "cpu"               # "cpu" | "cuda:0" | "mps"
    num_workers: int = 4
    save_annotated: bool = False
    annotated_output_dir: str = "outputs/annotated"


class InspectionPipeline:
    """
    High-throughput, configurable machine vision inspection pipeline.

    Combines classical OpenCV pre-processing with deep learning detection
    and classification, exposing a single `.inspect()` call.

    Parameters
    ----------
    config:
        PipelineConfig instance (defaults work out of the box).
    detector:
        Optional pre-loaded AnomalyDetector. Built from scratch if None.
    classifier:
        Optional pre-loaded DefectClassifier. Built from scratch if None.

    Example
    -------
    >>> pipeline = InspectionPipeline()
    >>> result = pipeline.inspect("part_001.jpg")
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detector: Optional[AnomalyDetector] = None,
        classifier: Optional[DefectClassifier] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.target_size,
            normalize_illumination=self.config.normalize_illumination,
            denoise_strength=self.config.denoise_strength,
        )
        self.detector = detector or AnomalyDetector(
            threshold=self.config.detection_threshold,
            nms_iou=self.config.nms_iou_threshold,
            device=self.config.device,
        )
        self.classifier = classifier or DefectClassifier(
            device=self.config.device
        )
        logger.info("InspectionPipeline ready (device=%s)", self.config.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(
        self,
        source: str | Path | np.ndarray,
        label: Optional[str] = None,
    ) -> InspectionResult:
        """
        Run the full inspection pipeline on *source*.

        Parameters
        ----------
        source:
            File path (str/Path) or pre-loaded BGR image (np.ndarray).
        label:
            Human-readable identifier attached to the result.

        Returns
        -------
        InspectionResult
        """
        t0 = time.perf_counter()

        # --- 1. Load -------------------------------------------------------
        image = self._load(source)
        label = label or (str(source) if not isinstance(source, np.ndarray) else "array")

        # --- 2. Pre-process ------------------------------------------------
        processed, meta = self.preprocessor.run(image)

        # --- 3. Detect anomalies -------------------------------------------
        raw_detections = self.detector.detect(processed)

        # --- 4. Filter small blobs -----------------------------------------
        detections = [
            d for d in raw_detections
            if d["area"] >= self.config.min_defect_area_px
        ]

        # --- 5. Classify each detection ------------------------------------
        defects: list[Defect] = []
        for det in detections:
            roi = self._crop_roi(processed, det["bbox"])
            label_pred, confidence = self.classifier.classify(roi)
            severity = self._score_to_severity(det["score"], confidence)
            defects.append(
                Defect(
                    bbox=det["bbox"],
                    label=label_pred,
                    confidence=float(confidence),
                    severity=severity,
                    area=int(det["area"]),
                )
            )

        # --- 6. Assemble result --------------------------------------------
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        result = InspectionResult(
            source_label=label,
            image_shape=image.shape[:2],
            defects=defects,
            preprocessing_meta=meta,
            inference_ms=round(elapsed_ms, 2),
        )

        # --- 7. Optional annotation ----------------------------------------
        if self.config.save_annotated:
            self._save_annotated(image, result)

        logger.debug("Inspected %s → %d defect(s) in %.1f ms", label, len(defects), elapsed_ms)
        return result

    def inspect_batch(
        self,
        sources: Sequence[str | Path | np.ndarray],
    ) -> list[InspectionResult]:
        """Inspect a list of images, returning results in the same order."""
        return [self.inspect(s) for s in sources]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source.copy()
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {source}")
        return img

    @staticmethod
    def _crop_roi(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    @staticmethod
    def _score_to_severity(anomaly_score: float, classifier_conf: float) -> DefectSeverity:
        combined = anomaly_score * 0.6 + classifier_conf * 0.4
        if combined > 0.80:
            return DefectSeverity.CRITICAL
        if combined > 0.55:
            return DefectSeverity.MAJOR
        if combined > 0.30:
            return DefectSeverity.MINOR
        return DefectSeverity.COSMETIC

    def _save_annotated(self, image: np.ndarray, result: InspectionResult) -> None:
        out_dir = Path(self.config.annotated_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = self._draw_detections(image.copy(), result.defects)
        fname = out_dir / f"{Path(result.source_label).stem}_annotated.jpg"
        cv2.imwrite(str(fname), annotated)

    @staticmethod
    def _draw_detections(image: np.ndarray, defects: list[Defect]) -> np.ndarray:
        _COLOR = {
            DefectSeverity.CRITICAL: (0, 0, 220),
            DefectSeverity.MAJOR:    (0, 100, 255),
            DefectSeverity.MINOR:    (0, 200, 255),
            DefectSeverity.COSMETIC: (0, 220, 0),
        }
        for d in defects:
            x1, y1, x2, y2 = d.bbox
            color = _COLOR[d.severity]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            tag = f"{d.label} {d.confidence:.0%}"
            cv2.putText(image, tag, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image
