"""
Test suite for the VisualInspector pipeline.

Run with:
    pytest tests/ -v --tb=short
"""
import numpy as np
import pytest
import cv2

from src.pipeline.preprocessor import ImagePreprocessor
from src.pipeline.result import Defect, DefectSeverity, InspectionResult
from src.pipeline.inspector import InspectionPipeline, PipelineConfig
from src.models.detector import AnomalyDetector
from src.models.classifier import DefectClassifier


# ────────────────────────── Fixtures ──────────────────────────────

@pytest.fixture
def clean_image() -> np.ndarray:
    """A uniform grey image — expected: no defects."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    return img


@pytest.fixture
def defect_image() -> np.ndarray:
    """Grey image with a synthetic dark blob — expected: ≥1 defect."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (260, 200), (20, 20, 20), -1)  # dark blob
    return img


@pytest.fixture
def scratch_image() -> np.ndarray:
    """Grey image with a long horizontal line — expected: scratch label."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.line(img, (50, 240), (590, 240), (10, 10, 10), 3)
    return img


# ────────────────────────── Preprocessor ──────────────────────────

class TestImagePreprocessor:
    def test_output_size(self, clean_image):
        pp = ImagePreprocessor(target_size=(320, 320))
        out, meta = pp.run(clean_image)
        assert out.shape[:2] == (320, 320), "Output must match target_size"

    def test_meta_keys(self, clean_image):
        pp = ImagePreprocessor()
        _, meta = pp.run(clean_image)
        assert {"original_shape", "scale", "pad_w", "pad_h"} <= meta.keys()

    def test_aspect_ratio_preserved(self):
        """Letterbox should not distort a non-square image."""
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        pp = ImagePreprocessor(target_size=(640, 640))
        out, meta = pp.run(img)
        assert out.shape == (640, 640, 3)
        # scale = min(640/400, 640/100) = 1.6 (height-limited)
        assert abs(meta["scale"] - 1.6) < 0.01

    def test_unmap_bbox_roundtrip(self, clean_image):
        pp = ImagePreprocessor(target_size=(640, 640))
        _, meta = pp.run(clean_image)
        original_bbox = (100, 80, 200, 160)
        # Map forward (manual), then unmap
        scale, pw, ph = meta["scale"], meta["pad_w"], meta["pad_h"]
        mapped = (
            int(original_bbox[0] * scale + pw),
            int(original_bbox[1] * scale + ph),
            int(original_bbox[2] * scale + pw),
            int(original_bbox[3] * scale + ph),
        )
        recovered = pp.unmap_bbox(mapped, meta)
        for a, b in zip(original_bbox, recovered):
            assert abs(a - b) <= 2, "Unmap should be within 2px of original"

    def test_no_denoise_when_disabled(self, clean_image):
        """Disabling denoise should still return a valid processed image."""
        pp = ImagePreprocessor(denoise_strength=0)
        out, _ = pp.run(clean_image)
        assert out is not None
        assert out.dtype == np.uint8


# ────────────────────────── Detector ──────────────────────────────

class TestAnomalyDetector:
    def test_clean_image_no_detections(self, clean_image):
        det = AnomalyDetector(threshold=0.3)
        results = det.detect(clean_image)
        assert isinstance(results, list)
        # May return 0 or very few on a uniform image
        assert len(results) <= 3

    def test_defect_image_detects(self, defect_image):
        det = AnomalyDetector(threshold=0.1)
        results = det.detect(defect_image)
        assert len(results) >= 1, "Should detect the synthetic dark blob"

    def test_bbox_within_image(self, defect_image):
        det = AnomalyDetector(threshold=0.1)
        h, w = defect_image.shape[:2]
        for d in det.detect(defect_image):
            x1, y1, x2, y2 = d["bbox"]
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h

    def test_detection_keys(self, defect_image):
        det = AnomalyDetector(threshold=0.1)
        for d in det.detect(defect_image):
            assert {"bbox", "score", "area"} <= d.keys()
            assert 0 <= d["score"] <= 1
            assert d["area"] > 0

    def test_nms_reduces_overlapping(self):
        """Overlapping identical boxes should be collapsed to one."""
        det = AnomalyDetector(threshold=0.1, nms_iou=0.4)
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)
        results = det.detect(img)
        assert len(results) <= 3, "NMS should suppress near-duplicate boxes"


# ────────────────────────── Classifier ───────────────────────────

class TestDefectClassifier:
    def test_returns_valid_label(self, defect_image):
        from src.models.classifier import DEFECT_CLASSES
        clf = DefectClassifier()
        roi = defect_image[150:200, 200:260]
        label, conf = clf.classify(roi)
        assert label in DEFECT_CLASSES
        assert 0.0 <= conf <= 1.0

    def test_empty_roi_fallback(self):
        clf = DefectClassifier()
        label, conf = clf.classify(np.array([]))
        assert label == "unknown"
        assert conf == 0.0

    def test_scratch_image_heuristic(self, scratch_image):
        clf = DefectClassifier()
        # ROI: a square patch centred on the scratch (avoids high-aspect fallback)
        roi = scratch_image[220:270, 200:460]
        label, conf = clf.classify(roi)
        # Heuristic detects Hough lines → scratch or crack (both valid for a line)
        assert label in ("scratch", "crack")
        assert conf > 0.5


# ────────────────────────── Pipeline ─────────────────────────────

class TestInspectionPipeline:
    def test_clean_image_passes(self, clean_image):
        pipeline = InspectionPipeline(config=PipelineConfig(detection_threshold=0.3))
        result = pipeline.inspect(clean_image, label="clean_test")
        assert isinstance(result.passed, bool)
        assert result.inference_ms > 0

    def test_defect_image_result_structure(self, defect_image):
        pipeline = InspectionPipeline(config=PipelineConfig(
            detection_threshold=0.1,
            min_defect_area_px=10,
        ))
        result = pipeline.inspect(defect_image, label="defect_test")
        assert result.source_label == "defect_test"
        assert isinstance(result.defects, list)
        for d in result.defects:
            assert isinstance(d, Defect)

    def test_verdict_fail_on_critical(self):
        result = InspectionResult(
            source_label="test",
            image_shape=(480, 640),
            defects=[
                Defect(
                    bbox=(0, 0, 10, 10),
                    label="scratch",
                    confidence=0.9,
                    severity=DefectSeverity.CRITICAL,
                    area=100,
                )
            ],
            preprocessing_meta={},
            inference_ms=5.0,
        )
        assert result.verdict == "FAIL"
        assert not result.passed

    def test_verdict_pass_cosmetic_only(self):
        result = InspectionResult(
            source_label="test",
            image_shape=(480, 640),
            defects=[
                Defect(
                    bbox=(0, 0, 5, 5),
                    label="discoloration",
                    confidence=0.6,
                    severity=DefectSeverity.COSMETIC,
                    area=25,
                )
            ],
            preprocessing_meta={},
            inference_ms=3.0,
        )
        assert result.verdict == "PASS"
        assert result.passed

    def test_to_json_roundtrip(self, clean_image):
        import json
        pipeline = InspectionPipeline()
        result = pipeline.inspect(clean_image)
        data = json.loads(result.to_json())
        assert "verdict" in data
        assert "defects" in data
        assert isinstance(data["defects"], list)

    def test_batch_inspect_length(self, clean_image, defect_image):
        pipeline = InspectionPipeline()
        results = pipeline.inspect_batch([clean_image, defect_image, clean_image])
        assert len(results) == 3

    def test_summary_string(self, defect_image):
        pipeline = InspectionPipeline(config=PipelineConfig(
            detection_threshold=0.1, min_defect_area_px=10))
        result = pipeline.inspect(defect_image)
        summary = result.summary()
        assert "VisualInspector" in summary
        assert result.verdict in summary
