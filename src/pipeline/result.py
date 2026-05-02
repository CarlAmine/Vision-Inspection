"""
Result data structures for the inspection pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DefectSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR    = "major"
    MINOR    = "minor"
    COSMETIC = "cosmetic"

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "major": "🟠", "minor": "🟡", "cosmetic": "🟢"}[self.value]


@dataclass
class Defect:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    label: str
    confidence: float
    severity: DefectSeverity
    area: int                         # pixels

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "area_px": self.area,
        }


@dataclass
class InspectionResult:
    source_label: str
    image_shape: tuple[int, int]      # (height, width)
    defects: list[Defect]
    preprocessing_meta: dict[str, Any]
    inference_ms: float

    @property
    def passed(self) -> bool:
        """True if no CRITICAL or MAJOR defects found."""
        return not any(
            d.severity in {DefectSeverity.CRITICAL, DefectSeverity.MAJOR}
            for d in self.defects
        )

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def summary(self) -> str:
        lines = [
            f"{'─' * 52}",
            f"  VisualInspector Result — {self.source_label}",
            f"{'─' * 52}",
            f"  Verdict   : {self.verdict}",
            f"  Defects   : {len(self.defects)}",
            f"  Image     : {self.image_shape[1]}×{self.image_shape[0]} px",
            f"  Inference : {self.inference_ms:.1f} ms",
        ]
        if self.defects:
            lines.append("")
            lines.append("  Detections:")
            for i, d in enumerate(self.defects, 1):
                lines.append(
                    f"    {i}. [{d.severity.emoji} {d.severity.value:8s}] "
                    f"{d.label} ({d.confidence:.0%})  area={d.area}px"
                )
        lines.append("─" * 52)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_label,
            "verdict": self.verdict,
            "defect_count": len(self.defects),
            "defects": [d.to_dict() for d in self.defects],
            "inference_ms": self.inference_ms,
            "image_shape": list(self.image_shape),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
