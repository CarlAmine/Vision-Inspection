"""
VisualInspector Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end machine vision inspection pipeline.
"""
from .inspector import InspectionPipeline
from .preprocessor import ImagePreprocessor
from .result import InspectionResult, Defect, DefectSeverity

__all__ = [
    "InspectionPipeline",
    "ImagePreprocessor",
    "InspectionResult",
    "Defect",
    "DefectSeverity",
]
