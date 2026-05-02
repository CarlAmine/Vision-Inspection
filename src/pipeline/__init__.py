"""
VisualInspector Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end machine vision inspection pipeline.
"""
from .inspector import InspectionPipeline, PipelineConfig
from .preprocessor import ImagePreprocessor
from .result import InspectionResult, Defect, DefectSeverity

__all__ = [
    "InspectionPipeline",
    "PipelineConfig",
    "ImagePreprocessor",
    "InspectionResult",
    "Defect",
    "DefectSeverity",
]
