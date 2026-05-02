"""
VisualInspector REST API
~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI app exposing /inspect (single image) and /inspect/batch endpoints.

Run:
    uvicorn src.api.app:app --reload --port 8000
"""
from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..pipeline import InspectionPipeline, PipelineConfig

logger = logging.getLogger(__name__)
_pipeline: Optional[InspectionPipeline] = None


# ------------------------------------------------------------------
# Lifespan: load pipeline once at startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Initialising InspectionPipeline…")
    _pipeline = InspectionPipeline(config=PipelineConfig(save_annotated=False))
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="VisualInspector API",
    description="Machine vision inspection as a service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class DefectOut(BaseModel):
    bbox:       list[int]
    label:      str
    confidence: float
    severity:   str
    area_px:    int


class InspectionOut(BaseModel):
    source:        str
    verdict:       str
    defect_count:  int
    defects:       list[DefectOut]
    inference_ms:  float
    image_shape:   list[int]


class Base64Request(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG image data.")
    label: Optional[str] = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.post("/inspect", response_model=InspectionOut)
async def inspect_upload(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
):
    """Upload an image file for inspection."""
    _require_pipeline()
    img = _decode_upload(await file.read())
    result = _pipeline.inspect(img, label=label or file.filename)
    return _result_to_out(result)


@app.post("/inspect/base64", response_model=InspectionOut)
async def inspect_base64(req: Base64Request):
    """Submit a base64-encoded image for inspection (e.g. from IoT devices)."""
    _require_pipeline()
    try:
        raw = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data.")
    img = _decode_upload(raw)
    result = _pipeline.inspect(img, label=req.label or "b64_upload")
    return _result_to_out(result)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    return img


def _result_to_out(result) -> InspectionOut:
    return InspectionOut(
        source=result.source_label,
        verdict=result.verdict,
        defect_count=len(result.defects),
        defects=[
            DefectOut(
                bbox=list(d.bbox),
                label=d.label,
                confidence=round(d.confidence, 4),
                severity=d.severity.value,
                area_px=d.area,
            )
            for d in result.defects
        ],
        inference_ms=result.inference_ms,
        image_shape=list(result.image_shape),
    )
