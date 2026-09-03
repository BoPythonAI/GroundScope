from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    queued = "queued"
    decoding = "decoding"
    selecting = "selecting"
    loading_model = "loading_model"
    inferencing = "inferencing"
    rendering = "rendering"
    complete = "complete"
    failed = "failed"


class SelectorName(StrEnum):
    uniform = "uniform"
    motion = "motion"
    query = "query"
    hybrid = "hybrid"


class TrackingMode(StrEnum):
    analysis = "analysis"
    dense = "dense"


class Artifact(BaseModel):
    name: str
    kind: str
    url: str
    bytes: int = 0


class FrameResult(BaseModel):
    slot: int
    source_index: int
    timestamp: float
    thumbnail_url: str
    overlay_url: str | None = None
    mask_url: str | None = None
    selection_score: float = 0.0
    relevance_score: float = 0.0
    motion_score: float = 0.0
    coverage: float = 0.0
    target_coverages: dict[str, float] = Field(default_factory=dict)
    temporal_iou: float | None = None


class DenseSummary(BaseModel):
    frame_count: int = 0
    fps: float = 0.0
    mean_coverage: float = 0.0
    empty_frame_ratio: float = 1.0
    mean_temporal_iou: float = 1.0
    target_mean_coverage: dict[str, float] = Field(default_factory=dict)
    target_presence_ratio: dict[str, float] = Field(default_factory=dict)
    identity_rejections: int = 0
    reacquisitions: int = 0
    filter_version: str | None = None
    reid_backend: str | None = None
    backward_refined_frames: int = 0
    backward_refinement_seeds: dict[str, int] = Field(default_factory=dict)
    target_filter_summary: dict[str, dict[str, float | int | str]] = Field(default_factory=dict)


class JobRecord(BaseModel):
    id: str
    filename: str
    prompt: str
    targets: list[str] = Field(default_factory=list)
    selector: SelectorName
    tracking_mode: TrackingMode = TrackingMode.analysis
    frame_count: int
    status: JobStatus = JobStatus.queued
    progress: int = 0
    stage: str = "Queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    prediction: str | None = None
    predictions: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float | None = None
    source_fps: float | None = None
    source_frames: int | None = None
    inference_seconds: float | None = None
    peak_vram_gib: float | None = None
    tracked_frames: int = 0
    propagation_seconds: float | None = None
    identity_filter_seconds: float | None = None
    dense_summary: DenseSummary | None = None
    selector_backend: str | None = None
    frames: list[FrameResult] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class CorrectionRequest(BaseModel):
    frame_slot: int = Field(ge=0)
    target_index: int = Field(default=0, ge=0)
    operation: Literal["add", "erase"]
    shape: Literal["brush", "polygon"] = "brush"
    points: list[Point] = Field(min_length=1)
    brush_radius: float = Field(default=0.02, ge=0.002, le=0.2)


class HealthResponse(BaseModel):
    status: str
    gpu: str | None
    cuda_available: bool
    model_ready: bool
    model_path: str
    queue_depth: int
    disk_free_gib: float
