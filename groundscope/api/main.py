from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiofiles
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..inference.visualization import rerender_from_disk
from ..jobs import JobManager
from ..schemas import Artifact, CorrectionRequest, HealthResponse, JobRecord, SelectorName, TrackingMode

settings.ensure_directories()
manager = JobManager(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.restore()
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(
    title="GroundScope API",
    version="0.3.0",
    description="Interactive dense multi-target open-vocabulary video segmentation with Sa2VA and SAM2",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=settings.runs_dir), name="artifacts")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    disk = shutil.disk_usage(settings.root)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="ok",
        gpu=gpu,
        cuda_available=torch.cuda.is_available(),
        model_ready=manager.sa2va.ready,
        model_path=str(settings.model_path),
        queue_depth=manager.queue.qsize(),
        disk_free_gib=round(disk.free / 1024**3, 2),
    )


@app.get("/api/jobs", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return manager.list()


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    try:
        return manager.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error


@app.post("/api/jobs", response_model=JobRecord, status_code=202)
async def create_job(
    video: UploadFile = File(...),  # noqa: B008 - FastAPI dependency declaration
    prompt: str = Form(..., min_length=2, max_length=1000),
    selector: SelectorName = Form(SelectorName.hybrid),  # noqa: B008
    tracking_mode: TrackingMode = Form(TrackingMode.dense),  # noqa: B008
    frame_count: int = Form(24, ge=4, le=48),
) -> JobRecord:
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload a supported video file")
    job_id = uuid.uuid4().hex[:12]
    input_path = settings.uploads_dir / f"{job_id}{suffix}"
    max_bytes = int(settings.max_upload_gib * 1024**3)
    written = 0
    async with aiofiles.open(input_path, "wb") as handle:
        while chunk := await video.read(8 * 1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                await handle.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Video exceeds {settings.max_upload_gib:g} GiB")
            await handle.write(chunk)
    if written == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    job = JobRecord(
        id=job_id,
        filename=video.filename or input_path.name,
        prompt=prompt.strip(),
        selector=selector,
        tracking_mode=tracking_mode,
        frame_count=frame_count,
    )
    manager.add(job, input_path)
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> EventSourceResponse:
    try:
        manager.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    async def stream():
        last = ""
        while True:
            job = manager.get(job_id)
            payload = job.model_dump_json()
            if payload != last:
                yield {"event": "job", "data": payload}
                last = payload
            if job.status.value in {"complete", "failed"}:
                break
            await asyncio.sleep(0.75)

    return EventSourceResponse(stream())


@app.post("/api/jobs/{job_id}/corrections", response_model=JobRecord)
async def correct_mask(job_id: str, correction: CorrectionRequest) -> JobRecord:
    try:
        job = manager.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    if job.status.value != "complete" or correction.frame_slot >= len(job.frames):
        raise HTTPException(status_code=409, detail="Correction target is not ready")
    if correction.target_index >= len(job.targets):
        raise HTTPException(status_code=422, detail="Correction target index is invalid")

    run_dir = settings.runs_dir / job_id
    target_mask_path = run_dir / "masks" / f"target_{correction.target_index:02d}" / f"{correction.frame_slot:04d}.png"
    mask_path = target_mask_path if target_mask_path.exists() else run_dir / "masks" / f"{correction.frame_slot:04d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask artifact not found")
    height, width = mask.shape
    points = np.asarray(
        [[round(item.x * (width - 1)), round(item.y * (height - 1))] for item in correction.points],
        dtype=np.int32,
    )
    value = 255 if correction.operation == "add" else 0
    radius = max(1, round(correction.brush_radius * max(width, height)))
    if correction.shape == "polygon" and len(points) >= 3:
        cv2.fillPoly(mask, [points], value)
    elif len(points) == 1:
        cv2.circle(mask, tuple(points[0]), radius, value, thickness=-1, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(mask, [points], False, value, thickness=radius * 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(mask_path), mask)
    records, summary = await asyncio.to_thread(rerender_from_disk, run_dir)
    for record in records:
        frame = job.frames[record["slot"]]
        frame.coverage = record["coverage"]
        frame.target_coverages = record["target_coverages"]
        frame.temporal_iou = record["temporal_iou"]

    log_path = run_dir / "corrections.json"
    corrections = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    corrections.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            **correction.model_dump(mode="json"),
        }
    )
    log_path.write_text(json.dumps(corrections, indent=2), encoding="utf-8")
    if not any(item.url.endswith("corrections.json") for item in job.artifacts):
        job.artifacts.append(
            Artifact(
                name="Correction log",
                kind="audit",
                url=f"/artifacts/{job_id}/corrections.json",
                bytes=log_path.stat().st_size,
            )
        )
    for artifact in job.artifacts:
        artifact_path = run_dir / Path(artifact.url).name
        if artifact_path.exists():
            artifact.bytes = artifact_path.stat().st_size
    manager._save(job)

    report_path = run_dir / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["job"] = job.model_dump(mode="json")
    report["summary"] = summary
    report["corrections"] = corrections
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for artifact in job.artifacts:
        artifact_path = run_dir / Path(artifact.url).name
        if artifact_path.exists():
            artifact.bytes = artifact_path.stat().st_size
    manager._save(job)
    return job


@app.get("/api/jobs/{job_id}/download/{filename}")
def download_artifact(job_id: str, filename: str) -> FileResponse:
    allowed = {Path(item.url).name for item in get_job(job_id).artifacts}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = settings.runs_dir / job_id / filename
    return FileResponse(path, filename=f"groundscope-{job_id}-{filename}")


@app.get("/", response_class=HTMLResponse)
def root() -> Response:
    index = settings.frontend_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>GroundScope API</h1><p>Frontend has not been built. Open <a href='/docs'>/docs</a>.</p>"
    )


if settings.frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="frontend-assets")
