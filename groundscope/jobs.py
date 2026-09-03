from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .inference.sa2va import Sa2VARunner, empty_masks_for
from .inference.selectors import (
    ClipScorer,
    decode_all_frames,
    decode_candidates,
    select_frames,
    select_target_anchors,
)
from .inference.tracking import RawMaskStore, filter_presence_tracks
from .inference.visualization import DenseArtifactWriter, combine_object_masks, write_artifacts
from .schemas import Artifact, DenseSummary, FrameResult, JobRecord, JobStatus, TrackingMode

logger = logging.getLogger(__name__)


def parse_targets(prompt: str, maximum: int = 4) -> list[str]:
    """Parse one target expression per line while preserving user wording."""
    raw_items = prompt.replace("；", "\n").splitlines()
    if len(raw_items) == 1 and ";" in prompt:
        raw_items = prompt.split(";")
    targets: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        target = item.strip().lstrip("-•").strip()
        key = target.casefold()
        if len(target) < 2 or key in seen:
            continue
        targets.append(target)
        seen.add(key)
        if len(targets) == maximum:
            break
    if not targets:
        raise ValueError("Provide at least one target expression")
    return targets


def choose_dense_anchors(selected, maximum: int = 5) -> list[int]:
    """Keep frame zero for complete propagation and add the strongest semantic anchors."""
    strongest = sorted(selected, key=lambda item: item.selection_score, reverse=True)
    anchors = {0}
    for item in strongest:
        anchors.add(int(item.source_index))
        if len(anchors) == maximum:
            break
    return sorted(anchors)


class JobManager:
    def __init__(self, config: Settings):
        self.config = config
        self.jobs: dict[str, JobRecord] = {}
        self.inputs: dict[str, Path] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.sa2va = Sa2VARunner(config.model_path)
        self.clip = ClipScorer(config.clip_model_path)

    def restore(self) -> None:
        for path in self.config.runs_dir.glob("*/job.json"):
            try:
                job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
                if not job.targets:
                    job.targets = parse_targets(job.prompt)
                dense_metrics_path = path.parent / "dense_metrics.json"
                if (
                    job.tracking_mode == TrackingMode.dense
                    and job.dense_summary is None
                    and dense_metrics_path.exists()
                ):
                    dense_payload = json.loads(dense_metrics_path.read_text(encoding="utf-8"))
                    job.dense_summary = DenseSummary.model_validate(dense_payload["summary"])
                if job.status not in {JobStatus.complete, JobStatus.failed}:
                    job.status = JobStatus.failed
                    job.error = "Server restarted before this job completed. Submit it again."
                self.jobs[job.id] = job
                matches = list(self.config.uploads_dir.glob(f"{job.id}.*"))
                if matches:
                    self.inputs[job.id] = matches[0]
            except (OSError, ValueError) as error:
                logger.warning("Skipping invalid persisted job %s: %s", path, error)
                continue

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker(), name="groundscope-gpu-worker")

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    def add(self, job: JobRecord, input_path: Path) -> None:
        self.jobs[job.id] = job
        self.inputs[job.id] = input_path
        self._save(job)
        self.queue.put_nowait(job.id)

    def get(self, job_id: str) -> JobRecord:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def list(self) -> list[JobRecord]:
        return sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)

    def _save(self, job: JobRecord) -> None:
        job.updated_at = datetime.now(UTC)
        run_dir = self.config.runs_dir / job.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def update(self, job_id: str, progress: int, stage: str, status: JobStatus | None = None) -> None:
        job = self.get(job_id)
        job.progress = max(job.progress, min(100, progress))
        job.stage = stage
        if status is not None:
            job.status = status
        self._save(job)

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await asyncio.to_thread(self._process, job_id)
            except Exception as error:  # noqa: BLE001 - worker boundary persists failures
                job = self.get(job_id)
                job.status = JobStatus.failed
                job.stage = "Failed"
                job.error = f"{type(error).__name__}: {error}"
                (self.config.runs_dir / job_id / "traceback.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                self._save(job)
            finally:
                self.queue.task_done()

    def _process(self, job_id: str) -> None:
        job = self.get(job_id)
        video_path = self.inputs[job_id]
        run_dir = self.config.runs_dir / job_id

        self.update(job_id, 8, "Decoding candidate frames", JobStatus.decoding)
        info, candidates = decode_candidates(video_path, self.config.max_candidate_frames)
        job.duration_seconds = info.duration_seconds
        job.source_fps = info.fps
        job.source_frames = info.total_frames
        self._save(job)

        job.targets = parse_targets(job.prompt)
        self.update(job_id, 20, f"Localizing {job.frame_count} analysis frames", JobStatus.selecting)
        selection = select_frames(
            candidates,
            job.frame_count,
            job.selector.value,
            "; ".join(job.targets),
            self.clip,
            lambda progress, stage: self.update(job_id, progress, stage),
        )
        selected = selection.frames
        job.selector_backend = selection.backend
        selection_data = {
            "selector": job.selector.value,
            "backend": selection.backend,
            "targets": job.targets,
            "source_indices": [item.source_index for item in selected],
            "timestamps": [item.timestamp for item in selected],
            "selection_scores": [item.selection_score for item in selected],
            "relevance_scores": [item.relevance_score for item in selected],
            "motion_scores": [item.motion_score for item in selected],
            "tracking_mode": job.tracking_mode.value,
        }

        self.update(job_id, 42, "Loading segmentation runtime", JobStatus.loading_model)
        self.sa2va.load(lambda progress, stage: self.update(job_id, progress, stage))
        images = [item.image for item in selected]
        shapes = [(image.height, image.width) for image in images]
        job.predictions = {}
        dense_summary = None
        if job.tracking_mode == TrackingMode.dense:
            self.update(job_id, 44, "Decoding every source frame for dense tracking", JobStatus.decoding)
            _, dense_frames = decode_all_frames(
                video_path,
                self.config.max_dense_frames,
                self.config.dense_max_side,
            )
            target_anchor_indices = select_target_anchors(candidates, job.targets, self.clip)
            selection_data["anchor_indices_by_target"] = {
                job.targets[target_index]: indices
                for target_index, indices in target_anchor_indices.items()
            }
            raw_store = RawMaskStore(run_dir, len(dense_frames), len(job.targets))
            writer: DenseArtifactWriter | None = None
            try:
                prediction = self.sa2va.predict_dense(
                    dense_frames,
                    target_anchor_indices,
                    job.targets,
                    raw_store.write,
                    lambda progress, stage: self.update(
                        job_id, progress, stage, JobStatus.inferencing
                    ),
                    retain_state=True,
                )
                identity_started = time.perf_counter()
                presence = filter_presence_tracks(
                    dense_frames,
                    raw_store,
                    job.targets,
                    self.clip,
                    lambda progress, stage: self.update(
                        job_id, progress, stage, JobStatus.inferencing
                    ),
                )
                refinement_seconds = 0.0
                refinement_seeds: dict[int, tuple[int, np.ndarray]] = {}
                for target_index, track in enumerate(presence.tracks):
                    if not track.validated_anchors:
                        continue
                    seed_index = int(track.validated_anchors[0])
                    if seed_index <= 0 or not track.accepted[seed_index]:
                        continue
                    seed_frame = dense_frames[seed_index]
                    seed_mask = presence.masks_for_frame(
                        seed_index,
                        (seed_frame.height, seed_frame.width),
                    )[target_index]
                    if seed_mask.any():
                        refinement_seeds[target_index] = (
                            seed_index,
                            np.asarray(seed_mask, dtype=bool).copy(),
                        )
                if refinement_seeds:
                    selection_data["backward_mask_refinement_seeds"] = {
                        job.targets[target_index]: seed_index
                        for target_index, (seed_index, _) in refinement_seeds.items()
                    }

                    def merge_refined_masks(
                        frame_index: int,
                        masks: list[np.ndarray],
                        confidences: list[float],
                    ) -> None:
                        for target_index, (seed_index, _) in refinement_seeds.items():
                            if frame_index >= seed_index or target_index >= len(masks):
                                continue
                            raw_store.replace(target_index, frame_index, masks[target_index])
                            raw_store.confidences[target_index, frame_index] = (
                                float(confidences[target_index])
                                if target_index < len(confidences)
                                else 0.0
                            )

                    refinement = self.sa2va.refine_dense_from_masks(
                        dense_frames,
                        refinement_seeds,
                        merge_refined_masks,
                        lambda progress, stage: self.update(
                            job_id, progress, stage, JobStatus.inferencing
                        ),
                    )
                    refinement_seconds = float(refinement["inference_seconds"])
                    prediction["inference_seconds"] = (
                        float(prediction["inference_seconds"]) + refinement_seconds
                    )
                    prediction["peak_vram_gib"] = max(
                        float(prediction["peak_vram_gib"]),
                        float(refinement["peak_vram_gib"]),
                    )
                    prediction["refined_frames"] = int(refinement["refined_frames"])
                    self.sa2va.release_dense_state()
                    presence = filter_presence_tracks(
                        dense_frames,
                        raw_store,
                        job.targets,
                        self.clip,
                        lambda _progress, stage: self.update(
                            job_id,
                            83,
                            f"Revalidating refined masks: {stage}",
                            JobStatus.inferencing,
                        ),
                    )
                else:
                    self.sa2va.release_dense_state()
                identity_elapsed = time.perf_counter() - identity_started
                job.identity_filter_seconds = round(
                    max(0.0, identity_elapsed - refinement_seconds),
                    3,
                )
                selection_data["validated_anchor_indices_by_target"] = {
                    target: presence.tracks[index].validated_anchors
                    for index, target in enumerate(job.targets)
                }
                self.update(
                    job_id,
                    84,
                    "Rendering presence-aware dense outputs",
                    JobStatus.rendering,
                )
                writer = DenseArtifactWriter(run_dir, dense_frames, job.targets, info.fps)
                for frame_index, frame in enumerate(dense_frames):
                    shape = (frame.height, frame.width)
                    writer.write(
                        frame_index,
                        presence.masks_for_frame(frame_index, shape),
                        presence.diagnostics_for_frame(frame_index),
                    )
                presence_summary = presence.summary
                refined_frames = int(prediction.get("refined_frames", 0))
                dense_summary = writer.close(
                    {
                        "filter_version": (
                            "mask-refine-v3"
                            if refined_frames
                            else presence_summary["filter_version"]
                        ),
                        "reid_backend": presence_summary["backend"],
                        "backward_refined_frames": refined_frames,
                        "backward_refinement_seeds": selection_data.get(
                            "backward_mask_refinement_seeds", {}
                        ),
                        "target_filter_summary": presence_summary["target_summaries"],
                    }
                )
                raw_store.cleanup()
            except Exception:
                self.sa2va.release_dense_state()
                if writer is not None and not writer.closed:
                    writer.abort()
                raw_store.cleanup()
                raise
            job.predictions = prediction["predictions"]
            job.inference_seconds = round(float(prediction["inference_seconds"]), 3)
            job.propagation_seconds = job.inference_seconds
            job.peak_vram_gib = round(float(prediction["peak_vram_gib"]), 3)
            job.tracked_frames = int(prediction["tracked_frames"])
            job.dense_summary = dense_summary
            selection_data["active_targets"] = prediction["active_targets"]
            target_masks = []
            for target_index in range(len(job.targets)):
                masks = []
                for slot, item in enumerate(selected):
                    path = run_dir / "dense_masks" / f"target_{target_index:02d}" / f"{item.source_index:06d}.png"
                    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        masks.append(empty_masks_for([images[slot]])[0])
                        continue
                    if mask.shape != shapes[slot]:
                        mask = cv2.resize(
                            mask,
                            (shapes[slot][1], shapes[slot][0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    masks.append(mask > 127)
                target_masks.append(masks)
            del dense_frames
        else:
            target_masks = []
            total_inference_seconds = 0.0
            peak_vram_gib = 0.0
            for target_index, target in enumerate(job.targets):
                progress = 52 + round(24 * target_index / len(job.targets))
                self.update(
                    job_id,
                    progress,
                    f"Segmenting target {target_index + 1}/{len(job.targets)}: {target}",
                    JobStatus.inferencing,
                )
                prediction = self.sa2va.predict(images, target)
                job.predictions[target] = prediction["prediction"]
                raw_masks = prediction["prediction_masks"]
                target_masks.append(
                    combine_object_masks(raw_masks, len(images), shapes) if raw_masks else empty_masks_for(images)
                )
                total_inference_seconds += float(prediction["inference_seconds"])
                peak_vram_gib = max(peak_vram_gib, float(prediction["peak_vram_gib"]))
            job.inference_seconds = round(total_inference_seconds, 3)
            job.peak_vram_gib = round(peak_vram_gib, 3)
            job.tracked_frames = len(images)
            job.propagation_seconds = None
            job.dense_summary = None
        job.prediction = " | ".join(f"{target}: {response}" for target, response in job.predictions.items())
        self._save(job)
        (run_dir / "selection.json").write_text(json.dumps(selection_data, indent=2), encoding="utf-8")

        self.update(job_id, 86, "Rendering analysis and audit artifacts", JobStatus.rendering)
        records, summary = write_artifacts(
            run_dir,
            images,
            target_masks,
            job.targets,
            selection_data["source_indices"],
            selection_data["timestamps"],
            selection_data["selection_scores"],
            selection_data["relevance_scores"],
            selection_data["motion_scores"],
        )

        job.frames = [
            FrameResult(
                **record,
                thumbnail_url=f"/artifacts/{job_id}/frames/{record['slot']:04d}.jpg",
                overlay_url=f"/artifacts/{job_id}/overlays/{record['slot']:04d}.jpg",
                mask_url=f"/artifacts/{job_id}/masks/{record['slot']:04d}.png",
            )
            for record in records
        ]
        artifact_specs = (
            ("Overlay video", "video", "overlay.mp4"),
            ("Binary-mask video", "video", "mask.mp4"),
            ("Selected frames", "video", "selected_frames.mp4"),
            ("Mask PNG archive", "archive", "masks.zip"),
            ("Frame metrics", "table", "metrics.csv"),
        )
        if job.tracking_mode == TrackingMode.dense:
            artifact_specs += (
                ("Dense source video", "video", "dense_source.mp4"),
                ("Dense overlay video", "video", "dense_overlay.mp4"),
                ("Dense mask video", "video", "dense_mask.mp4"),
                ("Dense target masks", "archive", "dense_masks.zip"),
                ("Dense frame metrics", "table", "dense_metrics.csv"),
                ("Dense metrics summary", "report", "dense_metrics.json"),
            )
        job.artifacts = [
            Artifact(name=name, kind=kind, url=f"/artifacts/{job_id}/{filename}", bytes=(run_dir / filename).stat().st_size)
            for name, kind, filename in artifact_specs
            if (run_dir / filename).exists()
        ]
        job.artifacts.append(
            Artifact(name="Audit report", kind="report", url=f"/artifacts/{job_id}/result.json", bytes=0)
        )
        job.status = JobStatus.complete
        job.stage = "Complete"
        job.progress = 100
        self._save(job)
        result_payload = {
            "job": job.model_dump(mode="json"),
            "summary": summary,
            "selection": selection_data,
            "dense_summary": dense_summary,
            "corrections": [],
            "model": {
                "name": "ByteDance/Sa2VA-1B",
                "revision": "82faf06c93f6ce3fdc0ad3d45b57fd52c463daeb",
                "dtype": "bfloat16",
            },
        }
        report_path = run_dir / "result.json"
        report_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        job.artifacts[-1].bytes = report_path.stat().st_size
        self._save(job)
        result_payload["job"] = job.model_dump(mode="json")
        report_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
