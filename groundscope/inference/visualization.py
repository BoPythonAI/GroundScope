from __future__ import annotations

import csv
import json
import subprocess
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import compute_frame_metrics, mask_iou, summarize_metrics

COLORS = ((35, 211, 238), (255, 199, 44), (255, 91, 112), (126, 231, 135))
FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "NotoSansSC-Regular.ttf"


def _draw_legend_text(rgb: np.ndarray, text: str, position: tuple[int, int]) -> np.ndarray:
    if FONT_PATH.exists():
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (position[0], position[1] - 14),
            text,
            font=ImageFont.truetype(str(FONT_PATH), 15),
            fill=(235, 239, 236),
        )
        return np.asarray(canvas)
    cv2.putText(
        rgb,
        text.encode("ascii", errors="replace").decode("ascii")[:34],
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 239, 236),
        1,
        cv2.LINE_AA,
    )
    return rgb


def combine_object_masks(raw_masks: list, frame_count: int, shapes: list[tuple[int, int]]) -> list[np.ndarray]:
    combined = [np.zeros(shape, dtype=bool) for shape in shapes]
    for object_masks in raw_masks:
        for frame_index in range(min(frame_count, len(object_masks))):
            mask = np.asarray(object_masks[frame_index]).squeeze().astype(np.uint8)
            if mask.shape != shapes[frame_index]:
                mask = cv2.resize(
                    mask,
                    (shapes[frame_index][1], shapes[frame_index][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            combined[frame_index] |= mask.astype(bool)
    return combined


def render_overlay(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.56,
    color_index: int = 0,
) -> np.ndarray:
    rgb = np.asarray(image).copy() if isinstance(image, Image.Image) else image.copy()
    mask = mask.astype(bool)
    color_tuple = COLORS[color_index % len(COLORS)]
    color = np.asarray(color_tuple, dtype=np.float32)
    if mask.any():
        rgb[mask] = (rgb[mask].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color_tuple, 2, cv2.LINE_AA)
    return rgb


def render_multi_overlay(
    image: Image.Image | np.ndarray,
    masks: list[np.ndarray],
    target_names: list[str],
    alpha: float = 0.52,
    target_states: list[str] | None = None,
) -> np.ndarray:
    rgb = np.asarray(image).copy() if isinstance(image, Image.Image) else image.copy()
    for target_index, mask in enumerate(masks):
        rgb = render_overlay(rgb, mask, alpha=alpha, color_index=target_index)
    active_indices = [index for index, mask in enumerate(masks) if np.asarray(mask).any()]
    if active_indices:
        legend_height = 10 + 22 * len(active_indices)
        legend_width = min(rgb.shape[1] - 16, 300)
        cv2.rectangle(rgb, (8, 8), (8 + legend_width, 8 + legend_height), (9, 13, 14), thickness=-1)
        for row, target_index in enumerate(active_indices):
            name = target_names[target_index]
            y = 25 + row * 22
            cv2.rectangle(rgb, (17, y - 10), (27, y), COLORS[target_index % len(COLORS)], thickness=-1)
            state = target_states[target_index] if target_states and target_index < len(target_states) else ""
            suffix = " · REACQUIRED" if state == "reacquired" else ""
            rgb = _draw_legend_text(rgb, f"{name[:30]}{suffix}", (36, y))
    return rgb


def _write_h264(frames_rgb: list[np.ndarray], output: Path, fps: float) -> None:
    if not frames_rgb:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{max(1.0, fps):.4f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames_rgb:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {error[-1000:]}")


def _start_h264(output: Path, fps: float, width: int, height: int) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{max(1.0, fps):.4f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class DenseArtifactWriter:
    """Stream full-frame masks and videos to the data disk during propagation."""

    def __init__(
        self,
        run_dir: Path,
        frames: list[Image.Image],
        target_names: list[str],
        fps: float,
    ) -> None:
        if not frames:
            raise ValueError("Cannot render an empty dense video")
        self.run_dir = run_dir
        self.frames = frames
        self.target_names = target_names
        self.fps = fps
        self.mask_dir = run_dir / "dense_masks"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for target_index in range(len(target_names)):
            (self.mask_dir / f"target_{target_index:02d}").mkdir(parents=True, exist_ok=True)
        width, height = frames[0].width, frames[0].height
        self.processes = {
            "source": _start_h264(run_dir / "dense_source.mp4", fps, width, height),
            "overlay": _start_h264(run_dir / "dense_overlay.mp4", fps, width, height),
            "mask": _start_h264(run_dir / "dense_mask.mp4", fps, width, height),
        }
        self.records: list[dict] = []
        self.previous_mask: np.ndarray | None = None
        self.closed = False

    def write(
        self,
        frame_index: int,
        masks: list[np.ndarray],
        diagnostics: list[dict[str, float | str]] | None = None,
    ) -> None:
        image = np.asarray(self.frames[frame_index])
        if not masks:
            masks = [np.zeros(image.shape[:2], dtype=bool)]
        masks = [np.asarray(mask, dtype=bool) for mask in masks]
        combined = np.logical_or.reduce(masks)
        states = [str(item.get("state", "tracked")) for item in diagnostics] if diagnostics else None
        overlay = render_multi_overlay(image, masks, self.target_names, target_states=states)
        mask_rgb = np.repeat((combined.astype(np.uint8) * 255)[..., None], 3, axis=2)
        for name, frame in (("source", image), ("overlay", overlay), ("mask", mask_rgb)):
            process = self.processes[name]
            assert process.stdin is not None
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        target_coverages: dict[str, float] = {}
        for target_index, target_mask in enumerate(masks):
            target_name = self.target_names[target_index]
            target_coverages[target_name] = float(target_mask.mean())
            cv2.imwrite(
                str(self.mask_dir / f"target_{target_index:02d}" / f"{frame_index:06d}.png"),
                target_mask.astype(np.uint8) * 255,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
        temporal_iou = None if self.previous_mask is None else mask_iou(self.previous_mask, combined)
        self.records.append(
            {
                "frame_index": frame_index,
                "timestamp": frame_index / self.fps,
                "coverage": float(combined.mean()),
                "target_coverages": target_coverages,
                "temporal_iou": temporal_iou,
                "target_diagnostics": diagnostics or [],
            }
        )
        self.previous_mask = combined

    def abort(self) -> None:
        for process in self.processes.values():
            if process.stdin:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
        self.closed = True

    def close(self, extra_summary: dict[str, object] | None = None) -> dict:
        if self.closed:
            raise RuntimeError("Dense artifact writer is already closed")
        errors: list[str] = []
        for name, process in self.processes.items():
            if process.stdin:
                process.stdin.close()
            error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            code = process.wait()
            if code != 0:
                errors.append(f"{name}: {error[-600:]}")
        self.closed = True
        if errors:
            raise RuntimeError("Dense video encoding failed: " + " | ".join(errors))

        with (self.run_dir / "dense_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["frame_index", "timestamp", "coverage", "temporal_iou"]
            for index in range(len(self.target_names)):
                fieldnames.extend(
                    [
                        f"coverage_target_{index}",
                        f"state_target_{index}",
                        f"confidence_target_{index}",
                        f"appearance_target_{index}",
                        f"semantic_target_{index}",
                        f"component_count_target_{index}",
                        f"component_margin_target_{index}",
                        f"reason_target_{index}",
                    ]
                )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                row = {
                    key: value
                    for key, value in record.items()
                    if key not in {"target_coverages", "target_diagnostics"}
                }
                for index, name in enumerate(self.target_names):
                    diagnostic = (
                        record["target_diagnostics"][index]
                        if index < len(record["target_diagnostics"])
                        else {}
                    )
                    row.update(
                        {
                            f"coverage_target_{index}": record["target_coverages"].get(name, 0.0),
                            f"state_target_{index}": diagnostic.get("state", "tracked"),
                            f"confidence_target_{index}": diagnostic.get("confidence", 0.0),
                            f"appearance_target_{index}": diagnostic.get("appearance_similarity", 0.0),
                            f"semantic_target_{index}": diagnostic.get("semantic_similarity", 0.0),
                            f"component_count_target_{index}": diagnostic.get("component_count", 0.0),
                            f"component_margin_target_{index}": diagnostic.get(
                                "component_score_margin", 0.0
                            ),
                            f"reason_target_{index}": diagnostic.get("reason", ""),
                        }
                    )
                writer.writerow(row)

        non_initial_ious = [record["temporal_iou"] for record in self.records if record["temporal_iou"] is not None]
        summary = {
            "frame_count": len(self.records),
            "fps": self.fps,
            "mean_coverage": float(np.mean([record["coverage"] for record in self.records])),
            "empty_frame_ratio": float(np.mean([record["coverage"] == 0 for record in self.records])),
            "mean_temporal_iou": float(np.mean(non_initial_ious)) if non_initial_ious else 1.0,
            "target_mean_coverage": {
                name: float(np.mean([record["target_coverages"].get(name, 0.0) for record in self.records]))
                for name in self.target_names
            },
            "target_presence_ratio": {
                name: float(
                    np.mean(
                        [
                            record["target_coverages"].get(name, 0.0) > 0
                            for record in self.records
                        ]
                    )
                )
                for name in self.target_names
            },
            "identity_rejections": int(
                sum(
                    diagnostic.get("state") == "rejected"
                    for record in self.records
                    for diagnostic in record["target_diagnostics"]
                )
            ),
            "reacquisitions": int(
                sum(
                    diagnostic.get("state") == "reacquired"
                    for record in self.records
                    for diagnostic in record["target_diagnostics"]
                )
            ),
        }
        if extra_summary:
            summary.update(extra_summary)
        (self.run_dir / "dense_metrics.json").write_text(
            json.dumps({"summary": summary, "targets": self.target_names}, indent=2), encoding="utf-8"
        )
        with zipfile.ZipFile(
            self.run_dir / "dense_masks.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(self.mask_dir.rglob("*.png")):
                archive.write(path, arcname=str(Path("dense_masks") / path.relative_to(self.mask_dir)))
        return summary


def write_artifacts(
    run_dir: Path,
    frames: list[Image.Image],
    target_masks: list[list[np.ndarray]],
    target_names: list[str],
    source_indices: list[int],
    timestamps: list[float],
    selection_scores: list[float],
    relevance_scores: list[float],
    motion_scores: list[float],
    playback_fps: float = 4.0,
) -> tuple[list[dict], dict]:
    frames_dir = run_dir / "frames"
    overlays_dir = run_dir / "overlays"
    masks_dir = run_dir / "masks"
    for path in (frames_dir, overlays_dir, masks_dir):
        path.mkdir(parents=True, exist_ok=True)

    originals: list[np.ndarray] = []
    overlays: list[np.ndarray] = []
    mask_viz: list[np.ndarray] = []
    if not target_masks:
        target_masks = [[np.zeros((image.height, image.width), dtype=bool) for image in frames]]
        target_names = ["target"]
    combined_masks = [
        np.logical_or.reduce([masks[slot] for masks in target_masks])
        for slot in range(len(frames))
    ]
    frame_metrics = compute_frame_metrics(combined_masks)
    target_metrics = [compute_frame_metrics(masks) for masks in target_masks]
    for target_index in range(len(target_masks)):
        (masks_dir / f"target_{target_index:02d}").mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for slot, (image, combined_mask) in enumerate(zip(frames, combined_masks, strict=True)):
        rgb = np.asarray(image)
        masks_at_slot = [masks[slot] for masks in target_masks]
        overlay = render_multi_overlay(rgb, masks_at_slot, target_names)
        mask_rgb = np.repeat((combined_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        Image.fromarray(rgb).save(frames_dir / f"{slot:04d}.jpg", quality=90)
        Image.fromarray(overlay).save(overlays_dir / f"{slot:04d}.jpg", quality=92)
        Image.fromarray(combined_mask.astype(np.uint8) * 255).save(masks_dir / f"{slot:04d}.png")
        for target_index, mask in enumerate(masks_at_slot):
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                masks_dir / f"target_{target_index:02d}" / f"{slot:04d}.png"
            )
        originals.append(rgb)
        overlays.append(overlay)
        mask_viz.append(mask_rgb)
        metric = frame_metrics[slot]
        target_coverages = {
            name: float(target_metrics[target_index][slot]["coverage"] or 0)
            for target_index, name in enumerate(target_names)
        }
        records.append(
            {
                "slot": slot,
                "source_index": source_indices[slot],
                "timestamp": timestamps[slot],
                "selection_score": selection_scores[slot],
                "relevance_score": relevance_scores[slot],
                "motion_score": motion_scores[slot],
                "coverage": metric["coverage"],
                "target_coverages": target_coverages,
                "temporal_iou": metric["temporal_iou"],
            }
        )

    _write_h264(originals, run_dir / "selected_frames.mp4", playback_fps)
    _write_h264(overlays, run_dir / "overlay.mp4", playback_fps)
    _write_h264(mask_viz, run_dir / "mask.mp4", playback_fps)

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_records = []
        for record in records:
            flattened = {key: value for key, value in record.items() if key != "target_coverages"}
            flattened.update(
                {f"coverage_target_{index}": record["target_coverages"].get(name, 0) for index, name in enumerate(target_names)}
            )
            csv_records.append(flattened)
        writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)
    summary = summarize_metrics(frame_metrics)
    summary["target_mean_coverage"] = {
        name: float(np.mean([metric["coverage"] for metric in target_metrics[index]]))
        for index, name in enumerate(target_names)
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "targets": target_names, "frames": records}, handle, indent=2)
    with zipfile.ZipFile(run_dir / "masks.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(masks_dir.rglob("*.png")):
            archive.write(path, arcname=str(Path("masks") / path.relative_to(masks_dir)))
    return records, summary


def rerender_from_disk(run_dir: Path, playback_fps: float = 4.0) -> tuple[list[dict], dict]:
    metadata_path = run_dir / "selection.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_paths = sorted((run_dir / "frames").glob("*.jpg"))
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    target_names = metadata.get("targets") or ["target"]
    target_dirs = sorted((run_dir / "masks").glob("target_*"))
    if target_dirs:
        target_masks = [
            [np.asarray(Image.open(path).convert("L")) > 127 for path in sorted(target_dir.glob("*.png"))]
            for target_dir in target_dirs
        ]
    else:
        mask_paths = sorted((run_dir / "masks").glob("*.png"))
        target_masks = [[np.asarray(Image.open(path).convert("L")) > 127 for path in mask_paths]]
    return write_artifacts(
        run_dir,
        frames,
        target_masks,
        target_names,
        metadata["source_indices"],
        metadata["timestamps"],
        metadata["selection_scores"],
        metadata["relevance_scores"],
        metadata["motion_scores"],
        playback_fps,
    )
