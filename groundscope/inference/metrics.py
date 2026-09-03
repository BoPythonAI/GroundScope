from __future__ import annotations

import numpy as np


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(bool)
    right = right.astype(bool)
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def compute_frame_metrics(masks: list[np.ndarray]) -> list[dict[str, float | None]]:
    metrics: list[dict[str, float | None]] = []
    previous: np.ndarray | None = None
    for mask in masks:
        mask = mask.astype(bool)
        temporal_iou = None if previous is None else mask_iou(previous, mask)
        metrics.append(
            {
                "coverage": float(mask.mean()),
                "temporal_iou": temporal_iou,
            }
        )
        previous = mask
    return metrics


def summarize_metrics(frame_metrics: list[dict[str, float | None]]) -> dict[str, float]:
    coverage = np.asarray([float(item["coverage"] or 0) for item in frame_metrics], dtype=np.float32)
    temporal = [float(item["temporal_iou"]) for item in frame_metrics if item["temporal_iou"] is not None]
    return {
        "mean_coverage": float(coverage.mean()) if len(coverage) else 0.0,
        "area_stability": float(max(0.0, 1.0 - coverage.std())) if len(coverage) else 0.0,
        "mean_temporal_iou": float(np.mean(temporal)) if temporal else 0.0,
        "empty_frame_ratio": float(np.mean(coverage == 0)) if len(coverage) else 1.0,
    }
