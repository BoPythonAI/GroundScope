import numpy as np

from groundscope.inference.metrics import compute_frame_metrics, mask_iou, summarize_metrics


def test_mask_iou_handles_empty_union() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    assert mask_iou(empty, empty) == 1.0


def test_metrics_are_auditable() -> None:
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[:2, :2] = True
    second[1:3, 1:3] = True
    frames = compute_frame_metrics([first, second])
    summary = summarize_metrics(frames)
    assert frames[0]["coverage"] == 0.25
    assert frames[1]["temporal_iou"] == 1 / 7
    assert summary["empty_frame_ratio"] == 0.0
