"""End-to-end Sa2VA smoke test for one video on a single GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Please segment the rabbit.")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=640)
    return parser.parse_args()


def read_uniform_frames(path: Path, count: int, max_side: int) -> tuple[list[Image.Image], list[int], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    if total <= 0:
        raise RuntimeError(f"Video reports no frames: {path}")

    indices = np.linspace(0, total - 1, min(count, total), dtype=np.int64).tolist()
    frames: list[Image.Image] = []
    actual_indices: list[int] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = capture.read()
        if not ok:
            continue
        height, width = bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        actual_indices.append(int(index))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return frames, actual_indices, fps


def color_for_object(index: int) -> np.ndarray:
    palette = ((35, 211, 238), (255, 199, 44), (255, 91, 112), (126, 231, 135))
    return np.asarray(palette[index % len(palette)], dtype=np.float32)


def render_results(
    frames: list[Image.Image], masks: list, output: Path, fps: float
) -> list[dict[str, float]]:
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    mask_dir = output / "masks"
    overlay_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    metrics: list[dict[str, float]] = []

    for frame_index, image in enumerate(frames):
        rgb = np.asarray(image).copy()
        union = np.zeros(rgb.shape[:2], dtype=bool)
        object_count = 0
        for object_index, object_masks in enumerate(masks):
            if frame_index >= len(object_masks):
                continue
            mask = np.asarray(object_masks[frame_index]).squeeze().astype(bool)
            if mask.shape != union.shape:
                mask = cv2.resize(mask.astype(np.uint8), (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                continue
            object_count += 1
            union |= mask
            color = color_for_object(object_index)
            rgb[mask] = (rgb[mask].astype(np.float32) * 0.44 + color * 0.56).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA)

        coverage = float(union.mean())
        cv2.imwrite(str(mask_dir / f"{frame_index:04d}.png"), union.astype(np.uint8) * 255)
        cv2.imwrite(str(overlay_dir / f"{frame_index:04d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        metrics.append({"frame": frame_index, "coverage": coverage, "objects": object_count})

    height, width = np.asarray(frames[0]).shape[:2]
    writer = cv2.VideoWriter(
        str(output / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (width, height),
    )
    for frame_index in range(len(frames)):
        rendered = cv2.imread(str(overlay_dir / f"{frame_index:04d}.jpg"))
        writer.write(rendered)
    writer.release()
    return metrics


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames, source_indices, source_fps = read_uniform_frames(args.video, args.frames, args.max_side)

    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval()
    load_seconds = time.perf_counter() - started

    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        result = model.predict_forward(
            video=frames,
            text=f"<image>{args.prompt}",
            tokenizer=tokenizer,
        )
    inference_seconds = time.perf_counter() - inference_started

    prediction = str(result.get("prediction", ""))
    prediction_masks = result.get("prediction_masks") or []
    metrics = render_results(frames, prediction_masks, args.output, source_fps / max(1, len(source_indices))) if prediction_masks else []
    report = {
        "model": str(args.model),
        "video": str(args.video),
        "prompt": args.prompt,
        "prediction": prediction,
        "sampled_source_frames": source_indices,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "mask_objects": len(prediction_masks),
        "frame_metrics": metrics,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
