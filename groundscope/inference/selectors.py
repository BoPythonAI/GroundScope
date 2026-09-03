from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class VideoInfo:
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


@dataclass
class CandidateFrame:
    source_index: int
    timestamp: float
    image: Image.Image
    feature: np.ndarray
    motion_score: float = 0.0
    relevance_score: float = 0.0
    selection_score: float = 0.0


@dataclass
class SelectionResult:
    frames: list[CandidateFrame]
    backend: str


ProgressCallback = Callable[[int, str], None]


def minmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    span = float(values.max(initial=0) - values.min(initial=0))
    if span < 1e-8:
        return np.zeros_like(values)
    return (values - values.min()) / span


def inspect_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path.name}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if total <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {path.name}")
    return VideoInfo(total, fps, width, height, total / fps)


def _compact_feature(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32).reshape(-1) / 255.0
    gray = gray[::4]
    return np.concatenate((hist, gray)).astype(np.float32)


def decode_candidates(path: Path, limit: int, max_side: int = 768) -> tuple[VideoInfo, list[CandidateFrame]]:
    info = inspect_video(path)
    candidate_count = min(max(limit, 2), info.total_frames)
    indices = np.linspace(0, info.total_frames - 1, candidate_count, dtype=np.int64)
    capture = cv2.VideoCapture(str(path))
    candidates: list[CandidateFrame] = []
    previous_gray: np.ndarray | None = None
    for source_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, bgr = capture.read()
        if not ok:
            continue
        height, width = bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            bgr = cv2.resize(
                bgr,
                (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
        motion = 0.0 if previous_gray is None else float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
        previous_gray = gray
        candidates.append(
            CandidateFrame(
                source_index=int(source_index),
                timestamp=float(source_index / info.fps),
                image=Image.fromarray(rgb),
                feature=_compact_feature(rgb),
                motion_score=motion,
            )
        )
    capture.release()
    if not candidates:
        raise ValueError(f"No frames decoded from: {path.name}")
    motion_scores = minmax(np.asarray([item.motion_score for item in candidates]))
    for item, score in zip(candidates, motion_scores, strict=True):
        item.motion_score = float(score)
    return info, candidates


def decode_all_frames(path: Path, max_frames: int, max_side: int = 960) -> tuple[VideoInfo, list[Image.Image]]:
    """Decode every source frame sequentially for SAM2 propagation.

    Frames are resized only when their longest side exceeds ``max_side``. The
    original temporal resolution and frame order are preserved exactly.
    """
    info = inspect_video(path)
    if info.total_frames > max_frames:
        raise ValueError(
            f"Dense tracking supports up to {max_frames} frames per job; "
            f"this video contains {info.total_frames}."
        )
    capture = cv2.VideoCapture(str(path))
    frames: list[Image.Image] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        height, width = bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            output_width = max(2, round(width * scale / 2) * 2)
            output_height = max(2, round(height * scale / 2) * 2)
            bgr = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {path.name}")
    return info, frames


class ClipScorer:
    """Lazy local-only CLIP scoring; query selection remains auditable if unavailable."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._processor = None

    @property
    def available(self) -> bool:
        return (self.model_path / "config.json").exists()

    def _ensure_model(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"CLIP model not found at {self.model_path}")
        from transformers import CLIPModel, CLIPProcessor

        if self._model is None:
            self._processor = CLIPProcessor.from_pretrained(
                self.model_path, local_files_only=True, use_fast=True
            )
            self._model = CLIPModel.from_pretrained(
                self.model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
            ).eval().cuda()

    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Return normalized CLIP image features without retaining GPU outputs."""
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        self._ensure_model()
        assert self._processor is not None and self._model is not None
        image_vectors: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
                vectors = self._model.get_image_features(**inputs)
                vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                image_vectors.append(vectors.cpu())
        return torch.cat(image_vectors).numpy()

    def embed_texts(self, prompts: list[str]) -> np.ndarray:
        """Return normalized CLIP text features for target-specific ranking."""
        if not prompts:
            return np.empty((0, 512), dtype=np.float32)
        self._ensure_model()
        assert self._processor is not None and self._model is not None
        with torch.inference_mode():
            text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {key: value.cuda(non_blocking=True) for key, value in text_inputs.items()}
            text_vectors = self._model.get_text_features(**text_inputs)
            text_vectors = torch.nn.functional.normalize(text_vectors.float(), dim=-1).cpu()
        return text_vectors.numpy()

    def score_prompts(
        self,
        frames: list[CandidateFrame],
        prompts: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score several targets against one shared image-embedding pass."""
        embeddings = self.embed_images([item.image for item in frames], batch_size=16)
        text_vectors = self.embed_texts(prompts)
        similarities = embeddings @ text_vectors.T
        normalized = np.column_stack(
            [minmax(similarities[:, index]) for index in range(len(prompts))]
        )
        return embeddings, normalized

    def score(self, frames: list[CandidateFrame], prompt: str) -> tuple[np.ndarray, np.ndarray]:
        embeddings, relevance = self.score_prompts(frames, [prompt])
        return embeddings, relevance[:, 0]


def select_target_anchors(
    candidates: list[CandidateFrame],
    targets: list[str],
    clip_scorer: ClipScorer,
    maximum: int = 5,
) -> dict[int, list[int]]:
    """Choose temporally separated, target-specific grounding anchors."""
    if not candidates or not targets:
        return {}
    maximum = min(maximum, len(candidates))
    if clip_scorer.available:
        _, relevance = clip_scorer.score_prompts(candidates, targets)
    else:
        fallback = np.asarray([item.selection_score for item in candidates], dtype=np.float32)
        relevance = np.repeat(fallback[:, None], len(targets), axis=1)
    anchors: dict[int, list[int]] = {}
    for target_index in range(len(targets)):
        chosen = _temporal_top(relevance[:, target_index], maximum)
        anchors[target_index] = [int(candidates[index].source_index) for index in chosen]
    return anchors


def _uniform_indices(length: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, length - 1, min(count, length), dtype=np.int64).tolist()))


def _temporal_top(scores: np.ndarray, count: int) -> list[int]:
    if count >= len(scores):
        return list(range(len(scores)))
    radius = max(1, len(scores) // (count * 2))
    remaining = scores.copy()
    selected: list[int] = []
    for _ in range(count):
        index = int(np.argmax(remaining))
        selected.append(index)
        remaining[max(0, index - radius) : min(len(scores), index + radius + 1)] = -np.inf
        if not np.isfinite(remaining).any():
            break
    if len(selected) < count:
        selected.extend(index for index in _uniform_indices(len(scores), count) if index not in selected)
    return sorted(selected[:count])


def select_frames(
    candidates: list[CandidateFrame],
    count: int,
    strategy: str,
    prompt: str,
    clip_scorer: ClipScorer,
    progress: ProgressCallback | None = None,
) -> SelectionResult:
    count = min(max(2, count), len(candidates))
    compact = np.stack([item.feature for item in candidates])
    compact /= np.maximum(np.linalg.norm(compact, axis=1, keepdims=True), 1e-8)
    motion = np.asarray([item.motion_score for item in candidates], dtype=np.float32)
    semantic_embeddings: np.ndarray | None = None
    relevance = np.zeros(len(candidates), dtype=np.float32)
    backend = "visual-motion"

    if strategy in {"query", "hybrid"}:
        if progress:
            progress(25, "Computing image-text relevance")
        if clip_scorer.available:
            semantic_embeddings, relevance = clip_scorer.score(candidates, prompt)
            backend = "CLIP ViT-B/32"
        else:
            backend = "visual-motion (CLIP unavailable)"

    if strategy == "uniform":
        selected = _uniform_indices(len(candidates), count)
        scores = np.ones(len(candidates), dtype=np.float32) * 0.5
        backend = "deterministic-uniform"
    elif strategy == "motion":
        scores = 0.85 * motion + 0.15 * minmax(np.arange(len(candidates), dtype=np.float32))
        selected = _temporal_top(scores, count)
    elif strategy == "query":
        scores = relevance if clip_scorer.available else 0.75 * motion
        selected = _temporal_top(scores, count)
    elif strategy == "hybrid":
        features = semantic_embeddings if semantic_embeddings is not None else compact
        base = 0.55 * relevance + 0.35 * motion + 0.10
        selected = [int(np.argmax(base))]
        while len(selected) < count:
            similarity = features @ features[selected].T
            visual_diversity = 1.0 - similarity.max(axis=1)
            temporal_distance = np.min(
                np.abs(np.arange(len(candidates))[:, None] - np.asarray(selected)[None, :]), axis=1
            ) / max(1, len(candidates) - 1)
            greedy_score = 0.55 * base + 0.30 * minmax(visual_diversity) + 0.15 * temporal_distance
            greedy_score[selected] = -np.inf
            selected.append(int(np.argmax(greedy_score)))
        selected.sort()
        scores = base
    else:
        raise ValueError(f"Unknown selector: {strategy}")

    for index, item in enumerate(candidates):
        item.relevance_score = float(relevance[index])
        item.selection_score = float(scores[index])
    return SelectionResult([candidates[index] for index in selected], backend)
