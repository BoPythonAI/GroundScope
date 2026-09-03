from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .selectors import ClipScorer


@dataclass(frozen=True)
class PresenceFilterConfig:
    min_area: float = 0.00008
    max_area: float = 0.65
    min_component_area: float = 0.00004
    max_components_per_frame: int = 12
    min_component_appearance: float = 0.70
    min_component_score: float = 0.62
    component_ambiguity_margin: float = 0.025
    min_sam_confidence: float = 0.53
    min_appearance_similarity: float = 0.72
    max_appearance_similarity: float = 0.88
    initial_margin: float = 0.04
    initial_semantic_threshold: float = 0.12
    initial_confirmation: int = 4
    reacquire_margin: float = 0.045
    missing_patience: int = 2
    reacquire_confirmation: int = 2
    prototype_limit: int = 16


@dataclass
class PresenceTrack:
    accepted: np.ndarray
    states: list[str]
    reasons: list[str]
    appearance_similarity: np.ndarray
    semantic_similarity: np.ndarray
    confidence: np.ndarray
    validated_anchors: list[int]
    summary: dict[str, float | int | str]
    component_counts: np.ndarray | None = None
    component_score_margins: np.ndarray | None = None


@dataclass
class PresenceFilterResult:
    store: RawMaskStore
    tracks: list[PresenceTrack]
    targets: list[str]
    backend: str
    filter_version: str = "component-reid-v2"

    def masks_for_frame(self, frame_index: int, shape: tuple[int, int]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for target_index, track in enumerate(self.tracks):
            if track.accepted[frame_index]:
                masks.append(self.store.load(target_index, frame_index, shape))
            else:
                masks.append(np.zeros(shape, dtype=bool))
        return masks

    def diagnostics_for_frame(self, frame_index: int) -> list[dict[str, float | str]]:
        diagnostics: list[dict[str, float | str]] = []
        for track in self.tracks:
            item: dict[str, float | str] = {
                "state": track.states[frame_index],
                "reason": track.reasons[frame_index],
                "confidence": float(track.confidence[frame_index]),
                "appearance_similarity": float(track.appearance_similarity[frame_index]),
                "semantic_similarity": float(track.semantic_similarity[frame_index]),
            }
            if track.component_counts is not None:
                item["component_count"] = float(track.component_counts[frame_index])
            if track.component_score_margins is not None:
                item["component_score_margin"] = float(
                    track.component_score_margins[frame_index]
                )
            diagnostics.append(item)
        return diagnostics

    @property
    def summary(self) -> dict[str, object]:
        return {
            "filter_version": self.filter_version,
            "backend": self.backend,
            "target_summaries": {
                target: track.summary for target, track in zip(self.targets, self.tracks, strict=True)
            },
            "identity_rejections": int(
                sum(int(track.summary["identity_rejections"]) for track in self.tracks)
            ),
            "reacquisitions": int(sum(int(track.summary["reacquisitions"]) for track in self.tracks)),
        }


class RawMaskStore:
    """Data-disk-backed raw masks so dense propagation remains memory bounded."""

    def __init__(self, run_dir: Path, frame_count: int, target_count: int) -> None:
        self.run_dir = run_dir.resolve()
        self.root = (self.run_dir / ".raw_dense_masks").resolve()
        if self.root.parent != self.run_dir or self.root.name != ".raw_dense_masks":
            raise ValueError("Unsafe raw-mask store path")
        self.frame_count = frame_count
        self.target_count = target_count
        self.confidences = np.zeros((target_count, frame_count), dtype=np.float32)
        self.root.mkdir(parents=True, exist_ok=True)
        for target_index in range(target_count):
            (self.root / f"target_{target_index:02d}").mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame_index: int,
        masks: list[np.ndarray],
        confidences: list[float] | None = None,
    ) -> None:
        for target_index in range(self.target_count):
            mask = (
                np.asarray(masks[target_index], dtype=bool)
                if target_index < len(masks)
                else np.zeros((1, 1), dtype=bool)
            )
            confidence = (
                float(confidences[target_index])
                if confidences and target_index < len(confidences)
                else 0.0
            )
            if confidence >= self.confidences[target_index, frame_index]:
                cv2.imwrite(
                    str(self.path(target_index, frame_index)),
                    mask.astype(np.uint8) * 255,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                self.confidences[target_index, frame_index] = confidence

    def path(self, target_index: int, frame_index: int) -> Path:
        return self.root / f"target_{target_index:02d}" / f"{frame_index:06d}.png"

    def load(self, target_index: int, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
        mask = cv2.imread(str(self.path(target_index, frame_index)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(shape, dtype=bool)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask > 127

    def replace(self, target_index: int, frame_index: int, mask: np.ndarray) -> None:
        """Replace a raw union mask with the single validated component candidate."""
        cv2.imwrite(
            str(self.path(target_index, frame_index)),
            np.asarray(mask, dtype=np.uint8) * 255,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

    def cleanup(self) -> None:
        if self.root.exists() and self.root.parent == self.run_dir and self.root.name == ".raw_dense_masks":
            shutil.rmtree(self.root)


@dataclass
class _ComponentCandidate:
    frame_index: int
    bbox: tuple[int, int, int, int]
    local_mask: np.ndarray
    coverage: float
    centroid: np.ndarray
    shape_score: float

    def materialize(self, shape: tuple[int, int]) -> np.ndarray:
        x, y, width, height = self.bbox
        mask = np.zeros(shape, dtype=bool)
        mask[y : y + height, x : x + width] = self.local_mask
        return mask


def _component_crop(
    image: Image.Image,
    candidate: _ComponentCandidate,
    size: int = 224,
) -> Image.Image:
    rgb = np.asarray(image)
    x, y, width, height = candidate.bbox
    pad = max(3, round(max(width, height) * 0.12))
    x0, x1 = max(0, x - pad), min(rgb.shape[1], x + width + pad)
    y0, y1 = max(0, y - pad), min(rgb.shape[0], y + height + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = np.zeros(crop.shape[:2], dtype=bool)
    offset_x, offset_y = x - x0, y - y0
    crop_mask[offset_y : offset_y + height, offset_x : offset_x + width] = (
        candidate.local_mask
    )
    crop[~crop_mask] = 114
    crop_height, crop_width = crop.shape[:2]
    side = max(crop_height, crop_width)
    square = np.full((side, side, 3), 114, dtype=np.uint8)
    top, left = (side - crop_height) // 2, (side - crop_width) // 2
    square[top : top + crop_height, left : left + crop_width] = crop
    return Image.fromarray(square).resize((size, size), Image.Resampling.BICUBIC)


def _split_mask_components(
    mask: np.ndarray,
    frame_index: int,
    config: PresenceFilterConfig,
) -> list[_ComponentCandidate]:
    """Split a possibly disjoint SAM mask into bounded component candidates."""
    binary = np.asarray(mask, dtype=np.uint8)
    height, width = binary.shape
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    minimum_pixels = max(12, round(height * width * config.min_component_area))
    maximum_pixels = round(height * width * config.max_area)
    component_ids = [
        label
        for label in range(1, label_count)
        if minimum_pixels <= int(stats[label, cv2.CC_STAT_AREA]) <= maximum_pixels
    ]
    component_ids.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    candidates: list[_ComponentCandidate] = []
    for label in component_ids[: config.max_components_per_frame]:
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        local_mask = labels[y : y + component_height, x : x + component_width] == label
        aspect = component_height / max(1, component_width)
        fill = area / max(1, component_width * component_height)
        aspect_score = float(np.exp(-0.38 * abs(np.log(max(aspect, 1e-3) / 1.8))))
        fill_score = float(np.clip(1.0 - abs(fill - 0.48) / 0.48, 0.0, 1.0))
        candidates.append(
            _ComponentCandidate(
                frame_index=frame_index,
                bbox=(x, y, component_width, component_height),
                local_mask=local_mask,
                coverage=area / float(height * width),
                centroid=np.asarray(
                    [
                        float(centroids[label, 0] / max(1, width - 1)),
                        float(centroids[label, 1] / max(1, height - 1)),
                    ],
                    dtype=np.float32,
                ),
                shape_score=0.72 * aspect_score + 0.28 * fill_score,
            )
        )
    return candidates


def _normalize_scores(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    selected = values[valid]
    if not len(selected):
        return result
    low, high = np.percentile(selected, [5, 95])
    if high - low < 1e-6:
        result[valid] = 0.5
        return result
    result[valid] = np.clip((selected - low) / (high - low), 0.0, 1.0)
    return result


def _masked_crop(image: Image.Image, mask: np.ndarray, size: int = 224) -> Image.Image:
    rgb = np.asarray(image)
    ys, xs = np.where(mask)
    if not len(xs):
        return Image.new("RGB", (size, size), (114, 114, 114))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(3, round(max(x1 - x0, y1 - y0) * 0.12))
    x0, x1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 114
    height, width = crop.shape[:2]
    side = max(height, width)
    square = np.full((side, side, 3), 114, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = crop
    return Image.fromarray(square).resize((size, size), Image.Resampling.BICUBIC)


def _compact_embeddings(crops: list[Image.Image]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for crop in crops:
        rgb = np.asarray(crop.resize((32, 32), Image.Resampling.BILINEAR))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[::4, ::4].flatten() / 255.0
        vector = np.concatenate((hist, gray)).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        vectors.append(vector)
    return np.stack(vectors) if vectors else np.empty((0, 192), dtype=np.float32)


def _select_component_sequence(
    frames: list[Image.Image],
    store: RawMaskStore,
    target_index: int,
    target: str,
    clip_scorer: ClipScorer,
    config: PresenceFilterConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | float],
    str,
]:
    """Choose at most one identity-consistent component for every video frame."""
    frame_count = len(frames)
    frame_candidates: list[list[int]] = [[] for _ in frames]
    candidates: list[_ComponentCandidate] = []
    crops: list[Image.Image] = []
    component_counts = np.zeros(frame_count, dtype=np.int16)
    for frame_index, frame in enumerate(frames):
        shape = (frame.height, frame.width)
        raw_mask = store.load(target_index, frame_index, shape)
        components = _split_mask_components(raw_mask, frame_index, config)
        component_counts[frame_index] = len(components)
        for component in components:
            candidate_index = len(candidates)
            candidates.append(component)
            frame_candidates[frame_index].append(candidate_index)
            crops.append(_component_crop(frame, component))

    if clip_scorer.available and crops:
        candidate_embeddings = clip_scorer.embed_images(crops)
        text_embedding = clip_scorer.embed_texts([target])[0]
        semantic_raw = (candidate_embeddings @ text_embedding).astype(np.float32)
        backend = "CLIP ViT-B/32 component-level appearance re-ID"
    else:
        candidate_embeddings = _compact_embeddings(crops)
        semantic_raw = np.full(len(candidates), 0.5, dtype=np.float32)
        backend = "compact component-level appearance"

    feature_width = candidate_embeddings.shape[1] if len(candidate_embeddings) else 192
    coverages = np.zeros(frame_count, dtype=np.float32)
    centroids = np.zeros((frame_count, 2), dtype=np.float32)
    embeddings = np.zeros((frame_count, feature_width), dtype=np.float32)
    semantic_scores = np.zeros(frame_count, dtype=np.float32)
    score_margins = np.zeros(frame_count, dtype=np.float32)
    if not candidates:
        return (
            coverages,
            centroids,
            embeddings,
            semantic_scores,
            component_counts,
            score_margins,
            {
                "candidate_components": 0,
                "discarded_components": 0,
                "multi_component_frames": 0,
                "ambiguous_component_frames": 0,
            },
            backend,
        )

    valid_candidates = np.ones(len(candidates), dtype=bool)
    semantic = _normalize_scores(semantic_raw, valid_candidates)
    temporal_support = np.zeros(len(candidates), dtype=np.float32)
    for candidate_index, candidate in enumerate(candidates):
        neighbor_scores: list[float] = []
        for neighbor_frame in (candidate.frame_index - 1, candidate.frame_index + 1):
            if not 0 <= neighbor_frame < frame_count:
                continue
            neighbor_indices = frame_candidates[neighbor_frame]
            if not neighbor_indices:
                continue
            neighbor_array = np.asarray(neighbor_indices, dtype=np.int64)
            similarities = candidate_embeddings[neighbor_array] @ candidate_embeddings[candidate_index]
            neighbor_centroids = np.stack(
                [candidates[index].centroid for index in neighbor_indices]
            )
            distances = np.linalg.norm(neighbor_centroids - candidate.centroid, axis=1)
            spatial_support = np.exp(-distances / 0.16)
            neighbor_scores.append(
                float(np.max(np.clip(similarities, 0.0, 1.0) * spatial_support))
            )
        temporal_support[candidate_index] = (
            float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
        )
    temporal = _normalize_scores(temporal_support, valid_candidates)
    confidence_norm = np.asarray(
        [
            np.clip((store.confidences[target_index, candidate.frame_index] - 0.5) / 0.35, 0.0, 1.0)
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    shape_scores = np.asarray([candidate.shape_score for candidate in candidates], dtype=np.float32)
    seed_scores = 0.56 * semantic + 0.27 * temporal + 0.10 * confidence_norm + 0.07 * shape_scores
    seed_index = int(np.argmax(seed_scores))
    seed_similarity = candidate_embeddings @ candidate_embeddings[seed_index]
    prototype_pool = np.flatnonzero((seed_similarity >= 0.80) & (semantic >= 0.18))
    if not len(prototype_pool):
        prototype_pool = np.asarray([seed_index])
    prototype_rank = (
        0.48 * seed_scores[prototype_pool]
        + 0.34 * np.clip(seed_similarity[prototype_pool], 0.0, 1.0)
        + 0.18 * temporal[prototype_pool]
    )
    ranked_pool = prototype_pool[np.argsort(prototype_rank)[::-1]]
    chosen_prototype: list[int] = []
    radius = max(1, frame_count // max(12, config.prototype_limit * 3))
    for candidate_index in ranked_pool:
        frame_index = candidates[int(candidate_index)].frame_index
        if all(
            abs(frame_index - candidates[existing].frame_index) >= radius
            for existing in chosen_prototype
        ):
            chosen_prototype.append(int(candidate_index))
        if len(chosen_prototype) == config.prototype_limit:
            break
    if not chosen_prototype:
        chosen_prototype = [seed_index]
    prototype = candidate_embeddings[np.asarray(chosen_prototype)].mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    appearance = (candidate_embeddings @ prototype).astype(np.float32)
    candidate_scores = (
        0.55 * np.clip(appearance, 0.0, 1.0)
        + 0.28 * semantic
        + 0.12 * temporal
        + 0.05 * shape_scores
    )

    selected_count = 0
    ambiguous_frames = 0
    for frame_index, candidate_indices in enumerate(frame_candidates):
        frame = frames[frame_index]
        shape = (frame.height, frame.width)
        if not candidate_indices:
            store.replace(target_index, frame_index, np.zeros(shape, dtype=bool))
            continue
        ranked = sorted(candidate_indices, key=lambda index: float(candidate_scores[index]), reverse=True)
        best_index = ranked[0]
        second_score = float(candidate_scores[ranked[1]]) if len(ranked) > 1 else 0.0
        margin = float(candidate_scores[best_index] - second_score)
        score_margins[frame_index] = margin
        ambiguous = (
            len(ranked) > 1
            and margin < config.component_ambiguity_margin
            and appearance[best_index] < 0.90
        )
        reliable = (
            candidate_scores[best_index] >= config.min_component_score
            and appearance[best_index] >= config.min_component_appearance
            and not ambiguous
        )
        if not reliable:
            ambiguous_frames += int(ambiguous)
            store.replace(target_index, frame_index, np.zeros(shape, dtype=bool))
            continue
        candidate = candidates[best_index]
        selected_mask = candidate.materialize(shape)
        store.replace(target_index, frame_index, selected_mask)
        coverages[frame_index] = candidate.coverage
        centroids[frame_index] = candidate.centroid
        embeddings[frame_index] = candidate_embeddings[best_index]
        semantic_scores[frame_index] = semantic_raw[best_index]
        selected_count += 1

    return (
        coverages,
        centroids,
        embeddings,
        semantic_scores,
        component_counts,
        score_margins,
        {
            "candidate_components": len(candidates),
            "discarded_components": len(candidates) - selected_count,
            "multi_component_frames": int(np.count_nonzero(component_counts > 1)),
            "ambiguous_component_frames": ambiguous_frames,
            "component_prototype_frame": candidates[seed_index].frame_index,
        },
        backend,
    )


def _suppress_disconnected_bootstrap(
    accepted: np.ndarray,
    states: list[str],
    reasons: list[str],
    seed_index: int,
    maximum_gap: int,
) -> int:
    """Drop an early false acquisition that is disconnected from the trusted seed."""
    accepted_indices = np.flatnonzero(accepted)
    if not len(accepted_indices):
        return 0
    anchor_position = int(np.argmin(np.abs(accepted_indices - seed_index)))
    canonical_start_position = anchor_position
    while canonical_start_position > 0:
        current = int(accepted_indices[canonical_start_position])
        previous = int(accepted_indices[canonical_start_position - 1])
        if current - previous > maximum_gap:
            break
        canonical_start_position -= 1
    canonical_start = int(accepted_indices[canonical_start_position])
    rejected = accepted_indices[accepted_indices < canonical_start]
    for frame_index in rejected:
        accepted[frame_index] = False
        states[frame_index] = "rejected"
        reasons[frame_index] = "pre-anchor-bootstrap-mismatch"
    states[canonical_start] = "acquired"
    reasons[canonical_start] = "anchor-connected-bootstrap"
    return len(rejected)


def classify_presence_track(
    coverages: np.ndarray,
    centroids: np.ndarray,
    confidences: np.ndarray,
    embeddings: np.ndarray,
    semantic_scores: np.ndarray,
    config: PresenceFilterConfig | None = None,
) -> PresenceTrack:
    """Classify raw masks into tracked/absent/reacquired states.

    The prototype is selected from semantically relevant, temporally supported
    masks. Re-entry requires two consecutive appearance-consistent candidates,
    which prevents one-frame identity jumps after the target leaves the scene.
    """
    cfg = config or PresenceFilterConfig()
    frame_count = len(coverages)
    valid = (
        (coverages >= cfg.min_area)
        & (coverages <= cfg.max_area)
        & (np.linalg.norm(embeddings, axis=1) > 0)
    )
    accepted = np.zeros(frame_count, dtype=bool)
    states = ["absent"] * frame_count
    reasons = ["empty-mask"] * frame_count
    appearance = np.zeros(frame_count, dtype=np.float32)
    semantic = _normalize_scores(semantic_scores.astype(np.float32), valid)
    if not valid.any():
        return PresenceTrack(
            accepted,
            states,
            reasons,
            appearance,
            semantic,
            confidences,
            [],
            {
                "presence_ratio": 0.0,
                "raw_presence_ratio": 0.0,
                "identity_rejections": 0,
                "reacquisitions": 0,
                "appearance_threshold": 1.0,
                "status": "no-candidate",
            },
        )

    temporal_support = np.zeros(frame_count, dtype=np.float32)
    for index in np.flatnonzero(valid):
        neighbors: list[float] = []
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < frame_count and valid[neighbor]:
                neighbors.append(float(embeddings[index] @ embeddings[neighbor]))
        temporal_support[index] = float(np.mean(neighbors)) if neighbors else 0.0
    temporal_norm = _normalize_scores(temporal_support, valid)
    confidence_norm = np.clip((confidences - 0.5) / 0.35, 0.0, 1.0)
    seed_scores = 0.52 * semantic + 0.30 * temporal_norm + 0.18 * confidence_norm
    seed_scores[~valid] = -1
    seed_index = int(np.argmax(seed_scores))
    seed_similarity = embeddings @ embeddings[seed_index]
    prototype_candidates = np.flatnonzero(
        valid & (seed_similarity >= 0.82) & (semantic >= 0.15)
    )
    if not len(prototype_candidates):
        prototype_candidates = np.asarray([seed_index])
    ranked = prototype_candidates[np.argsort(seed_scores[prototype_candidates])[::-1]]
    ranked = ranked[: cfg.prototype_limit]
    prototype = embeddings[ranked].mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    appearance = (embeddings @ prototype).astype(np.float32)
    appearance[~valid] = 0.0
    reference_area = float(np.median(coverages[ranked]))
    prototype_similarities = appearance[ranked]
    appearance_threshold = float(
        np.clip(
            np.percentile(prototype_similarities, 15) - 0.08,
            cfg.min_appearance_similarity,
            cfg.max_appearance_similarity,
        )
    )
    reacquire_threshold = min(0.94, appearance_threshold + cfg.reacquire_margin)
    initial_threshold = max(
        cfg.min_appearance_similarity,
        appearance_threshold - cfg.initial_margin,
    )

    mode = "missing"
    missing_count = cfg.missing_patience
    pending: list[int] = []
    last_centroid: np.ndarray | None = None
    last_area = reference_area
    last_accepted = -1
    reacquisitions = 0
    ever_tracked = False

    for index in range(frame_count):
        if not valid[index]:
            missing_count += 1
            pending.clear()
            states[index] = "absent"
            reasons[index] = "empty-mask" if coverages[index] < cfg.min_area else "invalid-area"
            if missing_count >= cfg.missing_patience:
                mode = "missing"
            continue

        area_ratio = coverages[index] / max(reference_area, 1e-8)
        area_ok = 0.16 <= area_ratio <= 6.0
        active_appearance_threshold = initial_threshold if index <= seed_index else appearance_threshold
        appearance_ok = appearance[index] >= active_appearance_threshold
        confidence_ok = confidences[index] >= cfg.min_sam_confidence
        semantic_ok = semantic[index] >= 0.04
        jump_ok = True
        if last_centroid is not None and index - last_accepted <= cfg.missing_patience:
            jump = float(np.linalg.norm(centroids[index] - last_centroid))
            allowed_jump = max(0.12, 5.0 * np.sqrt(max(last_area, reference_area)))
            jump_ok = jump <= allowed_jump or appearance[index] >= reacquire_threshold

        if mode == "tracking":
            if area_ok and appearance_ok and confidence_ok and semantic_ok and jump_ok:
                accepted[index] = True
                states[index] = "tracked"
                reasons[index] = "accepted"
                missing_count = 0
            else:
                states[index] = "rejected"
                if not appearance_ok:
                    reasons[index] = "appearance-mismatch"
                elif not jump_ok:
                    reasons[index] = "identity-jump"
                elif not area_ok:
                    reasons[index] = "area-shift"
                elif not confidence_ok:
                    reasons[index] = "low-confidence"
                else:
                    reasons[index] = "semantic-mismatch"
                missing_count += 1
                if missing_count >= cfg.missing_patience:
                    mode = "missing"
                    pending.clear()
        else:
            entry_threshold = reacquire_threshold if ever_tracked else initial_threshold
            entry_semantic_threshold = (
                0.12 if ever_tracked else cfg.initial_semantic_threshold
            )
            strong_reentry = (
                area_ok
                and appearance[index] >= entry_threshold
                and confidence_ok
                and semantic[index] >= entry_semantic_threshold
            )
            if strong_reentry:
                if pending and index != pending[-1] + 1:
                    pending.clear()
                if pending:
                    previous_index = pending[-1]
                    candidate_similarity = float(
                        embeddings[index] @ embeddings[previous_index]
                    )
                    candidate_jump = float(
                        np.linalg.norm(centroids[index] - centroids[previous_index])
                    )
                    candidate_area_ratio = coverages[index] / max(
                        coverages[previous_index], 1e-8
                    )
                    pending_consistent = (
                        candidate_similarity >= 0.80
                        and candidate_jump <= 0.16
                        and 0.35 <= candidate_area_ratio <= 2.85
                    )
                    if not pending_consistent:
                        pending.clear()
                pending.append(index)
                states[index] = "reacquiring"
                reasons[index] = "confirming-reentry"
                required_confirmation = (
                    cfg.reacquire_confirmation if ever_tracked else cfg.initial_confirmation
                )
                if len(pending) >= required_confirmation:
                    is_reentry = ever_tracked
                    for pending_index in pending:
                        accepted[pending_index] = True
                        states[pending_index] = "tracked"
                        reasons[pending_index] = "confirmed-reentry"
                    states[index] = "reacquired" if is_reentry else "acquired"
                    mode = "tracking"
                    missing_count = 0
                    pending.clear()
                    reacquisitions += int(is_reentry)
                    ever_tracked = True
            else:
                pending.clear()
                states[index] = "rejected"
                reasons[index] = "reentry-mismatch"

        if accepted[index]:
            ever_tracked = True
            last_centroid = centroids[index]
            last_area = float(coverages[index])
            last_accepted = index

    bootstrap_rejections = _suppress_disconnected_bootstrap(
        accepted,
        states,
        reasons,
        seed_index,
        cfg.missing_patience + 1,
    )
    reacquisitions = sum(
        accepted[index] and states[index] == "reacquired"
        for index in range(frame_count)
    )
    rejected_count = sum(state == "rejected" for state in states)
    validated = np.flatnonzero(accepted & (semantic >= 0.25) & (appearance >= appearance_threshold))
    if len(validated) > 5:
        validation_score = 0.55 * semantic[validated] + 0.45 * appearance[validated]
        chosen: list[int] = []
        for candidate in validated[np.argsort(validation_score)[::-1]]:
            radius = max(1, frame_count // 12)
            if all(abs(int(candidate) - existing) >= radius for existing in chosen):
                chosen.append(int(candidate))
            if len(chosen) == 5:
                break
        validated_anchors = sorted(chosen)
    else:
        validated_anchors = [int(index) for index in validated]

    return PresenceTrack(
        accepted=accepted,
        states=states,
        reasons=reasons,
        appearance_similarity=appearance,
        semantic_similarity=semantic,
        confidence=confidences,
        validated_anchors=validated_anchors,
        summary={
            "presence_ratio": float(accepted.mean()),
            "raw_presence_ratio": float(valid.mean()),
            "identity_rejections": int(rejected_count),
            "reacquisitions": int(reacquisitions),
            "bootstrap_rejections": bootstrap_rejections,
            "appearance_threshold": appearance_threshold,
            "initial_appearance_threshold": initial_threshold,
            "reacquire_appearance_threshold": reacquire_threshold,
            "prototype_frame": int(seed_index),
            "validated_anchor_count": len(validated_anchors),
            "status": "filtered",
        },
    )


def filter_presence_tracks(
    frames: list[Image.Image],
    store: RawMaskStore,
    targets: list[str],
    clip_scorer: ClipScorer,
    progress: Callable[[int, str], None] | None = None,
    config: PresenceFilterConfig | None = None,
) -> PresenceFilterResult:
    cfg = config or PresenceFilterConfig()
    tracks: list[PresenceTrack] = []
    backend = "compact component-level appearance"
    for target_index, target in enumerate(targets):
        if progress:
            progress(
                80 + round(4 * target_index / max(1, len(targets))),
                f"Splitting masks and validating identity {target_index + 1}/{len(targets)}: {target}",
            )
        (
            coverages,
            centroids,
            embeddings,
            semantic_scores,
            component_counts,
            component_score_margins,
            component_summary,
            target_backend,
        ) = _select_component_sequence(
            frames,
            store,
            target_index,
            target,
            clip_scorer,
            cfg,
        )
        backend = target_backend
        track = classify_presence_track(
            coverages,
            centroids,
            store.confidences[target_index],
            embeddings,
            semantic_scores,
            cfg,
        )
        track.component_counts = component_counts
        track.component_score_margins = component_score_margins
        track.summary.update(component_summary)
        track.summary["initial_confirmation_frames"] = cfg.initial_confirmation
        track.summary["single_component_output"] = "enabled"
        tracks.append(track)
    return PresenceFilterResult(store=store, tracks=tracks, targets=targets, backend=backend)
