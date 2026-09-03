import numpy as np
from PIL import Image

from groundscope.inference.tracking import (
    RawMaskStore,
    _suppress_disconnected_bootstrap,
    classify_presence_track,
    filter_presence_tracks,
)


class _ColorClipScorer:
    available = True

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for image in images:
            rgb = np.asarray(image, dtype=np.float32)
            red = max(float(rgb[..., 0].mean() - rgb[..., 2].mean()), 0.0) + 1e-4
            blue = max(float(rgb[..., 2].mean() - rgb[..., 0].mean()), 0.0) + 1e-4
            vector = np.asarray([red, blue], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)

    def embed_texts(self, prompts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(prompts), 1))


def test_presence_filter_rejects_identity_switch_and_confirms_reentry() -> None:
    frame_count = 12
    coverages = np.full(frame_count, 0.01, dtype=np.float32)
    coverages[4:6] = 0
    centroids = np.column_stack(
        (np.linspace(0.2, 0.5, frame_count), np.full(frame_count, 0.5))
    ).astype(np.float32)
    confidences = np.full(frame_count, 0.9, dtype=np.float32)
    embeddings = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (frame_count, 1))
    embeddings[4:6] = 0
    embeddings[6] = np.asarray([0.0, 1.0], dtype=np.float32)
    semantic = np.full(frame_count, 0.9, dtype=np.float32)
    semantic[4:6] = 0
    semantic[6] = 0.8

    track = classify_presence_track(
        coverages,
        centroids,
        confidences,
        embeddings,
        semantic,
    )

    assert track.accepted[0]
    assert not track.accepted[4]
    assert not track.accepted[6]
    assert track.reasons[6] == "reentry-mismatch"
    assert track.accepted[7]
    assert track.accepted[8]
    assert track.states[8] == "reacquired"
    assert track.summary["reacquisitions"] == 1
    assert int(track.summary["identity_rejections"]) >= 1


def test_presence_filter_returns_absent_without_candidates() -> None:
    track = classify_presence_track(
        np.zeros(5, dtype=np.float32),
        np.zeros((5, 2), dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.zeros((5, 2), dtype=np.float32),
        np.zeros(5, dtype=np.float32),
    )
    assert not track.accepted.any()
    assert track.summary["status"] == "no-candidate"


def test_component_filter_rejects_wrong_start_and_keeps_one_region(tmp_path) -> None:
    frame_count = 8
    frames: list[Image.Image] = []
    store = RawMaskStore(tmp_path, frame_count, 1)
    for frame_index in range(frame_count):
        rgb = np.full((64, 64, 3), 114, dtype=np.uint8)
        rgb[14:50, 8:22] = np.asarray([235, 25, 25], dtype=np.uint8)
        rgb[12:52, 42:57] = np.asarray([25, 25, 235], dtype=np.uint8)
        frames.append(Image.fromarray(rgb))
        mask = np.zeros((64, 64), dtype=bool)
        if frame_index >= 2:
            mask[14:50, 8:22] = True
        mask[12:52, 42:57] = True
        store.write(frame_index, [mask], [0.9])

    result = filter_presence_tracks(
        frames,
        store,
        ["the red person"],
        _ColorClipScorer(),  # type: ignore[arg-type]
    )
    track = result.tracks[0]

    assert not track.accepted[:2].any()
    assert track.accepted[2:].all()
    assert track.summary["multi_component_frames"] == 6
    assert int(track.summary["discarded_components"]) >= 8
    assert track.summary["single_component_output"] == "enabled"
    selected = result.masks_for_frame(4, (64, 64))[0]
    assert selected[20, 12]
    assert not selected[20, 48]
    assert result.filter_version == "component-reid-v2"


def test_bootstrap_filter_removes_segment_disconnected_from_seed() -> None:
    accepted = np.asarray(
        [True, True, True, False, False, False, True, True, True, True],
        dtype=bool,
    )
    states = ["tracked" if value else "absent" for value in accepted]
    reasons = ["accepted" if value else "empty-mask" for value in accepted]

    rejected = _suppress_disconnected_bootstrap(
        accepted,
        states,
        reasons,
        seed_index=8,
        maximum_gap=3,
    )

    assert rejected == 3
    assert not accepted[:6].any()
    assert accepted[6:].all()
    assert states[6] == "acquired"
    assert reasons[0] == "pre-anchor-bootstrap-mismatch"
