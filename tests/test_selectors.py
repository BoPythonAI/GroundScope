from pathlib import Path

import numpy as np
from PIL import Image

from groundscope.inference.sa2va import segmentation_instruction
from groundscope.inference.selectors import (
    CandidateFrame,
    ClipScorer,
    select_frames,
    select_target_anchors,
)
from groundscope.jobs import choose_dense_anchors, parse_targets


def candidates(count: int = 10) -> list[CandidateFrame]:
    return [
        CandidateFrame(
            source_index=index * 10,
            timestamp=float(index),
            image=Image.new("RGB", (32, 32), (index * 10, 20, 30)),
            feature=np.eye(count, dtype=np.float32)[index],
            motion_score=index / (count - 1),
        )
        for index in range(count)
    ]


def test_uniform_selector_is_deterministic() -> None:
    result = select_frames(candidates(), 4, "uniform", "person", ClipScorer(Path("missing")))
    assert [item.source_index for item in result.frames] == [0, 30, 60, 90]


def test_hybrid_falls_back_without_claiming_semantics() -> None:
    result = select_frames(candidates(), 4, "hybrid", "person", ClipScorer(Path("missing")))
    assert len(result.frames) == 4
    assert "CLIP unavailable" in result.backend
    assert [item.source_index for item in result.frames] == sorted(item.source_index for item in result.frames)


def test_target_parser_supports_multiline_and_deduplicates() -> None:
    assert parse_targets("the woman\nthe dragon\nThe Woman") == ["the woman", "the dragon"]


def test_dense_anchors_include_frame_zero_and_strongest_frames() -> None:
    items = candidates(8)
    for index, item in enumerate(items):
        item.selection_score = float(index)
    assert choose_dense_anchors(items, maximum=5) == [0, 40, 50, 60, 70]


def test_segmentation_instruction_is_explicit_and_idempotent() -> None:
    assert segmentation_instruction("the woman") == "Please segment the woman."
    assert segmentation_instruction("Please segment the dragon.") == "Please segment the dragon."


def test_target_anchors_are_selected_independently() -> None:
    items = candidates(6)

    class FakeClip:
        available = True

        def score_prompts(self, frames, prompts):
            embeddings = np.eye(len(frames), dtype=np.float32)
            relevance = np.zeros((len(frames), len(prompts)), dtype=np.float32)
            relevance[1, 0] = 1
            relevance[4, 1] = 1
            return embeddings, relevance

    anchors = select_target_anchors(items, ["first", "second"], FakeClip(), maximum=1)
    assert anchors[0] == [items[1].source_index]
    assert anchors[1] == [items[4].source_index]
