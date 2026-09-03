from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def segmentation_instruction(target: str) -> str:
    target = target.strip()
    if target.casefold().startswith(("segment ", "please segment ")):
        return target
    return f"Please segment {target.rstrip('.?!')}."


class Sa2VARunner:
    """Lazy, process-local Sa2VA runtime. One instance is shared by the GPU queue."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()
        self.load_seconds: float | None = None
        self._dense_state: dict | None = None
        self._dense_target_count = 0

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self, progress: Callable[[int, str], None] | None = None) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            if not (self.model_path / "config.json").exists():
                raise FileNotFoundError(f"Sa2VA model missing: {self.model_path}")
            if progress:
                progress(43, "Loading Sa2VA-1B onto RTX GPU")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            torch.set_float32_matmul_precision("high")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
            ).eval()
            self.load_seconds = time.perf_counter() - started

    def predict(self, frames: list[Image.Image], prompt: str) -> dict:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.predict_forward(
                video=frames,
                text=f"<image>{segmentation_instruction(prompt)}",
                tokenizer=self.tokenizer,
            )
        return {
            "prediction": str(result.get("prediction", "")),
            "prediction_masks": result.get("prediction_masks") or [],
            "inference_seconds": time.perf_counter() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def _extract_language_embedding(self, frames: list[Image.Image], prompt: str) -> tuple[str, torch.Tensor | None]:
        """Run the MLLM on anchor frames and retain its first [SEG] embedding."""
        assert self.model is not None and self.tokenizer is not None
        grounding_encoder = self.model.grounding_encoder
        original_inference = grounding_encoder.language_embd_inference
        captured: list[torch.Tensor] = []

        def capture(inference_state, language_embeddings):
            if language_embeddings:
                captured.append(language_embeddings[0].detach().cpu().clone())
            return original_inference(inference_state, language_embeddings)

        grounding_encoder.language_embd_inference = capture
        try:
            with torch.inference_mode():
                result = self.model.predict_forward(
                    video=frames,
                    text=f"<image>{segmentation_instruction(prompt)}",
                    tokenizer=self.tokenizer,
                )
        finally:
            grounding_encoder.language_embd_inference = original_inference
        prediction = str(result.get("prediction", ""))
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return prediction, captured[0] if captured else None

    def predict_dense(
        self,
        frames: list[Image.Image],
        anchor_indices: list[int] | dict[int, list[int]],
        targets: list[str],
        on_frame: Callable[[int, list[np.ndarray], list[float]], None],
        progress: Callable[[int, str], None] | None = None,
        retain_state: bool = False,
    ) -> dict:
        """Jointly propagate language targets across every supplied video frame.

        The MLLM sees at most five semantic anchor frames. All active targets are
        then inserted into one SAM2 inference state so image features and temporal
        memory are shared during dense propagation.
        """
        self.load()
        self.release_dense_state()
        assert self.model is not None and self.tokenizer is not None
        if not frames:
            raise ValueError("Dense tracking requires at least one decoded frame")
        if isinstance(anchor_indices, dict):
            target_anchors = {
                target_index: sorted(
                    {
                        max(0, min(len(frames) - 1, index))
                        for index in anchor_indices.get(target_index, [])
                    }
                )[:5]
                for target_index in range(len(targets))
            }
        else:
            shared = sorted(
                {max(0, min(len(frames) - 1, index)) for index in anchor_indices}
            )[:5]
            target_anchors = {target_index: shared for target_index in range(len(targets))}
        for target_index in range(len(targets)):
            if not target_anchors[target_index]:
                target_anchors[target_index] = [0]
        anchors = sorted({index for values in target_anchors.values() for index in values})
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        predictions: dict[str, str] = {}
        embeddings: list[torch.Tensor | None] = []
        for target_index, target in enumerate(targets):
            if progress:
                progress(
                    48 + round(12 * target_index / max(1, len(targets))),
                    f"Grounding target {target_index + 1}/{len(targets)}: {target}",
                )
            target_frames = [frames[index] for index in target_anchors[target_index]]
            prediction, embedding = self._extract_language_embedding(target_frames, target)
            predictions[target] = prediction
            embeddings.append(embedding)

        grounding_encoder = self.model.grounding_encoder
        if progress:
            progress(61, f"Preparing {len(frames)} frames for shared SAM2 memory")
        processed_frames: list[torch.Tensor] = []
        for frame in frames:
            grounding_image = self.model.extra_image_processor.apply_image(np.asarray(frame))
            pixel = torch.from_numpy(grounding_image).permute(2, 0, 1).contiguous()
            processed_frames.append(
                grounding_encoder.preprocess_image(pixel).to(self.model.torch_dtype)
            )
        inference_state = grounding_encoder.get_sam2_embeddings(processed_frames)

        active_target_ids: list[int] = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for target_index, embedding in enumerate(embeddings):
                if embedding is None:
                    continue
                active_target_ids.append(target_index)
                language_embedding = embedding.cuda(non_blocking=True)[0][None][None]
                for frame_index in target_anchors[target_index]:
                    grounding_encoder.sam2_model.add_language_embd(
                        inference_state,
                        frame_index,
                        100 + target_index,
                        language_embedding,
                        inference=True,
                    )

            if active_target_ids:
                first_anchor = min(anchors)
                passes = [
                    grounding_encoder.sam2_model.propagate_in_video(
                        inference_state,
                        start_frame_idx=first_anchor,
                    )
                ]
                last_anchor = max(anchors)
                if last_anchor > 0:
                    passes.append(
                        grounding_encoder.sam2_model.propagate_in_video(
                            inference_state,
                            start_frame_idx=last_anchor,
                            max_frame_num_to_track=last_anchor + 1,
                            reverse=True,
                        )
                    )
                emitted: set[int] = set()
                for propagated in passes:
                    for frame_index, object_ids, logits in propagated:
                        height, width = frames[frame_index].height, frames[frame_index].width
                        resized = F.interpolate(
                            logits.float(), size=(height, width), mode="bilinear", align_corners=False
                        )[:, 0]
                        probabilities = resized.sigmoid()
                        binary_tensor = probabilities > 0.5
                        binary = binary_tensor.cpu().numpy()
                        object_rows = {int(object_id) - 100: row for row, object_id in enumerate(object_ids)}
                        masks: list[np.ndarray] = []
                        confidences: list[float] = []
                        for target_index in range(len(targets)):
                            if target_index not in object_rows:
                                masks.append(np.zeros((height, width), dtype=bool))
                                confidences.append(0.0)
                                continue
                            row = object_rows[target_index]
                            target_binary = binary[row]
                            masks.append(target_binary)
                            target_probabilities = probabilities[row]
                            confidence = (
                                target_probabilities[binary_tensor[row]].mean()
                                if binary_tensor[row].any()
                                else target_probabilities.max()
                            )
                            confidences.append(float(confidence.item()))
                        on_frame(frame_index, masks, confidences)
                        emitted.add(int(frame_index))
                        if progress and len(emitted) % max(1, len(frames) // 20) == 0:
                            progress(
                                64 + round(14 * len(emitted) / max(1, len(frames))),
                                f"Bidirectional SAM2 propagation: {len(emitted)}/{len(frames)} frames",
                            )
            else:
                for frame_index, frame in enumerate(frames):
                    empty = [np.zeros((frame.height, frame.width), dtype=bool) for _ in targets]
                    on_frame(frame_index, empty, [0.0] * len(targets))

        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        if retain_state:
            self._dense_state = inference_state
            self._dense_target_count = len(targets)
            del processed_frames, embeddings
        else:
            del inference_state, processed_frames, embeddings
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "predictions": predictions,
            "inference_seconds": elapsed,
            "peak_vram_gib": peak_vram,
            "tracked_frames": len(frames),
            "anchor_indices": anchors,
            "anchor_indices_by_target": target_anchors,
            "active_targets": active_target_ids,
        }

    def refine_dense_from_masks(
        self,
        frames: list[Image.Image],
        seeds_by_target: dict[int, tuple[int, np.ndarray]],
        on_frame: Callable[[int, list[np.ndarray], list[float]], None],
        progress: Callable[[int, str], None] | None = None,
    ) -> dict[str, float | int]:
        """Correct retained SAM2 memory with trusted masks and propagate backward."""
        if self._dense_state is None:
            raise RuntimeError("Dense SAM2 state was not retained for mask refinement")
        if not seeds_by_target:
            return {
                "inference_seconds": 0.0,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "refined_frames": 0,
            }
        assert self.model is not None
        sam2_model = self.model.grounding_encoder.sam2_model
        processed_frames = self._dense_state["images"]
        started = time.perf_counter()
        if progress:
            progress(81, "Correcting early frames from trusted target masks")
        emitted_frames: set[tuple[int, int]] = set()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Use a clean visual-only state for every target. Reusing the dense
            # language state also reuses its incorrect early conditioning frames,
            # which can overpower a later trusted mask during reverse propagation.
            for target_index, (seed_frame, mask) in seeds_by_target.items():
                inference_state = sam2_model.init_state(processed_frames)
                self._add_mask_prompt_compat(
                    sam2_model,
                    inference_state,
                    int(seed_frame),
                    100 + int(target_index),
                    np.asarray(mask, dtype=bool),
                )
                propagated = sam2_model.propagate_in_video(
                    inference_state,
                    start_frame_idx=int(seed_frame),
                    max_frame_num_to_track=int(seed_frame) + 1,
                    reverse=True,
                )
                for frame_index, object_ids, logits in propagated:
                    height, width = frames[frame_index].height, frames[frame_index].width
                    resized = F.interpolate(
                        logits.float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    probabilities = resized.sigmoid()
                    binary_tensor = probabilities > 0.5
                    binary = binary_tensor.cpu().numpy()
                    object_rows = {
                        int(object_id) - 100: row
                        for row, object_id in enumerate(object_ids)
                    }
                    masks = [
                        np.zeros((height, width), dtype=bool)
                        for _ in range(self._dense_target_count)
                    ]
                    confidences = [0.0] * self._dense_target_count
                    row = object_rows.get(int(target_index))
                    if row is not None:
                        target_binary = binary[row]
                        masks[target_index] = target_binary
                        target_probabilities = probabilities[row]
                        confidence = (
                            target_probabilities[binary_tensor[row]].mean()
                            if binary_tensor[row].any()
                            else target_probabilities.max()
                        )
                        confidences[target_index] = float(confidence.item())
                    on_frame(int(frame_index), masks, confidences)
                    emitted_frames.add((int(target_index), int(frame_index)))
                    if (
                        progress
                        and len(emitted_frames)
                        % max(1, sum(seed + 1 for seed, _ in seeds_by_target.values()) // 8)
                        == 0
                    ):
                        expected = sum(
                            seed + 1 for seed, _ in seeds_by_target.values()
                        )
                        progress(
                            81 + round(len(emitted_frames) / max(1, expected)),
                            f"Clean backward refinement: {len(emitted_frames)}/{expected} target-frames",
                        )
                del inference_state
        return {
            "inference_seconds": time.perf_counter() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "refined_frames": len({frame for _, frame in emitted_frames}),
        }

    @staticmethod
    def _add_mask_prompt_compat(
        sam2_model,
        inference_state: dict,
        frame_index: int,
        object_id: int,
        mask: np.ndarray,
    ) -> None:
        """Inject a mask into upstream SAM2 or Sa2VA's reduced predictor API."""
        if hasattr(sam2_model, "add_new_mask"):
            sam2_model.add_new_mask(inference_state, frame_index, object_id, mask)
            return
        object_index = inference_state["obj_id_to_idx"].get(object_id)
        if object_index is None:
            if inference_state["tracking_has_started"]:
                raise RuntimeError(
                    f"Cannot add object {object_id} after SAM2 tracking started"
                )
            object_index = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][object_id] = object_index
            inference_state["obj_idx_to_id"][object_index] = object_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][object_index] = {}
            inference_state["mask_inputs_per_obj"][object_index] = {}
            inference_state["output_dict_per_obj"][object_index] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][object_index] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32)[None, None]
        mask_tensor = mask_tensor.to(inference_state["device"])
        if mask_tensor.shape[-2:] != (sam2_model.image_size, sam2_model.image_size):
            mask_tensor = F.interpolate(
                mask_tensor,
                size=(sam2_model.image_size, sam2_model.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mask_tensor = (mask_tensor >= 0.5).float()
        inference_state["mask_inputs_per_obj"][object_index][frame_index] = mask_tensor
        inference_state["point_inputs_per_obj"][object_index].pop(frame_index, None)
        is_initial = frame_index not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_initial
            else inference_state["frames_already_tracked"][frame_index]["reverse"]
        )
        object_output = inference_state["output_dict_per_obj"][object_index]
        temporary_output = inference_state["temp_output_dict_per_obj"][object_index]
        is_conditioning = is_initial or sam2_model.add_all_frames_to_correct_as_cond
        storage_key = (
            "cond_frame_outputs" if is_conditioning else "non_cond_frame_outputs"
        )
        current_output, _ = sam2_model._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=object_output,
            frame_idx=frame_index,
            batch_size=1,
            is_init_cond_frame=is_initial,
            point_inputs=None,
            mask_inputs=mask_tensor,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        temporary_output[storage_key][frame_index] = current_output

    def release_dense_state(self) -> None:
        if self._dense_state is not None:
            del self._dense_state
        self._dense_state = None
        self._dense_target_count = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.release_dense_state()
        if self.model is not None:
            self.model.cpu()
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def empty_masks_for(frames: list[Image.Image]) -> list[np.ndarray]:
    return [np.zeros((image.height, image.width), dtype=bool) for image in frames]
