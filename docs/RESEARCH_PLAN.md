# GroundScope research and execution plan

## Positioning

GroundScope is an interactive, presence-aware dense multi-target open-vocabulary video segmentation and analysis system. Sa2VA is the frozen base model. The research contribution is how language-defined targets are grounded independently, propagated bidirectionally, decomposed into identity-scored mask components, protected from false initial acquisition, and reacquired after verified re-entry without losing object-level provenance.

GPU capacity is an implementation constraint, not the research premise. The RTX 5090 is used to increase analysis density, keep Sa2VA and CLIP resident, and execute multiple target passes quickly.

## Core research question

How can an open-vocabulary video system maintain multiple language-defined objects through temporal change, target absence, scene transitions, and human revisions while preserving object-level provenance?

## Implemented method

1. The user enters up to four free-form target expressions, one per line.
2. CLIP selects up to five temporally separated semantic anchors independently for each target; absent frame zero is no longer forced into every target memory.
3. The persistent Sa2VA runtime extracts one `[SEG]` grounding embedding per target and injects it only at that target's anchors.
4. One shared SAM2 state propagates forward from the earliest anchor and backward from the latest anchor. Per-frame directional candidates are fused by SAM confidence.
5. Raw masks are streamed to the data disk and split into connected components. Every component receives CLIP semantic, appearance, temporal-support, confidence, and shape scores; at most one component is retained per target and frame.
6. Masked component crops form a target appearance prototype. A state machine classifies each frame as absent, acquiring, tracked, rejected, or reacquired, and requires four consistent frames for initial acquisition.
7. An initial trajectory is accepted only when it remains connected to the globally trusted prototype segment. Earlier disconnected false starts are rolled back to empty masks. Re-entry still requires consecutive appearance-consistent candidates.
8. Masks remain separate by target and export state, confidence, appearance similarity, semantic similarity, component count, selection margin, rejection reason, PNGs, H.264 videos, and audit metadata.
9. The visual workspace exposes presence ratio, rejected identity jumps, discarded components, false-bootstrap rejections, reacquisitions, dense playback, artifacts, and target-aware editing.

Temporal localisation remains a useful supporting module for long and edited videos. It should be described as semantic localisation, not as a method whose purpose is merely to save GPU memory.

## Evaluation matrix

Evaluate three capabilities separately:

| Capability | Suggested data | Primary measures |
|---|---|---|
| Single-target segmentation | DAVIS / Ref-DAVIS | J, F, J&F |
| Referring multi-object segmentation | MeViS / ReVOS | J&F, per-target failure rate |
| Interactive correction | curated difficult clips | clicks, edit time, downstream improvement |

GroundScope's coverage, empty ratio, area stability, and unwarped temporal IoU are diagnostics. They must not be described as ground-truth segmentation accuracy.

## Experiments

- One, two, three, and four simultaneous targets on the same video.
- Analysis resolutions 8, 16, 24, 32, and 48 frames.
- Uniform, motion, query, and hybrid temporal localisation.
- Per-target Sa2VA output versus a joint multi-object prompt.
- Target absent, reappearing, occluded, visually similar, and scene-cut cases.
- Presence filter ablation: no filter, confidence only, appearance only, and full state machine.
- Component ablation: union-mask filtering versus single-component selection versus component selection plus trusted-bootstrap suppression.
- Forward-only versus confidence-fused bidirectional propagation.
- Automatic output versus target-specific corrected output.
- Sa2VA-1B versus a larger checkpoint if later available.

Report mean and 95% bootstrap confidence intervals over videos. Keep target identity switches, hallucinated masks, empty masks, and edit counts as separate failure categories.

## Portfolio milestones

1. **Application:** stable multi-target upload → queue → Sa2VA grounding → dense SAM2 propagation → colored artifacts path. **Complete.**
2. **Object provenance:** independent target masks, responses, presence states, re-identification metrics, and correction log. **Complete.**
3. **Dataset evaluation:** official J/F metrics on one public referring-video dataset.
4. **Interactive study:** clicks and edit time for 15–30 difficult clips.
5. **Communication:** 90-second demo, architecture figure, multi-target qualitative grid, failure taxonomy, and technical report.

## Honest claims

- Say “built on” or “integrated” Sa2VA, not “developed Sa2VA.”
- Describe CLIP/motion selection as temporal localisation.
- Say “temporal consistency proxy” for unwarped mask IoU.
- Keep system timing separate from benchmark accuracy.
- Never report diagnostic scores as dataset accuracy.

## Strong next research extension

Benchmark presence-aware tracking on target-absence and re-entry subsets, then connect each browser edit to the dense SAM2 state so one correction triggers controlled neighbouring-frame re-propagation. Evaluate J&F, identity switches, false-positive masks during absence, reacquisition delay, improvement per click, and incorrect-propagation rate.
