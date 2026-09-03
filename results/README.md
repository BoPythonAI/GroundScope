# GroundScope result tables

These lightweight tables preserve the aggregate diagnostics used by the public
project report without publishing uploaded videos, per-frame masks, or model
weights.

## System diagnostics

| Version | Main change | Targets | Frames | Inference | Peak VRAM |
|:---|:---|---:|---:|---:|---:|
| v0.3 | Sa2VA-1B + forward SAM2 | 2 | 1,253 / 1,253 | 87.55 s | 4.97 GiB |
| v0.4 | Bidirectional SAM2 + presence re-ID | 1 | 476 / 476 | 36.89 s | 3.28 GiB |
| v0.5 | Component re-ID + bootstrap suppression | 1 | 476 / 476 | 37.17 s | 3.29 GiB |
| v0.6 | Clean trusted-mask backward replay | 1 | 476 / 476 | 36.88 s | 3.42 GiB |

The first row is a two-target, 52.2-second Sintel run; the later rows are
single-target regressions on a 19.0-second crowded-pedestrian clip. Timing
differences between these rows are therefore not direct speed comparisons.

## Presence-aware re-identification regression

| Filter | Raw presence | Accepted presence | Identity rejections | Candidate components | Discarded | Backward-refined frames | Temporal IoU |
|:---|---:|---:|---:|---:|---:|---:|---:|
| presence-reid-v1 | 85.1% | 41.8% | 206 | — | — | 0 | 0.895 |
| component-reid-v2 | 71.8% | 30.7% | 194 | 600 | 252 | 0 | 0.952 |
| mask-refine-v3 | 64.5% | 43.3% | 102 | 357 | 49 | 92 | 0.938 |

These values describe the system's own mask continuity and rejection behavior;
they are not benchmark accuracy scores. Ground-truth video annotations are
still required before claiming an accuracy improvement.

## Temporal selector ablation

| Selector | Backend | Inference | Mean coverage | Area stability | Temporal IoU | Empty frames |
|:---|:---|---:|---:|---:|---:|---:|
| Uniform | deterministic | 1.295 s | 3.74% | 0.962 | 0.017 | 50% |
| Motion | visual motion | 0.437 s | 8.78% | 0.911 | 0.013 | 25% |
| Query | CLIP ViT-B/32 | 0.372 s | 9.10% | 0.931 | 0.051 | 25% |
| Hybrid | CLIP ViT-B/32 | 0.374 s | 6.95% | 0.914 | 0.002 | 25% |

The four selectors use the same frozen model and four sampled frames from the
same source clip. This experiment studies sampling behavior, not segmentation
accuracy.

## Reproducibility anchors

- Hardware: NVIDIA GeForce RTX 5090, 32 GB, compute capability 12.0.
- Sa2VA source commit: `19de49b`.
- Sa2VA-1B model revision: `82faf06c93f6ce3fdc0ad3d45b57fd52c463daeb`.
- CLIP ViT-B/32 revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f02688`.
- Precision: BF16.
- Public snapshot test result: 12 tests passed.
