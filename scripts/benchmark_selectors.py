from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from groundscope.config import settings
from groundscope.jobs import JobManager
from groundscope.schemas import JobRecord, SelectorName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible four-selector ablation.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    manager = JobManager(settings)
    experiment_id = time.strftime("ablation-%Y%m%d-%H%M%S")
    experiment_dir = settings.runs_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for selector in SelectorName:
        job_id = f"{experiment_id}-{selector.value}"
        job = JobRecord(
            id=job_id,
            filename=args.video.name,
            prompt=args.prompt,
            selector=selector,
            frame_count=args.frames,
        )
        manager.jobs[job_id] = job
        manager.inputs[job_id] = args.video.resolve()
        manager._save(job)
        manager._process(job_id)
        metric_data = json.loads((settings.runs_dir / job_id / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "selector": selector.value,
                "selector_backend": job.selector_backend,
                "inference_seconds": job.inference_seconds,
                "peak_vram_gib": job.peak_vram_gib,
                **metric_data["summary"],
            }
        )

    with (experiment_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (experiment_dir / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"experiment": experiment_id, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
