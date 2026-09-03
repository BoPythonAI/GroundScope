from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    project_root: Path
    model_path: Path
    clip_model_path: Path
    runs_dir: Path
    uploads_dir: Path
    frontend_dist: Path
    max_upload_gib: float
    max_candidate_frames: int
    max_dense_frames: int
    dense_max_side: int

    @classmethod
    def from_env(cls) -> Settings:
        project_root = Path(__file__).resolve().parents[1]
        root = Path(os.getenv("GS_ROOT", "/root/autodl-tmp/groundscope")).resolve()
        return cls(
            root=root,
            project_root=project_root,
            model_path=Path(os.getenv("GS_MODEL_PATH", root / "models" / "Sa2VA-1B")).resolve(),
            clip_model_path=Path(
                os.getenv("GS_CLIP_MODEL_PATH", root / "models" / "clip-vit-base-patch32")
            ).resolve(),
            runs_dir=Path(os.getenv("GS_RUNS_DIR", root / "runs")).resolve(),
            uploads_dir=Path(os.getenv("GS_UPLOADS_DIR", root / "data" / "uploads")).resolve(),
            frontend_dist=Path(os.getenv("GS_FRONTEND_DIST", project_root / "frontend" / "dist")).resolve(),
            max_upload_gib=float(os.getenv("GS_MAX_UPLOAD_GIB", "2")),
            max_candidate_frames=int(os.getenv("GS_MAX_CANDIDATES", "48")),
            max_dense_frames=int(os.getenv("GS_MAX_DENSE_FRAMES", "3600")),
            dense_max_side=int(os.getenv("GS_DENSE_MAX_SIDE", "960")),
        )

    def ensure_directories(self) -> None:
        for path in (self.runs_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
