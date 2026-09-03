from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


def main() -> None:
    root = Path(os.getenv("GS_ROOT", "/root/autodl-tmp/groundscope")).resolve()
    model = Path(os.getenv("GS_MODEL_PATH", root / "models" / "Sa2VA-1B")).resolve()
    clip = Path(os.getenv("GS_CLIP_MODEL_PATH", root / "models" / "clip-vit-base-patch32")).resolve()
    disk = shutil.disk_usage(root)
    report = {
        "root": str(root),
        "root_on_data_disk": str(root).startswith("/root/autodl-tmp/"),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "sa2va_model": (model / "model.safetensors").exists(),
        "clip_model": (clip / "pytorch_model.bin").exists(),
    }
    print(json.dumps(report, indent=2))
    if not all((report["root_on_data_disk"], report["cuda_available"], report["sa2va_model"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
