#!/usr/bin/env bash
set -euo pipefail

export GS_ROOT="${GS_ROOT:-/root/autodl-tmp/groundscope}"
export GS_MODEL_PATH="${GS_MODEL_PATH:-$GS_ROOT/models/Sa2VA-1B}"
export GS_CLIP_MODEL_PATH="${GS_CLIP_MODEL_PATH:-$GS_ROOT/models/clip-vit-base-patch32}"
export GS_RUNS_DIR="${GS_RUNS_DIR:-$GS_ROOT/runs}"
export GS_UPLOADS_DIR="${GS_UPLOADS_DIR:-$GS_ROOT/data/uploads}"
export HF_HOME="${HF_HOME:-$GS_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$GS_ROOT/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$GS_ROOT/cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GS_ROOT/cache/uv}"
export TMPDIR="${TMPDIR:-$GS_ROOT/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GS_BIN_DIR="${GS_BIN_DIR:-$(dirname "$GS_ROOT")/bin}"
export PATH="$GS_BIN_DIR:$PATH"

mkdir -p "$GS_RUNS_DIR" "$GS_UPLOADS_DIR" "$TMPDIR"
