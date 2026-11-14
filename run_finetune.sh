#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_finetune.sh
# One-time setup in your Python environment: pip install -e .
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

TRAIN_JSONL="${TRAIN_JSONL:-english.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-outputs/slideon-alpha05-neg7}"
DATASET_ID="lightonai/embeddings-fine-tuning-filtered-en"
DATASET_REVISION="3464b0de66da5834f165a7eded49f1cc63d0b5eb"

# 1. Prepare every example, its source, all ten negatives and teacher scores.
# This CPU-only step runs once. Existing data is reused; it must have the new
# source column and at least seven negatives per example.
if [[ ! -f "$TRAIN_JSONL" ]]; then
    "$PYTHON" scripts/finetune/preprocess.py \
        --dataset "$DATASET_ID" --revision "$DATASET_REVISION" \
        --output "$TRAIN_JSONL"
fi

# 2. Fine-tune from the released SAE pinned in the original training config.
# All model/loss/optimizer settings remain in that config. Only data sampling
# changes here. Extra arguments are forwarded, e.g. --resume path/to/checkpoint.
exec "$PYTHON" scripts/finetune/train_slideon.py \
    --train "$TRAIN_JSONL" --output "$SFT_OUTPUT" \
    --source-sampling-alpha 0.5 --sample-negatives 7 \
    "$@"
