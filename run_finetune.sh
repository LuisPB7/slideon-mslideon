#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_finetune.sh
# One-time setup in your Python environment: pip install -e '.[eval]'
# Single GPU (default: GPU 0): CUDA_VISIBLE_DEVICES=2 bash run_finetune.sh
# Multiple GPUs on one node: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_finetune.sh
# Preprocessing runs once; evaluation uses the first selected GPU after training.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

# Keep the training and evaluation paths in agreement. Use the environment
# variable below to relocate a run, rather than overriding only the trainer.
for argument in "$@"; do
    case "$argument" in
        -h|--hel|--help) exec "$PYTHON" scripts/finetune/train_slideon.py "$@" ;;
        --ou*) printf '%s\n' 'Set SFT_OUTPUT to change the output directory; do not pass --output to this launcher.' >&2; exit 2 ;;
    esac
done

TRAIN_JSONL="${TRAIN_JSONL:-english.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-outputs/slideon-alpha05-neg7}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$SFT_OUTPUT/beir-q64-d512}"
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
# A list of visible GPUs selects one distributed worker per available GPU.
# Use the active Python environment for torchrun too. The trainer derives
# accumulation from GPU count to preserve the configured global batch size.
TRAIN_COMMAND=("$PYTHON")
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
fi
"${TRAIN_COMMAND[@]}" scripts/finetune/train_slideon.py \
    --train "$TRAIN_JSONL" --output "$SFT_OUTPUT" \
    --source-sampling-alpha 0.5 --sample-negatives 7 \
    "$@"

# 3. Evaluate the just-trained local model on BEIR-15 with q64/d512 caps.
# set -e above prevents this step from running after failed training.
# torchrun has finished, so evaluation runs once outside the distributed job.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$PYTHON" scripts/eval/evaluate.py \
    --model "$SFT_OUTPUT/final" --trust-remote-code --suite beir \
    --query-max-active-dims 64 --document-max-active-dims 512 \
    --output "$EVAL_OUTPUT"
