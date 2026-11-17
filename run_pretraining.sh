#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_pretraining.sh
# One-time setup in your Python environment: pip install -e .
# Edit the settings below, save this file, then run: bash run_pretraining.sh
# These values are fixed for each run; environment variables do not override them.
set -euo pipefail

# DEVICE AND WORKFLOW: use true or false for the switches.
PYTHON="python3"
# One GPU: "0" or "2". Multiple GPUs: "0,1" or "0,1,2,3".
# Preparation uses only the first selected GPU; SAE training uses the full list.
# An empty string selects CPU only.
export CUDA_VISIBLE_DEVICES="0"
PREPARE_TOKENS=true
TRAIN_SAE=true

# DATA: list the raw-text JSONL files in their intended reading order.
# Each row requires source/text; id and role (query/document; default document) are optional.
# Different files may share a source. This is not the fine-tuning english.jsonl.
TEXTS_JSONL=(
    "data/msmarco-v1.jsonl"
    "data/msmarco-v2.jsonl"
    "data/fineweb.jsonl"
    # Uncomment any generated query parts you want to include:
    # "data/msmarco-v1-artificial-queries-0.jsonl"
    # "data/msmarco-v1-artificial-queries-1.jsonl"
    # "data/msmarco-v1-artificial-queries-2.jsonl"
    # "data/msmarco-v1-artificial-queries-3.jsonl"
    # "data/msmarco-v1-artificial-queries-4.jsonl"
)
# Set TEXTS_JSONL=() to prepare from the original authors' HF data instead.
# With PREPARE_TOKENS=false, training reads only the existing CACHE_DIR.

# MODEL AND OUTPUT: relative paths are resolved from this repository.
# Architecture, DenseOn revision, prompts and other SAE settings remain in JSON.
TRAIN_CONFIG="scripts/training-configs/SlideOn-SAE/config.json"
CACHE_DIR="token-cache/slideon-texts"
SAE_OUTPUT="outputs/slideon-sae-texts"
# Use a fresh output for a new run. Resume points to a file, not its directory.
RESUME_CHECKPOINT=""  # e.g. outputs/slideon-sae-texts/checkpoint-100000/training_state.pt

# TRAINING: these values match the existing English SAE configuration.
# The batch counts TOKEN VECTORS globally, not queries/documents per GPU.
GLOBAL_TOKEN_BATCH_SIZE=4096
LEARNING_RATE=0.0003
EPOCHS=3
MAX_STEPS=""  # "" keeps the JSON default (uncapped); a positive integer caps the schedule.
QUERY_LENGTH=8192
DOCUMENT_LENGTH=8192
BF16=true
SEED=42
ENCODE_BATCH_SIZE=4  # Whole texts per frozen-backbone batch during preparation.
BUFFER_TOKENS=4000000
SAVE_STEPS=100000
LOG_STEPS=100

# Settings end here. Preparation writes the token cache; training consumes it.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if (( $# > 0 )); then
    printf '%s\n' 'Edit the settings at the top of run_pretraining.sh, then run it without arguments.' >&2
    exit 2
fi

COMMON_ARGS=(--config "$TRAIN_CONFIG" --cache "$CACHE_DIR"
    --batch-size "$GLOBAL_TOKEN_BATCH_SIZE" --learning-rate "$LEARNING_RATE"
    --epochs "$EPOCHS" --query-length "$QUERY_LENGTH" --document-length "$DOCUMENT_LENGTH"
    --seed "$SEED" --encode-batch-size "$ENCODE_BATCH_SIZE" --buffer-tokens "$BUFFER_TOKENS"
    --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS")
[[ -z "$MAX_STEPS" ]] || COMMON_ARGS+=(--max-steps "$MAX_STEPS")
if [[ "$BF16" == true ]]; then
    COMMON_ARGS+=(--bf16)
else
    COMMON_ARGS+=(--no-bf16)
fi

# 1. Encode raw texts once with frozen DenseOn, retaining the reusable token cache.
# The Python preparation path checks compatibility before reusing cached streams.
if [[ "$PREPARE_TOKENS" == true ]]; then
    PREPARE_ARGS=(--prepare "${COMMON_ARGS[@]}")
    if (( ${#TEXTS_JSONL[@]} > 0 )); then
        PREPARE_ARGS+=(--texts-jsonl "${TEXTS_JSONL[@]}")
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$PYTHON" \
        scripts/sae-pretrain/train_slideon_sae.py "${PREPARE_ARGS[@]}"
fi

# 2. Learn the SAE from cached token vectors. Raw text files are not read here.
# set -e above prevents training after failed preparation. The global token batch
# must divide evenly across the GPU workers; the trainer validates this.
if [[ "$TRAIN_SAE" == true ]]; then
    TRAIN_ARGS=("${COMMON_ARGS[@]}" --output "$SAE_OUTPUT")
    [[ -z "$RESUME_CHECKPOINT" ]] || TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
    TRAIN_COMMAND=("$PYTHON")
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
    fi
    "${TRAIN_COMMAND[@]}" scripts/sae-pretrain/train_slideon_sae.py "${TRAIN_ARGS[@]}"
fi

# A completed run writes SAE_OUTPUT/final. To fine-tune that SAE, edit
# run_finetune.sh: SAE_ID_OR_PATH="outputs/slideon-sae-texts/final" (or your path).
# Its BACKBONE_REVISION must match TRAIN_CONFIG's base_model_revision; both
# currently default to cb9947ebccb33862d24e3c7ca2edb25e51acd887 for DenseOn.
