#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_pretraining.sh
# One-time setup in your Python environment: pip install -e .
# Edit the settings below, save this file, then run: bash run_pretraining.sh
# These values are fixed for each run; environment variables do not override them.
set -euo pipefail

# DEVICE AND WORKFLOW: use true or false for the training switch.
PYTHON="python3"
# One GPU: "0" or "2". Multiple GPUs: "0,1" or "0,1,2,3".
# Each selected GPU runs frozen DenseOn and trains its own SAE replica.
# An empty string selects CPU only.
export CUDA_VISIBLE_DEVICES="0"
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
# Online training requires at least one file. Texts are shuffled during training.

# MODEL AND OUTPUT: relative paths are resolved from this repository.
# Architecture, DenseOn revision, prompts and other SAE settings remain in JSON.
TRAIN_CONFIG="scripts/training-configs/SlideOn-SAE/config.json"
# This small index stores offsets/counts/provenance, never dense token embeddings.
INDEX_DIR="data/slideon-text-index"
SAE_OUTPUT="outputs/slideon-sae-texts"
# Use a fresh output for a new run. Resume points to a file, not its directory.
RESUME_CHECKPOINT=""  # e.g. outputs/slideon-sae-texts/checkpoint-100000/training_state.pt

# TRAINING: SAE/optimizer defaults match the existing English configuration.
# This launcher's 192-token limits override the JSON's longer context limits.
# The batch counts TOKEN VECTORS globally, not queries/documents per GPU.
GLOBAL_TOKEN_BATCH_SIZE=4096
LEARNING_RATE=0.0003
EPOCHS=3
MAX_STEPS=""  # "" uses the full epoch schedule; a positive integer caps its updates.
QUERY_LENGTH=192
DOCUMENT_LENGTH=192
BF16=true
SEED=42
# Whole texts in each DenseOn forward PER GPU, independent of the SAE token batch.
# Reduce this if encoder memory is insufficient; it is not an extra microbatch knob.
ENCODE_BATCH_SIZE=512
SAVE_STEPS=100000
LOG_STEPS=100

# Settings end here. Training encodes texts online and keeps token vectors in RAM.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if (( $# > 0 )); then
    printf '%s\n' 'Edit the settings at the top of run_pretraining.sh, then run it without arguments.' >&2
    exit 2
fi

# The compact text index must be ready before distributed workers start, because
# indexing a large corpus can exceed their synchronization timeout. Single-GPU
# training builds it directly. The global token batch must divide across workers.
if [[ "$TRAIN_SAE" == true ]]; then
    if (( ${#TEXTS_JSONL[@]} == 0 )); then
        printf '%s\n' 'Set TEXTS_JSONL to at least one input file for online SAE training.' >&2
        exit 2
    fi
    TRAIN_ARGS=(--config "$TRAIN_CONFIG" --online --texts-jsonl "${TEXTS_JSONL[@]}"
        --index-dir "$INDEX_DIR" --output "$SAE_OUTPUT"
        --batch-size "$GLOBAL_TOKEN_BATCH_SIZE" --learning-rate "$LEARNING_RATE"
        --epochs "$EPOCHS" --query-length "$QUERY_LENGTH" --document-length "$DOCUMENT_LENGTH"
        --seed "$SEED" --encode-batch-size "$ENCODE_BATCH_SIZE"
        --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS")
    [[ -z "$MAX_STEPS" ]] || TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
    if [[ "$BF16" == true ]]; then
        TRAIN_ARGS+=(--bf16)
    else
        TRAIN_ARGS+=(--no-bf16)
    fi
    [[ -z "$RESUME_CHECKPOINT" ]] || TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
    TRAIN_COMMAND=("$PYTHON")
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        # Tokenizer/index work only: no DenseOn model or GPU allocation. A failure
        # stops this launcher before torchrun; a compatible index is reused.
        CUDA_VISIBLE_DEVICES="" "$PYTHON" scripts/sae-pretrain/train_slideon_sae.py \
            "${TRAIN_ARGS[@]}" --index-only
        TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
    fi
    "${TRAIN_COMMAND[@]}" scripts/sae-pretrain/train_slideon_sae.py "${TRAIN_ARGS[@]}"
fi

# A completed run writes SAE_OUTPUT/final. To fine-tune that SAE, edit
# run_finetune.sh: SAE_ID_OR_PATH="outputs/slideon-sae-texts/final" (or your path).
# Its BACKBONE_REVISION must match TRAIN_CONFIG's base_model_revision; both
# currently default to cb9947ebccb33862d24e3c7ca2edb25e51acd887 for DenseOn.
