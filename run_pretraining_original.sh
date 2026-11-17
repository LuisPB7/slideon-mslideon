#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_pretraining_original.sh
# One-time setup in your Python environment: pip install -e .
# Edit the settings below, save this file, then run it without arguments.
# These fixed settings override the JSON config; environment variables do not override them.
# This uses the original English SAE text selection with ONLINE frozen encoding.
# Dense token embeddings are never saved to disk.
set -euo pipefail

# DEVICE AND WORKFLOW: use true or false for the switches.
PYTHON="python3"
# One GPU: "0" or "2". Multiple GPUs: "0,1" or "0,1,2,3".
# Export uses CPU only. Each training GPU runs frozen DenseOn and an SAE replica.
export CUDA_VISIBLE_DEVICES="0"
PREPARE_TEXTS=true
TRAIN_SAE=true

# DATA AND MODEL: the JSON selects DenseOn and all nine English training sources
# from lightonai/embeddings-fine-tuning-filtered-en. No english.jsonl is needed.
# Export keeps queries, positives and seven sampled negatives after filtering,
# deduplicates IDs within each source, and writes raw text with source/role labels.
TRAIN_CONFIG="scripts/training-configs/SlideOn-SAE/config.json"
TEXTS_JSONL="data/slideon-original-texts.jsonl"
# Only text offsets and token counts are stored here, never DenseOn embeddings.
INDEX_DIR="data/slideon-original-text-index"
# Export-only pins: use an exact HF commit, or a JSON mapping dataset IDs to commits.
# Keep dataset_revisions={} in TRAIN_CONFIG, since training consumes local JSONL.
# With no pin, the first export resolves main and records its immutable revision;
# later launches reuse that verified export instead of refreshing its contents.
DATASET_REVISION=""
DATASET_REVISIONS=""

# OUTPUT: use a fresh directory for each new experiment.
SAE_OUTPUT="outputs/slideon-sae-original-online"
# Resume takes a FILE, not a directory. Use the same settings and GPU count.
RESUME_CHECKPOINT=""  # e.g. outputs/slideon-sae-original-online/checkpoint-100000/training_state.pt

# TRAINING: model/loss/optimizer defaults match the original English SAE config.
# Online epochs visit every exported text once. Token order and the exact update
# budget differ from the original cached path's approximate raw-token schedule.
GLOBAL_TOKEN_BATCH_SIZE=4096  # Token vectors across all GPUs per SAE update.
ENCODE_BATCH_SIZE=4          # Whole texts per DenseOn forward PER GPU, now during training.
# Start conservatively at the original 8K limits. Increase encoding batch if it fits.
QUERY_LENGTH=8192
DOCUMENT_LENGTH=8192
LEARNING_RATE=0.0003
EPOCHS=3
MAX_STEPS=""  # "" uses the configured schedule; a positive integer caps its updates.
BF16=true
SEED=42
SAVE_STEPS=100000
LOG_STEPS=100
# Vocabulary, token Top-K, AuxK, initialization and other optimizer settings stay in JSON.

# Settings end here. Paths below are resolved relative to this repository.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if (( $# > 0 )); then
    printf '%s\n' 'Edit the settings at the top of run_pretraining_original.sh, then run it without arguments.' >&2
    exit 2
fi

# 1. Export the authors' selected texts without tokenization or a DenseOn forward.
# Compatible completed JSONL is checksum-verified and reused; source downloads
# are needed only for a new export. This step never creates an embedding cache.
if [[ "$PREPARE_TEXTS" == true ]]; then
    EXPORT_ARGS=(--config "$TRAIN_CONFIG" --output "$TEXTS_JSONL" --seed "$SEED")
    [[ -z "$DATASET_REVISION" ]] || EXPORT_ARGS+=(--revision "$DATASET_REVISION")
    [[ -z "$DATASET_REVISIONS" ]] || EXPORT_ARGS+=(--dataset-revisions "$DATASET_REVISIONS")
    CUDA_VISIBLE_DEVICES="" "$PYTHON" scripts/sae-pretrain/export_original_texts.py "${EXPORT_ARGS[@]}"
fi

# 2. Use the same online provider as run_pretraining.sh: frozen DenseOn produces
# a bounded pool of shuffled token vectors, consumed in fixed-size SAE batches.
if [[ "$TRAIN_SAE" == true ]]; then
    TRAIN_ARGS=(--config "$TRAIN_CONFIG" --online --texts-jsonl "$TEXTS_JSONL"
        --index-dir "$INDEX_DIR" --output "$SAE_OUTPUT"
        --batch-size "$GLOBAL_TOKEN_BATCH_SIZE" --encode-batch-size "$ENCODE_BATCH_SIZE"
        --query-length "$QUERY_LENGTH" --document-length "$DOCUMENT_LENGTH"
        --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS" --seed "$SEED"
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
        # Finish the CPU index scan before workers start, avoiding synchronization
        # timeouts on large JSONL files. No DenseOn weights are loaded here.
        CUDA_VISIBLE_DEVICES="" "$PYTHON" scripts/sae-pretrain/train_slideon_sae.py \
            "${TRAIN_ARGS[@]}" --index-only
        TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
    fi
    "${TRAIN_COMMAND[@]}" scripts/sae-pretrain/train_slideon_sae.py "${TRAIN_ARGS[@]}"

    printf '\nSAE training complete. In run_finetune.sh, set:\nSAE_ID_OR_PATH="%s/final"\n' "$SAE_OUTPUT"
fi

# The final raw SAE is compatible with run_finetune.sh without conversion.
# Keep its BACKBONE_REVISION matched to TRAIN_CONFIG's base_model_revision;
# both currently default to cb9947ebccb33862d24e3c7ca2edb25e51acd887 for DenseOn.
