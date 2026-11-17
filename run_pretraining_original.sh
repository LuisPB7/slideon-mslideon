#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_pretraining_original.sh
# One-time setup in your Python environment: pip install -e .
# Edit the settings below, save this file, then run it without arguments.
# These fixed settings override the JSON config; environment variables do not override them.
# This runs the repository's original English SAE recipe, without online JSONL input.
set -euo pipefail

# DEVICE AND WORKFLOW: use true or false for the switches.
PYTHON="python3"
# One GPU: "0" or "2". Multiple GPUs: "0,1" or "0,1,2,3".
# Preparation uses only the first selected GPU. SAE training uses all selected GPUs.
export CUDA_VISIBLE_DEVICES="0"
PREPARE_TOKENS=true
TRAIN_SAE=true

# DATA AND MODEL: the JSON selects DenseOn and all nine English training sources
# from lightonai/embeddings-fine-tuning-filtered-en. No english.jsonl is needed.
# Preparation keeps queries, positives and seven sampled negatives after filtering,
# deduplicates IDs within each source, and writes their frozen token embeddings.
TRAIN_CONFIG="scripts/training-configs/SlideOn-SAE/config.json"
# This is the FULL embedding cache on disk, not the small online text index.
# Each retained token takes 1,536 bytes with the default 768-dimensional FP16 vectors.
# Choose a filesystem with enough space. Completed compatible streams are reused.
CACHE_DIR="data/slideon-original-token-cache"
# Optional JSON file mapping dataset IDs to exact commit SHAs. "" uses the config.
# Otherwise, the first preparation resolves and records revisions in the cache manifest.
DATASET_REVISIONS=""

# OUTPUT: use a fresh directory for each new experiment.
SAE_OUTPUT="outputs/slideon-sae-original"
# Resume takes a FILE, not a directory. Use the same settings and GPU count.
RESUME_CHECKPOINT=""  # e.g. outputs/slideon-sae-original/checkpoint-100000/training_state.pt

# TRAINING: defaults match the existing English SAE JSON configuration.
# The original cached path retains its original token-based epoch/update schedule.
GLOBAL_TOKEN_BATCH_SIZE=4096  # Token vectors across all GPUs per SAE update.
ENCODE_BATCH_SIZE=4          # Whole texts per frozen DenseOn forward during preparation.
BUFFER_TOKENS=4000000        # Approximate in-memory shuffle buffer target during training.
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

# Both stages receive the same token-producing settings, so preparation and
# training cannot accidentally disagree about DenseOn preprocessing or selection.
COMMON_ARGS=(--config "$TRAIN_CONFIG" --cache "$CACHE_DIR"
    --batch-size "$GLOBAL_TOKEN_BATCH_SIZE" --encode-batch-size "$ENCODE_BATCH_SIZE"
    --buffer-tokens "$BUFFER_TOKENS" --query-length "$QUERY_LENGTH"
    --document-length "$DOCUMENT_LENGTH" --learning-rate "$LEARNING_RATE"
    --epochs "$EPOCHS" --seed "$SEED" --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS")
[[ -z "$MAX_STEPS" ]] || COMMON_ARGS+=(--max-steps "$MAX_STEPS")
[[ -z "$DATASET_REVISIONS" ]] || COMMON_ARGS+=(--dataset-revisions "$DATASET_REVISIONS")
if [[ "$BF16" == true ]]; then
    COMMON_ARGS+=(--bf16)
else
    COMMON_ARGS+=(--no-bf16)
fi

# 1. Fetch/filter the authors' data and encode its texts into the reusable cache.
# Keep this single-process even when SAE training will use several GPUs.
# Set PREPARE_TOKENS=false to train directly from an already prepared cache.
if [[ "$PREPARE_TOKENS" == true ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$PYTHON" \
        scripts/sae-pretrain/train_slideon_sae.py "${COMMON_ARGS[@]}" --prepare
fi

# 2. Train the SAE on cached tokens. A failed preparation stops the script above.
# The trainer validates the cache, preserves its original losses/optimizer, and
# writes SAE_OUTPUT/final only after completing the requested training schedule.
if [[ "$TRAIN_SAE" == true ]]; then
    TRAIN_ARGS=("${COMMON_ARGS[@]}" --output "$SAE_OUTPUT")
    [[ -z "$RESUME_CHECKPOINT" ]] || TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
    TRAIN_COMMAND=("$PYTHON")
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
    fi
    "${TRAIN_COMMAND[@]}" scripts/sae-pretrain/train_slideon_sae.py "${TRAIN_ARGS[@]}"

    printf '\nSAE training complete. In run_finetune.sh, set:\nSAE_ID_OR_PATH="%s/final"\n' "$SAE_OUTPUT"
fi

# The final raw SAE is compatible with run_finetune.sh without conversion.
# Keep its BACKBONE_REVISION matched to TRAIN_CONFIG's base_model_revision;
# both currently default to cb9947ebccb33862d24e3c7ca2edb25e51acd887 for DenseOn.
