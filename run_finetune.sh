#!/usr/bin/env bash
# Run from any directory: bash /path/to/slideon/run_finetune.sh
# One-time setup in your Python environment: pip install -e '.[eval]'
# Edit the settings below, save this file, then run: bash run_finetune.sh
# These values are fixed for each run; environment variables do not override them.
set -euo pipefail

# DEVICE AND WORKFLOW: use true or false for the switches.
PYTHON="python3"
# One GPU: "0" or "2". Multiple GPUs: "0,1" or "0,1,2,3".
# Multiple GPUs automatically use torchrun; evaluation uses the first GPU only.
export CUDA_VISIBLE_DEVICES="0"
PREPARE_DATA_IF_MISSING=true
RUN_EVALUATION=true

# DATA: existing JSONL is reused. Preparation preserves all ten negatives,
# their teacher scores and source labels; sampling happens during training.
TRAIN_JSONL="english.jsonl"
DATASET_ID="lightonai/embeddings-fine-tuning-filtered-en"
DATASET_REVISION="3464b0de66da5834f165a7eded49f1cc63d0b5eb"
SOURCE_SAMPLING_ALPHA=0.5  # "" disables source sampling; 0 means uniform sources.
SAMPLE_NEGATIVES=7        # "" uses all negatives stored in the JSONL.

# MODEL AND OUTPUT: relative paths are resolved from this repository.
TRAIN_CONFIG="scripts/training-configs/SlideOn/config.json"
SAE_ID_OR_PATH="lightonai/SlideOn-SAE"  # Or our raw SAE: outputs/slideon-sae/final
SAE_REVISION="9372da2e84fb87035514469d00982ca7f13ef4d1"  # Ignored for local SAE paths.
# This DenseOn revision must match the backbone used by the chosen SAE.
BACKBONE_REVISION="cb9947ebccb33862d24e3c7ca2edb25e51acd887"
SFT_OUTPUT="outputs/slideon-alpha05-neg7"
RESUME_CHECKPOINT=""  # Or a Trainer checkpoint, e.g. outputs/my-run/checkpoint-500

# TRAINING: these values match the existing English training configuration.
# The values here override TRAIN_CONFIG; other loss/optimizer settings stay there.
BATCH_SIZE=4         # Query examples per GPU, each with a positive and negatives.
GLOBAL_BATCH_SIZE=128
# Accumulation is automatic: global batch / (batch per GPU * number of GPUs).
# For example, batch 4 on two GPUs gives 16 accumulation steps.
QUERY_LENGTH=8192
DOCUMENT_LENGTH=8192
HEAD_CHUNK_SIZE=2048
LEARNING_RATE=0.00004
MAX_STEPS=24544       # Set -1 to use EPOCHS instead of a fixed update count.
EPOCHS=1
WARMUP_RATIO=0.05
GRADIENT_CHECKPOINTING=false
BF16=true
SEED=42
LOGGING_STEPS=1       # Increase this to print losses less often.
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=2

# EVALUATION: runs on SFT_OUTPUT/final after successful training.
EVAL_SUITE="beir"  # Or "BEIR-Small".
EVAL_QUERY_LENGTH=""     # "" preserves the saved training query length.
EVAL_DOCUMENT_LENGTH=""  # "" preserves the saved training document length.
EVAL_QUERY_MAX_ACTIVE_DIMS=64     # "" means unpruned.
EVAL_DOCUMENT_MAX_ACTIVE_DIMS=512 # "" means unpruned.
EVAL_BATCH_SIZE=8
EVAL_OUTPUT=""  # "" creates a results directory inside SFT_OUTPUT from eval settings.

# Settings end here. The rest prepares data, launches training and evaluates.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if (( $# > 0 )); then
    printf '%s\n' 'Edit the settings at the top of run_finetune.sh, then run it without arguments.' >&2
    exit 2
fi

# 1. Prepare the JSONL only if it is missing and preparation is enabled.
if [[ ! -f "$TRAIN_JSONL" ]]; then
    if [[ "$PREPARE_DATA_IF_MISSING" != true ]]; then
        printf 'Training data not found: %s. Enable PREPARE_DATA_IF_MISSING or set TRAIN_JSONL to an existing file.\n' "$TRAIN_JSONL" >&2
        exit 1
    fi
    "$PYTHON" scripts/finetune/preprocess.py \
        --dataset "$DATASET_ID" --revision "$DATASET_REVISION" \
        --output "$TRAIN_JSONL"
fi

# 2. Assemble training arguments from the settings above.
TRAIN_ARGS=(--config "$TRAIN_CONFIG" --train "$TRAIN_JSONL" --output "$SFT_OUTPUT"
    --sae "$SAE_ID_OR_PATH" --sae-revision "$SAE_REVISION" --backbone-revision "$BACKBONE_REVISION"
    --batch-size "$BATCH_SIZE" --global-batch-size "$GLOBAL_BATCH_SIZE"
    --query-length "$QUERY_LENGTH" --document-length "$DOCUMENT_LENGTH"
    --head-chunk-size "$HEAD_CHUNK_SIZE" --learning-rate "$LEARNING_RATE"
    --max-steps "$MAX_STEPS" --epochs "$EPOCHS" --warmup-ratio "$WARMUP_RATIO"
    --seed "$SEED" --logging-steps "$LOGGING_STEPS" --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT")
[[ -z "$SOURCE_SAMPLING_ALPHA" ]] || TRAIN_ARGS+=(--source-sampling-alpha "$SOURCE_SAMPLING_ALPHA")
[[ -z "$SAMPLE_NEGATIVES" ]] || TRAIN_ARGS+=(--sample-negatives "$SAMPLE_NEGATIVES")
[[ -z "$RESUME_CHECKPOINT" ]] || TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
if [[ "$GRADIENT_CHECKPOINTING" == true ]]; then
    TRAIN_ARGS+=(--gradient-checkpointing)
else
    TRAIN_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$BF16" == true ]]; then
    TRAIN_ARGS+=(--bf16)
else
    TRAIN_ARGS+=(--no-bf16)
fi

# A comma-separated GPU list selects one distributed worker per available GPU.
# Use the active Python environment for both direct Python and torchrun.
TRAIN_COMMAND=("$PYTHON")
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    TRAIN_COMMAND+=(-m torch.distributed.run --standalone --nproc_per_node=gpu)
fi
"${TRAIN_COMMAND[@]}" scripts/finetune/train_slideon.py "${TRAIN_ARGS[@]}"

# 3. Evaluate once, on the first selected GPU, after successful training.
# set -e above prevents evaluation after failed preprocessing or training.
if [[ "$RUN_EVALUATION" == true ]]; then
    if [[ -z "$EVAL_OUTPUT" ]]; then
        EVAL_TAG="${EVAL_SUITE}-q${EVAL_QUERY_MAX_ACTIVE_DIMS:-full}-d${EVAL_DOCUMENT_MAX_ACTIVE_DIMS:-full}"
        if [[ -n "$EVAL_QUERY_LENGTH" || -n "$EVAL_DOCUMENT_LENGTH" ]]; then
            EVAL_TAG+="-qlen${EVAL_QUERY_LENGTH:-saved}-dlen${EVAL_DOCUMENT_LENGTH:-saved}"
        fi
        EVAL_OUTPUT="$SFT_OUTPUT/$EVAL_TAG"
    fi
    EVAL_ARGS=(--model "$SFT_OUTPUT/final" --trust-remote-code --suite "$EVAL_SUITE"
        --batch-size "$EVAL_BATCH_SIZE" --output "$EVAL_OUTPUT")
    [[ -z "$EVAL_QUERY_LENGTH" ]] || EVAL_ARGS+=(--query-length "$EVAL_QUERY_LENGTH")
    [[ -z "$EVAL_DOCUMENT_LENGTH" ]] || EVAL_ARGS+=(--document-length "$EVAL_DOCUMENT_LENGTH")
    [[ -z "$EVAL_QUERY_MAX_ACTIVE_DIMS" ]] || EVAL_ARGS+=(--query-max-active-dims "$EVAL_QUERY_MAX_ACTIVE_DIMS")
    [[ -z "$EVAL_DOCUMENT_MAX_ACTIVE_DIMS" ]] || EVAL_ARGS+=(--document-max-active-dims "$EVAL_DOCUMENT_MAX_ACTIVE_DIMS")
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$PYTHON" scripts/eval/evaluate.py "${EVAL_ARGS[@]}"
fi
