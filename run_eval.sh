#!/usr/bin/env bash
# Run with: bash run_eval.sh
# One-time setup in your Python environment: pip install -e '.[eval]'
# Edit the settings below, save this file, then run: bash run_eval.sh
# These values are fixed for each run; environment variables do not override them.
set -euo pipefail

# MODEL: the same command evaluates SAE and fine-tuned weights.
# Released SAE:       lightonai/SlideOn-SAE
# Released fine-tune: lightonai/SlideOn
# Our raw SAE:        outputs/slideon-sae/final
# Our fine-tune:      outputs/slideon-alpha05-neg7/final
# Relative paths are resolved from this repository; absolute paths work too.
MODEL_ID_OR_PATH="lightonai/SlideOn"
# "auto" uses our existing pins for the two released models; local paths have
# no Hub revision. Set a Hub commit explicitly, or "" to resolve its latest.
MODEL_REVISION="auto"

# TOKEN LIMITS: "" preserves the checkpoint's saved limit for that role.
# Set either independently, e.g. QUERY_LENGTH=128 and DOCUMENT_LENGTH=512.
QUERY_LENGTH=""
DOCUMENT_LENGTH=""

# BENCHMARK: "beir" (15 datasets) or "BEIR-Small" (9 datasets), for example.
EVAL_SUITE="beir"
# Final-vector sparsity, separate from token limits. "" means unpruned.
QUERY_MAX_ACTIVE_DIMS=64
DOCUMENT_MAX_ACTIVE_DIMS=512
BATCH_SIZE=8  # Number of texts encoded together.

# RESULTS: set a directory, or leave "" to name it from the settings below.
# Use a separate directory for each checkpoint/settings combination.
EVAL_OUTPUT=""
PYTHON="python3"
# Evaluation uses one process and the first visible GPU; "" selects CPU only.
export CUDA_VISIBLE_DEVICES="0"

# Settings end here. The rest builds and runs the evaluation command.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ "$MODEL_REVISION" == auto ]]; then
    MODEL_REVISION=""
    # Prefer an existing local directory, just as the Python evaluator does.
    if [[ ! -d "$MODEL_ID_OR_PATH" ]]; then
        case "$MODEL_ID_OR_PATH" in
            lightonai/SlideOn-SAE) MODEL_REVISION="02d82ff083c61710ca8ec0ae378a150bc927860f" ;;
            lightonai/SlideOn) MODEL_REVISION="370c600e9886afc43ff8026695261e56ecfaa011" ;;
        esac
    fi
fi

MODEL_TAG="${MODEL_ID_OR_PATH//\//_}"
EVAL_TAG="${EVAL_SUITE}-qlen${QUERY_LENGTH:-saved}-dlen${DOCUMENT_LENGTH:-saved}-q${QUERY_MAX_ACTIVE_DIMS:-full}-d${DOCUMENT_MAX_ACTIVE_DIMS:-full}"
EVAL_OUTPUT="${EVAL_OUTPUT:-results/$MODEL_TAG-$EVAL_TAG}"

# Assemble one evaluation command. Format detection and raw-SAE backbone
# assembly belong to the Python loader; there is no training stage here.
ARGS=(--model "$MODEL_ID_OR_PATH" --trust-remote-code --suite "$EVAL_SUITE"
      --batch-size "$BATCH_SIZE" --output "$EVAL_OUTPUT")
[[ -z "$MODEL_REVISION" ]] || ARGS+=(--revision "$MODEL_REVISION")
[[ -z "$QUERY_LENGTH" ]] || ARGS+=(--query-length "$QUERY_LENGTH")
[[ -z "$DOCUMENT_LENGTH" ]] || ARGS+=(--document-length "$DOCUMENT_LENGTH")
[[ -z "$QUERY_MAX_ACTIVE_DIMS" ]] || ARGS+=(--query-max-active-dims "$QUERY_MAX_ACTIVE_DIMS")
[[ -z "$DOCUMENT_MAX_ACTIVE_DIMS" ]] || ARGS+=(--document-max-active-dims "$DOCUMENT_MAX_ACTIVE_DIMS")
exec "$PYTHON" scripts/eval/evaluate.py "${ARGS[@]}"
