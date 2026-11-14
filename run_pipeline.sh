#!/usr/bin/env bash
# Run with: bash run_pipeline.sh
# One-time setup, in your Python environment: pip install -e '.[eval]'
# Only stages 3 and 5 are enabled. Uncomment complete commands to enable others.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-python3}"
# Default to GPU 0; preserve an existing selection, including "" for CPU only.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

# Training hyperparameters stay in the existing, unmodified JSON configs.
TOKEN_CACHE="token-cache/slideon"
SAE_OUTPUT="outputs/slideon-sae"
SFT_OUTPUT="outputs/slideon"
TRAIN_JSONL="english.jsonl"  # Needed only if you enable stage 4.

# Pinned released checkpoints. These evaluation pins are independent of the
# historical SAE initializer pin already stored in the SFT training config.
RELEASED_SAE="lightonai/SlideOn-SAE"
RELEASED_SAE_REVISION="02d82ff083c61710ca8ec0ae378a150bc927860f"
RELEASED_SLIDEON="lightonai/SlideOn"
RELEASED_SLIDEON_REVISION="370c600e9886afc43ff8026695261e56ecfaa011"

# Default English benchmark; use a different output directory for each setting.
EVAL_SUITE="beir"
# EVAL_SUITE="mteb-eng-v2"
QUERY_MAX_ACTIVE_DIMS=64
DOCUMENT_MAX_ACTIVE_DIMS=512
EVAL_TAG="${EVAL_SUITE}-q${QUERY_MAX_ACTIVE_DIMS}-d${DOCUMENT_MAX_ACTIVE_DIMS}"
# Cap the final pooled vectors at 64 query / 512 document nonzeros. A vector
# with fewer positive features remains smaller. Per-token token_k, prompts,
# sequence lengths and the evaluator's batch/chunk defaults stay unchanged.


# 1. PRE-TRAIN: prepare the frozen DenseOn token cache (disabled).
# "$PYTHON" scripts/sae-pretrain/train_slideon_sae.py \
#     --prepare --cache "$TOKEN_CACHE"

# 2. PRE-TRAIN: learn the SAE from that token cache (disabled).
# "$PYTHON" scripts/sae-pretrain/train_slideon_sae.py \
#     --cache "$TOKEN_CACHE" --output "$SAE_OUTPUT"

# 3. EVALUATE SAE: released checkpoint (enabled).
"$PYTHON" scripts/eval/evaluate.py \
    --model "$RELEASED_SAE" --revision "$RELEASED_SAE_REVISION" \
    --trust-remote-code --suite "$EVAL_SUITE" \
    --query-max-active-dims "$QUERY_MAX_ACTIVE_DIMS" \
    --document-max-active-dims "$DOCUMENT_MAX_ACTIVE_DIMS" \
    --output "results/released-sae-$EVAL_TAG"
# Local raw-SAE evaluation needs loader support in the Python evaluator first.

# 4. FINE-TUNE: start from the config's pinned released SAE (disabled).
# Requires prepared JSONL: query, positive, negatives, teacher_scores.
# "$PYTHON" scripts/finetune/train_slideon.py \
#     --train "$TRAIN_JSONL" --output "$SFT_OUTPUT"
# Alternatively, enable this command to start from our own SAE:
# "$PYTHON" scripts/finetune/train_slideon.py \
#     --train "$TRAIN_JSONL" --sae "$SAE_OUTPUT/final" --output "$SFT_OUTPUT"

# 5. EVALUATE SLIDEON: released fine-tuned checkpoint (enabled).
"$PYTHON" scripts/eval/evaluate.py \
    --model "$RELEASED_SLIDEON" --revision "$RELEASED_SLIDEON_REVISION" \
    --trust-remote-code --suite "$EVAL_SUITE" \
    --query-max-active-dims "$QUERY_MAX_ACTIVE_DIMS" \
    --document-max-active-dims "$DOCUMENT_MAX_ACTIVE_DIMS" \
    --output "results/released-slideon-$EVAL_TAG"
# Local final/ evaluation also needs Python support: the current CLI is Hub-only.
