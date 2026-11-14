#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_FILE="${OUTPUT_FILE:-/result/submission.jsonl}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=== ZAC 2026 LifeMate Inference Starting ==="
echo "Data Directory: $DATA_DIR"
echo "Output: $OUTPUT_FILE"

python3 "$SCRIPT_DIR/predict.py" \
    --root "$DATA_DIR" \
    --output "$OUTPUT_FILE" \
    "$@"

echo "=== Inference Completed Successfully ==="
