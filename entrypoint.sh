#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_FILE="${OUTPUT_FILE:-/app/result/submission.jsonl}"
SPLIT="${SPLIT:-test}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=== ZAIC 2026 LifeMate Inference Starting ==="
echo "Data Directory: $DATA_DIR"
echo "Output: $OUTPUT_FILE"
echo "Split: $SPLIT"

# Chay batched ReAct infer tren offline models
python run_infer_batched.py \
    --root "$DATA_DIR" \
    --split "$SPLIT" \
    --output "$OUTPUT_FILE" \
    --model "${MODEL_PATH:-/app/models/qwen3-4b-merged}" \
    --asr-model "${ASR_PATH:-/app/models/whisper-large-v3-turbo}" \
    --backend vllm \
    --batch-size 16 \
    --preprocessor safe \
    --gpu-memory-utilization 0.72

echo "=== Inference Completed Successfully ==="
