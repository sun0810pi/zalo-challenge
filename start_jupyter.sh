#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec jupyter lab \
    --port 9777 \
    --ip 0.0.0.0 \
    --IdentityProvider.token='zac2026' \
    --ServerApp.password='' \
    --ServerApp.root_dir="$SCRIPT_DIR" \
    --allow-root \
    --no-browser
