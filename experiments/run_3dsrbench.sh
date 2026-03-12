#!/bin/bash
# SpatiO — 3DSRBench only
# Usage:
#   bash experiments/run_3dsrbench.sh
#   MODE=full bash experiments/run_3dsrbench.sh
# Optional: SPATIO_IMAGE_CACHE=/path/to/cache for image caching
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${MODE:-50}"
if [ "$MODE" = "full" ]; then
  python run_3dsrbench.py --full --test_only --output_dir results/3dsrbench --seed 42
else
  python run_3dsrbench.py --max_samples 50 --test_only --output_dir results/3dsrbench --seed 42
fi
