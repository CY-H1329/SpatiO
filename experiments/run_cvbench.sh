#!/bin/bash
# SpatiO — CV-Bench only
# Usage:
#   bash experiments/run_cvbench.sh
#   MODE=full bash experiments/run_cvbench.sh
#   MODE=50 bash experiments/run_cvbench.sh  (default)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${MODE:-50}"
if [ "$MODE" = "full" ]; then
  python run_cvbench.py --full --test_only --output_dir results/cvbench --seed 42
else
  python run_cvbench.py --max_samples 50 --test_only --output_dir results/cvbench --seed 42
fi
