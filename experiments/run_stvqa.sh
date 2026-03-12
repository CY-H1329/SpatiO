#!/bin/bash
# SpatiO — STVQA-7K only
# Usage:
#   bash experiments/run_stvqa.sh
#   MODE=full bash experiments/run_stvqa.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${MODE:-50}"
if [ "$MODE" = "full" ]; then
  python run_stvqa.py --full --test_only --output_dir results/stvqa --seed 42
else
  python run_stvqa.py --max_samples 50 --test_only --output_dir results/stvqa --seed 42
fi
