#!/bin/bash
# SpatiO — Run all benchmarks (supplementary materials)
#
# Usage:
#   cd <supplement_root>
#   bash experiments/run_all.sh
#
# Runs each benchmark with:
#   - 50 samples (test)
#   - Full dataset
#
# Output: results/spatio/<benchmark>/50/ and results/spatio/<benchmark>/full/
set -e

export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::UserWarning"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_BASE="results/spatio"
SEED=42

echo "=============================================="
echo "SpatiO — Supplementary Materials"
echo "Output: $OUTPUT_BASE"
echo "Modes: 50 samples + full"
echo "=============================================="

for BENCHMARK in cvbench 3dsrbench stvqa; do
  for MODE in 50 full; do
    echo ""
    echo ">>> $BENCHMARK | $MODE"
    echo "----------------------------------------------"
    if [ "$MODE" = "full" ]; then
      python "run_${BENCHMARK}.py" \
        --full \
        --test_only \
        --output_dir "$OUTPUT_BASE/$BENCHMARK" \
        --seed "$SEED"
    else
      python "run_${BENCHMARK}.py" \
        --max_samples 50 \
        --test_only \
        --output_dir "$OUTPUT_BASE/$BENCHMARK" \
        --seed "$SEED"
    fi
  done
done

echo ""
echo "=============================================="
echo "Done. Results in $OUTPUT_BASE/<benchmark>/50/ and .../full/"
echo "=============================================="
