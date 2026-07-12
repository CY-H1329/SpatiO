#!/usr/bin/env bash
# Create conda env "spatial_reasoning" (Python 3.10) + pip torch + SpatiO deps.
# Optional: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#           TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="spatial_reasoning"
REQ_AFTER="$SPATIO_ROOT/requirements-no-torch.txt"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Use: python3.10 -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install torch torchvision --index-url $TORCH_INDEX_URL" >&2
  echo "  pip install -r requirements-no-torch.txt" >&2
  exit 1
fi

export CONDA_ALWAYS_YES="${CONDA_ALWAYS_YES:-1}"
BASE="$(conda info --base)"

echo "SpatiO setup → env=${ENV_NAME}  torch_index=${TORCH_INDEX_URL}"

if [[ ! -d "$BASE/envs/$ENV_NAME" ]]; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

RUN=(conda run -n "$ENV_NAME" --no-capture-output)
"${RUN[@]}" python -m pip install -U pip setuptools wheel ${PIP_EXTRA_ARGS:-}
"${RUN[@]}" python -m pip install ${PIP_EXTRA_ARGS:-} torch torchvision --index-url "$TORCH_INDEX_URL"
"${RUN[@]}" python -m pip install ${PIP_EXTRA_ARGS:-} -r "$REQ_AFTER"

"${RUN[@]}" python -c "import transformers; t=transformers.__version__; assert tuple(map(int,t.split('.')[:2]))>=(4,51)"

echo "OK. Next:"
echo "  conda activate $ENV_NAME"
echo "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
echo "  python scripts/smoke_test.py"
