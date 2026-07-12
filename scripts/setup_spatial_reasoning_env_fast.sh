#!/usr/bin/env bash
# Installation RAPIDE de l’env « spatial_reasoning » : conda ne résout que Python,
# puis PyTorch + le reste via pip (souvent bien plus court qu’un gros conda env create).
#
# Variables optionnelles :
#   TORCH_INDEX_URL   — wheel PyTorch (défaut CUDA 12.1)
#     CPU : export TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
#     CUDA 12.4 : https://download.pytorch.org/whl/cu124
#   PIP_EXTRA_ARGS    — ex. --no-cache-dir
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="spatial_reasoning"
REQ_AFTER="$SPATIO_ROOT/requirements-spatial_reasoning-after-torch.txt"
# PyTorch wheels (ajuster CUDA / CPU selon la machine)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda introuvable. Alternative : python -m venv .venv && source .venv/bin/activate && pip install torch ... && pip install -r requirements-spatial_reasoning.txt" >&2
  exit 1
fi

if [[ ! -f "$REQ_AFTER" ]]; then
  echo "Fichier manquant: $REQ_AFTER" >&2
  exit 1
fi

export CONDA_ALWAYS_YES="${CONDA_ALWAYS_YES:-1}"
BASE="$(conda info --base)"

echo "========================================================================"
echo "  SpatiO — installation RAPIDE : ${ENV_NAME}"
echo "  Étape 1/3 : conda (Python 3.10 uniquement, ~1–2 min)"
echo "  Étape 2/3 : pip torch/torchvision depuis ${TORCH_INDEX_URL}"
echo "  Étape 3/3 : pip transformers, datasets, …"
echo "  Ne pas utiliser Ctrl+Z pendant pip (gros téléchargements)."
echo "========================================================================"
echo

if [[ -d "$BASE/envs/$ENV_NAME" ]]; then
  echo "[fast] Env existant — on réutilise : $ENV_NAME (pip met à jour les paquets)"
else
  echo "[fast] conda create -n $ENV_NAME python=3.10"
  conda create -n "$ENV_NAME" python=3.10 -y
fi

RUN=(conda run -n "$ENV_NAME" --no-capture-output)
"${RUN[@]}" python -m pip install --upgrade pip setuptools wheel ${PIP_EXTRA_ARGS:-}

echo "[fast] pip install torch torchvision …"
"${RUN[@]}" python -m pip install ${PIP_EXTRA_ARGS:-} torch torchvision --index-url "$TORCH_INDEX_URL"

echo "[fast] pip install dépendances SpatiO (hors torch) …"
"${RUN[@]}" python -m pip install ${PIP_EXTRA_ARGS:-} -r "$REQ_AFTER"

if ! "${RUN[@]}" python -c "import transformers; t=transformers.__version__; assert tuple(map(int,t.split('.')[:2]))>=(4,51)" 2>/dev/null; then
  echo "[fast] ERREUR: transformers>=4.51 non satisfait dans $ENV_NAME" >&2
  exit 1
fi

echo
echo "OK (rapide). Next steps:"
echo "  conda activate $ENV_NAME"
echo "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT   # required for paper (5 specialists)"
echo "  python \"$SPATIO_ROOT/scripts/verify_spatial_reasoning_env.py\""
echo "  python \"$SPATIO_ROOT/scripts/smoke_pipeline_mock.py\""
echo "  # Paper stack (all 5 specialists):"
echo "  python \"$SPATIO_ROOT/run_cvbench.py\" --max_samples 50 --test_only --top_k 5 \\"
echo "    --device_map 0,1,2,3,4,5,6 --output_dir results/cvbench_paper5"
