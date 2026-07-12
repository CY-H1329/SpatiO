#!/usr/bin/env bash
# Crée ou met à jour l’environnement conda spatial_reasoning pour SpatiO (+ MindCube TSV, pas de dépendance lourde supplémentaire).
# Si la résolution conda est trop lente : utiliser scripts/setup_spatial_reasoning_env_fast.sh (Python conda + pip).
#
# IMPORTANT : ne pas appuyer sur Ctrl+Z pendant l’exécution (suspend le job → l’env
# n’est pas créée / reste cassé). Laisser finir (plusieurs minutes) ou Ctrl+C pour
# annuler proprement, puis : conda env remove -n spatial_reasoning -y  (si partiel)
# et relancer ce script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YML="$SPATIO_ROOT/environment-spatial_reasoning.yml"
# Le nom doit rester « spatial_reasoning » (identique au champ `name:` du YAML).
ENV_NAME="spatial_reasoning"

# Réponses non interactives aux prompts conda
export CONDA_ALWAYS_YES="${CONDA_ALWAYS_YES:-1}"

echo "========================================================================"
echo "  SpatiO — création / mise à jour conda : ${ENV_NAME}"
echo "  Ne pas utiliser Ctrl+Z. Attendre la fin du téléchargement des paquets."
echo "========================================================================"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "conda introuvable. Installe Miniconda/Anaconda ou utilise : pip install -r $SPATIO_ROOT/requirements-spatial_reasoning.txt" >&2
  exit 1
fi

if [[ ! -f "$YML" ]]; then
  echo "Fichier manquant: $YML" >&2
  exit 1
fi

BASE="$(conda info --base)"
if [[ -d "$BASE/envs/$ENV_NAME" ]]; then
  echo "[setup] Mise à jour de l’env existant: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$YML" --prune
else
  echo "[setup] Création de l’env: $ENV_NAME (première fois : long)"
  conda env create -f "$YML"
fi

if ! conda run -n "$ENV_NAME" --no-capture-output python -c "import transformers; v=transformers.__version__; assert tuple(map(int,v.split('.')[:2]))>=(4,51)" 2>/dev/null; then
  echo "[setup] ERREUR: transformers>=4.51 introuvable dans l’env $ENV_NAME" >&2
  exit 1
fi

echo
echo "OK. Prochaines étapes :"
echo "  conda activate $ENV_NAME"
echo "  export MODEL_ROOT=\"\$(cd \"$SPATIO_ROOT/..\" && pwd)\"   # répertoire CY (contient src/)"
echo "  python \"$SPATIO_ROOT/scripts/verify_spatial_reasoning_env.py\""
