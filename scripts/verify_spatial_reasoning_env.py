#!/usr/bin/env python3
"""Vérifie les versions pip/conda utiles à SpatiO (Qwen3-VL, datasets) et la préparation MindCube (chemins)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python scripts/verify_*.py` from any cwd
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


def _need(msg: str) -> None:
    print("MANQUE:", msg)
    sys.exit(1)


def _warn(msg: str) -> None:
    print("AVIS:", msg)


def main() -> None:
    print("=== SpatiO — vérification environnement ===\n")

    # Versions
    try:
        import torch

        print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
        if torch.__version__.split("+")[0] < "2.2":
            _warn("torch < 2.2 peut poser problème avec les VLMs récents.")
    except ImportError:
        _need("torch non installé")

    try:
        import transformers

        v = transformers.__version__
        print("transformers:", v)
        major, minor = int(v.split(".")[0]), int(v.split(".")[1])
        if (major, minor) < (4, 51):
            print("MANQUE: transformers >= 4.51 requis pour Qwen3-VL (Qwen3VLForConditionalGeneration).")
            ce = os.environ.get("CONDA_DEFAULT_ENV", "")
            if ce in ("", "base"):
                print(
                    "\nTu sembles utiliser le Python du (base) ou hors conda : torch/transformers y sont souvent anciens.\n"
                    "1) Laisser aller au bout (sans Ctrl+Z) :  bash SpatiO/scripts/setup_spatial_reasoning_env.sh\n"
                    "2) Si tu avais suspendu le script (^Z) :  jobs  puis  kill %1  ou  fg  puis Ctrl+C\n"
                    "   Si l’env est à moitié créée :  conda env remove -n spatial_reasoning -y\n"
                    "3)  conda activate spatial_reasoning\n"
                    "4)  python SpatiO/scripts/verify_spatial_reasoning_env.py\n"
                )
            sys.exit(1)
    except ImportError:
        _need("transformers non installé")

    try:
        import huggingface_hub

        print("huggingface_hub:", huggingface_hub.__version__)
    except ImportError:
        _warn("huggingface_hub absent (souvent installé avec transformers).")

    try:
        import datasets

        print("datasets:", datasets.__version__)
    except ImportError:
        _need("datasets non installé (benchmarks HF + STVQA).")

    try:
        import pyarrow

        print("pyarrow:", pyarrow.__version__)
    except ImportError:
        _warn("pyarrow recommandé pour STVQA-7K (parquet).")

    try:
        import accelerate

        print("accelerate:", accelerate.__version__)
    except ImportError:
        _warn("accelerate recommandé (chargement modèles).")

    try:
        import importlib.util

        if importlib.util.find_spec("qwen_vl_utils") is not None:
            print("qwen-vl-utils (module qwen_vl_utils): OK")
        else:
            _warn("qwen-vl-utils manquant — Qwen3-VL peut échouer. pip install qwen-vl-utils")
    except Exception:
        _warn("qwen-vl-utils manquant — Qwen3-VL peut échouer. pip install qwen-vl-utils")

    # Qwen3 classe (sans télécharger les poids)
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        assert Qwen3VLForConditionalGeneration is not None
        assert AutoProcessor is not None
        print("Qwen3VLForConditionalGeneration + AutoProcessor: import OK")
    except Exception as e:
        _need(f"Import Qwen3-VL impossible: {e}")

    # Local backends (no MODEL_ROOT required)
    spatio_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend = os.path.join(spatio_root, "models", "_backend_qwen3.py")
    if os.path.isfile(backend):
        print("SpatiO models/_backend_qwen3.py: OK (self-contained)")
    else:
        _need(f"Missing vendored backend: {backend}")

    profile = os.environ.get("SPATIO_PROFILE", "paper")
    print("SPATIO_PROFILE:", profile)
    try:
        from config import SPECIALIST_LLMS, TOP_K_SPECIALISTS

        print("SPECIALIST_LLMS (%d):" % len(SPECIALIST_LLMS), ", ".join(SPECIALIST_LLMS))
        print("TOP_K_SPECIALISTS:", TOP_K_SPECIALISTS)
        if profile not in ("minimal", "qwen", "qwen_only") and len(SPECIALIST_LLMS) < 5:
            _warn("Paper reproduction expects 5 specialists; got %d." % len(SPECIALIST_LLMS))
    except Exception as e:
        _warn(f"Could not import config specialists: {e}")

    srgpt = os.environ.get("SPATIALRGPT_PATH", "")
    if profile in ("minimal", "qwen", "qwen_only"):
        _warn("SPATIO_PROFILE=minimal is for local smoke only — public reproduction uses all 5 specialists.")
    elif srgpt and os.path.isdir(srgpt):
        print("SPATIALRGPT_PATH:", srgpt, "OK")
    else:
        _warn(
            "Full stack needs SpatialRGPT — clone AnjieCheng/SpatialRGPT "
            "and export SPATIALRGPT_PATH=..."
        )
    print(
        "NOTE: Official Reasoning Agent is not in this release yet (coming later); "
        "models/reasoning.py is an interim Qwen3-VL stand-in."
    )

    print("\n=== MindCube (préparation chemins) ===\n")
    tsv = os.environ.get("SPATIO_MINDCUBE_TSV", "")
    root = os.environ.get("SPATIO_MINDCUBE_IMAGES_ROOT", "")
    if not tsv and not root:
        print(
            "MindCube: définir avant run_mindcube_tto / run_mindcube :\n"
            "  export SPATIO_MINDCUBE_TSV=/chemin/MindCubeBench_tiny_raw_qa.tsv\n"
            "  export SPATIO_MINDCUBE_IMAGES_ROOT=/chemin/MindCube   # contient data/other_all_image/..."
        )
    else:
        if tsv and os.path.isfile(tsv):
            print("SPATIO_MINDCUBE_TSV:", tsv, "OK")
        elif tsv:
            _warn(f"SPATIO_MINDCUBE_TSV fichier introuvable: {tsv}")
        if root and os.path.isdir(root):
            img_root = os.path.join(root, "data", "other_all_image")
            if os.path.isdir(img_root):
                print("SPATIO_MINDCUBE_IMAGES_ROOT:", root, "| data/other_all_image: OK")
            else:
                _warn(f"Sous-dossier data/other_all_image absent sous {root}")
        elif root:
            _warn(f"SPATIO_MINDCUBE_IMAGES_ROOT n'est pas un répertoire: {root}")

    print("\n=== Résumé ===\nEnvironnement utilisable pour SpatiO si aucune ligne MANQUE ci-dessus.\n")


if __name__ == "__main__":
    main()
