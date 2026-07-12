#!/usr/bin/env python3
"""Check torch / transformers / backends / SPATIALRGPT_PATH."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _fail(msg: str) -> None:
    print("FAIL:", msg)
    sys.exit(1)


def _warn(msg: str) -> None:
    print("WARN:", msg)


def main() -> None:
    print("=== SpatiO env check ===\n")
    try:
        import torch

        print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
    except ImportError:
        _fail("torch not installed")

    try:
        import transformers

        major, minor = map(int, transformers.__version__.split(".")[:2])
        print("transformers", transformers.__version__)
        if (major, minor) < (4, 51):
            _fail("need transformers>=4.51 for Qwen3-VL")
    except ImportError:
        _fail("transformers not installed")

    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

        print("Qwen3VLForConditionalGeneration: OK")
    except Exception as e:
        _fail(f"Qwen3-VL import failed: {e}")

    if not (ROOT / "spatio" / "models" / "_backend_qwen3.py").is_file():
        _fail("missing spatio/models/_backend_qwen3.py")
    print("backends: OK")

    from spatio.config import SPECIALIST_LLMS, TOP_K_SPECIALISTS

    print("specialists (%d):" % len(SPECIALIST_LLMS), ", ".join(SPECIALIST_LLMS))
    print("top_k:", TOP_K_SPECIALISTS)
    if len(SPECIALIST_LLMS) < 5:
        _warn("expected 5 specialists for the paper stack")

    srgpt = os.environ.get("SPATIALRGPT_PATH", "")
    if srgpt and Path(srgpt).is_dir():
        print("SPATIALRGPT_PATH:", srgpt, "OK")
    else:
        _warn("set SPATIALRGPT_PATH to the SpatialRGPT clone for spatial_rgpt")

    print("\nNOTE: official Reasoning Agent ships later; interim stand-in in spatio/models/reasoning.py")
    print("OK — env looks usable.\n")


if __name__ == "__main__":
    main()
