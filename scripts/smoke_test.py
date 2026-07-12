#!/usr/bin/env python3
"""CPU smoke: pipeline + TTO with mock VLMs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from config import ALL_CATEGORIES
from pipeline import run_step
from score_map import ScoreMap


def _head(_img, _prompt: str) -> str:
    return "spatial_relation"


def _spec(_llm: str, _img, prompt: str) -> str:
    letter = "B" if "(B)" in prompt else "A"
    return f"Answer: ({letter})\nReason: mock ({_llm})."


def _reason(prompt: str, img=None, image=None, **kwargs) -> str:
    return "Answer: (A)\nReason: mock reasoner."


def main() -> None:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    query = (
        "Which object is closer to the camera?\n"
        "OPTIONS:\n(A) left box\n(B) right box\n(C) equal\n(D) unknown"
    )
    out = run_step(
        image=img,
        query=query,
        gt="(A)",
        step=0,
        score_map=ScoreMap(categories=ALL_CATEGORIES),
        trust_state=None,
        head_generate=_head,
        specialist_generate=_spec,
        reasoning_generate=_reason,
        update_trust=False,
        answer_type="multiple_choice",
    )
    assert out.get("final_answer"), out
    print("OK smoke_test:", out["final_answer"], "|", out.get("category"))


if __name__ == "__main__":
    main()
