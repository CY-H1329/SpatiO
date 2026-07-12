"""
Public architecture overview only.

Detailed ``run_step`` (prompts, trust updates, tool calls, reasoner wiring)
is intentionally withheld from this snapshot and will be released later.
"""

from typing import Any, Callable, Dict, List, Optional

from config import HEAD_AGENT_MODEL, REASONING_AGENT_MODEL, ROLES, SPECIALIST_LLMS, TOP_K_SPECIALISTS


def pipeline_stages() -> List[Dict[str, Any]]:
    """Return the public Head → 5 specialists → Reasoning layout."""
    return [
        {
            "stage": "head",
            "model": HEAD_AGENT_MODEL,
            "role": "Classify query category and drive role selection.",
        },
        {
            "stage": "specialists",
            "models": list(SPECIALIST_LLMS),
            "roles": list(ROLES),
            "top_k": TOP_K_SPECIALISTS,
            "role": "Produce complementary spatial answers into shared memory.",
        },
        {
            "stage": "reasoning",
            "model": REASONING_AGENT_MODEL,
            "role": "TTO-weighted synthesis → final answer.",
            "status": "detailed implementation coming later",
        },
    ]


def run_step(*_args, **_kwargs) -> Dict[str, Any]:
    raise NotImplementedError(
        "Full pipeline implementation is withheld in the public architecture snapshot. "
        "See README.md — detailed code (prompts / TTO / official Reasoning Agent) will be released later."
    )


# Placeholders so imports from older docs do not claim a runnable private stack.
def build_public_generate_hooks() -> Dict[str, Callable]:
    def _missing(*_a, **_k):
        raise NotImplementedError("VLM hooks withheld in public architecture snapshot.")

    return {
        "head_generate": _missing,
        "specialist_generate": _missing,
        "reasoning_generate": _missing,
    }
