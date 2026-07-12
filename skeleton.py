"""
SpatiO — public architecture skeleton.

Exposes Head + 5 specialists + Reasoning (TTO) layout without proprietary details.
"""

from config import (
    KAPPA,
    MU,
    GAMMA,
    LAMBDA_F,
    LAMBDA_G,
    RAMP_TEMP,
    BETA,
    ROLES,
    ALL_CATEGORIES,
    SPECIALIST_LLMS,
    HEAD_AGENT_MODEL,
    REASONING_AGENT_MODEL,
    MODEL_CARD,
    TOP_K_SPECIALISTS,
)


def describe_pipeline() -> str:
    lines = [
        "SpatiO public architecture",
        "",
        "  Image + Query",
        f"    → Head Agent [{HEAD_AGENT_MODEL}]",
        "    → Specialist pool (paper, all 5):",
    ]
    for i, name in enumerate(SPECIALIST_LLMS, 1):
        lines.append(f"         {i}. {name:18s}  {MODEL_CARD.get(name, '')}")
    lines += [
        f"         roles (ids): {', '.join(ROLES)}",
        f"         top_k={TOP_K_SPECIALISTS}",
        f"    → Reasoning Agent [{REASONING_AGENT_MODEL}]  ** detailed code: later release **",
        "    → Final answer (TTO-weighted synthesis)",
        "",
        "Paper hyperparameters (public):",
        f"  kappa={KAPPA}, mu={MU}, gamma={GAMMA}",
        f"  lambda_f={LAMBDA_F}, lambda_g={LAMBDA_G}",
        f"  ramp_temp={RAMP_TEMP}, beta={BETA}",
        "",
        "Categories:",
        "  - " + "\n  - ".join(ALL_CATEGORIES),
        "",
        "Withheld in this snapshot:",
        "  - full prompt templates",
        "  - TTO / trust-update implementation details",
        "  - tool stacks & eval orchestration internals",
        "  - official Reasoning Agent",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_pipeline())
