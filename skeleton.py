"""
SpatiO — high-level layout check (no GPU required).

For real evals see REPRODUCTION.md / run_cvbench.py.
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
    TOP_K_SPECIALISTS,
)


def describe_pipeline() -> str:
    lines = [
        "SpatiO basic public pipeline",
        f"  Head: {HEAD_AGENT_MODEL}",
        f"  Specialists ({len(SPECIALIST_LLMS)}, top_k={TOP_K_SPECIALISTS}):",
    ]
    for i, name in enumerate(SPECIALIST_LLMS, 1):
        lines.append(f"    {i}. {name}")
    lines += [
        f"  Roles: {', '.join(ROLES)}",
        f"  Reasoning: {REASONING_AGENT_MODEL} (official agent later; interim stand-in in models/reasoning.py)",
        "",
        f"  kappa={KAPPA} mu={MU} gamma={GAMMA} lambda_f={LAMBDA_F} lambda_g={LAMBDA_G}",
        f"  ramp_temp={RAMP_TEMP} beta={BETA}",
        "  categories: " + ", ".join(ALL_CATEGORIES),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_pipeline())
