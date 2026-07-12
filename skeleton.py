"""Print Head + 5 specialists + Reasoning layout (no GPU)."""

from config import (
    ALL_CATEGORIES,
    BETA,
    GAMMA,
    HEAD_AGENT_MODEL,
    KAPPA,
    LAMBDA_F,
    LAMBDA_G,
    MU,
    RAMP_TEMP,
    REASONING_AGENT_MODEL,
    ROLES,
    SPECIALIST_LLMS,
    TOP_K_SPECIALISTS,
)


def main() -> None:
    print("SpatiO")
    print(f"  head:       {HEAD_AGENT_MODEL}")
    print(f"  specialists (top_k={TOP_K_SPECIALISTS}):")
    for i, name in enumerate(SPECIALIST_LLMS, 1):
        print(f"    {i}. {name}")
    print(f"  roles:      {', '.join(ROLES)}")
    print(f"  reasoning:  {REASONING_AGENT_MODEL}  [official agent later]")
    print(f"  hparams:    κ={KAPPA} μ={MU} γ={GAMMA} λf={LAMBDA_F} λg={LAMBDA_G} T={RAMP_TEMP} β={BETA}")
    print(f"  categories: {', '.join(ALL_CATEGORIES)}")


if __name__ == "__main__":
    main()
