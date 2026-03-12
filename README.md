# SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning

Supplementary materials for **SpatiO** — adaptive test-time orchestration and multi-agent system for spatial reasoning benchmarks.

## Architecture

- **Head Agent**: Category inference from image + query (Qwen3-VL-4B)
- **5 Specialists**: LLaVA-4D, Qwen3-VL-4B, SpatialRGPT, SpatialReasoner, Sa2VA
- **Reasoning Agent**: Final synthesis with reliability-aware weights (DeepSeek-R1 or Qwen3-VL-8B)

## Hyperparameters (Paper Defaults)

| Symbol | Name | Value | Description |
|--------|------|-------|-------------|
| k | kappa | 0.5 | Penalty strength when final answer diverges from GT |
| μ | mu | 0.3 | Balance between short-term (f) and long-term (g) EMA |
| γ | gamma | 0.3 | Direct reward injection into final score |
| λf | lambda_f | 0.3 | Short-term EMA decay |
| λg | lambda_g | 0.1 | Long-term EMA decay |
| T | ramp_temp | 5 | Ramp temperature for φ(N_c) = 1 − exp(−N_c/T) |
| β | beta | 5 | Weight sharpness: w = exp(β·s) / Σ exp(β·s') |

## Structure

```
SpatiO/
├── README.md
├── config.py          # Hyperparameters and categories
├── core/               # Model runners (load from your implementation)
│   ├── __init__.py
│   ├── base.py        # Abstract runner interface
│   └── runners.py     # Specialist runners
├── trust_score.py     # TTO: reward, φ scaling, Beta+EMA, weight computation
├── prompts.py        # Head, specialist, and reasoning prompts
├── score_map.py      # Per-category score maps
├── pipeline.py       # Main MAS pipeline
└── main.py           # Entry point
```

## Usage

1. Implement model runners in `core/runners.py` (or load from your model repository).
2. Run: `python main.py --benchmark cvbench --max_samples 100`

## Dependencies

- PyTorch, Transformers
- PIL, datasets (HuggingFace)
