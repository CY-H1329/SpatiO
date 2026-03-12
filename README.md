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
├── config.py              # Hyperparameters and categories
├── core/                   # Model runners (load from your implementation)
│   ├── base.py
│   └── runners.py
├── benchmarks/             # Benchmark loaders
│   ├── loaders.py         # cvbench, 3dsrbench, stvqa
├── trust_score.py         # TTO: reward, φ scaling, Beta+EMA, weight computation
├── prompts.py             # Head, specialist, and reasoning prompts
├── score_map.py
├── shared_memory.py
├── pipeline.py
├── main.py                # Generic entry point
├── run_cvbench.py         # CV-Bench evaluation
├── run_3dsrbench.py       # 3DSRBench evaluation
├── run_stvqa.py           # STVQA-7K evaluation
├── run_all.py             # Run all benchmarks
├── experiments/
│   ├── run_all.sh         # Batch run all benchmarks
│   ├── run_cvbench.sh
│   ├── run_3dsrbench.sh
│   └── run_stvqa.sh
└── tools/                  # Optional: 3D representation, scene graph
```

## Benchmarks

| Benchmark | HuggingFace | Notes |
|-----------|-------------|-------|
| CV-Bench | nyu-visionx/CV-Bench | Count, Relation, Depth, Distance |
| 3DSRBench | ccvl/3DSRBench | 12 fine-grained spatial categories |
| STVQA-7K | hunarbatra/STVQA-7K | Spatial reasoning QA |

## Usage

### Per-benchmark run scripts

```bash
# 50 samples (default)
python run_cvbench.py --max_samples 50 --test_only --output_dir results/cvbench
python run_3dsrbench.py --max_samples 50 --test_only --output_dir results/3dsrbench
python run_stvqa.py --max_samples 50 --test_only --output_dir results/stvqa

# Full dataset
python run_cvbench.py --full --test_only --output_dir results/cvbench
python run_3dsrbench.py --full --test_only --output_dir results/3dsrbench
python run_stvqa.py --full --test_only --output_dir results/stvqa
```

### Run all benchmarks

```bash
# 50 samples (default)
python run_all.py --benchmarks cvbench,3dsrbench,stvqa --max_samples 50 --test_only

# Full dataset
python run_all.py --benchmarks cvbench,3dsrbench,stvqa --full --test_only
```

### Shell scripts (experiments)

```bash
# Run all: 50 + full for each benchmark
bash experiments/run_all.sh

# Per benchmark: 50 (default) or full
bash experiments/run_cvbench.sh
MODE=full bash experiments/run_cvbench.sh
MODE=50 bash experiments/run_3dsrbench.sh
MODE=full bash experiments/run_3dsrbench.sh
```

Output: `results/spatio/<benchmark>/50/` and `results/spatio/<benchmark>/full/`

### Generic entry point

```bash
python main.py --benchmark cvbench --max_samples 100
python main.py --benchmark 3dsrbench --max_samples 50 --train
```

## Dependencies

- PyTorch, Transformers
- PIL, datasets (HuggingFace), requests
