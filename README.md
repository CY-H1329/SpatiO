# SpatiO

**Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning**

> **ECCV 2026 — Accepted**

Official basic code: **Head + 5 specialists + TTO reasoning slot**.

The paper’s **official Reasoning Agent** will be released later.  
This repo uses a temporary Qwen3-VL stand-in (or `--final_aggregator majority`).

```
Image + Query → Head → 5 Specialists → Reasoning (TTO) → Answer
```

## Figures

| Observation | Pipeline |
|:-----------:|:--------:|
| ![Fig.2](assets/figures/2_observation.png) | ![Fig.3](assets/figures/3_main.png) |

## Models

| Role | Id | Checkpoint |
|------|-----|------------|
| Head | `qwen3_4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| Specialist | `llava4d` | `llava-hf/llava-1.5-7b-hf` |
| Specialist | `sa2va` | `ByteDance/Sa2VA-4B` |
| Specialist | `qwen3_4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| Specialist | `spatial_rgpt` | `a8cheng/SpatialRGPT-VILA1.5-8B` |
| Specialist | `spatial_reasoner` | `ccvl/SpatialReasoner` |
| Reasoning | `deepseek_r1` | *later* (interim: Qwen3-VL-8B) |

Default `top_k=5`. SpatialRGPT needs [`AnjieCheng/SpatialRGPT`](https://github.com/AnjieCheng/SpatialRGPT) and `SPATIALRGPT_PATH`.

## Setup

```bash
git clone https://github.com/CY-H1329/SpatiO.git
cd SpatiO

bash scripts/setup_env.sh          # conda env, Python ≥ 3.10
conda activate spatial_reasoning

git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python scripts/smoke_test.py
```

## Run

```bash
# device_map: head, reasoner, then 5 specialists
python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/cvbench

python run_3dsrbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/3dsr
```

Also: `run_stvqa.py`, `run_mmsi.py`. Details: [REPRODUCTION.md](REPRODUCTION.md).

## Layout

```
SpatiO/
  pipeline.py, trust_score.py, prompts.py, config.py
  models/          # VLM backends (5 specialists + interim reasoner)
  roles/           # specialist role prompts
  benchmarks/      # HF loaders
  run_*.py         # eval entry points
  scripts/         # setup + smoke
  assets/figures/  # paper figures
```

## Hyperparameters

| κ | μ | γ | λf | λg | T | β |
|---|---|---|----|----|---|---|
| 0.5 | 0.3 | 0.3 | 0.3 | 0.1 | 5 | 5 |

See `config.py`.

## Citation

```bibtex
@inproceedings{spatio2026,
  title     = {SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT — [LICENSE](LICENSE).
