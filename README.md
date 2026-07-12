# SpatiO

**Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning**

> **ECCV 2026 — Accepted**

Basic code: **Head + 5 specialists + TTO**.  
Official **Reasoning Agent** comes later (interim Qwen3-VL / `--final_aggregator majority`).

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

## Setup & run

```bash
git clone https://github.com/CY-H1329/SpatiO.git && cd SpatiO
bash scripts/setup_env.sh && conda activate spatial_reasoning

git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python scripts/smoke_test.py
python evals/run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/cvbench
```

Full guide: [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Repository layout

```
SpatiO/
├── README.md
├── LICENSE
├── requirements.txt
├── assets/figures/     # paper figures
├── docs/               # reproduction notes
├── scripts/            # setup, smoke, verify
├── evals/              # benchmark entry points
└── spatio/             # core library (pipeline, models, roles, …)
```

## Hyperparameters

κ=0.5 · μ=0.3 · γ=0.3 · λf=0.3 · λg=0.1 · T=5 · β=5 — see `spatio/config.py`.

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
