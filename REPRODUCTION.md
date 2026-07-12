# Reproduction (basic public code)

Runnable **basic** stack: Head + **all 5 specialists** + TTO pipeline.  
Official **Reasoning Agent** comes later (interim Qwen3-VL stand-in or `--final_aggregator majority`).

## Setup

```bash
git clone https://github.com/CY-H1329/SpatiO.git
cd SpatiO
bash scripts/setup_spatial_reasoning_env_fast.sh   # needs conda, Python 3.10+
conda activate spatial_reasoning

git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python scripts/verify_spatial_reasoning_env.py
python scripts/smoke_pipeline_mock.py
```

## Run (5 specialists)

```bash
unset SPATIO_PROFILE
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 \
  --output_dir results/cvbench_basic

python run_3dsrbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 \
  --output_dir results/3dsr_basic
```

Without the interim reasoner VLM:

```bash
python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --final_aggregator majority \
  --device_map 0,0,1,2,3,4,5 \
  --output_dir results/cvbench_majority
```

## Datasets

HF: CV-Bench, 3DSRBench, STVQA-7K, MMSI-Bench — see [`docs/DATASETS.md`](docs/DATASETS.md).

## Scope

| Included | Not in this basic release |
|----------|---------------------------|
| Pipeline / TTO / prompts / roles | Large `experiments/` sweeps |
| 5 specialist backends | Official Reasoning Agent |
| Benchmark runners | MindCube full tooling dump |
| Env + smoke scripts | Private checkpoints |
