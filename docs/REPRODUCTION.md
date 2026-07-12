# Reproduction

Basic runnable release: Head + **5 specialists** + TTO.  
Official Reasoning Agent: **later** (interim Qwen3-VL, or `--final_aggregator majority`).

## 1. Environment

```bash
bash scripts/setup_env.sh
conda activate spatial_reasoning
```

Python ≥ 3.10, CUDA, conda. See `requirements.txt`.

## 2. SpatialRGPT

```bash
git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"
```

## 3. Checks

```bash
python scripts/smoke_test.py
python scripts/verify_env.py
python -m spatio.skeleton
```

## 4. Eval

`device_map`: `head, reasoner, llava4d, sa2va, qwen3_4b, spatial_rgpt, spatial_reasoner`.

```bash
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python evals/run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/cvbench

python evals/run_3dsrbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/3dsr

python evals/run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --final_aggregator majority \
  --device_map 0,0,1,2,3,4,5 --output_dir results/cvbench_majority
```

| Script | Dataset |
|--------|---------|
| `evals/run_cvbench.py` | [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| `evals/run_3dsrbench.py` | [3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) |
| `evals/run_stvqa.py` | [STVQA-7K](https://huggingface.co/datasets/hunarbatra/STVQA-7K) |
| `evals/run_mmsi.py` | [MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) |

Optional: `export SPATIO_IMAGE_CACHE=/path/to/cache`.

## Not in this release

Experiment sweeps, MindCube tooling, official Reasoning Agent, private checkpoints.
