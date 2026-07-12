# Reproduction

Basic runnable release: Head + **5 specialists** + TTO.  
Official Reasoning Agent: **later** (interim Qwen3-VL, or `--final_aggregator majority`).

## 1. Environment

```bash
bash scripts/setup_env.sh
conda activate spatial_reasoning
```

Requires **Python ≥ 3.10**, CUDA GPU(s), and conda.  
Deps: `requirements.txt` / `requirements-no-torch.txt`.

## 2. SpatialRGPT

```bash
git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"
```

## 3. Checks

```bash
python scripts/smoke_test.py
python scripts/verify_env.py
```

## 4. Eval

`device_map` order: `head, reasoner, llava4d, sa2va, qwen3_4b, spatial_rgpt, spatial_reasoner`.

```bash
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/cvbench

python run_3dsrbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 --output_dir results/3dsr

# optional: no reasoner VLM
python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --final_aggregator majority \
  --device_map 0,0,1,2,3,4,5 --output_dir results/cvbench_majority
```

| Script | Dataset |
|--------|---------|
| `run_cvbench.py` | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| `run_3dsrbench.py` | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) |
| `run_stvqa.py` | [hunarbatra/STVQA-7K](https://huggingface.co/datasets/hunarbatra/STVQA-7K) |
| `run_mmsi.py` | [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) |

Optional image cache: `export SPATIO_IMAGE_CACHE=/path/to/cache`.

## Not in this release

Experiment sweeps, MindCube tooling, official Reasoning Agent, private checkpoints.
