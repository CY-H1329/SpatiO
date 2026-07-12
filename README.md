# SpatiO

**Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning**

> **ECCV 2026 — Accepted** 🎉

Basic **runnable** code for SpatiO: Head Agent + **5 specialist VLMs** + Reasoning slot (TTO).

This is a **basic public release** for reproduction of the overall pipeline — not every internal experiment or proprietary detail.

> **Reasoning Agent:** the paper’s official Reasoning Agent will be released **later**.  
> This snapshot ships a **temporary Qwen3-VL stand-in** (or use `--final_aggregator majority`).

---

## Pipeline

```
Image + Query → Head → 5 Specialists → Reasoning (TTO) → Final Answer
```

| Specialist | Hugging Face id |
|------------|-----------------|
| `llava4d` | `llava-hf/llava-1.5-7b-hf` |
| `sa2va` | `ByteDance/Sa2VA-4B` |
| `qwen3_4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| `spatial_rgpt` | `a8cheng/SpatialRGPT-VILA1.5-8B` |
| `spatial_reasoner` | `ccvl/SpatialReasoner` |

Head: `qwen3_4b`. Default `top_k=5`.

## Figures (ECCV 2026)

### Figure 2 — Observation

![Figure 2: Observation](assets/figures/2_observation.png)

### Figure 3 — Main pipeline

![Figure 3: Main](assets/figures/3_main.png)

## Quick start

```bash
git clone https://github.com/CY-H1329/SpatiO.git
cd SpatiO

# Python ≥ 3.10
bash scripts/setup_spatial_reasoning_env_fast.sh
conda activate spatial_reasoning   # or the env name you created

git clone https://github.com/AnjieCheng/SpatialRGPT.git ../SpatialRGPT
export SPATIALRGPT_PATH="$(cd ../SpatialRGPT && pwd)"

python scripts/smoke_pipeline_mock.py
python scripts/verify_spatial_reasoning_env.py

# Full 5-specialist eval (multi-GPU recommended)
# device_map: head, reasoner*, llava4d, sa2va, qwen3_4b, spatial_rgpt, spatial_reasoner
python run_cvbench.py --max_samples 50 --test_only --top_k 5 \
  --device_map 0,1,2,3,4,5,6 \
  --output_dir results/cvbench_basic
```

See [REPRODUCTION.md](REPRODUCTION.md) and [docs/MODELS.md](docs/MODELS.md).

**Not included (on purpose):** large ablation sweeps, MindCube tooling dumps, official Reasoning Agent.

## Paper hyperparameters

| Symbol | Value |
|--------|-------|
| κ / μ / γ | 0.5 / 0.3 / 0.3 |
| λf / λg | 0.3 / 0.1 |
| T / β | 5 / 5 |

[`config.py`](config.py)

## Citation

```bibtex
@inproceedings{spatio2026,
  title     = {SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
