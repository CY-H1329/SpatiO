# SpatiO

**Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning**

> **ECCV 2026 — Accepted** 🎉

Public **architecture release**: Head Agent + **5 specialist VLMs** + Reasoning Agent (TTO).

Implementation **details** (full prompts, TTO internals, tool stacks, proprietary reasoner wiring, large experiment sweeps) are **not** open-sourced here.  
They will be released later (and/or kept for patent / institutional reporting).

---

## Pipeline (what this repo exposes)

```
Image + Query
    → Head Agent              (category routing & role selection)
    → 5 Specialist Agents     (complementary spatial views)
    → Reasoning Agent (TTO)   (reliability-aware synthesis → final answer)
```

| Stage | Public here | Detailed code |
|-------|-------------|---------------|
| Head Agent | Interface + role in the pipeline | Later |
| 5 Specialists | Names + HF ids + thin loaders | Prompt / tool details later |
| Reasoning Agent | Slot in the pipeline | **Later release** (paper reasoner) |
| TTO / trust update | Paper hyperparameters in `config.py` | Full update rules later |
| Eval runners | High-level entry stubs | Full scripts later |

### Specialist pool (paper)

1. `llava4d` — `llava-hf/llava-1.5-7b-hf`  
2. `sa2va` — `ByteDance/Sa2VA-4B`  
3. `qwen3_4b` — `Qwen/Qwen3-VL-4B-Instruct`  
4. `spatial_rgpt` — `a8cheng/SpatialRGPT-VILA1.5-8B` (+ [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT))  
5. `spatial_reasoner` — `ccvl/SpatialReasoner`  

Head: `qwen3_4b`. Reasoning Agent: **coming later** (slot reserved as `deepseek_r1`).

## Figures (ECCV 2026)

### Figure 2 — Observation

![Figure 2: Observation](assets/figures/2_observation.png)

### Figure 3 — Main pipeline

![Figure 3: Main](assets/figures/3_main.png)

## Quick look (no proprietary details)

```bash
git clone https://github.com/CY-H1329/SpatiO.git
cd SpatiO
python skeleton.py
```

This prints the public pipeline layout and paper hyperparameters.  
It does **not** run the full private eval stack.

## Paper hyperparameters (defaults)

| Symbol | Value | Meaning |
|--------|-------|---------|
| κ (kappa) | 0.5 | Divergence penalty (train) |
| μ (mu) | 0.3 | Short- vs long-term balance |
| γ (gamma) | 0.3 | Direct reward injection |
| λf / λg | 0.3 / 0.1 | EMA decays |
| T / β | 5 / 5 | Ramp temperature / weight sharpness |

See [`config.py`](config.py).

## Citation

```bibtex
@inproceedings{spatio2026,
  title     = {SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

*(Author list and proceedings details will be updated for the camera-ready version.)*

## License

MIT License — see [LICENSE](LICENSE).  
Architecture / interfaces in this snapshot; detailed proprietary modules intentionally withheld.
