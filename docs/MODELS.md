# Models

Default specialist pool (**all 5**):

| Config | Weights |
|--------|---------|
| `llava4d` | `llava-hf/llava-1.5-7b-hf` |
| `sa2va` | `ByteDance/Sa2VA-4B` |
| `qwen3_4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| `spatial_rgpt` | `a8cheng/SpatialRGPT-VILA1.5-8B` (+ `SPATIALRGPT_PATH`) |
| `spatial_reasoner` | `ccvl/SpatialReasoner` |

Head: `qwen3_4b`.  
Reasoning slot `deepseek_r1`: **official agent later**; interim Qwen3-VL in `models/reasoning.py`.

Backends live under `models/_backend_*.py`.
