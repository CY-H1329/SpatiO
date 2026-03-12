# SpatiO Reproduction Notes

Known issues in the supplementary code and fixes for reproduction.

---

## 1. Known Issues in Supplementary Scripts

### 1.1 `run_3dsrbench.py`, `run_cvbench.py`, `run_stvqa.py`

| Issue | Symptom |
|-------|---------|
| `reasoning_generate(prompt, image=image)` | Pipeline uses `image=` keyword; callback must accept it |
| `--device_map="1,3,4"` | Runners expect `cuda:1`, `cuda:3` format |

### 1.2 `core/runners.py` — model_id Mapping

- Config uses `HEAD_AGENT_MODEL = "qwen3_4b"` which is passed to HuggingFace
- `"qwen3_4b"` is not a valid model identifier → `RepositoryNotFoundError`
- Fix: Map `qwen3_4b` → `Qwen/Qwen3-VL-4B-Instruct` in model wrappers

### 1.3 `models/` Package

- The supplementary package does not include `models/` by default
- `from models.llava4d import LLaVA4DModel` → `ImportError`
- Fix: Implement `models/` with wrappers, or set `MODEL_ROOT` to the path of a host project containing VLM implementations

### 1.4 `device_map` Format

- `--device_map="1,3,4,5,6"` passes `"1"`, `"3"` as strings
- Runners expect `"cuda:1"`, `"cuda:3"` format
- Fix: Convert with `f"cuda:{d}"` when `d.isdigit()`

---

## 2. Applied Fixes

- Run scripts: `_reason_gen` signature to accept `image=` keyword; `device_map` formatting
- `core/runners.py`: `MODEL_ROOT` support for loading from host project

---

## 3. Reproduction Checklist

1. **models package**: Implement `models/` with wrappers for each VLM, or set `MODEL_ROOT` env to the path containing VLM implementations
2. **model_id mapping**: In model wrappers, map `qwen3_4b` → `Qwen/Qwen3-VL-4B-Instruct`
3. **device_map**: Use `cuda:N` format (e.g. `--device_map 0,1,2,3` or `--device_map cuda:0,cuda:1`)
