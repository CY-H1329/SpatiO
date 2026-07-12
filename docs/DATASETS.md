# Datasets

All primary benchmarks load from **Hugging Face** via [`benchmarks/loaders.py`](../benchmarks/loaders.py).

| Benchmark | HF id | Script |
|-----------|-------|--------|
| CV-Bench | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | `run_cvbench.py` |
| 3DSRBench | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) | `run_3dsrbench.py` |
| STVQA-7K | [hunarbatra/STVQA-7K](https://huggingface.co/datasets/hunarbatra/STVQA-7K) | `run_stvqa.py` |
| MMSI-Bench | [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) | `run_mmsi.py` |

Image URL caches (3DSR / some STVQA paths): set `SPATIO_IMAGE_CACHE=/path/to/cache`.

MindCube (optional, multi-view) needs a local TSV + image root — see scripts under `scripts/` when published with that tooling.
