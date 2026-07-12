import ast
import csv
import hashlib
import io
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
import requests

IMAGE_CACHE_DIR = os.environ.get("SPATIO_IMAGE_CACHE")

BENCHMARK_CONFIGS = {
    "cvbench": {
        "name": "nyu-visionx/CV-Bench",
        "split": "test",
        "image_key": "image",
        "question_key": "question",
        "options_key": "choices",
        "answer_key": "answer",
        "category_key": "task",
    },
    "3dsrbench": {
        "name": "ccvl/3DSRBench",
        "split": "test",
        "subset": "benchmark",
        "image_key": "image_url",
        "question_key": "question",
        "options_keys": ["A", "B", "C", "D"],
        "answer_key": "answer",
        "category_key": "category",
    },
    "stvqa": {
        "name": "hunarbatra/STVQA-7K",
        "split": "train",
        "image_key": "images",
        "question_key": "question_with_options",
        "question_fallback": "question_only",
        "options_key": "options",
        "answer_key": "answer",
        "answer_fallback": "answer_only",
        "category_key": "category",
    },
    "mmsi_bench": {
        "name": "RunsenXu/MMSI-Bench",
        "split": "test",
        "image_key": "images",  # list of PIL images (multi-image)
        "question_key": "question",
        "answer_key": "answer",
        "category_key": "question_type",
    },
    # MindCube: TSV/JSONL + répertoire racine des images (voir run_mindcube_tto.py)
    "mindcube": {
        "question_key": "question",
        "answer_key": "answer",
        "category_key": "category",
        "image_key": "image_paths",
    },
}


def _url_to_cache_path(url: str) -> Optional[Path]:
    if not IMAGE_CACHE_DIR:
        return None
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = ".jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else ".png"
    return Path(IMAGE_CACHE_DIR) / f"{key}{ext}"


def _fetch_image_from_url(url: str) -> Optional[Image.Image]:
    cache_path = _url_to_cache_path(url)
    if cache_path and cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if cache_path:
            Path(IMAGE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            img.save(cache_path, quality=95)
        return img
    except Exception:
        return None


def _load_mindcube_tsv(tsv_path: Path, images_root: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("question"):
                rows.append(row)
    rng = random.Random(seed)
    if max_samples and len(rows) > max_samples:
        pick = rng.sample(range(len(rows)), max_samples)
        pick.sort()
        rows = [rows[i] for i in pick]
    examples: List[Dict[str, Any]] = []
    for row in rows:
        raw_paths = (row.get("image_path") or "").strip()
        try:
            paths = ast.literal_eval(raw_paths) if raw_paths.startswith("[") else [raw_paths]
        except (SyntaxError, ValueError):
            paths = [raw_paths] if raw_paths else []
        paths = [str(p).strip() for p in paths if str(p).strip()]
        examples.append(
            {
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("answer") or "").strip(),
                "category": (row.get("category") or "unknown").strip(),
                "image_paths": paths,
            }
        )
    return examples


def load_benchmark(
    benchmark: str,
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    category_filter: Optional[List[str]] = None,
    seed: int = 42,
):
    """Load benchmark from HuggingFace."""
    if benchmark == "mindcube":
        tsv = os.environ.get("SPATIO_MINDCUBE_TSV")
        if not tsv:
            raise ValueError(
                "MindCube: définir SPATIO_MINDCUBE_TSV (chemin vers un TSV type MindCubeBench_tiny_raw_qa.tsv) "
                "et SPATIO_MINDCUBE_IMAGES_ROOT (racine où se trouve data/other_all_image/...)."
            )
        root = Path(os.environ.get("SPATIO_MINDCUBE_IMAGES_ROOT", ".")).resolve()
        return _load_mindcube_tsv(Path(tsv).resolve(), root, max_samples, seed)

    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

    cfg = BENCHMARK_CONFIGS[benchmark]
    name = cfg["name"]
    split = cfg["split"]
    subset = cfg.get("subset")

    load_kw = {"split": split}
    # trust_remote_code 비사용 (deprecated / loading script 없음)
    if benchmark not in ("mmsi_bench", "stvqa"):
        load_kw["trust_remote_code"] = True

    if benchmark == "stvqa":
        # STVQA-7K: parquet URL → pyarrow → Dataset (HF cache/metadata 호환 이슈 회피)
        try:
            import io
            import pyarrow as pa
            import pyarrow.parquet as pq
            from datasets import Dataset as HFDataset
            base = "https://huggingface.co/datasets/hunarbatra/STVQA-7K/resolve/main/data"
            url = f"{base}/{split}-00000-of-00001.parquet"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            table = pq.read_table(io.BytesIO(r.content))
            new_schema = pa.schema([pa.field(f.name, f.type) for f in table.schema])
            table = table.cast(new_schema)
            ds = HFDataset(table)
        except Exception as e:
            raise RuntimeError(f"STVQA-7K load failed: {e}") from e
    else:
        try:
            if subset:
                ds = load_dataset(name, subset, **load_kw)
            else:
                ds = load_dataset(name, **load_kw)
        except Exception as e:
            if benchmark == "cvbench":
                try:
                    base = "https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/main"
                    data_files = {"test": [f"{base}/test_2d.parquet", f"{base}/test_3d.parquet"]}
                    ds = load_dataset("parquet", data_files=data_files, split="test")
                except Exception as parquet_err:
                    raise RuntimeError(f"CV-Bench load failed: {e}, parquet: {parquet_err}") from parquet_err
            else:
                raise

    rng = random.Random(seed)
    cat_key = cfg.get("category_key")

    if category_filter and cat_key and cat_key in ds.features:
        cats_set = set(c.strip() for c in category_filter)
        indices = [i for i in range(len(ds)) if (ds[i].get(cat_key) or "").strip() in cats_set]
        if not indices and len(ds) > 0:
            # Fallback: case-insensitive match (dataset may use different casing)
            indices = [i for i in range(len(ds)) if str(ds[i].get(cat_key) or "").strip().lower() in {c.lower() for c in cats_set}]
        if indices:
            ds = ds.select(indices)
        # else: category_filter matched nothing; continue with full ds (or empty)

    if max_per_category and cat_key and cat_key in ds.features:
        by_cat = {}
        for i in range(len(ds)):
            c = ds[i].get(cat_key) or "unknown"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            idx_list = by_cat[c]
            k = min(max_per_category, len(idx_list))
            indices.extend(rng.sample(idx_list, k))
        indices.sort()
        ds = ds.select(indices)
    elif max_samples:
        n = min(max_samples, len(ds))
        indices = rng.sample(range(len(ds)), n)
        indices.sort()
        ds = ds.select(indices)

    return ds


def _concatenate_images(images: List) -> Optional[Image.Image]:
    """Concatenate multiple images horizontally for multi-image benchmarks (MMSI-Bench)."""
    if not images:
        return None
    if len(images) == 1:
        img = images[0]
        return img.convert("RGB") if hasattr(img, "convert") else img
    imgs_rgb = []
    for img in images:
        if hasattr(img, "convert"):
            imgs_rgb.append(img.convert("RGB") if img.mode != "RGB" else img)
        else:
            return None
    target_h = min(max(img.height for img in imgs_rgb), 512)
    max_w = 384
    resized = []
    for img in imgs_rgb:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        if new_w > max_w:
            ratio = max_w / img.width
            new_w = max_w
            new_h = int(img.height * ratio)
        else:
            new_h = target_h
        resized.append(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
    h = max(r.height for r in resized)
    total_w = sum(r.width for r in resized)
    out = Image.new("RGB", (total_w, h))
    x = 0
    for r in resized:
        out.paste(r, (x, (h - r.height) // 2))
        x += r.width
    return out


def get_benchmark_image(example: Dict, benchmark: str) -> Optional[Image.Image]:
    if benchmark == "mindcube":
        paths = example.get("image_paths") or []
        root = Path(os.environ.get("SPATIO_MINDCUBE_IMAGES_ROOT", ".")).resolve()
        imgs: List[Image.Image] = []
        for p in paths:
            full = Path(p) if Path(p).is_absolute() else (root / p)
            if not full.exists():
                continue
            try:
                imgs.append(Image.open(full).convert("RGB"))
            except Exception:
                continue
        return _concatenate_images(imgs) if imgs else None

    cfg = BENCHMARK_CONFIGS[benchmark]
    img_key = cfg["image_key"]

    if img_key == "image_url":
        url = example.get(img_key)
        if url:
            return _fetch_image_from_url(url)
        return None

    img = example.get("images") or example.get("image")
    if img is None:
        return None
    # Parquet/Dataset from Arrow can store image as dict (e.g. {"bytes": b"..."}) or raw bytes
    if isinstance(img, dict):
        if "bytes" in img:
            try:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            except Exception:
                return None
        elif "path" in img:
            try:
                img = Image.open(img["path"]).convert("RGB")
            except Exception:
                return None
        else:
            return None
    elif isinstance(img, bytes):
        try:
            img = Image.open(io.BytesIO(img)).convert("RGB")
        except Exception:
            return None
    if isinstance(img, (list, tuple)):
        return _concatenate_images(list(img))
    if hasattr(img, "convert"):
        return img.convert("RGB")
    return img


def get_benchmark_prompt(example: Dict, benchmark: str, include_options: bool = True) -> str:
    cfg = BENCHMARK_CONFIGS[benchmark]
    q_key = cfg["question_key"]
    question = (example.get(q_key) or "").strip()
    if not question and cfg.get("question_fallback"):
        question = (example.get(cfg["question_fallback"]) or "").strip()

    if not include_options:
        return question

    opts_key = cfg.get("options_key")
    opts_keys = cfg.get("options_keys")

    if opts_key and opts_key in example:
        opts = example[opts_key]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)
    elif opts_keys:
        opts = [example.get(k) for k in opts_keys if example.get(k)]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)

    return question


def get_benchmark_answer(example: Dict, benchmark: str) -> str:
    cfg = BENCHMARK_CONFIGS[benchmark]
    ans_key = cfg["answer_key"]
    ans = example.get(ans_key) or ""
    if not ans and cfg.get("answer_fallback"):
        ans = example.get(cfg["answer_fallback"]) or ""
    s = str(ans).strip()
    if benchmark == "mindcube":
        u = s.upper()
        if len(u) == 1 and u in "ABCDEF":
            return f"({u})"
        for c in "ABCDEF":
            if f"({c})" in u or u == c:
                return f"({c})"
        return s
    if cfg.get("options_key") or cfg.get("options_keys"):
        for c in "ABCDEF":
            if f"({c})" in s.upper() or s.upper() == c:
                return f"({c})"
    return s


def get_benchmark_category(example: Dict, benchmark: str) -> Optional[str]:
    cfg = BENCHMARK_CONFIGS[benchmark]
    cat_key = cfg.get("category_key")
    if cat_key and cat_key in example:
        return str(example[cat_key])
    return None
