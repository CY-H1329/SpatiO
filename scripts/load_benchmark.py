#!/usr/bin/env python3
"""Load a few HF benchmark samples (needs network)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import get_benchmark_answer, get_benchmark_image, get_benchmark_prompt, load_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="cvbench", choices=["cvbench", "3dsrbench", "stvqa", "mmsi_bench"])
    p.add_argument("--max_samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = load_benchmark(args.benchmark, max_samples=args.max_samples, seed=args.seed)
    for i, ex in enumerate(data):
        img = get_benchmark_image(ex, args.benchmark)
        prompt = get_benchmark_prompt(ex, args.benchmark)
        ans = get_benchmark_answer(ex, args.benchmark)
        print(f"[{i}] image={None if img is None else img.size} ans={ans!r} prompt_len={len(prompt or '')}")
    print(f"OK load_benchmark: {args.benchmark} n={len(data)}")


if __name__ == "__main__":
    main()
