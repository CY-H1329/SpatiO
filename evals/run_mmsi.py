#!/usr/bin/env python3
"""
SpatiO — MMSI-Bench evaluation.
Multi-image benchmark: images are concatenated horizontally.

Usage:
  python evals/run_mmsi.py --max_samples 500 --test_only
  python evals/run_mmsi.py --max_samples 500 --test_only --device_map="1,3,4,5,6"
  python evals/run_mmsi.py --max_samples 500 --test_only
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


from spatio.config import ALL_CATEGORIES, SPECIALIST_LLMS, HEAD_AGENT_MODEL, REASONING_AGENT_MODEL
from spatio.score_map import ScoreMap
from spatio.pipeline import run_step
from spatio.core import get_runner
from spatio.benchmarks import load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_answer, get_benchmark_category

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK = "mmsi_bench"


def main():
    parser = argparse.ArgumentParser(description="SpatiO — MMSI-Bench")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/spatio/mmsi_bench")
    parser.add_argument("--device_map", type=str, default=None, help="Comma-separated GPU IDs, e.g. 1,3,4,5,6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    update_trust = args.train and not args.test_only

    device_map = {}
    if args.device_map:
        devices = [d.strip() for d in args.device_map.split(",")]
        # Format as cuda:N for each device
        device_ids = []
        for d in devices:
            try:
                n = int(d)
                device_ids.append(f"cuda:{n}")
            except ValueError:
                device_ids.append(d)
        models = [HEAD_AGENT_MODEL, REASONING_AGENT_MODEL] + SPECIALIST_LLMS
        for i, m in enumerate(models):
            device_map[m] = device_ids[i % len(device_ids)]

    score_map = ScoreMap(categories=ALL_CATEGORIES)
    trust_state = {} if update_trust else None

    runners = {}

    def _head_gen(img, p):
        r = runners.get(HEAD_AGENT_MODEL)
        if r is None:
            r = get_runner(HEAD_AGENT_MODEL, device=device_map.get(HEAD_AGENT_MODEL, "cuda:0"))
            r.load()
            runners[HEAD_AGENT_MODEL] = r
        return r.generate(img, p)

    def _spec_gen(name, img, p):
        r = runners.get(name)
        if r is None:
            r = get_runner(name, device=device_map.get(name, "cuda:0"))
            r.load()
            runners[name] = r
        return r.generate(img, p)

    def _reason_gen(p, img=None, image=None, **kwargs):
        im = img if img is not None else image
        r = runners.get(REASONING_AGENT_MODEL)
        if r is None:
            r = get_runner(REASONING_AGENT_MODEL, device=device_map.get(REASONING_AGENT_MODEL, "cuda:0"))
            r.load()
            runners[REASONING_AGENT_MODEL] = r
        return r.generate(im, p)

    max_samples = None if args.full else args.max_samples
    ds = load_benchmark(BENCHMARK, max_samples=max_samples, seed=args.seed)
    N_c_per_category = {}
    results = []
    correct_count = 0
    total = 0

    suffix = "full" if args.full else str(max_samples or args.max_samples)
    output_dir = Path(args.output_dir) / suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(len(ds)):
        ex = ds[step]
        image = get_benchmark_image(ex, BENCHMARK)
        query = get_benchmark_prompt(ex, BENCHMARK)
        gt = get_benchmark_answer(ex, BENCHMARK)

        if image is None:
            logger.warning("Step %d: no image, skipping", step)
            continue

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=step,
            score_map=score_map,
            trust_state=trust_state,
            head_generate=_head_gen,
            specialist_generate=_spec_gen,
            reasoning_generate=_reason_gen,
            N_c_per_category=N_c_per_category,
            update_trust=update_trust,
            use_beta_weights=True,
        )

        total += 1
        if result.get("correct"):
            correct_count += 1
        N_c_per_category[result["category"]] = N_c_per_category.get(result["category"], 0) + 1
        r = {"step": step, "final": result["final_answer"], "gt": gt, "correct": result.get("correct"), "category": result["category"]}
        qt = get_benchmark_category(ex, BENCHMARK)
        if qt is not None:
            r["question_type"] = qt
        if "timing" in result:
            r["timing"] = result["timing"]
        if "assignments" in result:
            r["assignments"] = result["assignments"]
        results.append(r)

        logger.info("Step %d | cat=%s | final=%s | gt=%s | ok=%s", step, result["category"], result["final_answer"], gt, result.get("correct"))

    acc = 100.0 * correct_count / total if total > 0 else 0.0
    logger.info("MMSI-Bench Accuracy: %.2f%% (%d/%d)", acc, correct_count, total)

    summary = {
        "benchmark": BENCHMARK,
        "max_samples": max_samples,
        "full": args.full,
        "total": total,
        "correct": correct_count,
        "accuracy": acc,
        "train": update_trust,
        "seed": args.seed,
        "device_map": args.device_map,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
