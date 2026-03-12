"""
SpatialTTO / SpatialMAS — Entry point for supplement.

Usage:
  python main.py --benchmark cvbench --max_samples 100
  python main.py --benchmark 3dsrbench --device_map cuda:0,cuda:1,cuda:2

Models are loaded from core/runners. Implement load() in each runner
to load from your model repository. Use device='cuda:X' for multi-GPU.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ALL_CATEGORIES,
    SPECIALIST_LLMS,
    HEAD_AGENT_MODEL,
    REASONING_AGENT_MODEL,
)
from score_map import ScoreMap
from pipeline import run_step
from core import get_runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_head_generate(runners, device_map):
    """Head agent uses Qwen3-VL-4B."""
    device = device_map.get(HEAD_AGENT_MODEL, "cuda:0") if device_map else "cuda:0"
    runner = get_runner(HEAD_AGENT_MODEL, device=device)
    runner.load()

    def fn(image, prompt):
        return runner.generate(image, prompt)

    return fn


def _make_specialist_generate(runners, device_map):
    """Specialist generate: llm_name -> calls appropriate runner."""
    def fn(llm_name, image, prompt):
        device = device_map.get(llm_name, "cuda:0") if device_map else "cuda:0"
        if llm_name not in runners:
            runners[llm_name] = get_runner(llm_name, device=device)
            runners[llm_name].load()
        return runners[llm_name].generate(image, prompt)

    return fn


def _make_reasoning_generate(runners, device_map):
    """Reasoning agent: DeepSeek-R1 or Qwen3-VL. Implement load() in core/runners."""
    device = device_map.get(REASONING_AGENT_MODEL, "cuda:0") if device_map else "cuda:0"
    model = REASONING_AGENT_MODEL
    if model not in runners:
        runner = get_runner(model, device=device)
        runner.load()
        runners[model] = runner

    def fn(prompt, image=None):
        return runners[model].generate(image, prompt)

    return fn


def main():
    parser = argparse.ArgumentParser(description="SpatialTTO / SpatialMAS Supplement")
    parser.add_argument("--benchmark", type=str, default="cvbench", choices=["cvbench", "3dsrbench"])
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--device_map", type=str, default=None,
                        help="Comma-separated device map, e.g. cuda:0,cuda:1,cuda:2")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train", action="store_true", help="Train phase (update trust)")
    args = parser.parse_args()

    device_map = {}
    if args.device_map:
        devices = [d.strip() for d in args.device_map.split(",")]
        models = [HEAD_AGENT_MODEL, REASONING_AGENT_MODEL] + SPECIALIST_LLMS
        for i, m in enumerate(models):
            device_map[m] = devices[i % len(devices)]

    score_map = ScoreMap(categories=ALL_CATEGORIES)
    trust_state = {} if args.train else None

    runners = {}
    head_generate = _make_head_generate(runners, device_map)
    specialist_generate = _make_specialist_generate(runners, device_map)
    reasoning_generate = _make_reasoning_generate(runners, device_map)

    # Placeholder dataset — replace with your benchmark loader
    try:
        from datasets import load_dataset
        if args.benchmark == "cvbench":
            ds = load_dataset("cvbench", split="test", trust_remote_code=True)
        else:
            ds = load_dataset("3dsrbench", split="test", trust_remote_code=True)
    except Exception:
        logger.warning("Benchmark dataset not found. Using placeholder.")
        ds = [{"image": None, "question": "How many objects?", "answer": "(A)"}] * min(args.max_samples, 5)

    N_c_per_category = {}
    correct_count = 0
    total = 0

    for step in range(min(args.max_samples, len(ds))):
        ex = ds[step]
        image = ex.get("image")
        query = ex.get("question", ex.get("prompt", "How many objects?"))
        gt = ex.get("answer")

        if image is None:
            logger.warning("Step %d: no image, skipping", step)
            continue

        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB") if hasattr(image, "__fspath__") else None
        if image is None:
            continue

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=step,
            score_map=score_map,
            trust_state=trust_state,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            N_c_per_category=N_c_per_category,
            update_trust=args.train,
            use_beta_weights=True,
        )

        total += 1
        if result.get("correct"):
            correct_count += 1
        N_c_per_category[result["category"]] = N_c_per_category.get(result["category"], 0) + 1

        logger.info(
            "Step %d | category=%s | final=%s | gt=%s | correct=%s",
            step, result["category"], result["final_answer"], gt, result.get("correct"),
        )

    if total > 0:
        logger.info("Accuracy: %.2f%% (%d/%d)", 100.0 * correct_count / total, correct_count, total)


if __name__ == "__main__":
    main()
