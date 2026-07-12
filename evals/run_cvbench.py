#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional

from spatio.config import (
    ALL_CATEGORIES,
    SPECIALIST_LLMS,
    HEAD_AGENT_MODEL,
    REASONING_AGENT_MODEL,
    BETA as DEFAULT_BETA,
    TOP_K_SPECIALISTS,
)
from spatio.score_map import ScoreMap
from spatio.pipeline import run_step
from spatio.core import get_runner
from spatio.benchmarks import load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK = "cvbench"


def _parse_fixed_role_map(s: Optional[str]) -> Optional[Dict[str, str]]:
    if not s:
        return None
    out: Dict[str, str] = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--fixed_role_map invalide (attendu role=model): {part!r}")
        role, model = part.split("=", 1)
        role, model = role.strip(), model.strip()
        if not role or not model:
            raise ValueError(f"--fixed_role_map invalide (role/model vides): {part!r}")
        out[role] = model
    return out or None


def main():
    parser = argparse.ArgumentParser(description="SpatiO — CV-Bench")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--parallel_specialists", action="store_true", help="Exécute les 3 spécialistes en parallèle (threads).")
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_SPECIALISTS,
        help="Top-k specialist candidates (default: full pool of 5).",
    )
    parser.add_argument("--beta", type=float, default=None, help="Beta softmax des poids TTO (défaut: config.BETA).")
    parser.add_argument(
        "--role_assignment",
        type=str,
        default="default",
        choices=["default", "fixed", "random"],
        help="Assignation des rôles: default (trust/TTO), fixed, random.",
    )
    parser.add_argument(
        "--fixed_role_map",
        type=str,
        default=None,
        help='Si --role_assignment fixed : "role=model,role=model".',
    )
    parser.add_argument(
        "--final_aggregator",
        type=str,
        default="reasoner",
        choices=["reasoner", "majority", "weighted"],
        help="Agrégation finale: reasoner (défaut), majority, weighted.",
    )
    parser.add_argument("--no_short_term_ema", action="store_true", help="Ablation: lambda_f=0.")
    parser.add_argument("--no_long_term_ema", action="store_true", help="Ablation: lambda_g=0.")
    parser.add_argument("--no_delta_penalty", action="store_true", help="Ablation: sans pénalité delta_i.")
    parser.add_argument(
        "--uniform_role_weights",
        action="store_true",
        help="Poids affichés / raisonnement uniformes (w=0.5) : désactive les poids beta TTO dans le prompt final.",
    )
    parser.add_argument("--output_dir", type=str, default="results/cvbench")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    update_trust = args.train and not args.test_only
    fixed_role_map = _parse_fixed_role_map(args.fixed_role_map)
    use_beta_weights = not bool(args.uniform_role_weights)
    beta = float(args.beta) if args.beta is not None else float(DEFAULT_BETA)

    device_map = {}
    if args.device_map:
        raw = [d.strip() for d in args.device_map.split(",")]
        devices = [f"cuda:{d}" if d.isdigit() else d for d in raw]
        models = [HEAD_AGENT_MODEL, REASONING_AGENT_MODEL] + SPECIALIST_LLMS
        for i, m in enumerate(models):
            device_map[m] = devices[i % len(devices)]

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

    # Évite from_pretrained / .to(cuda) concurrents sur Sa2VA ou Qwen3 (meta tensor).
    if args.parallel_specialists:
        logger.info("Pré-chargement des spécialistes sur le thread principal (%d).", len(SPECIALIST_LLMS))
        for name in list(SPECIALIST_LLMS):
            if runners.get(name) is None:
                r = get_runner(name, device=device_map.get(name, "cuda:0"))
                r.load()
                runners[name] = r

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
            use_beta_weights=use_beta_weights,
            parallel_specialists=bool(args.parallel_specialists),
            final_aggregator=args.final_aggregator,
            use_delta_penalty=(not bool(args.no_delta_penalty)),
            no_short_term_ema=bool(args.no_short_term_ema),
            no_long_term_ema=bool(args.no_long_term_ema),
            top_k=int(args.top_k),
            beta=beta,
            role_assignment=str(args.role_assignment),
            fixed_role_map=fixed_role_map,
        )

        total += 1
        if result.get("correct"):
            correct_count += 1
        N_c_per_category[result["category"]] = N_c_per_category.get(result["category"], 0) + 1
        results.append({"step": step, "final": result["final_answer"], "gt": gt, "correct": result.get("correct")})

        logger.info("Step %d | cat=%s | final=%s | gt=%s | ok=%s", step, result["category"], result["final_answer"], gt, result.get("correct"))

    acc = 100.0 * correct_count / total if total > 0 else 0.0
    logger.info("CV-Bench Accuracy: %.2f%% (%d/%d)", acc, correct_count, total)

    summary = {
        "benchmark": BENCHMARK,
        "max_samples": max_samples,
        "full": args.full,
        "total": total,
        "correct": correct_count,
        "accuracy": acc,
        "train": update_trust,
        "uniform_role_weights": bool(args.uniform_role_weights),
        "final_aggregator": args.final_aggregator,
        "no_short_term_ema": bool(args.no_short_term_ema),
        "no_long_term_ema": bool(args.no_long_term_ema),
        "no_delta_penalty": bool(args.no_delta_penalty),
        "parallel_specialists": bool(args.parallel_specialists),
        "top_k": int(args.top_k),
        "beta": beta,
        "role_assignment": str(args.role_assignment),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
