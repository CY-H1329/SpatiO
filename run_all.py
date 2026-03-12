#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARKS = ["cvbench", "3dsrbench", "stvqa"]
RUN_SCRIPTS = {
    "cvbench": "run_cvbench.py",
    "3dsrbench": "run_3dsrbench.py",
    "stvqa": "run_stvqa.py",
}


def main():
    parser = argparse.ArgumentParser(description="SpatiO — Run all benchmarks")
    parser.add_argument("--benchmarks", type=str, default="cvbench,3dsrbench,stvqa")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_base", type=str, default="results")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    for bench in benchmarks:
        if bench not in RUN_SCRIPTS:
            print(f"Unknown benchmark: {bench}. Skipping.")
            continue
        script = root / RUN_SCRIPTS[bench]
        if not script.exists():
            print(f"Script not found: {script}. Skipping.")
            continue

        cmd = [
            sys.executable,
            str(script),
            "--output_dir", f"{args.output_base}/{bench}",
            "--seed", str(args.seed),
        ]
        if args.full:
            cmd.append("--full")
        else:
            cmd.extend(["--max_samples", str(args.max_samples)])
        if args.test_only:
            cmd.append("--test_only")
        if args.device_map:
            cmd.extend(["--device_map", args.device_map])

        print(f"\n{'='*60}")
        print(f">>> Running {bench}")
        print(f"{'='*60}")
        subprocess.run(cmd, cwd=str(root))
        print()

    print("Done. Check results in", args.output_base)


if __name__ == "__main__":
    main()
