"""Run the whole CUMCM 2024 B pipeline.

Usage:
    python main.py            # Q1-Q4
    python main.py --skip-q4  # Q1-Q3 (Q4 is slower)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SRC / script), *args]
    print("\n" + "=" * 78, flush=True)
    print("RUN", " ".join(cmd), flush=True)
    print("=" * 78, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-q4", action="store_true")
    parser.add_argument("--q4-reps", type=int, default=40)
    parser.add_argument("--q4-draws", type=int, default=40)
    args = parser.parse_args()

    run("q1_sampling.py")
    run("q2_decision.py")
    run("q3_decision.py")
    if not args.skip_q4:
        run("q4_uncertainty.py", "--reps", str(args.q4_reps),
            "--draws", str(args.q4_draws), "--q3-top-k", "100")
        run("figures.py")


if __name__ == "__main__":
    main()
