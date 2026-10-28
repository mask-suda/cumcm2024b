"""Q3: two processes, 8 components, 3 semi-finished products, 1 product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .model import enumerate_q3, evaluate_q3_strategy
except ImportError:
    from model import enumerate_q3, evaluate_q3_strategy


DATA = Path(__file__).resolve().parent.parent / "data" / "q3_case.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def label(rec: dict) -> str:
    return (
        "components=" + "".join(str(x) for x in rec["components"])
        + "; semi_inspect=" + "".join(str(x) for x in rec["semi_inspect"])
        + "; semi_disassemble=" + "".join(str(x) for x in rec["semi_disassemble"])
        + f"; product_inspect={rec['root_inspect']}; product_disassemble={rec['root_disassemble']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    row = pd.read_csv(DATA).iloc[0]
    top = enumerate_q3(row, top_k=args.top_k)

    OUT.mkdir(exist_ok=True)
    for rec in top:
        rec["label"] = label(rec)
    (OUT / "q3_top_strategies.json").write_text(
        json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nQ3 top 10 strategies (expected profit per good product)\n")
    for i, rec in enumerate(top[:10], 1):
        print(f"{i:2d}. profit={rec['profit']:.3f} | {rec['label']}")

    # Named reference strategies for the report.
    named = {
        "best": top[0],
        "all_inspect": {
            "components": [1] * 8, "semi_inspect": [1] * 3,
            "semi_disassemble": [1] * 3, "root_inspect": 1, "root_disassemble": 1,
        },
        "components_only": {
            "components": [1] * 8, "semi_inspect": [0] * 3,
            "semi_disassemble": [1] * 3, "root_inspect": 1, "root_disassemble": 1,
        },
        "final_only": {
            "components": [0] * 8, "semi_inspect": [0] * 3,
            "semi_disassemble": [1] * 3, "root_inspect": 1, "root_disassemble": 1,
        },
        "no_inspect": {
            "components": [0] * 8, "semi_inspect": [0] * 3,
            "semi_disassemble": [0] * 3, "root_inspect": 0, "root_disassemble": 0,
        },
        "all_no_disassemble": {
            "components": [1] * 8, "semi_inspect": [1] * 3,
            "semi_disassemble": [0] * 3, "root_inspect": 1, "root_disassemble": 0,
        },
    }
    named_out = {}
    for name, rec in named.items():
        profit, cost, mode = evaluate_q3_strategy(row, rec)
        named_out[name] = {"profit": profit, "cost": cost, "mode": mode, "strategy": rec}
        print(f"\n{name}: profit={profit:.3f}, cost={cost:.3f}, mode={mode}")

    (OUT / "q3_results.json").write_text(
        json.dumps({"top": top, "named": named_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten: {OUT / 'q3_top_strategies.json'}")


if __name__ == "__main__":
    main()
