"""Q2: six production cases, 16 strategies each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .model import enumerate_q2
except ImportError:  # allow direct execution
    from model import enumerate_q2


DATA = Path(__file__).resolve().parent.parent / "data" / "q2_cases.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def strategy_label(rec: dict) -> str:
    return (
        f"inspect c1={rec['x1']}, c2={rec['x2']}, "
        f"product={rec['inspect_product']}, disassemble={rec['disassemble']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    all_rows, best_rows = enumerate_q2(df)

    OUT.mkdir(exist_ok=True)
    all_df = pd.DataFrame(all_rows)
    all_df["strategy"] = all_df.apply(strategy_label, axis=1)
    all_df.to_csv(OUT / "q2_all_strategies.csv", index=False, encoding="utf-8-sig")

    best_df = pd.DataFrame(best_rows)
    best_df["strategy"] = best_df.apply(strategy_label, axis=1)
    best_df.to_csv(OUT / "q2_best.csv", index=False, encoding="utf-8-sig")

    print("\nQ2 optimal decisions (expected profit per good product)\n")
    for rec in best_rows:
        print(
            f"case {rec['case']}: profit={rec['profit']:.3f}, cost={rec['cost']:.3f}, "
            f"q={rec['q']:.3f} | x1={rec['x1']}, x2={rec['x2']}, "
            f"inspect_product={rec['inspect_product']}, disassemble={rec['disassemble']}"
        )

    summary = {
        "best": best_rows,
        "n_strategies": 16,
        "n_cases": len(best_rows),
    }
    (OUT / "q2_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwritten: {OUT / 'q2_all_strategies.csv'}")
    print(f"written: {OUT / 'q2_best.csv'}")


if __name__ == "__main__":
    main()
