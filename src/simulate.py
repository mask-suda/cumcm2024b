"""Monte-Carlo validation of the Q2 expected-cost model.

The simulator follows exactly the assumptions used by model.py.  It is slow,
so it is only used to validate a handful of strategies, not to optimise.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

try:
    from .model import build_q2_tree, evaluate, root_profit
except ImportError:
    from model import build_q2_tree, evaluate, root_profit


DATA = Path(__file__).resolve().parent.parent / "data" / "q2_cases.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def simulate_order(row, x1: int, x2: int, y: int, d: int,
                   rng: random.Random) -> float:
    r1, p1, c1 = float(row.r1), float(row.p1), float(row.c1)
    r2, p2, c2 = float(row.r2), float(row.p2), float(row.c2)
    rp, a, t = float(row.rp), float(row.a), float(row.t)
    L, D = float(row.L), float(row.D)

    def component(r: float, p: float, c: float, inspect: bool):
        cost = 0.0
        while True:
            cost += p
            if inspect:
                cost += c
            defective = rng.random() < r
            if not inspect:
                return cost, defective
            if not defective:
                return cost, False

    state_good_components = False
    cost = 0.0
    while True:
        if not state_good_components:
            cc1, def1 = component(r1, p1, c1, bool(x1))
            cc2, def2 = component(r2, p2, c2, bool(x2))
            cost += cc1 + cc2 + a
            defective = def1 or def2 or (rng.random() < rp)
        else:
            cost += a
            defective = rng.random() < rp
            def1 = def2 = False

        if y:
            cost += t
            if not defective:
                return cost
        else:
            if not defective:
                return cost
            cost += L

        if not d:
            state_good_components = False
            continue

        cost += D
        if not state_good_components:
            # Inspect recovered components; replace the defective ones.
            for r, p, c, was_defective in ((r1, p1, c1, def1), (r2, p2, c2, def2)):
                cost += c
                if was_defective:
                    cc, _ = component(r, p, c, True)
                    cost += cc
        else:
            cost += c1 + c2
        state_good_components = True


def validate(case: int, x1: int, x2: int, y: int, d: int,
             n_sims: int = 100_000, seed: int = 2024) -> dict:
    row = pd.read_csv(DATA).query("case == @case").iloc[0]
    root = build_q2_tree(row, x1, x2, y, d)
    evaluate(root)
    _, exact, _ = root_profit(root, float(row.P), float(row.L))

    rng = random.Random(seed)
    vals = [simulate_order(row, x1, x2, y, d, rng) for _ in range(n_sims)]
    mc = sum(vals) / len(vals)
    var = sum((v - mc) ** 2 for v in vals) / (len(vals) - 1)
    se = (var / len(vals)) ** 0.5
    return {
        "case": case, "x1": x1, "x2": x2, "y": y, "d": d,
        "exact_cost": exact, "mc_cost": mc, "mc_se": se,
        "diff": mc - exact, "z": (mc - exact) / se if se else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=50_000)
    args = parser.parse_args()
    # Validate the optimal strategy of each case.
    best_df = pd.read_csv(OUT / "q2_best.csv")
    rows = []
    for _, rec in best_df.iterrows():
        r = validate(int(rec.case), int(rec.x1), int(rec.x2),
                     int(rec.inspect_product), int(rec.disassemble),
                     n_sims=args.sims)
        rows.append(r)
        print(
            f"case {r['case']}: exact={r['exact_cost']:.3f}, "
            f"MC={r['mc_cost']:.3f} +/- {1.96*r['mc_se']:.3f}, z={r['z']:.2f}"
        )
    pd.DataFrame(rows).to_csv(OUT / "q2_validation.csv", index=False, encoding="utf-8-sig")
    print(f"written: {OUT / 'q2_validation.csv'}")


if __name__ == "__main__":
    main()
