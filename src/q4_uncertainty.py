"""Q4: decision-making when defect rates are estimated from samples.

Model
-----
For every defect rate p we assume a binomial sample of size n.  With a
Jeffreys prior Beta(0.5, 0.5), the posterior is

    p | data ~ Beta(0.5 + k, 0.5 + n - k),   k ~ Binomial(n, p_true).

We draw posterior samples and evaluate each candidate strategy.  The selected
strategy maximises the posterior mean profit (sample average approximation,
SAA).  We also report the 5th percentile of profit as a risk measure, the
frequency with which each strategy is optimal, and the regret of using the
plug-in (true-rate) strategy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from .model import build_q2_tree, evaluate, root_profit, enumerate_q3, evaluate_q3_strategy
except ImportError:
    from model import build_q2_tree, evaluate, root_profit, enumerate_q3, evaluate_q3_strategy


BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

Q2_STRATEGIES = [
    (x1, x2, y, d)
    for x1 in (0, 1)
    for x2 in (0, 1)
    for y in (0, 1)
    for d in (0, 1)
]


def posterior_rate(p_true: float, n: int, rng: np.random.Generator,
                   prior_a: float = 0.5, prior_b: float = 0.5) -> float:
    k = int(rng.binomial(n, p_true))
    return float(rng.beta(prior_a + k, prior_b + n - k))


def q2_profits(row, rates: dict) -> np.ndarray:
    vals = []
    for x1, x2, y, d in Q2_STRATEGIES:
        obj = SimpleNamespace(**row)
        obj.r1, obj.r2, obj.rp = rates["r1"], rates["r2"], rates["rp"]
        root = build_q2_tree(obj, x1, x2, y, d)
        evaluate(root)
        profit, _, _ = root_profit(root, float(obj.P), float(obj.L))
        vals.append(profit)
    return np.array(vals, dtype=float)


def run_q2(df: pd.DataFrame, n_sample: int, reps: int, draws: int,
           seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for _, row in df.iterrows():
        case = int(row.case)
        true_rates = {"r1": float(row.r1), "r2": float(row.r2), "rp": float(row.rp)}
        true_profits = q2_profits(row, true_rates)
        plugin_idx = int(np.argmax(true_profits))

        all_profits = []
        best_indices = []
        for _ in range(reps):
            # One data set: one binomial observation per rate.
            observed = {
                "r1": int(rng.binomial(n_sample, true_rates["r1"])),
                "r2": int(rng.binomial(n_sample, true_rates["r2"])),
                "rp": int(rng.binomial(n_sample, true_rates["rp"])),
            }
            for _ in range(draws):
                rates = {
                    "r1": float(rng.beta(0.5 + observed["r1"], 0.5 + n_sample - observed["r1"])),
                    "r2": float(rng.beta(0.5 + observed["r2"], 0.5 + n_sample - observed["r2"])),
                    "rp": float(rng.beta(0.5 + observed["rp"], 0.5 + n_sample - observed["rp"])),
                }
                profits = q2_profits(row, rates)
                all_profits.append(profits)
                best_indices.append(int(np.argmax(profits)))

        P = np.vstack(all_profits)
        mean_profit = P.mean(axis=0)
        p05 = np.percentile(P, 5, axis=0)
        saa_idx = int(np.argmax(mean_profit))
        freq = Counter(best_indices)

        def describe(idx: int) -> dict:
            x1, x2, y, d = Q2_STRATEGIES[idx]
            return {
                "strategy": {"x1": x1, "x2": x2, "inspect_product": y, "disassemble": d},
                "true_profit": float(true_profits[idx]),
                "posterior_mean_profit": float(mean_profit[idx]),
                "posterior_p05_profit": float(p05[idx]),
                "optimal_frequency": freq.get(idx, 0) / len(best_indices),
            }

        results[str(case)] = {
            "plugin_best": describe(plugin_idx),
            "saa_best": describe(saa_idx),
            "saa_regret_vs_plugin": float(true_profits[plugin_idx] - true_profits[saa_idx]),
            "strategy_frequencies": {
                f"x1={Q2_STRATEGIES[i][0]},x2={Q2_STRATEGIES[i][1]},y={Q2_STRATEGIES[i][2]},d={Q2_STRATEGIES[i][3]}": c / len(best_indices)
                for i, c in sorted(freq.items(), key=lambda kv: -kv[1])
            },
            "n_posterior_samples": len(best_indices),
        }
        print(
            f"Q2 case {case}: plugin={plugin_idx}, SAA={saa_idx}, "
            f"freq={results[str(case)]['saa_best']['optimal_frequency']:.2%}, "
            f"regret={results[str(case)]['saa_regret_vs_plugin']:.3f}"
        )
    return results


def q3_profits(row, rates: dict, candidates: list[dict]) -> np.ndarray:
    obj = SimpleNamespace(**{k: v for k, v in row.items()})
    for i in range(1, 9):
        obj.__dict__[f"r{i}"] = rates[f"r{i}"]
    for j in range(1, 4):
        obj.__dict__[f"rh{j}"] = rates[f"rh{j}"]
    obj.rp = rates["rp"]
    vals = []
    for rec in candidates:
        profit, _, _ = evaluate_q3_strategy(obj, rec)
        vals.append(profit)
    return np.array(vals, dtype=float)


def run_q3(row, candidates: list[dict], n_sample: int, reps: int, draws: int,
           seed: int = 2025) -> dict:
    rng = np.random.default_rng(seed)
    rate_names = [f"r{i}" for i in range(1, 9)] + [f"rh{j}" for j in range(1, 4)] + ["rp"]
    true_rates = {name: float(getattr(row, name)) for name in rate_names}
    true_profits = q3_profits(row, true_rates, candidates)
    plugin_idx = int(np.argmax(true_profits))

    all_profits = []
    best_indices = []
    for _ in range(reps):
        observed = {name: int(rng.binomial(n_sample, true_rates[name])) for name in rate_names}
        for _ in range(draws):
            rates = {
                name: float(rng.beta(0.5 + observed[name], 0.5 + n_sample - observed[name]))
                for name in rate_names
            }
            profits = q3_profits(row, rates, candidates)
            all_profits.append(profits)
            best_indices.append(int(np.argmax(profits)))

    P = np.vstack(all_profits)
    mean_profit = P.mean(axis=0)
    p05 = np.percentile(P, 5, axis=0)
    saa_idx = int(np.argmax(mean_profit))
    freq = Counter(best_indices)

    def describe(idx: int) -> dict:
        rec = candidates[idx]
        return {
            "strategy": {k: rec[k] for k in ("components", "semi_inspect", "semi_disassemble", "root_inspect", "root_disassemble")},
            "true_profit": float(true_profits[idx]),
            "posterior_mean_profit": float(mean_profit[idx]),
            "posterior_p05_profit": float(p05[idx]),
            "optimal_frequency": freq.get(idx, 0) / len(best_indices),
        }

    print(
        f"Q3: plugin={plugin_idx}, SAA={saa_idx}, "
        f"freq={freq.get(saa_idx, 0)/len(best_indices):.2%}, "
        f"regret={true_profits[plugin_idx]-true_profits[saa_idx]:.3f}"
    )
    return {
        "plugin_best": describe(plugin_idx),
        "saa_best": describe(saa_idx),
        "saa_regret_vs_plugin": float(true_profits[plugin_idx] - true_profits[saa_idx]),
        "strategy_frequencies": {
            str(i): c / len(best_indices) for i, c in sorted(freq.items(), key=lambda kv: -kv[1])
        },
        "n_posterior_samples": len(best_indices),
        "n_candidates": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=100)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--q3-top-k", type=int, default=100)
    args = parser.parse_args()

    q2 = pd.read_csv(BASE / "data" / "q2_cases.csv")
    q2_results = run_q2(q2, args.n_sample, args.reps, args.draws)
    (OUT / "q4_q2.json").write_text(json.dumps(q2_results, ensure_ascii=False, indent=2), encoding="utf-8")

    q3_row = pd.read_csv(BASE / "data" / "q3_case.csv").iloc[0]
    top = enumerate_q3(q3_row, top_k=args.q3_top_k)
    q3_results = run_q3(q3_row, top, args.n_sample, max(10, args.reps // 2), max(10, args.draws // 2))
    (OUT / "q4_q3.json").write_text(json.dumps(q3_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written: outputs/q4_q2.json, outputs/q4_q3.json")


if __name__ == "__main__":
    main()
