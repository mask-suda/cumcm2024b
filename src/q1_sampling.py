"""Q1: sampling inspection plans for a supplier's defect rate.

Two complementary views are implemented:

* Fixed-sample normal approximation (the "98 / 60" plan used in many solutions).
* Exact binomial one-sided tests (conservative, no normal approximation).
* Wald sequential probability ratio test (SPRT), which minimises the expected
  number of inspections.

The problem statement does not specify a margin of error or an alternative
defect rate.  We therefore expose those choices as parameters and report the
plans for the common defaults (delta=5%, p1=15% / 5%).
"""

from __future__ import annotations

import json
import math
import random
from statistics import NormalDist
from typing import Iterable

P0 = 0.10


# ---------------------------------------------------------------------------
# Binomial helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def binom_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binom_pmf_list(n: int, p: float) -> list[float]:
    return [binom_pmf(n, k, p) for k in range(n + 1)]


def binom_cdf(n: int, k: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(binom_pmf(n, i, p) for i in range(0, k + 1))


def binom_sf(n: int, k: int, p: float) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return math.fsum(binom_pmf(n, i, p) for i in range(k, n + 1))


def exact_upper_threshold(n: int, p0: float, alpha: float) -> int:
    """Smallest c such that P(X >= c | p0) <= alpha (reject for large c)."""
    pmf = binom_pmf_list(n, p0)
    tail = 0.0
    # Walk from the top so that we can stop at the first c satisfying alpha.
    for c in range(n, -1, -1):
        tail += pmf[c]
        if tail > alpha:
            return c + 1
    return 0


def exact_lower_threshold(n: int, p0: float, alpha: float) -> int:
    """Largest c such that P(X <= c | p0) <= alpha (accept for small c)."""
    pmf = binom_pmf_list(n, p0)
    cdf = 0.0
    last = -1
    for c in range(n + 1):
        cdf += pmf[c]
        if cdf <= alpha + 1e-15:
            last = c
        else:
            break
    return last


# ---------------------------------------------------------------------------
# Fixed-sample normal approximation
# ---------------------------------------------------------------------------

def normal_sample_size(p0: float, confidence: float, margin: float) -> int:
    z = NormalDist().inv_cdf(confidence)
    n = (z * math.sqrt(p0 * (1.0 - p0)) / margin) ** 2
    return math.ceil(n)


def normal_plan(p0: float, confidence: float, margin: float) -> dict:
    n = normal_sample_size(p0, confidence, margin)
    z = NormalDist().inv_cdf(confidence)
    se = math.sqrt(p0 * (1.0 - p0) / n)
    upper = p0 + z * se
    lower = p0 - z * se
    return {
        "n": n,
        "z": z,
        "se": se,
        "upper_proportion": upper,
        "lower_proportion": lower,
        "upper_count": math.floor(upper * n + 1e-12) + 1,
        "lower_count": math.ceil(lower * n - 1e-12) - 1,
    }


# ---------------------------------------------------------------------------
# Exact fixed-sample design with a specified power
# ---------------------------------------------------------------------------

def exact_sample_size(p0: float, p1: float, alpha: float, power: float,
                      max_n: int = 2000) -> dict:
    """Smallest n and threshold with type-I <= alpha and power >= power."""
    if p1 > p0:
        for n in range(1, max_n + 1):
            c = exact_upper_threshold(n, p0, alpha)
            if c <= n and binom_sf(n, c, p1) >= power:
                return {"n": n, "threshold": c, "alpha_actual": binom_sf(n, c, p0),
                        "power_actual": binom_sf(n, c, p1), "direction": "upper"}
    elif p1 < p0:
        for n in range(1, max_n + 1):
            c = exact_lower_threshold(n, p0, alpha)
            if c >= 0 and binom_cdf(n, c, p1) >= power:
                return {"n": n, "threshold": c, "alpha_actual": binom_cdf(n, c, p0),
                        "power_actual": binom_cdf(n, c, p1), "direction": "lower"}
    raise ValueError("p1 must differ from p0")


# ---------------------------------------------------------------------------
# Wald SPRT
# ---------------------------------------------------------------------------

def sprt_boundaries(p0: float, p1: float, alpha: float, beta: float) -> dict:
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    llr_defect = math.log(p1 / p0)
    llr_good = math.log((1.0 - p1) / (1.0 - p0))
    return {"A": A, "B": B, "llr_defect": llr_defect, "llr_good": llr_good}


def sprt_run(p_true: float, p0: float, p1: float, alpha: float, beta: float,
             rng: random.Random | None = None, max_n: int = 10000) -> tuple[str, int]:
    """Return ('reject' or 'accept', number inspected)."""
    rng = rng or random.Random()
    b = sprt_boundaries(p0, p1, alpha, beta)
    llr = 0.0
    for n in range(1, max_n + 1):
        if rng.random() < p_true:
            llr += b["llr_defect"]
        else:
            llr += b["llr_good"]
        if llr >= b["A"]:
            return "reject", n
        if llr <= b["B"]:
            return "accept", n
    return "accept", max_n


def sprt_asn(p_true: float, p0: float, p1: float, alpha: float, beta: float,
             runs: int = 20000, seed: int = 2024) -> dict:
    rng = random.Random(seed)
    total = 0
    rejects = 0
    for _ in range(runs):
        decision, n = sprt_run(p_true, p0, p1, alpha, beta, rng)
        total += n
        rejects += decision == "reject"
    return {"p_true": p_true, "asn": total / runs, "reject_rate": rejects / runs}


# ---------------------------------------------------------------------------
# OC curve
# ---------------------------------------------------------------------------

def oc_curve(n: int, c: int, direction: str, ps: Iterable[float]) -> list[dict]:
    rows = []
    for p in ps:
        if direction == "upper":
            # Reject when X >= c; accept otherwise.
            pa = binom_cdf(n, c - 1, p)
        else:
            # Accept when X <= c.
            pa = binom_cdf(n, c, p)
        rows.append({"p": p, "accept_probability": pa})
    return rows


def run_q1() -> dict:
    margin = 0.05
    normal_upper = normal_plan(P0, 0.95, margin)
    normal_lower = normal_plan(P0, 0.90, margin)

    n_upper = normal_upper["n"]
    n_lower = normal_lower["n"]
    c_upper = exact_upper_threshold(n_upper, P0, 0.05)
    c_lower = exact_lower_threshold(n_lower, P0, 0.10)

    exact_upper_power = exact_sample_size(P0, 0.15, 0.05, 0.80)
    exact_lower_power = exact_sample_size(P0, 0.05, 0.10, 0.80)

    sprt_upper = sprt_boundaries(P0, 0.15, 0.05, 0.10)
    sprt_lower = sprt_boundaries(P0, 0.05, 0.10, 0.10)
    sprt_upper_asn = []
    for p in (0.05, 0.10, 0.15, 0.20):
        rec = sprt_asn(p, P0, 0.15, 0.05, 0.10)
        rec["reject_supplier_rate"] = rec.pop("reject_rate")
        sprt_upper_asn.append(rec)
    sprt_lower_asn = []
    for p in (0.03, 0.05, 0.08, 0.10):
        rec = sprt_asn(p, P0, 0.05, 0.10, 0.10)
        # In the lower test, rejecting H0: p >= p0 means accepting the batch.
        rec["accept_batch_rate"] = rec.pop("reject_rate")
        sprt_lower_asn.append(rec)

    ps = [round(0.01 * i, 2) for i in range(0, 31)]
    return {
        "p0": P0,
        "normal": {
            "margin": margin,
            "upper_95": normal_upper,
            "lower_90": normal_lower,
            "note": "n = ceil((z*sqrt(p0(1-p0))/margin)^2)",
        },
        "exact_fixed": {
            "upper_95": {
                "n": n_upper, "threshold": c_upper,
                "alpha_actual": binom_sf(n_upper, c_upper, P0),
                "decision": f"reject if defects >= {c_upper}",
            },
            "lower_90": {
                "n": n_lower, "threshold": c_lower,
                "alpha_actual": binom_cdf(n_lower, c_lower, P0),
                "decision": f"accept if defects <= {c_lower}",
            },
        },
        "exact_power_design": {
            "upper_95_p1_15": exact_upper_power,
            "lower_90_p1_05": exact_lower_power,
        },
        "sprt": {
            "upper_95_0_15": {
                "boundaries": sprt_upper,
                "asn": sprt_upper_asn,
            },
            "lower_90_0_05": {
                "boundaries": sprt_lower,
                "asn": sprt_lower_asn,
            },
        },
        "oc_curve": {
            "upper_95": oc_curve(n_upper, c_upper, "upper", ps),
            "lower_90": oc_curve(n_lower, c_lower, "lower", ps),
        },
    }


if __name__ == "__main__":
    from pathlib import Path
    result = run_q1()
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    (out / "q1_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Save the OC curves as a compact CSV.
    rows = []
    for name, curve in result["oc_curve"].items():
        for rec in curve:
            rows.append({"plan": name, **rec})
    import csv
    with (out / "q1_oc_curve.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["plan", "p", "accept_probability"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({
        "normal": result["normal"],
        "exact_fixed": result["exact_fixed"],
        "exact_power_design": result["exact_power_design"],
        "written": [str(out / "q1_results.json"), str(out / "q1_oc_curve.csv")],
    }, ensure_ascii=False, indent=2))
