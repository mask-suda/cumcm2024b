"""Lightweight regression tests for the CUMCM 2024 B implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import build_q2_tree, build_q3_tree, evaluate, root_profit, enumerate_q3  # noqa: E402
from q1_sampling import normal_plan, exact_upper_threshold, exact_lower_threshold  # noqa: E402
from simulate import simulate_order  # noqa: E402


class TestQ1(unittest.TestCase):
    def test_normal_plan(self):
        up = normal_plan(0.10, 0.95, 0.05)
        lo = normal_plan(0.10, 0.90, 0.05)
        self.assertEqual(up["n"], 98)
        self.assertEqual(lo["n"], 60)

    def test_exact_thresholds(self):
        self.assertEqual(exact_upper_threshold(98, 0.10, 0.05), 16)
        self.assertEqual(exact_lower_threshold(60, 0.10, 0.10), 2)


class TestQ2(unittest.TestCase):
    def test_case1_best(self):
        row = pd.read_csv(ROOT / "data" / "q2_cases.csv").query("case == 1").iloc[0]
        best = (-1e9, None)
        for x1 in (0, 1):
            for x2 in (0, 1):
                for y in (0, 1):
                    for d in (0, 1):
                        root = build_q2_tree(row, x1, x2, y, d)
                        evaluate(root)
                        profit, _, _ = root_profit(root, float(row.P), float(row.L))
                        if profit > best[0]:
                            best = (profit, (x1, x2, y, d))
        self.assertAlmostEqual(best[0], 18.3755555556, places=6)
        self.assertEqual(best[1], (0, 0, 0, 1))

    def test_monte_carlo_close(self):
        row = pd.read_csv(ROOT / "data" / "q2_cases.csv").query("case == 1").iloc[0]
        root = build_q2_tree(row, 0, 0, 0, 1)
        evaluate(root)
        _, exact, _ = root_profit(root, float(row.P), float(row.L))
        import random
        rng = random.Random(7)
        vals = [simulate_order(row, 0, 0, 0, 1, rng) for _ in range(5000)]
        mc = sum(vals) / len(vals)
        self.assertLess(abs(mc - exact), 0.8)


class TestQ3(unittest.TestCase):
    def test_best_profit(self):
        row = pd.read_csv(ROOT / "data" / "q3_case.csv").iloc[0]
        top = enumerate_q3(row, top_k=5)
        self.assertAlmostEqual(top[0]["profit"], 57.6884444444, places=6)
        for rec in top[1:]:
            self.assertLessEqual(rec["profit"], top[0]["profit"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
