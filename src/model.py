"""Core expected-cost model for CUMCM 2024 B.

The model treats one customer order as a renewal process: we keep producing /
reworking until one good final product is delivered.  All costs are expected
costs per delivered good product; profit = market price - expected cost.

Assumptions (stated explicitly in README):
1. Defects of components / assembly operations are independent.
2. Inspected components are replaced until a good one is obtained.
3. If a defective assembly is disassembled, all of its children are inspected
   and defective children are restored to good before reassembly.
4. A defective child is restored by the cheaper of (a) discarding it and
   producing a new good child, or (b) disassembling it and reworking its
   children, when disassembly is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import heapq
import math
from typing import Iterable


@dataclass
class Node:
    name: str
    kind: str  # "component" or "assembly"
    r: float = 0.0  # defect rate of the node's own operation (assembly only)
    p: float = 0.0  # purchase price (component only)
    c: float = 0.0  # inspection cost (component only)
    a: float = 0.0  # assembly cost (assembly only)
    t: float = 0.0  # inspection cost (assembly only)
    D: float = 0.0  # disassembly cost (assembly only)
    children: list["Node"] = field(default_factory=list)
    inspect: int = 0
    disassemble: int = 0

    # computed by evaluate()
    A: float = 0.0          # expected cost of children used + own assembly
    q: float = 0.0          # probability this node's output is defective
    good: float = 0.0       # cost to make one good node from good children
    Rdef: float = 0.0       # cost to restore children given node defective
    Rg: float = 0.0         # cost to re-inspect known-good children
    scratch: float = 0.0    # cost to make one good node from scratch
    prod: float = 0.0       # cost to make one node without inspecting it
    use: float = 0.0        # expected cost of one node as used by its parent
    defect: float = 0.0     # probability a used node is defective
    reverify: float = 0.0   # cost to re-inspect a known-good node
    extra: float = 0.0      # extra cost if this node is found defective


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 1e-15 else float("inf")


def evaluate(node: Node) -> None:
    """Bottom-up expected-cost evaluation.  Mutates the node tree."""
    for ch in node.children:
        evaluate(ch)

    if node.kind == "component":
        node.good = _safe_div(node.p + node.c, 1.0 - node.r)
        node.prod = node.p
        node.q = node.r
        node.A = node.p
        node.Rdef = 0.0
        node.Rg = 0.0
        node.reverify = node.c
        node.extra = node.good
        if node.inspect:
            node.use, node.defect = node.good, 0.0
        else:
            node.use, node.defect = node.p, node.r
        node.scratch = node.good
        return

    # Assembly node: expected cost of one assembly attempt.
    node.A = sum(ch.use for ch in node.children) + node.a
    node.Rg = sum(ch.reverify for ch in node.children)

    good_prob = 1.0 - node.r
    for ch in node.children:
        good_prob *= 1.0 - ch.defect
    node.q = 1.0 - good_prob

    # Cost to restore all children, given this node is defective.
    restore = 0.0
    for ch in node.children:
        pcond = _safe_div(ch.defect, node.q) if node.q > 1e-15 else 0.0
        if ch.kind == "component":
            extra = ch.good
        else:
            if ch.disassemble:
                extra = min(ch.scratch, ch.D + ch.Rdef + ch.good)
            else:
                extra = ch.scratch
        restore += ch.reverify + pcond * extra
    node.Rdef = restore

    node.reverify = node.t
    node.prod = node.A

    if node.disassemble:
        # Starting with all children good: only the assembly operation can fail.
        node.good = _safe_div(
            node.a + node.t + node.r * (node.D + node.Rg), 1.0 - node.r
        )
        node.scratch = node.A + node.t + node.q * (node.D + node.Rdef + node.good)
        node.extra = min(node.scratch, node.D + node.Rdef + node.good)
    else:
        # Discard a defective assembly and restart from scratch.
        node.scratch = _safe_div(node.A + node.t, 1.0 - node.q)
        node.good = node.a + node.t + node.r * node.scratch
        node.extra = node.scratch

    if node.inspect:
        node.use, node.defect = node.scratch, 0.0
    else:
        node.use, node.defect = node.prod, node.q


def root_profit(root: Node, price: float, replacement_loss: float) -> tuple[float, float, str]:
    """Return (profit, expected cost per good sale, mode) for the final product.

    If the root is inspected, the cost is simply the rework-until-good cost.
    If it is not inspected, defective units are sold, returned, and handled;
    the recursion below solves the renewal equation for one delivered good unit.
    """
    if root.inspect:
        return price - root.scratch, root.scratch, "inspect"

    A, q = root.A, root.q
    if q >= 1.0 - 1e-15:
        return float("-inf"), float("inf"), "no_inspect"

    if root.disassemble:
        # Starting from good children, a returned unit incurs L and is reworked.
        good_from_good = _safe_div(
            root.a + root.r * (replacement_loss + root.D + root.Rg), 1.0 - root.r
        )
        cost = A + q * (replacement_loss + root.D + root.Rdef + good_from_good)
    else:
        # Defective returned unit is discarded and a new one is started.
        cost = _safe_div(A + q * replacement_loss, 1.0 - q)

    return price - cost, cost, "no_inspect"


# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

def build_q2_tree(row, x1: int, x2: int, y: int, d: int) -> Node:
    c1 = Node("c1", "component", r=float(row.r1), p=float(row.p1), c=float(row.c1), inspect=x1)
    c2 = Node("c2", "component", r=float(row.r2), p=float(row.p2), c=float(row.c2), inspect=x2)
    root = Node(
        "product",
        "assembly",
        r=float(row.rp),
        a=float(row.a),
        t=float(row.t),
        D=float(row.D),
        children=[c1, c2],
        inspect=y,
        disassemble=d,
    )
    return root


def build_q3_tree(row, component_bits: Iterable[int], semi_inspect_bits: Iterable[int],
                  semi_disassemble_bits: Iterable[int], root_inspect: int,
                  root_disassemble: int) -> Node:
    component_bits = list(component_bits)
    semi_inspect_bits = list(semi_inspect_bits)
    semi_disassemble_bits = list(semi_disassemble_bits)
    components = [
        Node(
            f"c{i+1}",
            "component",
            r=float(getattr(row, f"r{i+1}")),
            p=float(getattr(row, f"p{i+1}")),
            c=float(getattr(row, f"c{i+1}")),
            inspect=component_bits[i],
        )
        for i in range(8)
    ]
    groups = [(0, 3), (3, 6), (6, 8)]
    semis = []
    for j, (lo, hi) in enumerate(groups, start=1):
        semis.append(
            Node(
                f"h{j}",
                "assembly",
                r=float(getattr(row, f"rh{j}")),
                a=float(getattr(row, f"ah{j}")),
                t=float(getattr(row, f"th{j}")),
                D=float(getattr(row, f"Dh{j}")),
                children=components[lo:hi],
                inspect=semi_inspect_bits[j - 1],
                disassemble=semi_disassemble_bits[j - 1],
            )
        )
    root = Node(
        "product",
        "assembly",
        r=float(row.rp),
        a=float(row.a),
        t=float(row.t),
        D=float(row.D),
        children=semis,
        inspect=root_inspect,
        disassemble=root_disassemble,
    )
    return root


# ---------------------------------------------------------------------------
# Enumerators
# ---------------------------------------------------------------------------

def enumerate_q2(df, top_k: int | None = None) -> tuple[list[dict], list[dict]]:
    """Evaluate all 16 Q2 strategies for every case.

    Returns (all_rows, best_rows).  Each row is a plain dict.
    """
    all_rows: list[dict] = []
    best_rows: list[dict] = []
    for _, row in df.iterrows():
        case_rows = []
        for x1, x2, y, d in product([0, 1], repeat=4):
            root = build_q2_tree(row, x1, x2, y, d)
            evaluate(root)
            profit, cost, mode = root_profit(root, float(row.P), float(row.L))
            rec = {
                "case": int(row.case),
                "x1": x1,
                "x2": x2,
                "inspect_product": y,
                "disassemble": d,
                "profit": profit,
                "cost": cost,
                "mode": mode,
                "q": root.q,
            }
            case_rows.append(rec)
            all_rows.append(rec)
        case_rows.sort(key=lambda r: r["profit"], reverse=True)
        best_rows.append(case_rows[0])
        if top_k is not None:
            # keep memory bounded when requested
            pass
    return all_rows, best_rows


def enumerate_q3(row, top_k: int = 100) -> list[dict]:
    """Enumerate all 2^16 Q3 strategies and return the top_k by profit."""
    components = [
        Node(
            f"c{i+1}",
            "component",
            r=float(getattr(row, f"r{i+1}")),
            p=float(getattr(row, f"p{i+1}")),
            c=float(getattr(row, f"c{i+1}")),
        )
        for i in range(8)
    ]
    groups = [(0, 3), (3, 6), (6, 8)]
    semis = []
    for j, (lo, hi) in enumerate(groups, start=1):
        semis.append(
            Node(
                f"h{j}",
                "assembly",
                r=float(getattr(row, f"rh{j}")),
                a=float(getattr(row, f"ah{j}")),
                t=float(getattr(row, f"th{j}")),
                D=float(getattr(row, f"Dh{j}")),
                children=components[lo:hi],
            )
        )
    root = Node(
        "product",
        "assembly",
        r=float(row.rp),
        a=float(row.a),
        t=float(row.t),
        D=float(row.D),
        children=semis,
    )

    def make_bits(bits: tuple[int, ...]) -> dict:
        return {
            "components": list(bits[:8]),
            "semi_inspect": list(bits[8:11]),
            "semi_disassemble": list(bits[11:14]),
            "root_inspect": bits[14],
            "root_disassemble": bits[15],
        }

    heap: list[tuple[float, int, tuple[int, ...]]] = []
    counter = 0
    for bits in product([0, 1], repeat=16):
        for i, ch in enumerate(components):
            ch.inspect = bits[i]
        for j, h in enumerate(semis):
            h.inspect = bits[8 + j]
            h.disassemble = bits[11 + j]
        root.inspect = bits[14]
        root.disassemble = bits[15]
        evaluate(root)
        profit, cost, mode = root_profit(root, float(row.P), float(row.L))
        item = (profit, counter, bits)
        counter += 1
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif profit > heap[0][0]:
            heapq.heapreplace(heap, item)

    out = []
    for profit, _, bits in sorted(heap, key=lambda t: t[0], reverse=True):
        rec = make_bits(bits)
        rec.update({"profit": profit, "bits": bits})
        out.append(rec)
    return out


def evaluate_q3_strategy(row, rec: dict) -> tuple[float, float, str]:
    root = build_q3_tree(
        row,
        rec["components"],
        rec["semi_inspect"],
        rec["semi_disassemble"],
        rec["root_inspect"],
        rec["root_disassemble"],
    )
    evaluate(root)
    return root_profit(root, float(row.P), float(row.L))
