"""Generate dependency-free SVG figures from the result files."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

FONT = "Microsoft YaHei, SimHei, Noto Sans CJK SC, sans-serif"
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _svg_header(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700">{escape(title)}</text>',
    ]


def _axes(w: int, h: int, left: int, right: int, top: int, bottom: int,
          xlab: str, ylab: str) -> list[str]:
    x0, x1 = left, w - right
    y0, y1 = h - bottom, top
    parts = [
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<text x="{(x0+x1)/2}" y="{h-18}" text-anchor="middle" font-family="{FONT}" font-size="13">{escape(xlab)}</text>',
        f'<text x="16" y="{(y0+y1)/2}" text-anchor="middle" font-family="{FONT}" font-size="13" transform="rotate(-90 16 {(y0+y1)/2})">{escape(ylab)}</text>',
    ]
    for i in range(6):
        y = y1 + (y0 - y1) * i / 5
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
    return parts


def line_chart(series: list[tuple[str, list[tuple[float, float]]]], path: Path,
               title: str, xlab: str, ylab: str, xmax: float = 0.3,
               ymax: float = 1.0) -> None:
    w, h, left, right, top, bottom = 820, 470, 75, 30, 55, 70
    x0, x1 = left, w - right
    y0, y1 = h - bottom, top

    def sx(x: float) -> float:
        return x0 + (x / xmax) * (x1 - x0)

    def sy(y: float) -> float:
        return y0 - (y / ymax) * (y0 - y1)

    parts = _svg_header(w, h, title)
    parts += _axes(w, h, left, right, top, bottom, xlab, ylab)
    # x ticks
    for i in range(7):
        xv = xmax * i / 6
        parts.append(f'<text x="{sx(xv):.2f}" y="{y0+18}" text-anchor="middle" font-family="{FONT}" font-size="11">{xv:.2f}</text>')
    # y ticks
    for i in range(6):
        yv = ymax * i / 5
        parts.append(f'<text x="{x0-8}" y="{sy(yv)+4:.2f}" text-anchor="end" font-family="{FONT}" font-size="11">{yv:.1f}</text>')
    for k, (name, pts) in enumerate(series):
        color = COLORS[k % len(COLORS)]
        d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        lx = x0 + 12
        ly = top + 12 + k * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{lx+30}" y="{ly+4}" font-family="{FONT}" font-size="12">{escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def bar_chart(labels: list[str], values: list[float], path: Path, title: str,
              xlab: str = "期望利润") -> None:
    w, h = 900, max(360, 90 + 42 * len(labels))
    left, right, top, bottom = 330, 50, 55, 45
    x0, x1 = left, w - right
    y0 = h - bottom
    vmin = min(0.0, min(values))
    vmax = max(0.0, max(values))
    span = max(vmax - vmin, 1e-9)
    if vmin < 0:
        zero = x0 + (-vmin / span) * (x1 - x0)
    else:
        zero = x0
    parts = _svg_header(w, h, title)
    parts.append(f'<line x1="{zero}" y1="{top}" x2="{zero}" y2="{y0}" stroke="#111827" stroke-width="1.2"/>')
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top + i * 42
        xv = x0 + ((val - vmin) / span) * (x1 - x0)
        color = "#16a34a" if val >= 0 else "#dc2626"
        xa, xb = (zero, xv) if xv >= zero else (xv, zero)
        parts.append(f'<rect x="{xa:.2f}" y="{y}" width="{max(xb-xa, 2):.2f}" height="24" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{left-10}" y="{y+17}" text-anchor="end" font-family="{FONT}" font-size="12">{escape(lab)}</text>')
        tx = xv + (6 if val >= 0 else -6)
        anchor = "start" if val >= 0 else "end"
        parts.append(f'<text x="{tx:.2f}" y="{y+17}" text-anchor="{anchor}" font-family="{FONT}" font-size="12">{val:.3f}</text>')
    parts.append(f'<text x="{(x0+x1)/2}" y="{h-12}" text-anchor="middle" font-family="{FONT}" font-size="13">{escape(xlab)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    # Q1 OC curves
    rows = list(csv.DictReader((OUT / "q1_oc_curve.csv").open(encoding="utf-8-sig")))
    series = []
    for plan in ("upper_95", "lower_90"):
        pts = [(float(r["p"]), float(r["accept_probability"])) for r in rows if r["plan"] == plan]
        name = "95% 拒收方案（n=98）" if plan == "upper_95" else "90% 接收方案（n=60）"
        series.append((name, pts))
    line_chart(series, FIG / "q1_oc.svg", "Q1 抽样方案 OC 曲线", "真实次品率 p", "接收概率", 0.30, 1.0)

    # Q3 named strategies
    q3 = json.loads((OUT / "q3_results.json").read_text(encoding="utf-8"))
    names = {
        "best": "最优", "all_inspect": "全检", "components_only": "只检零件",
        "final_only": "只检成品", "no_inspect": "全不检", "all_no_disassemble": "全检不拆",
    }
    labels = [names[k] for k in names]
    vals = [q3["named"][k]["profit"] for k in names]
    bar_chart(labels, vals, FIG / "q3_strategies.svg", "Q3 典型策略期望利润对比")

    # Q4 case 6 decision frequencies
    q4 = json.loads((OUT / "q4_q2.json").read_text(encoding="utf-8"))
    freq = q4["6"]["strategy_frequencies"]
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
    labels = [k for k, _ in top]
    vals = [v for _, v in top]
    bar_chart(labels, vals, FIG / "q4_case6_freq.svg", "Q4 情况 6：各策略成为最优的频率", "频率")
    print("written:", FIG / "q1_oc.svg")
    print("written:", FIG / "q3_strategies.svg")
    print("written:", FIG / "q4_case6_freq.svg")


if __name__ == "__main__":
    main()
