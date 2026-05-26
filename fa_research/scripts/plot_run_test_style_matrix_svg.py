#!/usr/bin/env python3
"""Draw a paper-style matrix figure from run_test_style_matrix.csv.

The layout follows the compact multi-panel style used in the referenced paper:
top panels are tolerance curves, bottom panels are horizontal bar charts.
No plotting libraries are required; the script writes SVG directly.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


MODELS = ["CTC", "CRCTC", "OTTC", "CROTTC"]
TARGETS = ["canonical", "perceived"]
COLORS = {
    "CTC": "#222222",
    "CRCTC": "#1f77b4",
    "OTTC": "#2ca02c",
    "CROTTC": "#d62728",
}
MARKERS = {
    "CTC": "circle",
    "CRCTC": "square",
    "OTTC": "triangle",
    "CROTTC": "diamond",
}


def read_matrix(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"model", "target"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value) if value != "" else value
            rows[(str(parsed["model"]), str(parsed["target"]))] = parsed
    return rows


def svg_text(x, y, text, size=12, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Times New Roman, Times, serif" font-weight="{weight}" '
        f'text-anchor="{anchor}" dominant-baseline="middle"{transform}>'
        f"{html.escape(str(text))}</text>"
    )


def marker_svg(kind: str, x: float, y: float, color: str, size: float = 4.5) -> str:
    if kind == "square":
        s = size * 1.5
        return f'<rect x="{x - s / 2:.2f}" y="{y - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" fill="{color}"/>'
    if kind == "triangle":
        return (
            f'<path d="M {x:.2f} {y - size:.2f} L {x + size:.2f} {y + size:.2f} '
            f'L {x - size:.2f} {y + size:.2f} Z" fill="{color}"/>'
        )
    if kind == "diamond":
        return (
            f'<path d="M {x:.2f} {y - size:.2f} L {x + size:.2f} {y:.2f} '
            f'L {x:.2f} {y + size:.2f} L {x - size:.2f} {y:.2f} Z" fill="{color}"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}"/>'


def draw_axes(x, y, w, h, x_ticks, y_ticks, x_label, y_label, title, x_min, x_max, y_min, y_max):
    out = []
    out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="white" stroke="#222" stroke-width="1"/>')
    for tick in x_ticks:
        tx = x + (tick - x_min) / (x_max - x_min) * w
        out.append(f'<line x1="{tx:.2f}" y1="{y:.2f}" x2="{tx:.2f}" y2="{y + h:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{tx:.2f}" y1="{y + h:.2f}" x2="{tx:.2f}" y2="{y + h + 4:.2f}" stroke="#222" stroke-width="1"/>')
        out.append(svg_text(tx, y + h + 15, int(tick), size=10))
    for tick in y_ticks:
        ty = y + h - (tick - y_min) / (y_max - y_min) * h
        out.append(f'<line x1="{x:.2f}" y1="{ty:.2f}" x2="{x + w:.2f}" y2="{ty:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{x - 4:.2f}" y1="{ty:.2f}" x2="{x:.2f}" y2="{ty:.2f}" stroke="#222" stroke-width="1"/>')
        out.append(svg_text(x - 9, ty, f"{tick:.0f}", size=10, anchor="end"))
    out.append(svg_text(x + w / 2, y - 13, title, size=13, weight="bold"))
    out.append(svg_text(x + w / 2, y + h + 34, x_label, size=12))
    out.append(svg_text(x - 38, y + h / 2, y_label, size=12, rotate=-90))
    return out


def draw_curve_panel(rows, target, x, y, w, h):
    taus = [10, 20, 25, 30, 40, 50]
    y_max = 75
    out = draw_axes(
        x,
        y,
        w,
        h,
        x_ticks=taus,
        y_ticks=[0, 15, 30, 45, 60, 75],
        x_label="Tolerance (ms)",
        y_label="Phone-level ACC (%)",
        title=f"Phone-level - {target.capitalize()}",
        x_min=10,
        x_max=50,
        y_min=0,
        y_max=y_max,
    )
    for model in MODELS:
        row = rows[(model, target)]
        points = []
        for tau in taus:
            px = x + (tau - 10) / 40 * w
            py = y + h - float(row[f"phone_acc_tau_{tau}"]) / y_max * h
            points.append((px, py))
        d = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        color = COLORS[model]
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for px, py in points:
            out.append(marker_svg(MARKERS[model], px, py, color))
    return out


def draw_bar_panel(rows, target, x, y, w, h):
    out = []
    values = [float(rows[(model, target)]["phone_tse_ms"]) for model in MODELS]
    x_max = max(180, int(max(values) / 20 + 2) * 20)
    left_label_w = 64
    plot_x = x + left_label_w
    plot_w = w - left_label_w
    out.append(svg_text(x + w / 2, y - 13, f"TSE - {target.capitalize()}", size=13, weight="bold"))
    out.append(f'<rect x="{plot_x:.2f}" y="{y:.2f}" width="{plot_w:.2f}" height="{h:.2f}" fill="white" stroke="#222" stroke-width="1"/>')
    for tick in range(0, x_max + 1, 30):
        tx = plot_x + tick / x_max * plot_w
        out.append(f'<line x1="{tx:.2f}" y1="{y:.2f}" x2="{tx:.2f}" y2="{y + h:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(svg_text(tx, y + h + 15, tick, size=10))
    bar_h = 17
    gap = (h - len(MODELS) * bar_h) / (len(MODELS) + 1)
    for idx, model in enumerate(MODELS):
        by = y + gap + idx * (bar_h + gap)
        val = float(rows[(model, target)]["phone_tse_ms"])
        bw = val / x_max * plot_w
        out.append(svg_text(plot_x - 8, by + bar_h / 2, model, size=11, anchor="end", weight="bold" if model in {"OTTC", "CROTTC"} else "normal"))
        out.append(f'<rect x="{plot_x:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bar_h:.2f}" fill="{COLORS[model]}"/>')
        out.append(svg_text(plot_x + bw + 6, by + bar_h / 2, f"{val:.1f}", size=10, anchor="start"))
    out.append(svg_text(plot_x + plot_w / 2, y + h + 34, "Phone TSE (ms)", size=12))
    return out


def draw_legend(x, y):
    out = []
    step = 120
    for i, model in enumerate(MODELS):
        lx = x + i * step
        color = COLORS[model]
        out.append(f'<line x1="{lx:.2f}" y1="{y:.2f}" x2="{lx + 28:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>')
        out.append(marker_svg(MARKERS[model], lx + 14, y, color))
        out.append(svg_text(lx + 36, y, model, size=12, anchor="start"))
    return out


def render(rows, out_path: Path):
    width, height = 900, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Run-test-style FA matrix on L2-ARCTIC", size=17, weight="bold"),
        *draw_legend(220, 62),
        *draw_curve_panel(rows, "canonical", 90, 105, 330, 185),
        *draw_curve_panel(rows, "perceived", 510, 105, 330, 185),
        *draw_bar_panel(rows, "canonical", 70, 380, 360, 130),
        *draw_bar_panel(rows, "perceived", 490, 380, 360, 130),
        svg_text(450, 575, "Silence phones are filtered before scoring; ACC requires both start and end boundaries within tolerance.", size=11),
        "</svg>",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("fa_research/results/l2arctic_fa_timestamp_metrics/run_test_style_matrix.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fa_research/figures/l2arctic_fa_timestamp_metrics/run_test_style_matrix_paper_like.svg"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_matrix(args.matrix)
    render(rows, args.output)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
