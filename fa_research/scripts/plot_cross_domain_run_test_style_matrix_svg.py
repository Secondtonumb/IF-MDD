#!/usr/bin/env python3
"""Draw a paper-style cross-domain FA matrix from run_test_style_matrix.csv."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


PREFERRED_DATASETS = ["l2arctic", "timit_dev", "timit_test", "buckeye"]
PREFERRED_MODELS = [
    "L2ARCTIC_CTC",
    "L2ARCTIC_CRCTC",
    "L2ARCTIC_OTTC",
    "L2ARCTIC_CROTTC",
    "LIBRISPEECH_CTC",
    "LIBRISPEECH_CRCTC",
    "LIBRISPEECH_OTTC",
    "LIBRISPEECH_CROTTC",
]
MODEL_LABELS = {
    "L2ARCTIC_CTC": "L2 CTC",
    "L2ARCTIC_CRCTC": "L2 CRCTC",
    "L2ARCTIC_OTTC": "L2 OTTC",
    "L2ARCTIC_CROTTC": "L2 CROTTC",
    "LIBRISPEECH_CTC": "LS CTC",
    "LIBRISPEECH_CRCTC": "LS CRCTC",
    "LIBRISPEECH_OTTC": "LS OTTC",
    "LIBRISPEECH_CROTTC": "LS CROTTC",
}
DATASET_LABELS = {
    "l2arctic": "L2-ARCTIC",
    "timit_dev": "TIMIT dev",
    "timit_test": "TIMIT test",
    "buckeye": "Buckeye",
}
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


def recipe(model: str) -> str:
    return model.split("_", 1)[1] if "_" in model else model


def source(model: str) -> str:
    return model.split("_", 1)[0] if "_" in model else ""


def read_matrix(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"dataset", "target", "model", "source", "recipe"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value) if value != "" else value
            rows[(str(parsed["dataset"]), str(parsed["model"]))] = parsed
    return rows


def filter_rows(
    rows: dict[tuple[str, str], dict[str, float | str]],
    datasets: list[str] | None,
    models: list[str] | None,
) -> dict[tuple[str, str], dict[str, float | str]]:
    dataset_filter = set(datasets or [])
    model_filter = set(models or [])
    return {
        (dataset, model): row
        for (dataset, model), row in rows.items()
        if (not dataset_filter or dataset in dataset_filter)
        and (not model_filter or model in model_filter)
    }


def ordered_values(values, preferred):
    values = list(dict.fromkeys(values))
    preferred_index = {value: idx for idx, value in enumerate(preferred)}
    return sorted(values, key=lambda value: (preferred_index.get(value, len(preferred)), str(value)))


def svg_text(x, y, text, size=12, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Times New Roman, Times, serif" font-weight="{weight}" '
        f'text-anchor="{anchor}" dominant-baseline="middle"{transform}>'
        f"{html.escape(str(text))}</text>"
    )


def marker_svg(kind: str, x: float, y: float, color: str, size: float = 4.2) -> str:
    if kind == "square":
        s = size * 1.45
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


def model_style(model: str) -> tuple[str, str, str]:
    rec = recipe(model)
    color = COLORS.get(rec, "#555555")
    marker = MARKERS.get(rec, "circle")
    dash = "" if source(model) == "L2ARCTIC" else ' stroke-dasharray="5 4"'
    return color, marker, dash


def draw_axes(x, y, w, h, x_ticks, y_ticks, x_label, y_label, title, x_min, x_max, y_min, y_max):
    out = []
    out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="white" stroke="#222" stroke-width="1"/>')
    for tick in x_ticks:
        tx = x + (tick - x_min) / (x_max - x_min) * w
        out.append(f'<line x1="{tx:.2f}" y1="{y:.2f}" x2="{tx:.2f}" y2="{y + h:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{tx:.2f}" y1="{y + h:.2f}" x2="{tx:.2f}" y2="{y + h + 4:.2f}" stroke="#222" stroke-width="1"/>')
        out.append(svg_text(tx, y + h + 14, int(tick), size=9))
    for tick in y_ticks:
        ty = y + h - (tick - y_min) / (y_max - y_min) * h
        out.append(f'<line x1="{x:.2f}" y1="{ty:.2f}" x2="{x + w:.2f}" y2="{ty:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{x - 4:.2f}" y1="{ty:.2f}" x2="{x:.2f}" y2="{ty:.2f}" stroke="#222" stroke-width="1"/>')
        out.append(svg_text(x - 9, ty, f"{tick:.0f}", size=9, anchor="end"))
    out.append(svg_text(x + w / 2, y - 12, title, size=12, weight="bold"))
    out.append(svg_text(x + w / 2, y + h + 31, x_label, size=11))
    out.append(svg_text(x - 36, y + h / 2, y_label, size=11, rotate=-90))
    return out


def draw_curve_panel(rows, dataset, models, x, y, w, h):
    taus = [10, 20, 25, 30, 40, 50]
    out = draw_axes(
        x,
        y,
        w,
        h,
        x_ticks=taus,
        y_ticks=[0, 20, 40, 60, 80, 100],
        x_label="Tolerance (ms)",
        y_label="Phone ACC (%)",
        title="Phone-level ACC",
        x_min=10,
        x_max=50,
        y_min=0,
        y_max=100,
    )
    for model in models:
        row = rows.get((dataset, model))
        if not row:
            continue
        points = []
        for tau in taus:
            value = row.get(f"phone_acc_tau_{tau}", "")
            if value == "":
                continue
            px = x + (tau - 10) / 40 * w
            py = y + h - float(value) / 100 * h
            points.append((px, py))
        if len(points) < 2:
            continue
        color, marker, dash = model_style(model)
        d = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')
        for px, py in points:
            out.append(marker_svg(marker, px, py, color))
    return out


def draw_bar_panel(rows, dataset, models, x, y, w, h):
    out = []
    present_models = [
        model
        for model in models
        if (dataset, model) in rows and rows[(dataset, model)].get("phone_tse_ms") != ""
    ]
    values = [
        float(rows[(dataset, model)]["phone_tse_ms"])
        for model in present_models
    ]
    if not values:
        return out
    x_max = max(80, int(max(values) / 25 + 2) * 25)
    left_label_w = 76
    plot_x = x + left_label_w
    plot_w = w - left_label_w
    out.append(svg_text(x + w / 2, y - 12, "Phone-level TSE", size=12, weight="bold"))
    out.append(f'<rect x="{plot_x:.2f}" y="{y:.2f}" width="{plot_w:.2f}" height="{h:.2f}" fill="white" stroke="#222" stroke-width="1"/>')
    tick_step = 50 if x_max > 200 else 25
    for tick in range(0, x_max + 1, tick_step):
        tx = plot_x + tick / x_max * plot_w
        out.append(f'<line x1="{tx:.2f}" y1="{y:.2f}" x2="{tx:.2f}" y2="{y + h:.2f}" stroke="#dddddd" stroke-width="0.7"/>')
        out.append(svg_text(tx, y + h + 14, tick, size=9))
    bar_h = 10.5
    gap = (h - len(present_models) * bar_h) / (len(present_models) + 1)
    for idx, model in enumerate(present_models):
        row = rows.get((dataset, model))
        if not row or row.get("phone_tse_ms") == "":
            continue
        by = y + gap + idx * (bar_h + gap)
        val = float(row["phone_tse_ms"])
        bw = val / x_max * plot_w
        color, _, _ = model_style(model)
        out.append(svg_text(plot_x - 8, by + bar_h / 2, MODEL_LABELS.get(model, model), size=8.5, anchor="end"))
        out.append(f'<rect x="{plot_x:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bar_h:.2f}" fill="{color}"/>')
        out.append(svg_text(plot_x + bw + 5, by + bar_h / 2, f"{val:.1f}", size=8.5, anchor="start"))
    out.append(svg_text(plot_x + plot_w / 2, y + h + 31, "Phone TSE (ms)", size=11))
    return out


def draw_legend(x, y, models, per_row=4):
    out = []
    step = 138
    for i, model in enumerate(models):
        lx = x + (i % per_row) * step
        ly = y + (i // per_row) * 24
        color, marker, dash = model_style(model)
        out.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{lx + 28:.2f}" y2="{ly:.2f}" stroke="{color}" stroke-width="1.8"{dash}/>')
        out.append(marker_svg(marker, lx + 14, ly, color))
        out.append(svg_text(lx + 36, ly, MODEL_LABELS.get(model, model), size=11, anchor="start"))
    return out


def render_row_layout(rows, out_path: Path, datasets: list[str], models: list[str]):
    width = 1180
    legend_per_row = 4
    legend_rows = (len(models) + legend_per_row - 1) // legend_per_row
    top = 80 + legend_rows * 24
    row_h = 205
    height = top + len(datasets) * row_h + 50
    legend_w = min(legend_per_row, len(models)) * 138
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Cross-domain forced-alignment phone timing", size=17, weight="bold"),
        svg_text(width / 2, 51, "Solid lines: L2-ARCTIC in-domain models; dashed lines: LibriSpeech out-of-domain models", size=11),
        *draw_legend((width - legend_w) / 2, 77, models, per_row=legend_per_row),
    ]
    for idx, dataset in enumerate(datasets):
        dataset_models = [model for model in models if (dataset, model) in rows]
        y = top + idx * row_h
        parts.append(svg_text(58, y + 74, DATASET_LABELS.get(dataset, dataset), size=13, weight="bold", rotate=-90))
        parts.extend(draw_curve_panel(rows, dataset, dataset_models, 105, y + 18, 455, 132))
        parts.extend(draw_bar_panel(rows, dataset, dataset_models, 650, y + 18, 440, 132))
    parts.append(svg_text(width / 2, height - 24, "Silence phones are filtered before scoring; ACC requires both start and end boundaries within tolerance.", size=10.5))
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def render_column_layout(rows, out_path: Path, datasets: list[str], models: list[str]):
    col_w = 325
    left = 58
    right = 58
    legend_per_row = min(4, max(1, len(models)))
    legend_rows = (len(models) + legend_per_row - 1) // legend_per_row
    top = 82 + legend_rows * 24
    panel = 220
    panel_gap = 70
    title_gap = 26
    bottom = 44
    width = max(760, left + right + len(datasets) * col_w)
    height = top + title_gap + panel * 2 + panel_gap + bottom
    legend_w = legend_per_row * 138

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Cross-domain forced-alignment phone timing", size=17, weight="bold"),
        svg_text(width / 2, 51, "Solid lines: L2-ARCTIC in-domain models; dashed lines: LibriSpeech out-of-domain models", size=11),
        *draw_legend((width - legend_w) / 2, 77, models, per_row=legend_per_row),
    ]

    total_cols_w = len(datasets) * col_w
    start_x = (width - total_cols_w) / 2
    for idx, dataset in enumerate(datasets):
        dataset_models = [model for model in models if (dataset, model) in rows]
        col_x = start_x + idx * col_w
        center_x = col_x + col_w / 2
        y0 = top
        acc_x = center_x - panel / 2
        bar_x = acc_x - 76
        parts.append(svg_text(center_x, y0, DATASET_LABELS.get(dataset, dataset), size=13, weight="bold"))
        parts.extend(draw_curve_panel(rows, dataset, dataset_models, acc_x, y0 + title_gap, panel, panel))
        parts.extend(draw_bar_panel(rows, dataset, dataset_models, bar_x, y0 + title_gap + panel + panel_gap, panel + 76, panel))

    parts.append(svg_text(width / 2, height - 24, "Silence phones are filtered before scoring; ACC requires both start and end boundaries within tolerance.", size=10.5))
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def render(rows, out_path: Path, column: bool = False):
    datasets = ordered_values((dataset for dataset, _ in rows), PREFERRED_DATASETS)
    models = ordered_values((model for _, model in rows), PREFERRED_MODELS)
    if not datasets or not models:
        raise ValueError("No dataset/model rows available to plot.")

    if column:
        render_column_layout(rows, out_path, datasets, models)
    else:
        render_row_layout(rows, out_path, datasets, models)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/run_test_style_matrix.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fa_research/figures/cross_domain_json_fa/run_test_style_matrix_paper_like.svg"),
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--column", action="store_true", help="Lay out datasets as vertical columns with near-square panels.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = filter_rows(read_matrix(args.matrix), args.datasets, args.models)
    render(rows, args.output, column=args.column)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
