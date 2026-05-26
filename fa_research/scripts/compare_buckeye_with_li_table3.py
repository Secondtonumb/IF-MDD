#!/usr/bin/env python3
"""Compare our Buckeye run-test-style metrics with Li et al. Table 3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PAPER_ROWS = [
    {"group": "Li Table 3", "model": "TDNN-F(Kaldi)", "p_tse": 26.327, "p_acc_25": 26.1, "p_acc_50": 63.6},
    {"group": "Li Table 3", "model": "WavLM(39)(CTC*)", "p_tse": 26.081, "p_acc_25": 27.4, "p_acc_50": 54.9},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)", "p_tse": 26.341, "p_acc_25": 21.0, "p_acc_50": 48.9},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(LP)", "p_tse": 26.084, "p_acc_25": 26.0, "p_acc_50": 52.8},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(EnCTC)", "p_tse": 26.319, "p_acc_25": 21.2, "p_acc_50": 48.8},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(LP+EnCTC)", "p_tse": 26.005, "p_acc_25": 27.1, "p_acc_50": 54.2},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(39)", "p_tse": 25.977, "p_acc_25": 23.5, "p_acc_50": 51.1},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(39)(LP)", "p_tse": 25.261, "p_acc_25": 32.8, "p_acc_50": 59.0},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(39)(EnCTC)", "p_tse": 26.231, "p_acc_25": 24.4, "p_acc_50": 51.6},
    {"group": "Li Table 3", "model": "WavLM(DWFST-CTC)(39)(LP+EnCTC)", "p_tse": 25.231, "p_acc_25": 32.9, "p_acc_50": 59.4},
]


def read_our_rows(matrix_path: Path, models: list[str] | None = None) -> list[dict]:
    rows = []
    model_filter = set(models or [])
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "buckeye":
                continue
            if model_filter and row.get("model") not in model_filter:
                continue
            rows.append(
                {
                    "group": "Ours",
                    "model": row["model"],
                    "p_tse": float(row["phone_tse_ms"]),
                    "p_acc_25": float(row["phone_acc_tau_25"]),
                    "p_acc_50": float(row["phone_acc_tau_50"]),
                }
            )
    return rows


def add_deltas(rows: list[dict], reference: dict) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["delta_tse_vs_best_table3"] = item["p_tse"] - reference["p_tse"]
        item["delta_acc25_vs_best_table3"] = item["p_acc_25"] - reference["p_acc_25"]
        item["delta_acc50_vs_best_table3"] = item["p_acc_50"] - reference["p_acc_50"]
        out.append(item)
    return out


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "model",
        "p_tse",
        "p_acc_25",
        "p_acc_50",
        "delta_tse_vs_best_table3",
        "delta_acc25_vs_best_table3",
        "delta_acc50_vs_best_table3",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Group | Model | P-TSE ↓ | P-ACC@25 ↑ | P-ACC@50 ↑ | ΔTSE vs best Table 3 | ΔACC@25 | ΔACC@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["model"]),
                    fmt(row["p_tse"]),
                    fmt(row["p_acc_25"]),
                    fmt(row["p_acc_50"]),
                    fmt(row["delta_tse_vs_best_table3"]),
                    fmt(row["delta_acc25_vs_best_table3"]),
                    fmt(row["delta_acc50_vs_best_table3"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/run_test_style_matrix.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/buckeye_li_table3_comparison.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/buckeye_li_table3_comparison.md"),
    )
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_table3 = min(PAPER_ROWS, key=lambda row: row["p_tse"])
    our_rows = read_our_rows(args.matrix, args.models)
    if not our_rows:
        print("No Buckeye rows from our matrix matched the requested filters; writing Li Table 3 rows only.")
    rows = add_deltas(PAPER_ROWS + our_rows, best_table3)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
