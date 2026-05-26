#!/usr/bin/env python3
"""Create L2-ARCTIC timestamp result tables from run-test-style metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE_LABELS = {
    "L2ARCTIC": "L2-Arctic-trained",
    "LIBRISPEECH": "LibriSpeech-trained",
}
RECIPE_ORDER = {"CTC": 0, "CRCTC": 1, "OTTC": 2, "CROTTC": 3}
ACC_TAUS = [10, 20, 25, 30, 40, 50]


def read_l2arctic_rows(matrix: Path) -> list[dict]:
    rows = []
    with matrix.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "l2arctic":
                continue
            item = {
                "source": row["source"],
                "source_label": SOURCE_LABELS.get(row["source"], row["source"]),
                "recipe": row["recipe"],
                "model": row["model"],
                "target": row.get("target", ""),
                "utterances": int(float(row["utterances"])),
                "phones": int(float(row["count"])),
                "p_tse": float(row["phone_tse_ms"]),
                "start_mae": float(row["start_mae_ms"]),
                "end_mae": float(row["end_mae_ms"]),
            }
            for tau in ACC_TAUS:
                item[f"p_acc_{tau}"] = float(row[f"phone_acc_tau_{tau}"])
            rows.append(item)
    rows.sort(key=lambda r: (r["source"], RECIPE_ORDER.get(r["recipe"], 99), r["recipe"]))
    return rows


def table_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = {
            "Source": row["source_label"],
            "Model": row["recipe"],
            "Target": row["target"],
            "Utterances": row["utterances"],
            "Phones": row["phones"],
            "P-TSE": row["p_tse"],
            "Start MAE": row["start_mae"],
            "End MAE": row["end_mae"],
        }
        for tau in ACC_TAUS:
            item[f"P-ACC@{tau}"] = row[f"p_acc_{tau}"]
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table_rows(rows)
    if not table:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def best_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    best_tse = min(rows, key=lambda r: r["p_tse"])
    best_acc25 = max(rows, key=lambda r: r["p_acc_25"])
    best_acc50 = max(rows, key=lambda r: r["p_acc_50"])
    return [
        f"- Best P-TSE: {best_tse['source_label']} {best_tse['recipe']} ({best_tse['p_tse']:.3f} ms)",
        f"- Best P-ACC@25: {best_acc25['source_label']} {best_acc25['recipe']} ({best_acc25['p_acc_25']:.3f}%)",
        f"- Best P-ACC@50: {best_acc50['source_label']} {best_acc50['recipe']} ({best_acc50['p_acc_50']:.3f}%)",
    ]


def write_markdown(path: Path, title: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table_rows(rows)
    headers = [
        "Source",
        "Model",
        "P-TSE ↓",
        "P-ACC@10 ↑",
        "P-ACC@20 ↑",
        "P-ACC@25 ↑",
        "P-ACC@30 ↑",
        "P-ACC@40 ↑",
        "P-ACC@50 ↑",
        "Start MAE",
        "End MAE",
        "Utterances",
        "Phones",
    ]
    lines = [f"# {title}", "", "Metrics match `run_test.version1.py`: silence-filtered per-utterance TSE/ACC, then utterance-averaged.", ""]
    lines.extend(best_lines(rows))
    lines.extend(["", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in table:
        values = [
            row["Source"],
            row["Model"],
            fmt(row["P-TSE"]),
            fmt(row["P-ACC@10"]),
            fmt(row["P-ACC@20"]),
            fmt(row["P-ACC@25"]),
            fmt(row["P-ACC@30"]),
            fmt(row["P-ACC@40"]),
            fmt(row["P-ACC@50"]),
            fmt(row["Start MAE"]),
            fmt(row["End MAE"]),
            fmt(row["Utterances"]),
            fmt(row["Phones"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/run_test_style_matrix.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa/l2arctic_timestamp_tables"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_l2arctic_rows(args.matrix)
    groups = {
        "l2arctic_trained": [row for row in rows if row["source"] == "L2ARCTIC"],
        "librispeech_trained": [row for row in rows if row["source"] == "LIBRISPEECH"],
        "combined": rows,
    }
    titles = {
        "l2arctic_trained": "L2-ARCTIC Timestamp Results: L2-Arctic-Trained Models",
        "librispeech_trained": "L2-ARCTIC Timestamp Results: LibriSpeech-Trained Models",
        "combined": "L2-ARCTIC Timestamp Results: Combined Model Comparison",
    }
    for name, group_rows in groups.items():
        write_csv(args.output_dir / f"{name}.csv", group_rows)
        write_markdown(args.output_dir / f"{name}.md", titles[name], group_rows)
        print(f"Saved {name}: {args.output_dir / f'{name}.md'}")


if __name__ == "__main__":
    main()
