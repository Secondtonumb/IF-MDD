#!/usr/bin/env python3
"""Print compact L2-ARCTIC FA metrics from a run_test_style_matrix.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = {
    "CTC": (0, 0),
    "OTTC": (0, 1),
    "CRCTC": (0, 2),
    "CROTTC": (0, 3),
    "dp_ctc": (1, 0),
    "dp_ottc": (1, 1),
    "dp_crctc": (1, 2),
    "dp_crottc": (1, 3),
    "tri_ctc": (2, 0),
    "tri_ottc": (2, 1),
    "tri_crctc": (2, 2),
    "tri_crottc": (2, 3),
    "bwt_ctc": (3, 0),
    "bwt_ottc": (3, 1),
    "bwt_crctc": (3, 2),
    "bwt_crottc": (3, 3),
    "wpu_ctc": (4, 0),
    "wpu_ottc": (4, 1),
    "wpu_crctc": (4, 2),
    "wpu_crottc": (4, 3),
}


def sort_key(row: dict):
    return (*METHOD_ORDER.get(row["model"], (99, 99)), row["model"], row["target"])


def fmt(value: str, digits: int = 2) -> str:
    if value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def print_table(matrix: Path, acc_taus: list[int]) -> None:
    with matrix.open("r", encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=sort_key)

    headers = ["model", "target", "utt", "phones", "TSE(ms)", *[f"ACC@{tau}" for tau in acc_taus]]
    widths = [max(len(h), 8) for h in headers]
    body = []
    for row in rows:
        line = [
            row["model"],
            row["target"],
            row.get("utterances", ""),
            row.get("count", ""),
            fmt(row.get("phone_tse_ms", ""), 2),
        ]
        for tau in acc_taus:
            line.append(fmt(row.get(f"phone_acc_tau_{tau}", ""), 2))
        body.append(line)
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, line)]

    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for line in body:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(line, widths)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--acc-taus", nargs="+", type=int, default=[25, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_table(args.matrix, args.acc_taus)


if __name__ == "__main__":
    main()
