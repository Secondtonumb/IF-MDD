#!/usr/bin/env python3
"""Recompute L2-ARCTIC FA metrics with run_test.version1.py semantics.

This consumes an alignments.jsonl produced by evaluate_l2arctic_fa_timestamps.py
and recomputes timestamp metrics in the same style as fa_research/run_test.version1.py:

  - filter silence labels from both reference and prediction
  - TSE per phone = abs(ref_start - pred_start) + abs(ref_end - pred_end)
  - ACC@tau = both start and end errors are within tau milliseconds
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SILENCE = {"SIL", "sil", "SP", "sp", "SPN", "spn", ""}
ALIGNMENT_TO_TARGET = {
    "canonical_fa": "canonical",
    "perceived_fa": "perceived",
}


def is_silence(label: str) -> bool:
    return label in SILENCE or label.upper() in SILENCE


def filtered_entries(record: dict, starts_key: str, ends_key: str) -> list[tuple[float, float, str]]:
    entries = zip(record["phones"], record[starts_key], record[ends_key])
    return [
        (float(start), float(end), str(phone))
        for phone, start, end in entries
        if not is_silence(str(phone))
    ]


def compute_run_test_metrics(record: dict, tolerances_ms: Iterable[int]) -> dict:
    ref = filtered_entries(record, "gt_starts", "gt_ends")
    pred = filtered_entries(record, "pred_starts", "pred_ends")
    if not ref or len(ref) != len(pred):
        raise ValueError(f"filtered length mismatch: ref={len(ref)}, pred={len(pred)}")

    start_abs = []
    end_abs = []
    tse_values = []
    acc_hits = {tau: 0 for tau in tolerances_ms}

    for (rs, re, _), (ps, pe, _) in zip(ref, pred):
        s_err = abs(rs - ps)
        e_err = abs(re - pe)
        start_abs.append(s_err)
        end_abs.append(e_err)
        tse_values.append(s_err + e_err)
        for tau in tolerances_ms:
            tau_sec = tau / 1000.0
            if s_err <= tau_sec and e_err <= tau_sec:
                acc_hits[tau] += 1

    count = len(ref)
    metrics = {
        "count": count,
        "phone_tse_ms": sum(tse_values) / count * 1000.0,
        "start_mae_ms": sum(start_abs) / count * 1000.0,
        "end_mae_ms": sum(end_abs) / count * 1000.0,
    }
    for tau in tolerances_ms:
        metrics[f"phone_acc_tau_{tau}"] = acc_hits[tau] / count * 100.0
    return metrics


def add_weighted(acc: dict, metrics: dict, tolerances_ms: Iterable[int]) -> None:
    n = metrics["count"]
    acc["utterances"] += 1
    acc["count"] += n
    acc["phone_tse_ms_sum"] += metrics["phone_tse_ms"] * n
    acc["start_mae_ms_sum"] += metrics["start_mae_ms"] * n
    acc["end_mae_ms_sum"] += metrics["end_mae_ms"] * n
    for tau in tolerances_ms:
        acc[f"phone_acc_tau_{tau}_hits"] += metrics[f"phone_acc_tau_{tau}"] / 100.0 * n


def finalize(acc: dict, tolerances_ms: Iterable[int]) -> dict:
    n = acc["count"]
    if n == 0:
        return {
            "utterances": acc["utterances"],
            "failures": acc["failures"],
            "count": 0,
        }

    out = {
        "utterances": acc["utterances"],
        "failures": acc["failures"],
        "count": n,
        "phone_tse_ms": acc["phone_tse_ms_sum"] / n,
        "start_mae_ms": acc["start_mae_ms_sum"] / n,
        "end_mae_ms": acc["end_mae_ms_sum"] / n,
        "phone_acc": {},
    }
    for tau in tolerances_ms:
        out["phone_acc"][f"tau_{tau}"] = acc[f"phone_acc_tau_{tau}_hits"] / n * 100.0
    return out


def new_accumulator(tolerances_ms: Iterable[int]) -> dict:
    acc = defaultdict(float)
    acc["utterances"] = 0
    acc["failures"] = 0
    acc["count"] = 0
    for tau in tolerances_ms:
        acc[f"phone_acc_tau_{tau}_hits"] = 0.0
    return acc


def recompute(alignments_path: Path, output_dir: Path, tolerances_ms: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accumulators: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: new_accumulator(tolerances_ms))
    )

    with alignments_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            target = ALIGNMENT_TO_TARGET.get(record.get("alignment_type"))
            if target is None:
                continue
            model = record["model"]
            acc = accumulators[model][target]
            try:
                metrics = compute_run_test_metrics(record, tolerances_ms)
                add_weighted(acc, metrics, tolerances_ms)
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "utt": record.get("utt", ""),
                        "wav": record.get("wav", ""),
                        **metrics,
                    }
                )
            except Exception as exc:
                acc["failures"] += 1
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "utt": record.get("utt", ""),
                        "wav": record.get("wav", ""),
                        "line": line_no,
                        "error": str(exc),
                    }
                )

    summary = {
        model: {
            target: finalize(acc, tolerances_ms)
            for target, acc in sorted(targets.items())
        }
        for model, targets in sorted(accumulators.items())
    }

    summary_path = output_dir / "run_test_style_summary.json"
    per_utt_path = output_dir / "run_test_style_per_utterance.csv"
    matrix_path = output_dir / "run_test_style_matrix.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row})
    with per_utt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matrix_rows = []
    for model, targets in summary.items():
        for target, metrics in targets.items():
            row = {
                "model": model,
                "target": target,
                "utterances": metrics.get("utterances", 0),
                "failures": metrics.get("failures", 0),
                "count": metrics.get("count", 0),
                "phone_tse_ms": metrics.get("phone_tse_ms", ""),
                "start_mae_ms": metrics.get("start_mae_ms", ""),
                "end_mae_ms": metrics.get("end_mae_ms", ""),
            }
            for tau in tolerances_ms:
                row[f"phone_acc_tau_{tau}"] = metrics.get("phone_acc", {}).get(f"tau_{tau}", "")
            matrix_rows.append(row)

    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0]) if matrix_rows else [])
        if matrix_rows:
            writer.writeheader()
            writer.writerows(matrix_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-utterance metrics: {per_utt_path}")
    print(f"Saved matrix: {matrix_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignments",
        type=Path,
        default=Path("fa_research/results/l2arctic_fa_timestamp_metrics/alignments.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fa_research/results/l2arctic_fa_timestamp_metrics"),
    )
    parser.add_argument("--tolerances-ms", nargs="+", type=int, default=[10, 20, 25, 30, 40, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recompute(args.alignments, args.output_dir, args.tolerances_ms)


if __name__ == "__main__":
    main()
