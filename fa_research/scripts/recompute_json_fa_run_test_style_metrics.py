#!/usr/bin/env python3
"""Recompute run-test-style metrics for JSON FA alignment records.

The evaluator writes one alignment record per utterance/model/dataset. This
script converts those records to the paper-style phone ACC/TSE metrics:

  - filter silence labels
  - TSE per phone = abs(start error) + abs(end error), in milliseconds
  - ACC@tau = both start and end boundaries are within tau milliseconds
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SILENCE = {"", "sil", "SIL", "sp", "SP", "spn", "SPN"}


def is_silence(label: str) -> bool:
    return label in SILENCE or label.upper() in SILENCE


def model_parts(model: str) -> tuple[str, str]:
    if "_" not in model:
        return "", model
    source, recipe = model.split("_", 1)
    return source, recipe


def filtered_entries(record: dict, starts_key: str, ends_key: str) -> list[tuple[float, float, str]]:
    return [
        (float(start), float(end), str(phone))
        for phone, start, end in zip(record["phones"], record[starts_key], record[ends_key])
        if not is_silence(str(phone))
    ]


def compute_run_test_metrics(record: dict, tolerances_ms: Iterable[int]) -> dict:
    ref = filtered_entries(record, "ref_starts", "ref_ends")
    pred = filtered_entries(record, "pred_starts", "pred_ends")
    if not ref or len(ref) != len(pred):
        raise ValueError(f"filtered length mismatch: ref={len(ref)}, pred={len(pred)}")

    start_abs = []
    end_abs = []
    tse_values = []
    acc_hits = {tau: 0 for tau in tolerances_ms}

    for (rs, re, _), (ps, pe, _) in zip(ref, pred):
        start_error = abs(rs - ps)
        end_error = abs(re - pe)
        start_abs.append(start_error)
        end_abs.append(end_error)
        tse_values.append(start_error + end_error)
        for tau in tolerances_ms:
            tau_sec = tau / 1000.0
            if start_error <= tau_sec and end_error <= tau_sec:
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


def new_accumulator(tolerances_ms: Iterable[int]) -> dict:
    acc = defaultdict(float)
    acc["utterances"] = 0
    acc["failures"] = 0
    acc["count"] = 0
    for tau in tolerances_ms:
        acc[f"phone_acc_tau_{tau}_hits"] = 0.0
    return acc


def add_metrics(acc: dict, metrics: dict, tolerances_ms: Iterable[int]) -> None:
    n = metrics["count"]
    acc["utterances"] += 1
    acc["count"] += n
    # Match run_test.version1.py: compute each utterance metric first, then
    # average utterances equally. Keep phone-weighted values only as audit cols.
    acc["phone_tse_ms_utt_sum"] += metrics["phone_tse_ms"]
    acc["start_mae_ms_utt_sum"] += metrics["start_mae_ms"]
    acc["end_mae_ms_utt_sum"] += metrics["end_mae_ms"]
    acc["phone_tse_ms_phone_sum"] += metrics["phone_tse_ms"] * n
    acc["start_mae_ms_phone_sum"] += metrics["start_mae_ms"] * n
    acc["end_mae_ms_phone_sum"] += metrics["end_mae_ms"] * n
    for tau in tolerances_ms:
        acc[f"phone_acc_tau_{tau}_utt_sum"] += metrics[f"phone_acc_tau_{tau}"]
        acc[f"phone_acc_tau_{tau}_phone_hits"] += metrics[f"phone_acc_tau_{tau}"] / 100.0 * n


def finalize(acc: dict, tolerances_ms: Iterable[int]) -> dict:
    n = acc["count"]
    u = acc["utterances"]
    out = {
        "utterances": int(u),
        "failures": int(acc["failures"]),
        "count": int(n),
    }
    if n == 0 or u == 0:
        return out
    out.update(
        {
            "phone_tse_ms": acc["phone_tse_ms_utt_sum"] / u,
            "start_mae_ms": acc["start_mae_ms_utt_sum"] / u,
            "end_mae_ms": acc["end_mae_ms_utt_sum"] / u,
            "phone_tse_ms_phone_weighted": acc["phone_tse_ms_phone_sum"] / n,
            "start_mae_ms_phone_weighted": acc["start_mae_ms_phone_sum"] / n,
            "end_mae_ms_phone_weighted": acc["end_mae_ms_phone_sum"] / n,
            "phone_acc": {
                f"tau_{tau}": acc[f"phone_acc_tau_{tau}_utt_sum"] / u
                for tau in tolerances_ms
            },
            "phone_acc_phone_weighted": {
                f"tau_{tau}": acc[f"phone_acc_tau_{tau}_phone_hits"] / n * 100.0
                for tau in tolerances_ms
            },
        }
    )
    return out


def recompute(
    alignments_path: Path,
    output_dir: Path,
    tolerances_ms: list[int],
    datasets: list[str] | None = None,
    models: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accumulators: dict[tuple[str, str, str], dict] = defaultdict(lambda: new_accumulator(tolerances_ms))
    dataset_filter = set(datasets or [])
    model_filter = set(models or [])

    with alignments_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = record.get("dataset", "")
            target = record.get("target", "")
            model = record.get("model", "")
            if dataset_filter and dataset not in dataset_filter:
                continue
            if model_filter and model not in model_filter:
                continue
            source, recipe = model_parts(model)
            key = (dataset, target, model)
            acc = accumulators[key]

            try:
                metrics = compute_run_test_metrics(record, tolerances_ms)
                add_metrics(acc, metrics, tolerances_ms)
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "model": model,
                        "source": source,
                        "recipe": recipe,
                        "utt": record.get("utt", ""),
                        "wav": record.get("wav", ""),
                        **metrics,
                    }
                )
            except Exception as exc:
                acc["failures"] += 1
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "model": model,
                        "source": source,
                        "recipe": recipe,
                        "utt": record.get("utt", ""),
                        "wav": record.get("wav", ""),
                        "line": line_no,
                        "error": str(exc),
                    }
                )

    summary = {}
    matrix_rows = []
    for (dataset, target, model), acc in sorted(accumulators.items()):
        metrics = finalize(acc, tolerances_ms)
        summary.setdefault(dataset, {}).setdefault(target, {})[model] = metrics
        source, recipe = model_parts(model)
        row = {
            "dataset": dataset,
            "target": target,
            "model": model,
            "source": source,
            "recipe": recipe,
            "utterances": metrics.get("utterances", 0),
            "failures": metrics.get("failures", 0),
            "count": metrics.get("count", 0),
            "phone_tse_ms": metrics.get("phone_tse_ms", ""),
            "start_mae_ms": metrics.get("start_mae_ms", ""),
            "end_mae_ms": metrics.get("end_mae_ms", ""),
            "phone_tse_ms_phone_weighted": metrics.get("phone_tse_ms_phone_weighted", ""),
            "start_mae_ms_phone_weighted": metrics.get("start_mae_ms_phone_weighted", ""),
            "end_mae_ms_phone_weighted": metrics.get("end_mae_ms_phone_weighted", ""),
        }
        for tau in tolerances_ms:
            row[f"phone_acc_tau_{tau}"] = metrics.get("phone_acc", {}).get(f"tau_{tau}", "")
            row[f"phone_acc_tau_{tau}_phone_weighted"] = metrics.get("phone_acc_phone_weighted", {}).get(f"tau_{tau}", "")
        matrix_rows.append(row)

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

    if matrix_rows:
        with matrix_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0]))
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
        default=Path("fa_research/results/cross_domain_json_fa/alignments.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fa_research/results/cross_domain_json_fa"),
    )
    parser.add_argument("--tolerances-ms", nargs="+", type=int, default=[10, 20, 25, 30, 40, 50])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recompute(args.alignments, args.output_dir, args.tolerances_ms, args.datasets, args.models)


if __name__ == "__main__":
    main()
