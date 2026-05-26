#!/usr/bin/env python3
"""Visualize L2-ARCTIC forced-alignment timestamp metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["CTC", "OTTC", "CRCTC", "CROTTC"]
TARGET_ORDER = ["canonical", "perceived"]
TARGET_COLORS = {
    "canonical": "#4C78A8",
    "perceived": "#F58518",
}


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("target") not in TARGET_ORDER:
                continue
            rows.append(row)
    return rows


def to_float(row, key, default=np.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def save_summary_bars(summary: dict, output_dir: Path):
    metrics = [
        ("boundary_mse", "Boundary MSE (s^2)", "Lower is better"),
        ("boundary_mae", "Boundary MAE (s)", "Lower is better"),
        ("segment_within_50ms", "Segments Within 50 ms", "Higher is better"),
        ("segment_within_100ms", "Segments Within 100 ms", "Higher is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    x = np.arange(len(MODEL_ORDER))
    width = 0.36

    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        for offset, target in [(-width / 2, "canonical"), (width / 2, "perceived")]:
            values = [
                summary.get(model, {}).get(target, {}).get(metric, np.nan)
                for model in MODEL_ORDER
            ]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=target,
                color=TARGET_COLORS[target],
                alpha=0.9,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    label = f"{value:.3f}" if "within" in metric else f"{value:.4f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

        ax.set_title(subtitle, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, ncols=2)
    fig.suptitle("L2-ARCTIC FA Timestamp Metrics", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "summary_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tolerance_curves(summary: dict, output_dir: Path):
    tolerances = []
    sample = next(iter(next(iter(summary.values())).values()))
    for key in sample:
        if key.startswith("segment_within_") and key.endswith("ms"):
            tolerances.append(int(key.removeprefix("segment_within_").removesuffix("ms")))
    tolerances = sorted(tolerances)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, target in zip(axes, TARGET_ORDER):
        for model in MODEL_ORDER:
            y = [
                summary.get(model, {}).get(target, {}).get(f"segment_within_{tol}ms", np.nan)
                for tol in tolerances
            ]
            ax.plot(tolerances, y, marker="o", linewidth=2, label=model)

        ax.set_title(target.capitalize())
        ax.set_xlabel("Tolerance (ms)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Segment Hit Rate")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Tolerance Curves", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "tolerance_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_utterance_boxplots(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["target"])].append(to_float(row, "boundary_mae"))

    for ax, target in zip(axes, TARGET_ORDER):
        data = [grouped[(model, target)] for model in MODEL_ORDER]
        bp = ax.boxplot(
            data,
            labels=MODEL_ORDER,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(TARGET_COLORS[target])
            patch.set_alpha(0.7)
        ax.set_title(target.capitalize())
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Per-Utterance Boundary MAE (s)")
    fig.suptitle("Boundary MAE Distribution", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "boundary_mae_boxplots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_canonical_vs_perceived_scatter(rows: list[dict], output_dir: Path):
    by_model_utt = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["utt"])
        by_model_utt[key][row["target"]] = to_float(row, "boundary_mae")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    all_values = []

    for ax, model in zip(axes, MODEL_ORDER):
        xs = []
        ys = []
        for (row_model, _utt), values in by_model_utt.items():
            if row_model != model:
                continue
            if "canonical" in values and "perceived" in values:
                xs.append(values["canonical"])
                ys.append(values["perceived"])
        all_values.extend(xs)
        all_values.extend(ys)
        ax.scatter(xs, ys, s=28, alpha=0.72, color="#54A24B", edgecolor="white", linewidth=0.4)
        ax.set_title(model)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    max_value = max(all_values) if all_values else 0.1
    limit = max_value * 1.08
    for ax in axes:
        ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)

    axes[2].set_xlabel("Canonical Boundary MAE (s)")
    axes[3].set_xlabel("Canonical Boundary MAE (s)")
    axes[0].set_ylabel("Perceived Boundary MAE (s)")
    axes[2].set_ylabel("Perceived Boundary MAE (s)")
    fig.suptitle("Canonical vs Perceived Per-Utterance Error", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "canonical_vs_perceived_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_vs_error(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, target in zip(axes, TARGET_ORDER):
        for model in MODEL_ORDER:
            xs = [
                to_float(row, "mean_alignment_score")
                for row in rows
                if row["target"] == target and row["model"] == model
            ]
            ys = [
                to_float(row, "boundary_mae")
                for row in rows
                if row["target"] == target and row["model"] == model
            ]
            ax.scatter(xs, ys, s=24, alpha=0.65, label=model)
        ax.set_title(target.capitalize())
        ax.set_xlabel("Mean FA Score")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Boundary MAE (s)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Alignment Confidence vs Timestamp Error", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "score_vs_boundary_mae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="l2arctic_fa_timestamp_metrics")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(metrics_dir / "summary.json", "r", encoding="utf-8"))
    rows = read_rows(metrics_dir / "per_utterance.csv")

    save_summary_bars(summary, output_dir)
    save_tolerance_curves(summary, output_dir)
    save_per_utterance_boxplots(rows, output_dir)
    save_canonical_vs_perceived_scatter(rows, output_dir)
    save_score_vs_error(rows, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
