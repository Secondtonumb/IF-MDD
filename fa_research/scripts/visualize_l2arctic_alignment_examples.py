#!/usr/bin/env python3
"""Per-example visualization for L2-ARCTIC canonical/perceived/inference alignments."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["CTC", "OTTC", "CRCTC", "CROTTC"]
CONTEXT_VARIANT_ORDER = ["dp", "tri", "bwt", "wpu"]
CONTEXT_LOSS_ORDER = ["ctc", "crctc", "ottc", "crottc"]
CONTEXT_MODE_ORDER = [
    "uniphone",
    "diphone",
    "triphone",
    "between_word_triphone",
    "word_position_uniphone",
]
CONTEXT_MODE_TITLES = {
    "uniphone": "Uniphone",
    "diphone": "Diphone",
    "triphone": "Triphone",
    "between_word_triphone": "Between-word Triphone",
    "word_position_uniphone": "Word-position Uniphone",
}
LANE_ORDER = ["canonical_fa", "perceived_fa"]
LANE_TITLES = {
    "canonical_fa": "Canonical FA",
    "perceived_fa": "Perceived FA",
}
STATUS_COLORS = {
    "correct": "#62B36F",
    "substitution": "#F28E2B",
    "deletion": "#D64F4F",
    "unknown": "#BAB0AC",
    "ground_truth": "#6F6A55",
}


def load_alignments(path: Path):
    grouped = defaultdict(lambda: defaultdict(dict))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            grouped[record["utt"]][record["model"]][record["alignment_type"]] = record
    return grouped


def edit_statuses(reference: list[str], hypothesis: list[str]) -> list[str]:
    """Return one status per reference token against hypothesis tokens."""
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [(dp[i - 1][j - 1], "eq")]
            else:
                candidates = [(dp[i - 1][j - 1] + 1, "sub")]
            candidates.extend(
                [
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                ]
            )
            cost, op = min(candidates, key=lambda x: x[0])
            dp[i][j] = cost
            back[i][j] = op

    statuses = ["unknown"] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "eq":
            statuses[i - 1] = "correct"
            i -= 1
            j -= 1
        elif op == "sub":
            statuses[i - 1] = "substitution"
            i -= 1
            j -= 1
        elif op == "del":
            statuses[i - 1] = "deletion"
            i -= 1
        elif op == "ins":
            j -= 1
        else:
            break
    return statuses


def format_phone_text(phones: list[str], max_chars: int = 120) -> str:
    text = " ".join(phones)
    if not text:
        return "(empty)"
    wrapped = textwrap.wrap(text, width=max_chars)
    if len(wrapped) > 2:
        return "\n".join(wrapped[:2]) + " ..."
    return "\n".join(wrapped)


def display_phones(record: dict) -> list[str]:
    labels = record.get("display_phones") or record.get("model_labels") or record.get("phones") or []
    return [str(label) for label in labels]


def model_sort_key(model: str):
    if model in MODEL_ORDER:
        return (0, MODEL_ORDER.index(model), model)
    variant, sep, loss = model.rpartition("_")
    if sep and variant in CONTEXT_VARIANT_ORDER and loss in CONTEXT_LOSS_ORDER:
        return (
            1,
            CONTEXT_VARIANT_ORDER.index(variant),
            CONTEXT_LOSS_ORDER.index(loss),
            model,
        )
    return (2, model)


def model_order(records_for_utt: dict) -> list[str]:
    return sorted(records_for_utt, key=model_sort_key)


def normalize_mode(mode: str | None) -> str:
    if not mode or mode == "mono":
        return "uniphone"
    return mode


def mode_sort_key(mode: str):
    if mode in CONTEXT_MODE_ORDER:
        return (CONTEXT_MODE_ORDER.index(mode), mode)
    return (len(CONTEXT_MODE_ORDER), mode)


def model_mode(model_records: dict) -> str:
    for record in model_records.values():
        return normalize_mode(record.get("context_phone_mode"))
    return "uniphone"


def model_groups(records_for_utt: dict) -> list[tuple[str, list[str]]]:
    grouped = defaultdict(list)
    for model in model_order(records_for_utt):
        grouped[model_mode(records_for_utt.get(model, {}))].append(model)
    return [(mode, grouped[mode]) for mode in sorted(grouped, key=mode_sort_key)]


def record_duration(records_for_utt: dict) -> float:
    max_end = 0.0
    for model_records in records_for_utt.values():
        for record in model_records.values():
            ends = record.get("gt_ends") or record.get("pred_ends") or []
            if ends:
                max_end = max(max_end, max(float(x) for x in ends))
    return max_end


def draw_inference_table(ax, utt: str, records_for_utt: dict):
    ax.axis("off")
    ax.text(0.0, 1.0, f"Utterance: {utt}", fontsize=14, fontweight="bold", va="top")
    models = model_order(records_for_utt)
    y = 0.86
    step = 0.82 / max(1, len(models))
    model_fontsize = 8 if len(models) > 8 else 11
    phone_fontsize = 7 if len(models) > 8 else 9
    for model in models:
        inference = records_for_utt.get(model, {}).get("inference_alignment")
        phones = display_phones(inference) if inference else []
        ax.text(0.0, y, model, fontsize=model_fontsize, fontweight="bold", va="top")
        ax.text(0.11, y, format_phone_text(phones), fontsize=phone_fontsize, va="top", family="monospace")
        y -= step


def draw_alignment_lane(ax, record: dict, inference_record: dict | None, title: str, duration: float):
    phones = display_phones(record)
    pred_starts = [float(x) for x in record.get("pred_starts", [])]
    pred_ends = [float(x) for x in record.get("pred_ends", [])]
    gt_starts = [float(x) for x in record.get("gt_starts", [])]
    gt_ends = [float(x) for x in record.get("gt_ends", [])]
    inference_phones = display_phones(inference_record) if inference_record else []
    statuses = edit_statuses(phones, inference_phones)

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9, labelpad=48)
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for idx, (phone, start, end) in enumerate(zip(phones, pred_starts, pred_ends)):
        status = statuses[idx] if idx < len(statuses) else "unknown"
        color = STATUS_COLORS[status]
        width = max(end - start, 0.002)
        ax.broken_barh([(start, width)], (0.25, 0.5), facecolors=color, edgecolors="white", linewidth=0.7)

        if idx < len(gt_starts) and idx < len(gt_ends):
            ax.vlines([gt_starts[idx], gt_ends[idx]], 0.18, 0.82, colors="black", linewidth=0.35, alpha=0.22)

        fontsize = 7 if width > 0.045 else 5
        ax.text(
            start + width / 2,
            0.52,
            phone,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            clip_on=True,
        )

    ax.tick_params(axis="x", labelbottom=False, length=0)


def first_record(model_records_list: list[dict], lane_type: str) -> dict | None:
    for model_records in model_records_list:
        record = model_records.get(lane_type)
        if record and record.get("gt_starts") and record.get("gt_ends"):
            return record
    return None


def draw_ground_truth_lane(ax, record: dict | None, title: str, duration: float):
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9, labelpad=62)
    ax.grid(axis="x", alpha=0.18)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    if not record:
        ax.text(0.0, 0.5, "missing ground truth", va="center")
        ax.tick_params(axis="x", labelbottom=False, length=0)
        return

    phones = display_phones(record)
    starts = [float(x) for x in record.get("gt_starts", [])]
    ends = [float(x) for x in record.get("gt_ends", [])]
    for phone, start, end in zip(phones, starts, ends):
        width = max(end - start, 0.002)
        ax.broken_barh(
            [(start, width)],
            (0.25, 0.5),
            facecolors=STATUS_COLORS["ground_truth"],
            edgecolors="white",
            linewidth=0.7,
        )
        fontsize = 7 if width > 0.045 else 5
        ax.text(
            start + width / 2,
            0.52,
            phone,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            clip_on=True,
        )
    ax.tick_params(axis="x", labelbottom=False, length=0)


def plot_utterance(utt: str, records_for_utt: dict, output_dir: Path):
    duration = record_duration(records_for_utt)
    groups = model_groups(records_for_utt)
    model_count = len(model_order(records_for_utt))
    lane_rows = sum(len(LANE_ORDER) + len(models) * len(LANE_ORDER) for _, models in groups)
    fig_height = max(12, 3.2 + lane_rows * 0.82)
    fig = plt.figure(figsize=(18, fig_height), facecolor="white")
    gs = fig.add_gridspec(
        nrows=lane_rows + 2,
        ncols=1,
        height_ratios=[max(2.2, 0.28 * model_count), *([1] * lane_rows), 0.45],
        hspace=0.15,
    )

    top_ax = fig.add_subplot(gs[0, 0])
    draw_inference_table(top_ax, utt, records_for_utt)

    row = 1
    for mode, models in groups:
        mode_title = CONTEXT_MODE_TITLES.get(mode, mode.replace("_", " ").title())
        model_records_list = [records_for_utt.get(model, {}) for model in models]
        for lane_type in LANE_ORDER:
            ax = fig.add_subplot(gs[row, 0])
            record = first_record(model_records_list, lane_type)
            title = f"Ground Truth\n{mode_title}\n{LANE_TITLES[lane_type]}"
            draw_ground_truth_lane(ax, record, title, duration)
            row += 1
        for model in models:
            model_records = records_for_utt.get(model, {})
            inference_record = model_records.get("inference_alignment")
            for lane_type in LANE_ORDER:
                ax = fig.add_subplot(gs[row, 0])
                record = model_records.get(lane_type)
                title = f"{model}\n{LANE_TITLES[lane_type]}"
                if record:
                    draw_alignment_lane(ax, record, inference_record, title, duration)
                else:
                    ax.axis("off")
                    ax.text(0.0, 0.5, f"{title}: missing", va="center")
                row += 1

    time_ax = fig.add_subplot(gs[-1, 0])
    time_ax.set_xlim(0, duration)
    time_ax.set_ylim(0, 1)
    time_ax.set_yticks([])
    time_ax.set_xlabel("Time (seconds)", fontsize=11)
    time_ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        time_ax.spines[spine].set_visible(False)

    legend_items = [
        mpatches.Patch(color=STATUS_COLORS["ground_truth"], label="ground truth timestamps"),
        mpatches.Patch(color=STATUS_COLORS["correct"], label="matched inference phone"),
        mpatches.Patch(color=STATUS_COLORS["substitution"], label="wrong phone"),
        mpatches.Patch(color=STATUS_COLORS["deletion"], label="not recognized"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("L2-ARCTIC FA vs Inference Alignment", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{utt}_alignment_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", default="l2arctic_fa_timestamp_metrics/alignments.jsonl")
    parser.add_argument("--output-dir", default="l2arctic_fa_timestamp_metrics/example_figures")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--utt", nargs="*", default=None)
    args = parser.parse_args()

    grouped = load_alignments(Path(args.alignments))
    utts = sorted(grouped)
    if args.utt:
        wanted = set(args.utt)
        utts = [utt for utt in utts if utt in wanted]
    if args.limit > 0:
        utts = utts[: args.limit]

    output_dir = Path(args.output_dir)
    for idx, utt in enumerate(utts, start=1):
        out_path = plot_utterance(utt, grouped[utt], output_dir)
        print(f"[{idx}/{len(utts)}] saved {out_path}")


if __name__ == "__main__":
    main()
