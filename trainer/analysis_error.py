#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 1. Phone → ID mapping (provided by user)
# =========================================================

PHONE2ID = {
    "<blank>": 0,
    "<eos>": 1,
    "<sil>": 2,
    "y": 3,
    "a": 4,
    "f": 5,
    "q": 6,
    "A": 7,
    "h": 8,
    "w": 9,
    "l": 10,
    "ii": 11,
    "H": 12,
    "$": 13,
    "r": 14,
    "n": 15,
    "aa": 16,
    "d": 17,
    "<": 18,
    "i": 19,
    "s": 20,
    "u": 21,
    "j": 22,
    "z": 23,
    "m": 24,
    "b": 25,
    "AA": 26,
    "Z": 27,
    "I": 28,
    "t": 29,
    "k": 30,
    "g": 31,
    "E": 32,
    "ll": 33,
    "uu": 34,
    "S": 35,
    "*": 36,
    "^": 37,
    "UU": 38,
    "zz": 39,
    "x": 40,
    "D": 41,
    "mm": 42,
    "ss": 43,
    "EE": 44,
    "ww": 45,
    "nn": 46,
    "U": 47,
    "T": 48,
    "**": 49,
    "bb": 50,
    "qq": 51,
    "dd": 52,
    "rr": 53,
    "ZZ": 54,
    "$$": 55,
    "jj": 56,
    "kk": 57,
    "SS": 58,
    "tt": 59,
    "yy": 60,
    "^^": 61,
    "TT": 62,
    "xx": 63,
    "ff": 64,
    "DD": 65,
    "II": 66,
    "hh": 67,
    "HH": 68,
    "<bos>": 69,
}


# =========================================================
# 2. Display rule for x-axis labels
# =========================================================

_ONLY_LETTERS = re.compile(r"^[A-Za-z]+$")

def display_label(phone: str) -> str:
    """
    If phone contains only letters: show phone
    Else: show its numeric ID
    """
    if _ONLY_LETTERS.match(phone):
        return phone
    if phone not in PHONE2ID:
        raise ValueError(f"Phone not in PHONE2ID: {phone}")
    return str(PHONE2ID[phone])


# =========================================================
# 3. Tokenization & alignment
# =========================================================

def tokenize(ph_str):
    if not ph_str:
        return []
    return ph_str.strip().split()


def levenshtein_ops(ref, hyp):
    """
    Return ops: (op, ref_phone_or_None, hyp_phone_or_None)
    op ∈ {eq, sub, del, ins}
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, None)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", None, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            candidates = [
                (dp[i - 1][j] + 1, ("del", i - 1, None)),
                (dp[i][j - 1] + 1, ("ins", None, j - 1)),
                (dp[i - 1][j - 1] + cost,
                 ("eq" if cost == 0 else "sub", i - 1, j - 1)),
            ]
            dp[i][j], bt[i][j] = min(candidates, key=lambda x: x[0])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        tag, ri, hj = bt[i][j]
        if tag == "del":
            ops.append(("del", ref[ri], None))
            i -= 1
        elif tag == "ins":
            ops.append(("ins", None, hyp[hj]))
            j -= 1
        elif tag == "eq":
            ops.append(("eq", ref[ri], hyp[hj]))
            i -= 1
            j -= 1
        else:
            ops.append(("sub", ref[ri], hyp[hj]))
            i -= 1
            j -= 1

    return ops[::-1]


# =========================================================
# 4. Analysis
# =========================================================

def analyze(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_ref = Counter()
    ins = Counter()
    dele = Counter()
    sub_in = Counter()
    sub_out = Counter()

    for item in data.values():
        ref = tokenize(item.get("phoneme_ref", ""))
        hyp = tokenize(item.get("phoneme_mis", ""))

        total_ref.update(ref)

        for op, r, h in levenshtein_ops(ref, hyp):
            if op == "ins":
                ins[h] += 1
            elif op == "del":
                dele[r] += 1
            elif op == "sub":
                sub_out[r] += 1
                sub_in[h] += 1

    phones = sorted(set(total_ref) | set(ins) | set(dele) | set(sub_in) | set(sub_out))

    rows = []
    for p in phones:
        rows.append({
            "phone": p,
            "count_total": total_ref.get(p, 0),
            "ins": ins.get(p, 0),
            "del": dele.get(p, 0),
            "sub_in": sub_in.get(p, 0),
            "sub_out": sub_out.get(p, 0),
        })

    df = pd.DataFrame(rows)
    df["err_total"] = df[["ins", "del", "sub_in", "sub_out"]].sum(axis=1)
    df["err_rate"] = df["err_total"] / df["count_total"].replace(0, pd.NA)
    df = df.sort_values(["count_total", "err_total"], ascending=[False, False]).reset_index(drop=True)

    return df


# =========================================================
# 5. Plot 1: counts + error rate (3 axes)
# =========================================================

def plot_combined(df, out_png: Path, top_k: int):
    d = df.head(top_k).copy()
    x = list(range(len(d)))

    fig, ax_total = plt.subplots(figsize=(max(12, top_k * 0.32), 6))

    # Total count (left)
    ax_total.bar(
        [i - 0.25 for i in x],
        d["count_total"],
        width=0.35,
        color="#1f77b4",
        label="Total count",
    )
    ax_total.set_ylabel("Total count", color="#1f77b4")
    ax_total.tick_params(axis="y", labelcolor="#1f77b4")

    # Error count (right 1)
    ax_err = ax_total.twinx()
    ax_err.bar(
        [i + 0.25 for i in x],
        d["err_total"],
        width=0.35,
        color="#ff7f0e",
        label="Error count",
    )
    ax_err.set_ylabel("Error count", color="#ff7f0e")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")

    # Error rate (right 2, shifted)
    ax_rate = ax_total.twinx()
    ax_rate.spines["right"].set_position(("axes", 1.08))
    ax_rate.set_frame_on(True)
    ax_rate.patch.set_visible(False)

    ax_rate.plot(
        x,
        d["err_rate"],
        color="#d62728",
        marker="o",
        linewidth=2.2,
        label="Error rate",
    )
    ax_rate.set_ylabel("Error rate", color="#d62728")
    ax_rate.tick_params(axis="y", labelcolor="#d62728")
    ax_rate.set_ylim(0, d["err_rate"].max() * 1.25)

    for i, v in enumerate(d["err_rate"]):
        if pd.notna(v):
            ax_rate.text(i, v, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=8, color="#d62728")

    ax_total.set_xticks(x)
    ax_total.set_xticklabels(
        [display_label(p) for p in d["phone"]],
        rotation=90
    )

    handles, labels = [], []
    for ax in [ax_total, ax_err, ax_rate]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    ax_total.legend(handles, labels, loc="upper left", fontsize=9)
    ax_total.set_title("Phoneme frequency, error count, and error rate")

    fig.subplots_adjust(bottom=0.42, right=0.85)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================================================
# 6. Plot 2: detailed error breakdown
# =========================================================

def plot_detailed(df, out_png: Path, top_k: int):
    d = df.head(top_k).copy()
    x = list(range(len(d)))
    w = 0.18

    fig, ax = plt.subplots(figsize=(max(12, top_k * 0.35), 6))

    ax.bar([i - 1.5 * w for i in x], d["ins"], width=w, label="ins")
    ax.bar([i - 0.5 * w for i in x], d["del"], width=w, label="del")
    ax.bar([i + 0.5 * w for i in x], d["sub_in"], width=w, label="sub_in")
    ax.bar([i + 1.5 * w for i in x], d["sub_out"], width=w, label="sub_out")

    ax.set_ylabel("Count")
    ax.set_title("Detailed error breakdown by phoneme")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_label(p) for p in d["phone"]],
        rotation=90
    )

    ax.legend()
    fig.subplots_adjust(bottom=0.42)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================================================
# 7. Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = analyze(Path(args.json))
    df.to_csv(out_dir / "phone_stats.csv", index=False, encoding="utf-8")

    k = min(args.top_k, len(df))
    plot_combined(df, out_dir / "phone_total_vs_errors.png", k)
    plot_detailed(df, out_dir / "phone_error_breakdown.png", k)

    print("Saved results to:", out_dir)


if __name__ == "__main__":
    main()
