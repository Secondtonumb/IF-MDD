#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
from pathlib import Path


# =========================================================
# Tokenization & alignment
# =========================================================

def tokenize(ph_str):
    if not ph_str:
        return []
    return ph_str.strip().split()


def levenshtein_ops(ref, hyp):
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
            cands = [
                (dp[i - 1][j] + 1, ("del", i - 1, None)),
                (dp[i][j - 1] + 1, ("ins", None, j - 1)),
                (dp[i - 1][j - 1] + cost,
                 ("eq" if cost == 0 else "sub", i - 1, j - 1)),
            ]
            dp[i][j], bt[i][j] = min(cands, key=lambda x: x[0])

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
# Main analysis
# =========================================================

def analyze(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_ref = Counter()
    total_mis = Counter()

    ins = Counter()
    dele = Counter()
    sub = Counter()   # (ref, hyp)

    for item in data.values():
        # ref = tokenize(item.get("phoneme_ref", ""))
        # hyp = tokenize(item.get("phoneme_mis", ""))
        ref = tokenize(item.get("canonical_aligned", ""))
        hyp = tokenize(item.get("perceived_aligned", ""))

        total_ref.update(ref)
        total_mis.update(hyp)

        for op, r, h in levenshtein_ops(ref, hyp):
            if op == "ins":
                ins[h] += 1
            elif op == "del":
                dele[r] += 1
            elif op == "sub":
                sub[(r, h)] += 1

    return total_ref, total_mis, ins, dele, sub


# =========================================================
# Reporting
# =========================================================

def report_ins(total_mis, ins, top_k):
    print("\n" + "=" * 80)
    print("Insertion errors (Top-K by COUNT, then RATE)")
    print("=" * 80)

    # ---- Top-K by count ----
    for p, cnt in ins.most_common(top_k):
        rate = cnt / total_mis[p] if total_mis[p] > 0 else 0
        print(f"[COUNT] {p:<10s}  count={cnt:<6d}  rate={rate:.4f}")

    print("\n--- Top-K by RATE ---")
    rates = [(p, cnt, cnt / total_mis[p])
             for p, cnt in ins.items() if total_mis[p] > 0]
    rates.sort(key=lambda x: x[2], reverse=True)

    for p, cnt, rate in rates[:top_k]:
        print(f"[RATE ] {p:<10s}  rate={rate:.4f}  count={cnt}")


def report_del(total_ref, dele, top_k):
    print("\n" + "=" * 80)
    print("Deletion errors (Top-K by COUNT, then RATE)")
    print("=" * 80)

    for p, cnt in dele.most_common(top_k):
        rate = cnt / total_ref[p] if total_ref[p] > 0 else 0
        print(f"[COUNT] {p:<10s}  count={cnt:<6d}  rate={rate:.4f}")

    print("\n--- Top-K by RATE ---")
    rates = [(p, cnt, cnt / total_ref[p])
             for p, cnt in dele.items() if total_ref[p] > 0]
    rates.sort(key=lambda x: x[2], reverse=True)

    for p, cnt, rate in rates[:top_k]:
        print(f"[RATE ] {p:<10s}  rate={rate:.4f}  count={cnt}")


def report_sub(total_ref, total_mis, sub, top_k):
    print("\n" + "=" * 80)
    print("Substitution errors (Top-K by COUNT, then RATE)")
    print("=" * 80)

    for (r, h), cnt in sub.most_common(top_k):
        denom = min(total_ref[r], total_mis[h])
        rate = cnt / denom if denom > 0 else 0
        print(f"[COUNT] {r:>4s} → {h:<4s}  count={cnt:<6d}  rate={rate:.4f}")

    print("\n--- Top-K by RATE ---")
    rates = []
    for (r, h), cnt in sub.items():
        denom = min(total_ref[r], total_mis[h])
        if denom > 0:
            rates.append((r, h, cnt, cnt / denom))

    rates.sort(key=lambda x: x[3], reverse=True)

    for r, h, cnt, rate in rates[:top_k]:
        print(f"[RATE ] {r:>4s} → {h:<4s}  rate={rate:.4f}  count={cnt}")


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    total_ref, total_mis, ins, dele, sub = analyze(Path(args.json))

    report_ins(total_mis, ins, args.top_k)
    report_del(total_ref, dele, args.top_k)
    report_sub(total_ref, total_mis, sub, args.top_k)


if __name__ == "__main__":
    main()
