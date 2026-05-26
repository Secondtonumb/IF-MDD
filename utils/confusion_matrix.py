#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


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


def build_confusion_matrix(json_path: Path, min_count=1):
    """
    Build confusion matrix from aligned phoneme data.
    Returns: confusion_map (dict): ref_phoneme -> Counter of hyp_phonemes
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    confusion_map = defaultdict(Counter)
    total_ref_count = Counter()

    for item in data.values():
        ref = tokenize(item.get("canonical_aligned", ""))
        hyp = tokenize(item.get("perceived_aligned", ""))

        total_ref_count.update(ref)

        for op, r, h in levenshtein_ops(ref, hyp):
            if op == "sub":
                confusion_map[r][h] += 1

    # Filter by minimum count
    filtered_map = {}
    for ref_ph, hyp_counter in confusion_map.items():
        filtered_counter = Counter({h: c for h, c in hyp_counter.items() if c >= min_count})
        if filtered_counter:
            filtered_map[ref_ph] = filtered_counter

    return filtered_map, total_ref_count


def print_confusion_matrix(confusion_map, total_ref_count, top_k=5, min_rate=0.0):
    """
    Print confusion matrix in the format:
    ref_phoneme: hyp1, hyp2, hyp3 (with counts and rates)
    """
    print("\n" + "=" * 80)
    print("Confusion Matrix (Substitution Errors)")
    print("=" * 80)
    print(f"Format: REF → [HYP (count, rate%), ...]")
    print("=" * 80 + "\n")

    # Sort by reference phoneme
    sorted_refs = sorted(confusion_map.keys())

    for ref_ph in sorted_refs:
        hyp_counter = confusion_map[ref_ph]
        total_ref = total_ref_count[ref_ph]

        # Sort by count (descending)
        most_common = hyp_counter.most_common(top_k)

        # Filter by minimum rate
        filtered = []
        for hyp_ph, count in most_common:
            rate = count / total_ref if total_ref > 0 else 0
            if rate >= min_rate:
                filtered.append((hyp_ph, count, rate))

        if filtered:
            # Format output
            confusions = ", ".join([f"{h} ({c}, {r*100:.1f}%)" for h, c, r in filtered])
            print(f"{ref_ph:>6s} → {confusions}")


def print_simple_confusion_matrix(confusion_map, top_k=5):
    """
    Print confusion matrix in simple format:
    ref_phoneme: hyp1, hyp2, hyp3
    """
    print("\n" + "=" * 80)
    print("Confusion Matrix (Simple Format)")
    print("=" * 80 + "\n")

    sorted_refs = sorted(confusion_map.keys())

    for ref_ph in sorted_refs:
        hyp_counter = confusion_map[ref_ph]
        most_common = hyp_counter.most_common(top_k)
        
        if most_common:
            hyp_list = ", ".join([h for h, _ in most_common])
            print(f"{ref_ph}: {hyp_list}")


def analyze_error_rates(confusion_map, total_ref_count):
    """
    Analyze error rates for each phoneme and identify phonemes that are easily confused.
    Returns a list of tuples: (phoneme, error_count, total_count, error_rate)
    sorted by error_rate in descending order.
    """
    error_rates = []
    
    for ref_ph in confusion_map.keys():
        error_count = sum(confusion_map[ref_ph].values())
        total_count = total_ref_count[ref_ph]
        error_rate = error_count / total_count if total_count > 0 else 0
        error_rates.append((ref_ph, error_count, total_count, error_rate))
    
    # Sort by error rate (descending)
    error_rates.sort(key=lambda x: x[3], reverse=True)
    
    return error_rates


def analyze_reverse_error_rates(confusion_map, total_ref_count):
    """
    Reverse analysis: For each CANONICAL phoneme, count how many times it's mispronounced.
    This helps identify which canonical phonemes are "easy to mispronounce".
    
    Returns: list of tuples (canonical_ph, mispronounce_count, total_count, mispronounce_rate)
    sorted by mispronounce_rate descending.
    """
    mispronounce_rates = []
    
    for canonical_ph in total_ref_count.keys():
        # Total times this canonical phoneme appears
        total_count = total_ref_count[canonical_ph]
        
        # How many times it was mispronounced (substituted)
        if canonical_ph in confusion_map:
            mispronounce_count = sum(confusion_map[canonical_ph].values())
        else:
            mispronounce_count = 0
        
        mispronounce_rate = mispronounce_count / total_count if total_count > 0 else 0
        mispronounce_rates.append((canonical_ph, mispronounce_count, total_count, mispronounce_rate))
    
    # Sort by mispronounce rate (descending)
    mispronounce_rates.sort(key=lambda x: x[3], reverse=True)
    
    return mispronounce_rates


def print_error_analysis(error_rates, top_n=10):
    """
    Print phonemes that are easily confused (high error rates).
    """
    print("\n" + "=" * 80)
    print("Error Rate Analysis - Phonemes Most Easily Confused")
    print("=" * 80)
    print(f"{'Phoneme':<10s} {'Errors':<10s} {'Total':<10s} {'Error Rate':<15s} {'Weight':<10s}")
    print("-" * 80)
    
    easy_confused = []
    for i, (phoneme, error_count, total_count, error_rate) in enumerate(error_rates[:top_n]):
        # Calculate weight (inverse of accuracy, normalized)
        weight = error_rate  # or could be (1 - error_rate) for other interpretations
        easy_confused.append((phoneme, error_rate, weight))
        print(f"{phoneme:<10s} {error_count:<10d} {total_count:<10d} {error_rate*100:<14.2f}% {weight:<10.4f}")
    
    print("\n" + "=" * 80)
    print("Recommended Class Weights (for weighted loss)")
    print("=" * 80)
    print("# Higher weight = more focus on correcting errors")
    print("class_weights = {")
    for phoneme, error_rate, weight in easy_confused:
        # Scale weight: if error_rate=0.5, weight should be higher than 1.0
        # Simple scaling: weight = 1 + error_rate
        scaled_weight = 1.0 + error_rate
        print(f'    "{phoneme}": {scaled_weight:.4f},  # error_rate={error_rate*100:.2f}%')
    print("}")
    
    return easy_confused


def print_reverse_error_analysis(mispronounce_rates, top_n=10):
    """
    Print canonical phonemes that are most easily mispronounced.
    For training: LOWER weight for hard-to-pronounce phonemes.
    """
    print("\n" + "=" * 80)
    print("Reverse Analysis - Canonical Phonemes Most Easily Mispronounced")
    print("=" * 80)
    print(f"{'Canonical':<10s} {'Mispro':<10s} {'Total':<10s} {'Mispro Rate':<15s} {'Suggested Weight':<15s}")
    print("-" * 80)
    
    easy_mispronounce = []
    for canonical_ph, mispro_count, total_count, mispro_rate in mispronounce_rates[:top_n]:
        # LOWER weight = hard to pronounce, don't be too strict
        # weight = 1 / (1 + mispro_rate)
        suggested_weight = 1.0 / (1.0 + mispro_rate)
        easy_mispronounce.append((canonical_ph, mispro_rate, suggested_weight))
        
        print(f"{canonical_ph:<10s} {mispro_count:<10d} {total_count:<10d} {mispro_rate*100:<14.2f}% {suggested_weight:<15.4f}")
    
    print("\n" + "=" * 80)
    print("Recommended Class Weights (LOWERED for hard-to-pronounce phonemes)")
    print("=" * 80)
    print("# Key insight: Hard-to-pronounce canonical phonemes get LOWER weights")
    print("# This allows model to be less strict on inherently difficult sounds")
    print("class_weights = {")
    for canonical_ph, mispro_rate, suggested_weight in easy_mispronounce:
        print(f'    "{canonical_ph}": {suggested_weight:.4f},  # mispro_rate={mispro_rate*100:.2f}% (hard to say)')
    print("}")
    print()
    
    return easy_mispronounce


def save_confusion_to_json(confusion_map, total_ref_count, output_path: Path, top_k=5):
    """
    Save confusion matrix to JSON file with detailed statistics.
    """
    output_data = {
        "confusion_matrix": {},
        "statistics": {
            "total_phonemes": len(confusion_map),
            "total_substitutions": sum(sum(c.values()) for c in confusion_map.values())
        }
    }
    
    for ref_ph in sorted(confusion_map.keys()):
        hyp_counter = confusion_map[ref_ph]
        total_ref = total_ref_count[ref_ph]
        most_common = hyp_counter.most_common(top_k)
        
        confusions = []
        for hyp_ph, count in most_common:
            rate = count / total_ref if total_ref > 0 else 0
            confusions.append({
                "phoneme": hyp_ph,
                "count": count,
                "rate": round(rate, 4)
            })
        
        output_data["confusion_matrix"][ref_ph] = {
            "confusions": confusions,
            "total_ref_count": total_ref
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Confusion matrix saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate phoneme confusion matrix from aligned data")
    ap.add_argument("--json", required=True, help="Path to JSON file with aligned phonemes")
    ap.add_argument("--top_k", type=int, default=5, help="Show top K confusions per phoneme")
    ap.add_argument("--min_count", type=int, default=1, help="Minimum confusion count to include")
    ap.add_argument("--min_rate", type=float, default=0.0, help="Minimum confusion rate to include")
    ap.add_argument("--simple", action="store_true", help="Use simple format (phoneme: list)")
    ap.add_argument("--output", type=str, help="Output JSON file path to save confusion matrix")
    ap.add_argument("--analyze", action="store_true", help="Analyze and print error rates for each phoneme")
    ap.add_argument("--analyze-reverse", action="store_true", help="Reverse analysis: canonical phonemes that are easy to mispronounce")
    ap.add_argument("--top_errors", type=int, default=10, help="Show top N phonemes with highest error rates")
    args = ap.parse_args()

    confusion_map, total_ref_count = build_confusion_matrix(Path(args.json), args.min_count)

    # Get all unique phonemes (from both ref and hyp)
    all_phonemes = set(confusion_map.keys())
    for hyp_counter in confusion_map.values():
        all_phonemes.update(hyp_counter.keys())

    if args.simple:
        print_simple_confusion_matrix(confusion_map, args.top_k)
    else:
        print_confusion_matrix(confusion_map, total_ref_count, args.top_k, args.min_rate)

    # Analyze error rates if requested
    if args.analyze:
        error_rates = analyze_error_rates(confusion_map, total_ref_count)
        easy_confused = print_error_analysis(error_rates, args.top_errors)

    # Reverse analysis: canonical phonemes easy to mispronounce
    if args.analyze_reverse:
        mispronounce_rates = analyze_reverse_error_rates(confusion_map, total_ref_count)
        easy_mispronounce = print_reverse_error_analysis(mispronounce_rates, args.top_errors)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total unique phonemes: {len(all_phonemes)}")
    print(f"Phonemes with confusions: {len(confusion_map)}")
    total_confusions = sum(sum(c.values()) for c in confusion_map.values())
    print(f"Total substitution errors: {total_confusions}")

    # Save to JSON if output path is provided
    if args.output:
        save_confusion_to_json(confusion_map, total_ref_count, Path(args.output), args.top_k)
        print(f"Total unique phonemes: {len(all_phonemes)}")


if __name__ == "__main__":
    main()
