#!/usr/bin/env python3
"""Check JSON annotation splits for exact wav overlap and timestamp length issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wav_key(key: str, item: dict) -> str:
    return str(item.get("wav") or item.get("wav_path") or key)


def split_wavs(data: dict) -> set[str]:
    return {wav_key(key, item) for key, item in data.items()}


def length_errors(data: dict) -> list[tuple[str, str, int, int, int]]:
    errors = []
    checks = [
        ("canonical_aligned", "canonical_starts", "canonical_ends"),
        ("perceived_train_target", "target_starts", "target_ends"),
    ]
    for key, item in data.items():
        utt = wav_key(key, item)
        for phone_field, start_field, end_field in checks:
            if start_field not in item or end_field not in item:
                continue
            phones = len(str(item.get(phone_field, "")).split())
            starts = len(item.get(start_field, []))
            ends = len(item.get(end_field, []))
            if not (phones == starts == ends):
                errors.append((utt, phone_field, phones, starts, ends))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {
        "train": read_json(args.train),
        "valid": read_json(args.valid),
        "test": read_json(args.test),
    }
    wavs = {name: split_wavs(data) for name, data in splits.items()}

    failed = False
    print("Split sizes:")
    for name, data in splits.items():
        bad_lengths = length_errors(data)
        print(f"  {name}: utterances={len(data)}, unique_wav={len(wavs[name])}, length_bad={len(bad_lengths)}")
        if bad_lengths:
            failed = True
            for item in bad_lengths[: args.max_examples]:
                print(f"    length mismatch: {item}")

    for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = sorted(wavs[left] & wavs[right])
        print(f"{left} vs {right}: exact_wav_overlap={len(overlap)}")
        if overlap:
            failed = True
            for wav in overlap[: args.max_examples]:
                print(f"  {wav}")

    if failed:
        raise SystemExit("Split overlap or timestamp length check failed.")


if __name__ == "__main__":
    main()
