#!/usr/bin/env python3
"""Build timestamped L2-Arctic train/dev/test splits without exact wav overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TIMESTAMP_FIELDS = [
    "canonical_starts",
    "canonical_ends",
    "target_starts",
    "target_ends",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def wav_key(key: str, item: dict) -> str:
    return str(item.get("wav") or key)


def by_wav(*datasets: dict) -> dict[str, dict]:
    out = {}
    for data in datasets:
        for key, item in data.items():
            out[wav_key(key, item)] = item
    return out


def timestamped_item(key: str, split_item: dict, timestamp_lookup: dict[str, dict]) -> dict:
    wav = wav_key(key, split_item)
    if wav not in timestamp_lookup:
        raise KeyError(f"Missing timestamp record for {wav}")

    ts_item = timestamp_lookup[wav]
    merged = dict(split_item)
    for field in TIMESTAMP_FIELDS:
        if field not in ts_item:
            raise KeyError(f"Missing {field} in timestamp record for {wav}")
        merged[field] = ts_item[field]

    for field in ["duration", "spk_id", "wrd", "wav"]:
        if field in ts_item:
            merged[field] = ts_item[field]

    for phone_field, start_field, end_field in [
        ("canonical_aligned", "canonical_starts", "canonical_ends"),
        ("perceived_train_target", "target_starts", "target_ends"),
    ]:
        phone_count = len(str(merged.get(phone_field, "")).split())
        start_count = len(merged.get(start_field, []))
        end_count = len(merged.get(end_field, []))
        if not (phone_count == start_count == end_count):
            raise ValueError(
                f"Length mismatch for {wav}: {phone_field}={phone_count}, "
                f"{start_field}={start_count}, {end_field}={end_count}"
            )

    return merged


def materialize_split(
    split_data: dict,
    timestamp_lookup: dict[str, dict],
    exclude_wavs: set[str] | None = None,
) -> dict:
    out = {}
    exclude_wavs = exclude_wavs or set()
    for key, item in split_data.items():
        wav = wav_key(key, item)
        if wav in exclude_wavs:
            continue
        out[key] = timestamped_item(key, item, timestamp_lookup)
    return out


def overlap_report(train: dict, dev: dict, test: dict) -> dict:
    splits = {"train-train": train, "train-dev": dev, "test": test}
    wavs = {
        name: {wav_key(key, item) for key, item in data.items()}
        for name, data in splits.items()
    }
    report = {
        "splits": {
            name: {
                "utterances": len(data),
                "unique_wav": len(wavs[name]),
            }
            for name, data in splits.items()
        },
        "overlaps": {},
    }
    names = list(splits)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            examples = sorted(wavs[left] & wavs[right])
            report["overlaps"][f"{left}_vs_{right}"] = {
                "exact_wav": len(examples),
                "examples": examples[:20],
            }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-new", type=Path, default=Path("data/train-train_new.json"))
    parser.add_argument("--dev-new", type=Path, default=Path("data/train-dev_new.json"))
    parser.add_argument("--train-times", type=Path, default=Path("data/train-train_times.json"))
    parser.add_argument("--dev-times", type=Path, default=Path("data/train-dev_times.json"))
    parser.add_argument("--test-times", type=Path, default=Path("data/test_times.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/l2arctic_nooverlap_times"))
    parser.add_argument(
        "--keep-train-overlap",
        action="store_true",
        help="Keep train entries whose wav is also in dev. Off by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_new = read_json(args.train_new)
    dev_new = read_json(args.dev_new)
    train_times = read_json(args.train_times)
    dev_times = read_json(args.dev_times)
    test_times = read_json(args.test_times)

    timestamp_lookup = by_wav(train_times, dev_times, test_times)
    dev_wavs = {wav_key(key, item) for key, item in dev_new.items()}
    exclude = set() if args.keep_train_overlap else dev_wavs

    train = materialize_split(train_new, timestamp_lookup, exclude_wavs=exclude)
    dev = materialize_split(dev_new, timestamp_lookup)
    test = materialize_split(test_times, timestamp_lookup)
    report = overlap_report(train, dev, test)
    report["source"] = {
        "train_new": str(args.train_new),
        "dev_new": str(args.dev_new),
        "train_times": str(args.train_times),
        "dev_times": str(args.dev_times),
        "test_times": str(args.test_times),
        "removed_train_wav_overlaps_with_dev": len(train_new) - len(train),
    }

    write_json(args.output_dir / "train-train.json", train)
    write_json(args.output_dir / "train-dev.json", dev)
    write_json(args.output_dir / "test.json", test)
    write_json(args.output_dir / "overlap_report.json", report)
    write_json(
        args.output_dir / "metadata.json",
        {
            "splits": report["splits"],
            "overlaps": report["overlaps"],
            "source": report["source"],
            "timestamp_fields": TIMESTAMP_FIELDS,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
