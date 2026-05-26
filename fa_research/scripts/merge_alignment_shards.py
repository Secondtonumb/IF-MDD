#!/usr/bin/env python3
"""Merge per-dataset/model alignment shards into one JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="Only merge these dataset shard names.")
    parser.add_argument("--models", nargs="+", default=None, help="Only merge these model shard names.")
    parser.add_argument("--list", action="store_true", help="List available dataset/model shards and exit.")
    return parser.parse_args()


def discover_shards(shards_dir: Path) -> list[tuple[str, str, Path]]:
    shards = []
    for shard_path in sorted(shards_dir.glob("*/*/alignments.jsonl")):
        model = shard_path.parent.name
        dataset = shard_path.parent.parent.name
        shards.append((dataset, model, shard_path))
    return shards


def filter_shards(
    shards: list[tuple[str, str, Path]],
    datasets: list[str] | None,
    models: list[str] | None,
) -> list[tuple[str, str, Path]]:
    dataset_filter = set(datasets or [])
    model_filter = set(models or [])
    return [
        (dataset, model, path)
        for dataset, model, path in shards
        if (not dataset_filter or dataset in dataset_filter)
        and (not model_filter or model in model_filter)
    ]


def print_available_shards(shards: list[tuple[str, str, Path]]) -> None:
    by_dataset: dict[str, list[str]] = {}
    for dataset, model, _ in shards:
        by_dataset.setdefault(dataset, []).append(model)
    for dataset in sorted(by_dataset):
        print(dataset)
        for model in sorted(set(by_dataset[dataset])):
            print(f"  {model}")


def main():
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    output = Path(args.output)

    shards = discover_shards(shards_dir)
    if not shards:
        raise SystemExit(f"No alignment shards found under {shards_dir}")
    if args.list:
        print_available_shards(shards)
        return

    selected_shards = filter_shards(shards, args.datasets, args.models)
    if not selected_shards:
        raise SystemExit(
            "No alignment shards matched the requested filters. "
            "Use --list to inspect available dataset/model names."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as out:
        for dataset, model, shard_path in selected_shards:
            count = 0
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line)
                    count += 1
            total += count
            print(f"{dataset}/{model}: {count} ({shard_path})")

    print(f"Merged {len(selected_shards)} shards, {total} alignments -> {output}")


if __name__ == "__main__":
    main()
