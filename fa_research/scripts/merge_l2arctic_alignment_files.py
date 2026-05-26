#!/usr/bin/env python3
"""Merge multiple L2-ARCTIC alignment JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def merge(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    total = 0
    with output.open("w", encoding="utf-8") as out_f:
        for path in inputs:
            with path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[path.as_posix()] += 1
                    total += 1
    for path in inputs:
        print(f"{path}: {counts[path.as_posix()]} records")
    print(f"merged: {total} records -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
