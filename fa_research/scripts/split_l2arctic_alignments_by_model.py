#!/usr/bin/env python3
"""Split an L2-ARCTIC alignments.jsonl into one shard per model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def split_alignments(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    counts = Counter()
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                model = str(record.get("model", "unknown"))
                if model not in handles:
                    model_dir = output_dir / safe_name(model)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    handles[model] = (model_dir / "alignments.jsonl").open("w", encoding="utf-8")
                handles[model].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[model] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for model, count in sorted(counts.items()):
        print(f"{model}: {count} records -> {output_dir / safe_name(model) / 'alignments.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_alignments(args.input, args.output_dir)


if __name__ == "__main__":
    main()
