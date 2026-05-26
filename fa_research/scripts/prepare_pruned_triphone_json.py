#!/usr/bin/env python3
"""Prepare senone-style pruned triphone JSON annotations.

Rare full triphones L~C~R are tied to shared backoff labels while preserving
the center phone C for decoding:

  L~C~R  ->  L~C~*  or  *~C~R  or  *~C~*

The mapping is learned from the training split only and then applied unchanged
to train/dev/test, so dev/test do not influence the tied label inventory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PHONE_FIELDS = ["perceived_train_target", "canonical_aligned", "perceived_aligned"]
SEP = "~"
BACKOFF = "*"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def split_triphone(token: str) -> tuple[str, str, str] | None:
    parts = str(token).split(SEP)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def left_backoff(token: str) -> str:
    left, center, _ = split_triphone(token) or (BACKOFF, token, BACKOFF)
    return f"{left}{SEP}{center}{SEP}{BACKOFF}"


def right_backoff(token: str) -> str:
    _, center, right = split_triphone(token) or (BACKOFF, token, BACKOFF)
    return f"{BACKOFF}{SEP}{center}{SEP}{right}"


def center_backoff(token: str) -> str:
    _, center, _ = split_triphone(token) or (BACKOFF, token, BACKOFF)
    return f"{BACKOFF}{SEP}{center}{SEP}{BACKOFF}"


def count_train_labels(train_data: dict) -> tuple[Counter, Counter, Counter, Counter]:
    full = Counter()
    left = Counter()
    right = Counter()
    center = Counter()
    for item in train_data.values():
        for field in PHONE_FIELDS:
            for token in str(item.get(field, "")).split():
                if split_triphone(token) is None:
                    continue
                full[token] += 1
                left[left_backoff(token)] += 1
                right[right_backoff(token)] += 1
                center[center_backoff(token)] += 1
    return full, left, right, center


def build_pruner(
    full_counts: Counter,
    left_counts: Counter,
    right_counts: Counter,
    *,
    min_full_count: int,
    min_backoff_count: int,
):
    def prune(token: str) -> str:
        if split_triphone(token) is None:
            return token
        if full_counts[token] >= min_full_count:
            return token

        left = left_backoff(token)
        right = right_backoff(token)
        left_count = left_counts[left]
        right_count = right_counts[right]

        if left_count >= min_backoff_count and left_count >= right_count:
            return left
        if right_count >= min_backoff_count:
            return right
        if left_count >= min_backoff_count:
            return left
        return center_backoff(token)

    return prune


def convert_dataset(data: dict, prune) -> tuple[dict, set[str], int]:
    converted = {}
    vocab = set()
    changed = 0
    for key, item in data.items():
        new_item = dict(item)
        for field in PHONE_FIELDS:
            if field not in item:
                continue
            old_tokens = str(item[field]).split()
            new_tokens = [prune(token) for token in old_tokens]
            changed += sum(1 for old, new in zip(old_tokens, new_tokens) if old != new)
            new_item[field] = " ".join(new_tokens)
            vocab.update(new_tokens)
        converted[key] = new_item
    return converted, vocab, changed


def write_label_encoder(path: Path, vocab: set[str], add_bos_eos: bool = True) -> int:
    labels = sorted(label for label in vocab if label not in {"<blank>", "<bos>", "<eos>"})
    lines = [f"'{label}' => {idx}" for idx, label in enumerate(labels, start=1)]
    max_index = len(labels)
    if add_bos_eos:
        bos_index = max_index + 1
        eos_index = max_index + 2
        lines.append(f"'<bos>' => {bos_index}")
        lines.append(f"'<eos>' => {eos_index}")
        max_index = eos_index
    lines.append("'<blank>' => 0")
    lines.extend(
        [
            "================",
            "'starting_index' => 0",
            "'blank_label' => '<blank>'",
        ]
    )
    if add_bos_eos:
        lines.insert(-1, "'eos_label' => '<eos>'")
        lines.insert(-1, "'bos_label' => '<bos>'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return max_index + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/context_phone_nooverlap_times/l2arctic_triphone"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/context_phone_nooverlap_times/l2arctic_triphone_pruned_min10"),
    )
    parser.add_argument("--min-full-count", type=int, default=10)
    parser.add_argument("--min-backoff-count", type=int, default=10)
    parser.add_argument("--no-bos-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_json(args.input_dir / "train-train.json")
    full_counts, left_counts, right_counts, center_counts = count_train_labels(train)
    prune = build_pruner(
        full_counts,
        left_counts,
        right_counts,
        min_full_count=args.min_full_count,
        min_backoff_count=args.min_backoff_count,
    )

    vocab = set()
    split_stats = {}
    pruning_map = {}
    for token in full_counts:
        pruned = prune(token)
        if token != pruned:
            pruning_map[token] = pruned

    for split in ["train-train", "train-dev", "test"]:
        data = read_json(args.input_dir / f"{split}.json")
        converted, split_vocab, changed = convert_dataset(data, prune)
        write_json(args.output_dir / f"{split}.json", converted)
        vocab.update(split_vocab)
        split_stats[split] = {
            "utterances": len(converted),
            "vocab": len(split_vocab),
            "changed_tokens": changed,
            "path": str(args.output_dir / f"{split}.json"),
        }

    output_neurons = write_label_encoder(
        args.output_dir / "label_encoder.txt",
        vocab,
        add_bos_eos=not args.no_bos_eos,
    )

    metadata = {
        "mode": "triphone",
        "pruning": {
            "method": "frequency_backoff_senone_style",
            "backoff_label": BACKOFF,
            "min_full_count": args.min_full_count,
            "min_backoff_count": args.min_backoff_count,
            "train_full_triphone_vocab": len(full_counts),
            "train_left_backoff_vocab": len(left_counts),
            "train_right_backoff_vocab": len(right_counts),
            "train_center_backoff_vocab": len(center_counts),
            "pruned_train_full_labels": len(pruning_map),
        },
        "phone_fields": PHONE_FIELDS,
        "vocab_size_without_specials": len(vocab),
        "add_bos_eos": not args.no_bos_eos,
        "output_neurons": output_neurons,
        "blank_index": 0,
        "label_encoder": str(args.output_dir / "label_encoder.txt"),
        "source_dir": str(args.input_dir),
        "splits": split_stats,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "pruning_map.json", pruning_map)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
