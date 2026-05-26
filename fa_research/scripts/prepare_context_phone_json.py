#!/usr/bin/env python3
"""Prepare diphone/triphone JSON annotations and label encoders."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.context_phone_codec import make_context_tokens, normalize_context_mode, word_position_name  # noqa: E402


PHONE_FIELDS = ["perceived_train_target", "canonical_aligned", "perceived_aligned"]
SILENCE_LABELS = {"", "sil", "sp", "spn"}
STRESS_RE = re.compile(r"\d+$")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, item) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, ensure_ascii=False)


def normalize_phone(phone: str) -> str:
    phone = STRESS_RE.sub("", str(phone).strip().lower())
    return "sil" if phone in SILENCE_LABELS else phone


def textgrid_path_for_item(item: dict, textgrid_root: Path | None = None) -> Path | None:
    wav = item.get("wav")
    if not wav:
        return None
    wav_path = Path(wav)
    stem = wav_path.with_suffix(".TextGrid").name
    if textgrid_root is not None:
        spk_id = item.get("spk_id")
        if spk_id:
            return textgrid_root / str(spk_id) / "annotation" / stem
        return textgrid_root / stem
    parts = list(wav_path.parts)
    if "wav" not in parts:
        return None
    parts[parts.index("wav")] = "annotation"
    return Path(*parts).with_suffix(".TextGrid")


def parse_textgrid_intervals(path: Path) -> dict[str, list[dict]]:
    """Parse simple Praat IntervalTier TextGrid files without extra deps."""
    tiers: dict[str, list[dict]] = {}
    current_tier: str | None = None
    current_interval: dict | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("name = "):
            match = re.match(r'name = "(.*)"', line)
            if match:
                current_tier = match.group(1).strip().lower()
                tiers.setdefault(current_tier, [])
        elif line.startswith("intervals [") and current_tier:
            current_interval = {}
        elif current_interval is not None and line.startswith("xmin = "):
            current_interval["xmin"] = float(line.split("=", 1)[1].strip())
        elif current_interval is not None and line.startswith("xmax = "):
            current_interval["xmax"] = float(line.split("=", 1)[1].strip())
        elif current_interval is not None and line.startswith("text = "):
            match = re.match(r'text = "(.*)"', line)
            current_interval["text"] = match.group(1).strip() if match else ""
            if current_tier:
                tiers[current_tier].append(current_interval)
            current_interval = None
    return tiers


def find_interval_tier(tiers: dict[str, list[dict]], names: list[str]) -> list[dict]:
    for name in names:
        if name in tiers:
            return tiers[name]
    for tier_name, intervals in tiers.items():
        if any(name in tier_name for name in names):
            return intervals
    return []


def word_id_for_midpoint(word_intervals: list[dict], midpoint: float) -> int | None:
    for word_idx, interval in enumerate(word_intervals):
        text = str(interval.get("text", "")).strip()
        if not text:
            continue
        if interval["xmin"] <= midpoint <= interval["xmax"]:
            return word_idx
    return None


def word_text_by_id(word_intervals: list[dict]) -> dict[int, str]:
    return {
        word_idx: str(interval.get("text", "")).strip()
        for word_idx, interval in enumerate(word_intervals)
        if str(interval.get("text", "")).strip()
    }


def split_l2arctic_phone_mark(mark: str) -> tuple[str, str]:
    mark = str(mark).strip()
    if normalize_phone(mark) == "sil":
        return "sil", "sil"
    parts = [part.strip() for part in mark.split(",")]
    canonical = normalize_phone(parts[0]) if parts else "sil"
    perceived = normalize_phone(parts[1]) if len(parts) > 1 and parts[1] else canonical
    return canonical, perceived


def textgrid_context_for_item(
    item: dict,
    textgrid_root: Path | None,
) -> dict | None:
    path = textgrid_path_for_item(item, textgrid_root)
    if path is None or not path.exists():
        return None
    tiers = parse_textgrid_intervals(path)
    word_intervals = find_interval_tier(tiers, ["words", "word"])
    phone_intervals = find_interval_tier(tiers, ["phones", "phone"])
    if not word_intervals or not phone_intervals:
        return None

    canonical_phones = []
    perceived_phones = []
    word_ids = []
    starts = []
    ends = []
    for interval in phone_intervals:
        canonical, perceived = split_l2arctic_phone_mark(interval.get("text", ""))
        canonical_phones.append(canonical)
        perceived_phones.append(perceived)
        starts.append(interval["xmin"])
        ends.append(interval["xmax"])
        if canonical == "sil" and perceived == "sil":
            word_ids.append(None)
        else:
            midpoint = (interval["xmin"] + interval["xmax"]) / 2.0
            word_ids.append(word_id_for_midpoint(word_intervals, midpoint))

    return {
        "canonical_phones": canonical_phones,
        "perceived_phones": perceived_phones,
        "word_ids": word_ids,
        "starts": starts,
        "ends": ends,
        "words": word_text_by_id(word_intervals),
    }


def project_aligned_labels(
    aligned_phones: list[str],
    aligned_labels: list[str],
    target_phones: list[str],
) -> list[str] | None:
    """Project aligned BWT labels onto a deduplicated target sequence."""
    projected = []
    start = 0
    for phone in target_phones:
        found = None
        for idx in range(start, len(aligned_phones)):
            if aligned_phones[idx] == phone:
                found = idx
                break
        if found is None:
            return None
        projected.append(aligned_labels[found])
        start = found + 1
    return projected


def project_aligned_indices(
    aligned_phones: list[str],
    target_phones: list[str],
) -> list[int] | None:
    """Project aligned phone indices onto a deduplicated target sequence."""
    projected = []
    start = 0
    for phone in target_phones:
        found = None
        for idx in range(start, len(aligned_phones)):
            if aligned_phones[idx] == phone:
                found = idx
                break
        if found is None:
            return None
        projected.append(found)
        start = found + 1
    return projected


def context_interval_records(
    labels: list[str],
    center_phones: list[str],
    source_indices: list[int],
    context: dict,
) -> list[dict]:
    """Build transparent interval metadata for word-position labels."""
    records = []
    words = context["words"]
    starts = context["starts"]
    ends = context["ends"]
    word_ids = context["word_ids"]
    for out_idx, src_idx in enumerate(source_indices):
        word_id = word_ids[src_idx] if src_idx < len(word_ids) else None
        records.append(
            {
                "index": out_idx,
                "source_index": src_idx,
                "label": labels[out_idx],
                "center_phone": center_phones[out_idx],
                "start": starts[src_idx] if src_idx < len(starts) else None,
                "end": ends[src_idx] if src_idx < len(ends) else None,
                "word_id": word_id,
                "word": words.get(word_id, "") if word_id is not None else "",
                "word_position": word_position_name(word_ids, src_idx),
            }
        )
    return records


def convert_phone_string(value: str, mode: str) -> str:
    phones = [normalize_phone(phone) for phone in str(value).split()]
    return " ".join(make_context_tokens(phones, mode))


def convert_field_with_textgrid(
    item: dict,
    field: str,
    mode: str,
    textgrid_context: dict | None,
) -> tuple[str, bool, list[dict] | None]:
    phones = [normalize_phone(phone) for phone in str(item[field]).split()]
    mode = normalize_context_mode(mode)
    textgrid_modes = {"between_word_triphone", "word_position_uniphone"}
    if mode not in textgrid_modes or textgrid_context is None:
        return " ".join(make_context_tokens(phones, mode)), False, None

    canonical_phones = textgrid_context["canonical_phones"]
    perceived_phones = textgrid_context["perceived_phones"]
    word_ids = textgrid_context["word_ids"]
    canonical_labels = make_context_tokens(canonical_phones, mode, word_ids)
    perceived_labels = make_context_tokens(perceived_phones, mode, word_ids)

    if field == "canonical_aligned" and phones == canonical_phones:
        records = context_interval_records(
            canonical_labels,
            canonical_phones,
            list(range(len(canonical_phones))),
            textgrid_context,
        )
        return " ".join(canonical_labels), True, records
    if field == "perceived_aligned" and phones == perceived_phones:
        records = context_interval_records(
            perceived_labels,
            perceived_phones,
            list(range(len(perceived_phones))),
            textgrid_context,
        )
        return " ".join(perceived_labels), True, records
    if field == "perceived_train_target":
        projected_indices = project_aligned_indices(perceived_phones, phones)
        if projected_indices is not None:
            projected = [perceived_labels[idx] for idx in projected_indices]
            records = context_interval_records(
                projected,
                phones,
                projected_indices,
                textgrid_context,
            )
            return " ".join(projected), True, records

    return " ".join(make_context_tokens(phones, mode)), False, None


def convert_dataset(
    data: dict,
    mode: str,
    textgrid_root: Path | None = None,
) -> tuple[dict, set[str], dict[str, int]]:
    converted = {}
    vocab = set()
    stats = {"textgrid_used": 0, "textgrid_fallback": 0}
    for key, item in data.items():
        new_item = dict(item)
        textgrid_context = textgrid_context_for_item(item, textgrid_root)
        used_any_textgrid = False
        for field in PHONE_FIELDS:
            if field not in item:
                continue
            new_item[field], used_textgrid, interval_records = convert_field_with_textgrid(
                item,
                field,
                mode,
                textgrid_context,
            )
            if interval_records is not None:
                suffix = (
                    "bwt_intervals"
                    if normalize_context_mode(mode) == "between_word_triphone"
                    else "word_position_intervals"
                )
                new_item[f"{field}_{suffix}"] = interval_records
            used_any_textgrid = used_any_textgrid or used_textgrid
            vocab.update(new_item[field].split())
        if normalize_context_mode(mode) in {"between_word_triphone", "word_position_uniphone"}:
            if used_any_textgrid:
                stats["textgrid_used"] += 1
            else:
                stats["textgrid_fallback"] += 1
        converted[key] = new_item
    return converted, vocab, stats


def write_label_encoder(path: Path, vocab: set[str], add_bos_eos: bool = True) -> int:
    """Write a SpeechBrain-style CTCTextEncoder label file.

    Index 0 is reserved for standalone CTC <blank>. Context labels start from
    1. Composite labels such as <blank>~sil are normal output classes, while
    standalone <blank> remains the CTC blank class.
    """
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
    parser.add_argument(
        "--mode",
        choices=[
            "diphone",
            "triphone",
            "between_word_triphone",
            "between-word-triphone",
            "between_word_uniphone",
            "between-word-uniphone",
            "bwt",
            "bwt_triphone",
            "bwt-triphone",
            "word_position_uniphone",
            "word-position-uniphone",
            "uniphone_word_position",
            "uniphone-word-position",
            "uniphone_state",
            "uniphone-state",
            "wpu",
        ],
        required=True,
    )
    parser.add_argument("--train-json", type=Path, default=REPO_ROOT / "data" / "train-train.json")
    parser.add_argument("--valid-json", type=Path, default=REPO_ROOT / "data" / "train-dev.json")
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "data" / "test.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--textgrid-root",
        type=Path,
        default=None,
        help="Optional L2-Arctic root. Defaults to inferring annotation paths from each wav path.",
    )
    parser.add_argument("--no-bos-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = normalize_context_mode(args.mode)
    splits = {
        "train-train": args.train_json,
        "train-dev": args.valid_json,
        "test": args.test_json,
    }
    vocab: set[str] = set()
    counts = {}
    textgrid_stats = {}
    for split, path in splits.items():
        data = read_json(path)
        converted, split_vocab, split_stats = convert_dataset(
            data,
            mode,
            textgrid_root=args.textgrid_root,
        )
        vocab.update(split_vocab)
        out_path = args.output_dir / f"{split}.json"
        write_json(out_path, converted)
        counts[split] = {"utterances": len(converted), "vocab": len(split_vocab), "path": str(out_path)}
        textgrid_stats[split] = split_stats

    label_encoder_path = args.output_dir / "label_encoder.txt"
    add_bos_eos = not args.no_bos_eos
    output_neurons = write_label_encoder(label_encoder_path, vocab, add_bos_eos=add_bos_eos)
    metadata = {
        "mode": mode,
        "phone_fields": PHONE_FIELDS,
        "vocab_size_without_specials": len(vocab),
        "add_bos_eos": add_bos_eos,
        "output_neurons": output_neurons,
        "blank_index": 0,
        "label_encoder": str(label_encoder_path),
        "splits": counts,
        "textgrid_stats": textgrid_stats,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
