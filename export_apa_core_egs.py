#!/usr/bin/env python3
"""
Export APA branch core datasets into ESPnet-style egs folders.

Output structure (default: ./egs_APA_core):
  egs_APA_core/
    l2_arctic/
      data/{train,dev,test}/{wav.scp,utt2spk,spk2utt,text,text.canonical,text.perceived,text.wrd}
      label_encoder.txt
      phones.txt
    so762/
      data/{train,dev,test}/...
      label_encoder.txt
      phones.txt
    iqra_eval/
      data/eval/...
      label_encoder.txt
      phones.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_SPECS = {
    "l2_arctic": {
        "train": "/home/m64000/work/IF-MDD/data/train-train.json",
        "dev": "/home/m64000/work/IF-MDD/data/train-dev.json",
        "test": "/home/m64000/work/IF-MDD/data/test.json",
    },
    "so762": {
        "train": "/home/m64000/work/dataset/speechocean762_orig_spk_open/train-train.json",
        "dev": "/home/m64000/work/dataset/speechocean762_orig_spk_open/train-dev.json",
        "test": "/home/m64000/work/dataset/speechocean762_orig_spk_open/test.json",
    },
    "iqra_eval": {
        "train": "/home/m64000/work/dataset/data_iqra/iqra_train.json",
        "dev": "/home/m64000/work/dataset/data_iqra/iqra_dev.json",
        "test": "/home/m64000/work/dataset/data_iqra/iqra_test_with_cano.json",
    },
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def make_utt_id(spk_id: str, wav_path: str, fallback_key: str, used: set[str]) -> str:
    base = Path(wav_path).stem if wav_path else Path(fallback_key).stem
    spk = (spk_id or "unk").strip() or "unk"
    if base.startswith(spk + "_"):
        utt = base
    else:
        utt = f"{spk}_{base}"

    if utt not in used:
        used.add(utt)
        return utt

    idx = 2
    while True:
        cand = f"{utt}_dup{idx}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


def choose_train_target(entry: Dict) -> str:
    for k in ["perceived_train_target", "perceived_aligned", "canonical_aligned", "wrd"]:
        value = normalize_text(str(entry.get(k, "")))
        if value:
            return value
    return "<empty>"


def write_kv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, val in rows:
            f.write(f"{key} {val}\n")


def write_label_encoder(dataset_dir: Path, phones: List[str]) -> None:
    phones_sorted = sorted(set(p for p in phones if p))

    with (dataset_dir / "phones.txt").open("w", encoding="utf-8") as f:
        for p in phones_sorted:
            f.write(p + "\n")

    lines: List[str] = []
    for idx, p in enumerate(phones_sorted, start=1):
        lines.append(f"'{p}' => {idx}")
    lines.append("'<blank>' => 0")
    lines.append(f"'<bos>' => {len(phones_sorted) + 1}")
    lines.append(f"'<eos>' => {len(phones_sorted) + 2}")
    lines.append("================")
    lines.append("'starting_index' => 0")
    lines.append("'blank_label' => '<blank>'")

    with (dataset_dir / "label_encoder.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_one_dataset(name: str, split_map: Dict[str, str], out_root: Path) -> Dict[str, int]:
    dataset_dir = out_root / name
    data_dir = dataset_dir / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if data_dir.exists():
        shutil.rmtree(data_dir)

    stats: Dict[str, int] = {}
    all_phones: List[str] = []

    for split, json_path in split_map.items():
        src = Path(json_path)
        if not src.exists():
            raise FileNotFoundError(f"Missing source json for {name}/{split}: {src}")

        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        wav_rows: List[Tuple[str, str]] = []
        utt2spk_rows: List[Tuple[str, str]] = []
        text_rows: List[Tuple[str, str]] = []
        cano_rows: List[Tuple[str, str]] = []
        perc_rows: List[Tuple[str, str]] = []
        wrd_rows: List[Tuple[str, str]] = []

        spk2utts: defaultdict[str, List[str]] = defaultdict(list)
        used_utt_ids: set[str] = set()

        for key, entry in data.items():
            wav = str(entry.get("wav", "")).strip()
            if not wav:
                continue
            spk = str(entry.get("spk_id", "unk")).strip() or "unk"
            utt = make_utt_id(spk, wav, key, used_utt_ids)

            canonical = normalize_text(str(entry.get("canonical_aligned", "")))
            perceived = normalize_text(str(entry.get("perceived_aligned", "")))
            wrd = normalize_text(str(entry.get("wrd", "")))
            train_target = choose_train_target(entry)

            wav_rows.append((utt, wav))
            utt2spk_rows.append((utt, spk))
            text_rows.append((utt, train_target))
            cano_rows.append((utt, canonical if canonical else "<empty>"))
            perc_rows.append((utt, perceived if perceived else canonical if canonical else "<empty>"))
            wrd_rows.append((utt, wrd if wrd else "<empty>"))
            spk2utts[spk].append(utt)

            if canonical:
                all_phones.extend(canonical.split())
            if perceived:
                all_phones.extend(perceived.split())
            if train_target and train_target != "<empty>":
                all_phones.extend(train_target.split())

        # Sort for reproducibility
        wav_rows.sort(key=lambda x: x[0])
        utt2spk_rows.sort(key=lambda x: x[0])
        text_rows.sort(key=lambda x: x[0])
        cano_rows.sort(key=lambda x: x[0])
        perc_rows.sort(key=lambda x: x[0])
        wrd_rows.sort(key=lambda x: x[0])

        spk2utt_rows: List[Tuple[str, str]] = []
        for spk, utts in sorted(spk2utts.items(), key=lambda x: x[0]):
            spk2utt_rows.append((spk, " ".join(sorted(utts))))

        write_kv(split_dir / "wav.scp", wav_rows)
        write_kv(split_dir / "utt2spk", utt2spk_rows)
        write_kv(split_dir / "spk2utt", spk2utt_rows)
        write_kv(split_dir / "text", text_rows)
        write_kv(split_dir / "text.canonical", cano_rows)
        write_kv(split_dir / "text.perceived", perc_rows)
        write_kv(split_dir / "text.wrd", wrd_rows)

        # Keep source snapshot path for traceability
        with (split_dir / "SOURCE_JSON").open("w", encoding="utf-8") as f:
            f.write(str(src) + "\n")

        stats[split] = len(wav_rows)

    write_label_encoder(dataset_dir, all_phones)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export core APA datasets to ESPnet-style egs")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/home/m64000/work/IF-MDD/egs_APA_core"),
        help="Output root for generated egs",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Exporting APA core egs to: {args.out_root}")
    print("=" * 70)

    summary: Dict[str, Dict[str, int]] = {}
    for dataset_name, split_map in DATASET_SPECS.items():
        print(f"\n[{dataset_name}]")
        stats = export_one_dataset(dataset_name, split_map, args.out_root)
        summary[dataset_name] = stats
        for split, count in stats.items():
            print(f"  - {split}: {count} utts")

    summary_path = args.out_root / "SUMMARY.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
