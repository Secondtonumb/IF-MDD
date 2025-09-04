#!/usr/bin/env python3
"""Convert SpeechBrain TIMIT recipe JSON (e.g. valid_timit.json) into
MDD train-dev.json format.

Source format example (valid_timit.json):
{
  "fdrw0_sx113": {
    "wav": "/common/db/TIMIT/timit/test/dr6/fdrw0/sx113.wav",
    "duration": 4.0704375,
    "spk_id": "fdrw0",
    "phn": "sil ey m ah s ...",
    "wrd": "a muscular abdomen is good for your back",
    "ground_truth_phn_ends": "2120 3592 ..."
  },
  ...
}

Target format (train-dev.json): key becomes wav path and phoneme string is
replicated into canonical_aligned / perceived_aligned / perceived_train_target.
{
  "/common/db/TIMIT/timit/test/dr6/fdrw0/sx113.wav": {
    "wav": "/common/db/TIMIT/timit/test/dr6/fdrw0/sx113.wav",
    "duration": 4.0704375,
    "spk_id": "fdrw0",
    "canonical_aligned": "sil ey m ah s ...",
    "perceived_aligned": "sil ey m ah s ...",
    "perceived_train_target": "sil ey m ah s ...",
    "wrd": "a muscular abdomen is good for your back"
  },
  ...
}

Usage:
  python convert_timit_json.py --input valid_timit.json --output train-dev.json

Optional:
  --limit N   (only convert first N utterances)
  --pretty    (pretty-print JSON with indentation)

Exit codes:
  0 success
  1 failure
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

def convert(src_path: Path, dst_path: Path, limit: int | None = None, pretty: bool = False) -> None:
    with src_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON root must be an object/dict")

    out: "OrderedDict[str, dict]" = OrderedDict()
    count = 0
    for utt_id, rec in data.items():
        if limit is not None and count >= limit:
            break
        if not isinstance(rec, dict):
            print(f"Skip {utt_id}: value not object", file=sys.stderr)
            continue
        wav = rec.get("wav")
        phn = rec.get("phn")
        if wav is None or phn is None:
            print(f"Skip {utt_id}: missing wav or phn", file=sys.stderr)
            continue
        if wav in out:
            print(f"Warning: duplicate wav path {wav}, overwriting previous entry", file=sys.stderr)
        # Build target record
        out[wav] = {
            "wav": wav,
            "duration": rec.get("duration"),
            "spk_id": rec.get("spk_id"),
            "canonical_aligned": phn,
            "perceived_aligned": phn,
            "perceived_train_target": phn,
            "wrd": rec.get("wrd"),
        }
        count += 1

    # Write output
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open('w', encoding='utf-8') as f:
        if pretty:
            json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=False)
            f.write('\n')
        else:
            json.dump(out, f, ensure_ascii=False, separators=(',', ':'), sort_keys=False)
    print(f"Converted {count} utterances -> {dst_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert TIMIT JSON to MDD train-dev format")
    p.add_argument('-i', '--input', required=True, type=Path, help='Input valid_timit.json')
    p.add_argument('-o', '--output', required=True, type=Path, help='Output train-dev.json path')
    p.add_argument('--limit', type=int, default=None, help='Only convert first N entries (for debugging)')
    p.add_argument('--pretty', action='store_true', help='Pretty-print output JSON')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        convert(args.input, args.output, args.limit, args.pretty)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
