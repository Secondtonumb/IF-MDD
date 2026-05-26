#!/usr/bin/env python3
"""Decode context-phone token sequences back to monophone sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.context_phone_codec import decode_context_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "mono",
            "diphone",
            "triphone",
            "between_word_triphone",
            "between-word-triphone",
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
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--text", default=None, help="Whitespace-separated context-phone tokens.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens:
        tokens = args.tokens
    elif args.text:
        tokens = args.text.split()
    else:
        tokens = sys.stdin.read().strip().split()
    print(" ".join(decode_context_tokens(tokens, args.mode)))


if __name__ == "__main__":
    main()
