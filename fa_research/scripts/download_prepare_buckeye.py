#!/usr/bin/env python3
"""
Download alexwengg/buckeye from Hugging Face and convert it to a text.json-like
manifest used in this repo.

The Buckeye HF dataset provides word-level timestamps, not phone-level labels.
For compatibility, this script writes canonical_aligned/perceived_aligned as
word-token sequences and also stores explicit word_aligned/word_starts/word_ends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HF_REPO = "alexwengg/buckeye"
HF_REVISION = "main"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}"


def hf_resolve_url(path: str) -> str:
    return f"{HF_BASE}/{urllib.parse.quote(path)}?download=true"


def download_file(url: str, dst: Path, retries: int = 3, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(dst)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def download_manifest(output_dir: Path, force: bool) -> dict:
    manifest_path = output_dir / "manifest.json"
    if force or not manifest_path.exists():
        print(f"Downloading manifest.json -> {manifest_path}")
        download_file(hf_resolve_url("manifest.json"), manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_word(word: str) -> str:
    return str(word).strip()


def sample_to_entry(sample: dict, wav_path: Path) -> dict:
    words = sample.get("words", [])
    word_tokens = [normalize_word(item.get("word", "")) for item in words]
    word_tokens = [word for word in word_tokens if word]
    word_aligned = " ".join(word_tokens)
    starts = [round(float(item["start_ms"]) / 1000.0, 6) for item in words]
    ends = [round(float(item["end_ms"]) / 1000.0, 6) for item in words]

    transcript = str(sample.get("transcript") or word_aligned)
    return {
        "wav": str(wav_path.resolve()),
        "duration": float(sample.get("duration_s", 0.0)),
        "spk_id": sample.get("speaker", ""),
        "utt_id": sample.get("id", wav_path.stem),
        "wrd": transcript.upper().strip() + "\n",
        "transcript": transcript,
        "word_aligned": word_aligned,
        "word_starts": starts,
        "word_ends": ends,
        "words": [
            {
                "word": normalize_word(item.get("word", "")),
                "start": round(float(item["start_ms"]) / 1000.0, 6),
                "end": round(float(item["end_ms"]) / 1000.0, 6),
                "start_ms": float(item["start_ms"]),
                "end_ms": float(item["end_ms"]),
            }
            for item in words
        ],
        # Word-level compatibility fields. These are not phone labels.
        "canonical_aligned": word_aligned,
        "perceived_aligned": word_aligned,
        "perceived_train_target": word_aligned,
    }


def download_audio_samples(samples: list[dict], output_dir: Path, workers: int, force: bool) -> list[Path]:
    audio_paths = []
    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for sample in samples:
            rel_audio = sample["audio"]
            dst = output_dir / rel_audio
            audio_paths.append(dst)
            if dst.exists() and dst.stat().st_size > 0 and not force:
                continue
            tasks.append((rel_audio, executor.submit(download_file, hf_resolve_url(rel_audio), dst)))

        total = len(tasks)
        if total == 0:
            print("All audio files already exist; skipping audio download.")
            return audio_paths

        print(f"Downloading {total} audio files with {workers} workers...")
        for index, (rel_audio, future) in enumerate(tasks, start=1):
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"audio download failed for {rel_audio}") from exc
            if index == 1 or index % 50 == 0 or index == total:
                print(f"  [{index}/{total}] {rel_audio}")
    return audio_paths


def write_text_json(samples: list[dict], audio_paths: list[Path], output_dir: Path) -> Path:
    text_json = {}
    for sample, wav_path in zip(samples, audio_paths):
        entry = sample_to_entry(sample, wav_path)
        text_json[entry["wav"]] = entry

    out_path = output_dir / "text.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(text_json, f, indent=2, ensure_ascii=False)
    return out_path


def write_metadata(manifest: dict, output_dir: Path, num_samples: int) -> None:
    metadata = {
        "source": f"https://huggingface.co/datasets/{HF_REPO}",
        "dataset": manifest.get("dataset"),
        "description": manifest.get("description"),
        "original_total_segments": manifest.get("total_segments"),
        "original_total_words": manifest.get("total_words"),
        "converted_segments": num_samples,
        "format_note": (
            "Buckeye provides word-level timestamps. canonical_aligned and "
            "perceived_aligned are word-level compatibility fields, not phones."
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/buckeye")
    parser.add_argument("--limit", type=int, default=0, help="0 means convert all samples.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-only", action="store_true", help="Do not download WAV files.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = download_manifest(output_dir, force=args.force)
    samples = list(manifest.get("samples", []))
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No samples found in manifest.json")

    if args.manifest_only:
        audio_paths = [output_dir / sample["audio"] for sample in samples]
    else:
        audio_paths = download_audio_samples(samples, output_dir, args.workers, args.force)

    out_path = write_text_json(samples, audio_paths, output_dir)
    write_metadata(manifest, output_dir, len(samples))
    print(f"Saved text.json: {out_path}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    print(f"Samples: {len(samples)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
