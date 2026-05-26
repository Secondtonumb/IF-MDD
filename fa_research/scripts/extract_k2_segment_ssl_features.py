#!/usr/bin/env python3
"""
Extract segment-level SSL features from K2 alignment records.

The input is an alignments.jsonl produced by evaluate_l2arctic_fa_timestamps.py
with --fa-backend native-k2. For each selected utterance/alignment record, this
script cuts each phone segment first, then runs the SSL encoder on that segment
alone and mean-pools over the encoder time axis.

Outputs:
  - one .npy per SSL model: (num_records, max_segments, feature_dim)
  - segment_ssl_features.h5, with /features/{ssl_model}
  - segment_mask.npy:       (num_records, max_segments)
  - metadata.json: selected rows, phones, timestamps, model names
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_DIR = REPO_ROOT / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from AutoSSLoader import AutoSSLLoader  # noqa: E402


SAMPLE_RATE = 16000
DEFAULT_SSL_MODELS = ["wav2vec2_large", "hubert_large", "wavlm_large"]
CONSONANTS = {
    "b",
    "ch",
    "d",
    "dh",
    "f",
    "g",
    "hh",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "v",
    "w",
    "y",
    "z",
    "zh",
}


def read_alignment_records(
    alignments_jsonl: Path,
    model: str,
    alignment_type: str,
    limit: int,
) -> list[dict]:
    records = []
    with alignments_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if model != "all" and record.get("model") != model:
                continue
            if alignment_type != "all" and record.get("alignment_type") != alignment_type:
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(
            f"No records matched model={model!r}, alignment_type={alignment_type!r} "
            f"in {alignments_jsonl}"
        )
    return records


def record_key(record: dict) -> tuple[str, str]:
    return str(record.get("model", "")), str(record.get("utt", ""))


def load_canonical_records(alignments_jsonl: Path, model: str, limit: int) -> dict[tuple[str, str], dict]:
    records = read_alignment_records(
        alignments_jsonl=alignments_jsonl,
        model=model,
        alignment_type="canonical_fa",
        limit=limit,
    )
    return {record_key(record): record for record in records}


def align_phone_sequences(canonical: list[str], observed: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment. Returns ops with canonical/observed indices."""
    n = len(canonical)
    m = len(observed)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    back: list[list[str | None]] = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if canonical[i - 1] == observed[j - 1] else 1
            candidates = [
                (dp[i - 1, j - 1] + sub_cost, "match" if sub_cost == 0 else "substitute"),
                (dp[i - 1, j] + 1, "delete"),
                (dp[i, j - 1] + 1, "insert"),
            ]
            cost, op = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost
            back[i][j] = op

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "substitute"}:
            ops.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "delete":
            ops.append((op, i - 1, None))
            i -= 1
        elif op == "insert":
            ops.append((op, None, j - 1))
            j -= 1
        else:
            raise RuntimeError(f"bad alignment backpointer at i={i}, j={j}: {op}")
    ops.reverse()
    return ops


def compute_pseudo_scores(
    records: list[dict],
    canonical_by_key: dict[tuple[str, str], dict],
    max_segments: int,
    exact_score: float,
    substitution_score: float,
    insertion_score: float,
    deletion_score: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    phone_scores = np.zeros((len(records), max_segments), dtype=np.float32)
    canonical_phone_scores = np.zeros((len(records), max_segments), dtype=np.float32)
    details = []

    for row_idx, record in enumerate(records):
        key = record_key(record)
        canonical_record = canonical_by_key.get(key)
        if canonical_record is None and record.get("alignment_type") == "canonical_fa":
            canonical_record = record
        if canonical_record is None:
            raise ValueError(f"No canonical_fa record found for model/utt={key}")

        canonical = canonical_record["phones"][:max_segments]
        observed = record["phones"][:max_segments]
        ops = align_phone_sequences(canonical, observed)
        detail_ops = []

        if record.get("alignment_type") == "canonical_fa":
            phone_scores[row_idx, : len(observed)] = exact_score
            canonical_phone_scores[row_idx, : len(canonical)] = exact_score
            ops = [("match", idx, idx) for idx in range(min(len(canonical), len(observed)))]
        else:
            for op, c_idx, o_idx in ops:
                score = exact_score if op == "match" else substitution_score
                if op == "delete":
                    score = deletion_score
                elif op == "insert":
                    score = insertion_score

                if o_idx is not None and o_idx < max_segments:
                    phone_scores[row_idx, o_idx] = score
                if c_idx is not None and c_idx < max_segments:
                    canonical_phone_scores[row_idx, c_idx] = score

        for op, c_idx, o_idx in ops:
            detail_ops.append(
                {
                    "op": op,
                    "canonical_index": c_idx,
                    "canonical_phone": canonical[c_idx] if c_idx is not None and c_idx < len(canonical) else None,
                    "observed_index": o_idx,
                    "observed_phone": observed[o_idx] if o_idx is not None and o_idx < len(observed) else None,
                }
            )

        details.append(
            {
                "utt": record.get("utt"),
                "model": record.get("model"),
                "alignment_type": record.get("alignment_type"),
                "ops": detail_ops,
            }
        )

    return phone_scores, canonical_phone_scores, details


def load_wav_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav.squeeze(0).contiguous()


def cut_segment(
    wav: torch.Tensor,
    start_sec: float,
    end_sec: float,
    phone: str,
    min_segment_ms: float,
    consonant_pad_ms: float,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(wav.numel(), int(round(end_sec * sample_rate)))
    if end <= start:
        end = min(wav.numel(), start + 1)
    segment = wav[start:end]

    target_ms = min_segment_ms
    if phone.lower() in CONSONANTS and consonant_pad_ms > 0:
        target_ms = max(target_ms, consonant_pad_ms)

    target_len = int(round(target_ms / 1000.0 * sample_rate))
    if segment.numel() < target_len:
        pad_total = target_len - segment.numel()
        left = pad_total // 2
        right = pad_total - left
        segment = torch.nn.functional.pad(segment, (left, right))

    return segment


def load_ssl_encoder(model_name: str, pretrained_model_path: str, device: str):
    encoder = AutoSSLLoader(
        model_name=model_name,
        freeze=True,
        freeze_feature_extractor=True,
        save_path=pretrained_model_path,
        output_all_hiddens=False,
        encoder_type=None,
    )
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def extract_mean_feature(encoder, segment: torch.Tensor, device: str) -> np.ndarray:
    wav = segment.to(device).unsqueeze(0)
    wav_lens = torch.tensor([1.0], device=device)
    with torch.no_grad():
        features = encoder(wav, wav_lens)
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim == 3:
        features = features[0]
    pooled = features.mean(dim=0)
    return pooled.detach().cpu().numpy().astype(np.float32)


def extract_mean_features_batch(encoder, segments: list[torch.Tensor], device: str) -> np.ndarray:
    if not segments:
        raise ValueError("empty segment batch")

    lengths = torch.tensor([segment.numel() for segment in segments], dtype=torch.float32, device=device)
    max_len = int(lengths.max().item())
    wavs = torch.zeros((len(segments), max_len), dtype=torch.float32, device=device)
    for idx, segment in enumerate(segments):
        wavs[idx, : segment.numel()] = segment.to(device)

    wav_lens = lengths / float(max_len)
    with torch.no_grad():
        features = encoder(wavs, wav_lens)
    if isinstance(features, (tuple, list)):
        features = features[-1]

    pooled = []
    for idx in range(features.shape[0]):
        valid_frames = max(1, int(round(float(wav_lens[idx].item()) * features.shape[1])))
        pooled.append(features[idx, :valid_frames].mean(dim=0))
    return torch.stack(pooled, dim=0).detach().cpu().numpy().astype(np.float32)


def infer_feature_dim(encoder, device: str) -> int:
    probe = torch.zeros(int(0.4 * SAMPLE_RATE), dtype=torch.float32)
    return int(extract_mean_feature(encoder, probe, device).shape[0])


def build_metadata(records: list[dict], max_segments: int, ssl_models: list[str]) -> dict:
    rows = []
    for record in records:
        num_segments = min(len(record["phones"]), max_segments)
        rows.append(
            {
                "utt": record.get("utt"),
                "wav": record.get("wav"),
                "model": record.get("model"),
                "alignment_type": record.get("alignment_type"),
                "num_segments": num_segments,
                "truncated": len(record["phones"]) > max_segments,
                "phones": record["phones"][:max_segments],
                "pred_starts": record["pred_starts"][:max_segments],
                "pred_ends": record["pred_ends"][:max_segments],
            }
        )
    return {
        "shape": {
            "num_rows": len(records),
            "max_segments": max_segments,
            "layout": "features[record_index, segment_index, feature_dim]",
        },
        "ssl_models": ssl_models,
        "rows": rows,
    }


def write_h5_static_metadata(
    h5_path: Path,
    records: list[dict],
    segment_mask: np.ndarray,
    phone_scores: np.ndarray,
    canonical_phone_scores: np.ndarray,
    metadata: dict,
    args,
    group_name: str | None = None,
):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "H5 output requested, but h5py is not installed in this environment. "
            "Please install h5py or run with --no-h5."
        ) from exc

    string_dtype = h5py.string_dtype(encoding="utf-8")
    phones = np.full((len(records), args.max_segments), "", dtype=object)
    pred_starts = np.zeros((len(records), args.max_segments), dtype=np.float32)
    pred_ends = np.zeros((len(records), args.max_segments), dtype=np.float32)
    utts = np.empty((len(records),), dtype=object)
    wavs = np.empty((len(records),), dtype=object)
    alignment_models = np.empty((len(records),), dtype=object)
    alignment_types = np.empty((len(records),), dtype=object)
    num_segments = np.zeros((len(records),), dtype=np.int32)

    for row_idx, record in enumerate(records):
        n = min(len(record["phones"]), args.max_segments)
        phones[row_idx, :n] = record["phones"][:n]
        pred_starts[row_idx, :n] = np.asarray(record["pred_starts"][:n], dtype=np.float32)
        pred_ends[row_idx, :n] = np.asarray(record["pred_ends"][:n], dtype=np.float32)
        utts[row_idx] = record.get("utt", "")
        wavs[row_idx] = record.get("wav", "")
        alignment_models[row_idx] = record.get("model", "")
        alignment_types[row_idx] = record.get("alignment_type", "")
        num_segments[row_idx] = n

    with h5py.File(h5_path, "a") as h5:
        root = h5
        if group_name:
            if group_name in h5:
                del h5[group_name]
            root = h5.create_group(group_name)

        root.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        root.attrs["layout"] = "features/{ssl_model}[record_index, segment_index, feature_dim]"
        root.attrs["sample_rate"] = SAMPLE_RATE
        root.attrs["max_segments"] = args.max_segments
        root.attrs["consonant_pad_ms"] = args.consonant_pad_ms
        root.attrs["min_segment_ms"] = args.min_segment_ms
        root.create_group("features")
        root.create_dataset("segment_mask", data=segment_mask.astype(np.bool_))
        root.create_dataset("phone_scores", data=phone_scores.astype(np.float32))
        root.create_dataset("canonical_phone_scores", data=canonical_phone_scores.astype(np.float32))
        root.create_dataset("phones", data=phones, dtype=string_dtype)
        root.create_dataset("pred_starts", data=pred_starts)
        root.create_dataset("pred_ends", data=pred_ends)
        root.create_dataset("num_segments", data=num_segments)
        root.create_dataset("utt", data=utts, dtype=string_dtype)
        root.create_dataset("wav", data=wavs, dtype=string_dtype)
        root.create_dataset("alignment_model", data=alignment_models, dtype=string_dtype)
        root.create_dataset("alignment_type", data=alignment_types, dtype=string_dtype)


def write_h5_feature_dataset(
    h5_path: Path,
    ssl_model_name: str,
    features: np.ndarray,
    compression: str | None,
    group_name: str | None = None,
):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "H5 output requested, but h5py is not installed in this environment. "
            "Please install h5py or run with --no-h5."
        ) from exc

    with h5py.File(h5_path, "a") as h5:
        root = h5[group_name] if group_name else h5
        group = root.require_group("features")
        if ssl_model_name in group:
            del group[ssl_model_name]
        kwargs = {}
        if compression != "none":
            kwargs["compression"] = compression
        dataset = group.create_dataset(ssl_model_name, data=features, **kwargs)
        dataset.attrs["layout"] = "record_index, segment_index, feature_dim"


def extract_for_model(
    records: list[dict],
    ssl_model_name: str,
    args,
    wav_cache: dict[str, torch.Tensor],
) -> np.ndarray:
    print(f"Loading SSL model: {ssl_model_name}")
    encoder = load_ssl_encoder(ssl_model_name, args.pretrained_model_path, args.device)
    feature_dim = infer_feature_dim(encoder, args.device)
    features = np.zeros((len(records), args.max_segments, feature_dim), dtype=np.float32)

    pending_segments: list[torch.Tensor] = []
    pending_indices: list[tuple[int, int]] = []

    def flush_batch():
        if not pending_segments:
            return
        batch_features = extract_mean_features_batch(encoder, pending_segments, args.device)
        for feat, (row_index, seg_index) in zip(batch_features, pending_indices):
            features[row_index, seg_index] = feat
        pending_segments.clear()
        pending_indices.clear()

    for row_idx, record in enumerate(records, start=0):
        wav_path = record["wav"]
        if wav_path not in wav_cache:
            wav_cache[wav_path] = load_wav_mono(wav_path, SAMPLE_RATE)
        wav = wav_cache[wav_path]

        phones = record["phones"][: args.max_segments]
        starts = record["pred_starts"][: args.max_segments]
        ends = record["pred_ends"][: args.max_segments]
        for seg_idx, (phone, start, end) in enumerate(zip(phones, starts, ends)):
            segment = cut_segment(
                wav=wav,
                start_sec=float(start),
                end_sec=float(end),
                phone=phone,
                min_segment_ms=args.min_segment_ms,
                consonant_pad_ms=args.consonant_pad_ms,
                sample_rate=SAMPLE_RATE,
            )
            pending_segments.append(segment)
            pending_indices.append((row_idx, seg_idx))
            if len(pending_segments) >= args.batch_size:
                flush_batch()

        print(f"  {row_idx + 1}/{len(records)} {record.get('utt')} segments={len(phones)}")

    flush_batch()

    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignments-jsonl",
        default="l2arctic_fa_timestamp_metrics_native_k2/alignments.jsonl",
        help="K2 alignments.jsonl from evaluate_l2arctic_fa_timestamps.py.",
    )
    parser.add_argument("--output-dir", default="fa_research/results/k2_segment_ssl_features")
    parser.add_argument("--model", default="CROTTC", help="Alignment-producing model name, or 'all'.")
    parser.add_argument(
        "--alignment-type",
        default="perceived_fa",
        choices=["canonical_fa", "perceived_fa", "inference_alignment", "all", "three"],
        help="'three' stores canonical_fa, perceived_fa, and inference_alignment separately.",
    )
    parser.add_argument("--ssl-models", nargs="+", default=DEFAULT_SSL_MODELS)
    parser.add_argument("--max-segments", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--h5-file", default="segment_ssl_features.h5")
    parser.add_argument("--h5-compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--no-h5", action="store_true")
    parser.add_argument("--no-npy", action="store_true")
    parser.add_argument("--no-scores", action="store_true")
    parser.add_argument("--score-only", action="store_true", help="Write score/mask/metadata files without SSL features.")
    parser.add_argument("--exact-score", type=float, default=2.0)
    parser.add_argument("--substitution-score", type=float, default=0.5)
    parser.add_argument("--insertion-score", type=float, default=0.0)
    parser.add_argument("--deletion-score", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--min-segment-ms",
        type=float,
        default=25.0,
        help="Zero-pad any shorter segment to this duration so SSL conv frontends can run.",
    )
    parser.add_argument("--consonant-pad-ms", type=float, default=400.0)
    parser.add_argument("--pretrained-model-path", default=str(REPO_ROOT / "pretrained_models"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_one_alignment_type(
    args,
    output_dir: Path,
    alignment_type: str,
    h5_path: Path,
    group_name: str | None,
    canonical_by_key: dict[tuple[str, str], dict],
):
    records = read_alignment_records(
        alignments_jsonl=Path(args.alignments_jsonl),
        model=args.model,
        alignment_type=alignment_type,
        limit=args.limit,
    )
    print(
        f"Selected {len(records)} records from {args.alignments_jsonl} "
        f"(model={args.model}, alignment_type={alignment_type})"
    )

    segment_mask = np.zeros((len(records), args.max_segments), dtype=np.bool_)
    for idx, record in enumerate(records):
        segment_mask[idx, : min(len(record["phones"]), args.max_segments)] = True
    prefix = f"{alignment_type}_" if group_name else ""
    np.save(output_dir / f"{prefix}segment_mask.npy", segment_mask)

    if args.no_scores:
        phone_scores = np.zeros((len(records), args.max_segments), dtype=np.float32)
        canonical_phone_scores = np.zeros((len(records), args.max_segments), dtype=np.float32)
        score_details = []
    else:
        phone_scores, canonical_phone_scores, score_details = compute_pseudo_scores(
            records=records,
            canonical_by_key=canonical_by_key,
            max_segments=args.max_segments,
            exact_score=args.exact_score,
            substitution_score=args.substitution_score,
            insertion_score=args.insertion_score,
            deletion_score=args.deletion_score,
        )
        np.save(output_dir / f"{prefix}phone_scores.npy", phone_scores)
        np.save(output_dir / f"{prefix}canonical_phone_scores.npy", canonical_phone_scores)
        with (output_dir / f"{prefix}phone_score_details.json").open("w", encoding="utf-8") as f:
            json.dump(score_details, f, indent=2, ensure_ascii=False)

    metadata = build_metadata(records, args.max_segments, args.ssl_models)
    metadata["pseudo_score_rule"] = {
        "exact": args.exact_score,
        "substitution": args.substitution_score,
        "insertion": args.insertion_score,
        "deletion": args.deletion_score,
        "phone_scores_layout": "aligned to this group's segment positions",
        "canonical_phone_scores_layout": "aligned to canonical_fa positions; deletions are visible here as 0",
    }
    with (output_dir / f"{prefix}metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if not args.no_h5:
        write_h5_static_metadata(
            h5_path,
            records,
            segment_mask,
            phone_scores,
            canonical_phone_scores,
            metadata,
            args,
            group_name=group_name,
        )
        print(f"Initialized H5 group: {group_name or '/'} in {h5_path}")

    if not args.score_only:
        wav_cache: dict[str, torch.Tensor] = {}
        for ssl_model_name in args.ssl_models:
            features = extract_for_model(records, ssl_model_name, args, wav_cache)
            if not args.no_npy:
                out_path = output_dir / f"{prefix}{ssl_model_name}.npy"
                np.save(out_path, features)
                print(f"Saved {out_path} shape={features.shape}")
            if not args.no_h5:
                write_h5_feature_dataset(h5_path, ssl_model_name, features, args.h5_compression, group_name=group_name)
                dataset_path = f"/{group_name}/features/{ssl_model_name}" if group_name else f"/features/{ssl_model_name}"
                print(f"Saved H5 dataset {dataset_path} shape={features.shape}")

    print(f"Saved segment mask: {output_dir / f'{prefix}segment_mask.npy'}")
    if not args.no_scores:
        print(f"Saved phone scores: {output_dir / f'{prefix}phone_scores.npy'}")
        print(f"Saved canonical phone scores: {output_dir / f'{prefix}canonical_phone_scores.npy'}")
    print(f"Saved metadata: {output_dir / f'{prefix}metadata.json'}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / args.h5_file
    if h5_path.exists() and not args.no_h5:
        h5_path.unlink()
    canonical_by_key = {} if args.no_scores else load_canonical_records(
        Path(args.alignments_jsonl),
        args.model,
        args.limit,
    )

    if args.alignment_type == "three":
        alignment_types = ["canonical_fa", "perceived_fa", "inference_alignment"]
        for alignment_type in alignment_types:
            run_one_alignment_type(
                args=args,
                output_dir=output_dir,
                alignment_type=alignment_type,
                h5_path=h5_path,
                group_name=alignment_type,
                canonical_by_key=canonical_by_key,
            )
    else:
        run_one_alignment_type(
            args=args,
            output_dir=output_dir,
            alignment_type=args.alignment_type,
            h5_path=h5_path,
            group_name=None,
            canonical_by_key=canonical_by_key,
        )

    if not args.no_h5:
        print(f"Saved H5 file: {h5_path}")


if __name__ == "__main__":
    main()
