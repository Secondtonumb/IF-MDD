#!/usr/bin/env python3
"""Batch inference + forced-alignment timestamp evaluation for TIMIT."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from torchaudio.functional import forced_align

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000

speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))


MODELS_CONFIG = {
    "CTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/CTCLoss_MLP",
    "OTTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/l2arctic_OTTC",
    "CRCTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/crctc",
    "CROTTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/crottc",
}


# Conservative TIMIT-to-IF-MDD phone normalization. Closure/noise labels are
# mapped to sil so FA can keep the full annotated timeline.
TIMIT_PHONE_MAP = {
    "h#": "sil",
    "pau": "sil",
    "epi": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "kcl": "sil",
    "pcl": "sil",
    "tcl": "sil",
    "q": "sil",
    "ax": "ah",
    "ax-h": "ah",
    "axr": "er",
    "ix": "ih",
    "ux": "uw",
    "hv": "hh",
    "el": "l",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "nx": "n",
    "zh": "sh",
}


def load_my_encoder_asr():
    local_path = REPO_ROOT / "trainer" / "MyEncoderASR.py"
    spec = importlib.util.spec_from_file_location("MyEncoderASR", local_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyEncoderASR


def normalize_phone(phone: str) -> str:
    phone = phone.strip().lower()
    return TIMIT_PHONE_MAP.get(phone, phone)


def read_timit_phn(path: Path, sample_rate: int = SAMPLE_RATE):
    phones = []
    starts = []
    ends = []
    raw_phones = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        start_sample, end_sample, phone = line.split()
        raw_phones.append(phone)
        phones.append(normalize_phone(phone))
        starts.append(int(start_sample) / float(sample_rate))
        ends.append(int(end_sample) / float(sample_rate))
    return phones, starts, ends, raw_phones


def read_timit_text(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split(maxsplit=2)
    return parts[2] if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() else text


def collect_items(root: Path, split: str):
    split_root = root / split if (root / split).exists() else root
    items = []
    for wav_path in sorted(split_root.rglob("*.wav")):
        phn_path = wav_path.with_suffix(".phn")
        if not phn_path.exists():
            continue
        txt_path = wav_path.with_suffix(".txt")
        items.append(
            {
                "utt": wav_path.stem,
                "speaker": wav_path.parent.name,
                "dialect": wav_path.parent.parent.name if wav_path.parent.parent else "",
                "wav": str(wav_path),
                "phn": str(phn_path),
                "txt": str(txt_path),
                "text": read_timit_text(txt_path),
            }
        )
    return items


def load_audio(path: str):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    return torch.from_numpy(wav).float()


def make_batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pad_waveforms(waveforms: list[torch.Tensor]):
    lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.float32)
    max_len = int(lengths.max().item())
    batch = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    for idx, wav in enumerate(waveforms):
        batch[idx, : wav.numel()] = wav
    wav_lens = lengths / float(max_len)
    return batch, wav_lens, lengths


def encode_phones(tokenizer, phones: Iterable[str], unk_policy: str):
    ids = []
    kept_phones = []
    dropped = []
    for phone in phones:
        if phone in tokenizer.lab2ind:
            ids.append(tokenizer.lab2ind[phone])
            kept_phones.append(phone)
        elif unk_policy == "drop":
            dropped.append(phone)
        elif unk_policy == "sil" and "sil" in tokenizer.lab2ind:
            ids.append(tokenizer.lab2ind["sil"])
            kept_phones.append("sil")
            dropped.append(phone)
        else:
            raise ValueError(f"phone not in tokenizer: {phone}")
    return ids, kept_phones, dropped


def forced_align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int):
    alignments, scores = forced_align(
        log_probs=ctc_p,
        targets=torch.tensor([target_ids], dtype=torch.int32, device=ctc_p.device),
        input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32, device=ctc_p.device),
        target_lengths=torch.tensor([len(target_ids)], dtype=torch.int32, device=ctc_p.device),
        blank=blank_id,
    )
    return alignments[0], scores[0].exp()


def dense_path_to_target_segments(
    dense_alignment: torch.Tensor,
    frame_scores: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    duration: float,
):
    labels = dense_alignment.detach().cpu().tolist()
    probs = frame_scores.detach().cpu().tolist()
    chunks = []
    active_token = None
    active_start = None

    for frame_idx, token_id in enumerate(labels):
        if token_id == blank_id:
            if active_token is not None:
                chunks.append((active_token, active_start, frame_idx, float(np.mean(probs[active_start:frame_idx]))))
                active_token = None
                active_start = None
            continue
        if active_token is None:
            active_token = token_id
            active_start = frame_idx
        elif token_id != active_token:
            chunks.append((active_token, active_start, frame_idx, float(np.mean(probs[active_start:frame_idx]))))
            active_token = token_id
            active_start = frame_idx

    if active_token is not None:
        chunks.append((active_token, active_start, len(labels), float(np.mean(probs[active_start:len(labels)]))))

    seconds_per_frame = duration / float(len(labels))
    segments = []
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and chunks[cursor][0] != target_id:
            cursor += 1
        if cursor >= len(chunks):
            raise ValueError(f"could not recover segment {target_index}/{len(target_ids)} from {len(chunks)} chunks")
        token_id, start_frame, end_frame, mean_score = chunks[cursor]
        segments.append(
            {
                "token_id": int(token_id),
                "start": start_frame * seconds_per_frame,
                "end": end_frame * seconds_per_frame,
                "mean_score": mean_score,
            }
        )
        cursor += 1
    return segments


def collapse_repeats(values: list[str]):
    out = []
    last = None
    for value in values:
        if value != last:
            out.append(value)
            last = value
    return out


def greedy_decode(ctc_log_probs: torch.Tensor, tokenizer, blank_id: int):
    ids = ctc_log_probs.argmax(dim=-1).detach().cpu().tolist()
    token_ids = [idx for idx in collapse_repeats(ids) if idx != blank_id]
    labels = [tokenizer.ind2lab.get(idx, str(idx)) for idx in token_ids]
    return " ".join(labels)


def timestamp_metrics(segments, starts, ends, tolerances):
    pred_starts = np.array([s["start"] for s in segments], dtype=np.float64)
    pred_ends = np.array([s["end"] for s in segments], dtype=np.float64)
    starts = np.array(starts, dtype=np.float64)
    ends = np.array(ends, dtype=np.float64)

    if not (len(pred_starts) == len(pred_ends) == len(starts) == len(ends)):
        raise ValueError(f"length mismatch: pred={len(pred_starts)}, ref={len(starts)}")

    start_err = pred_starts - starts
    end_err = pred_ends - ends
    boundary_err = np.concatenate([start_err, end_err])
    boundary_abs = np.abs(boundary_err)
    segment_abs_max = np.maximum(np.abs(start_err), np.abs(end_err))

    metrics = {
        "count": int(len(starts)),
        "start_mse": float(np.mean(start_err**2)),
        "end_mse": float(np.mean(end_err**2)),
        "boundary_mse": float(np.mean(boundary_err**2)),
        "start_mae": float(np.mean(np.abs(start_err))),
        "end_mae": float(np.mean(np.abs(end_err))),
        "boundary_mae": float(np.mean(boundary_abs)),
        "mean_alignment_score": float(np.mean([s["mean_score"] for s in segments])),
    }
    for tol in tolerances:
        ms = int(round(tol * 1000))
        metrics[f"boundary_within_{ms}ms"] = float(np.mean(boundary_abs <= tol))
        metrics[f"segment_within_{ms}ms"] = float(np.mean(segment_abs_max <= tol))
    return metrics


def new_acc(tolerances):
    acc = defaultdict(float)
    acc["utterances"] = 0
    acc["failures"] = 0
    acc["count"] = 0
    for tol in tolerances:
        ms = int(round(tol * 1000))
        acc[f"boundary_within_{ms}ms_hits"] = 0
        acc[f"segment_within_{ms}ms_hits"] = 0
    return acc


def add_metrics(acc, metrics, tolerances):
    n = metrics["count"]
    acc["utterances"] += 1
    acc["count"] += n
    acc["start_sq_sum"] += metrics["start_mse"] * n
    acc["end_sq_sum"] += metrics["end_mse"] * n
    acc["start_abs_sum"] += metrics["start_mae"] * n
    acc["end_abs_sum"] += metrics["end_mae"] * n
    acc["score_sum"] += metrics["mean_alignment_score"] * n
    for tol in tolerances:
        ms = int(round(tol * 1000))
        acc[f"boundary_within_{ms}ms_hits"] += metrics[f"boundary_within_{ms}ms"] * (2 * n)
        acc[f"segment_within_{ms}ms_hits"] += metrics[f"segment_within_{ms}ms"] * n


def finish_acc(acc, tolerances):
    n = int(acc["count"])
    if n == 0:
        return {"utterances": int(acc["utterances"]), "failures": int(acc["failures"]), "count": 0}
    out = {
        "utterances": int(acc["utterances"]),
        "failures": int(acc["failures"]),
        "count": n,
        "start_mse": acc["start_sq_sum"] / n,
        "end_mse": acc["end_sq_sum"] / n,
        "boundary_mse": (acc["start_sq_sum"] + acc["end_sq_sum"]) / (2 * n),
        "start_mae": acc["start_abs_sum"] / n,
        "end_mae": acc["end_abs_sum"] / n,
        "boundary_mae": (acc["start_abs_sum"] + acc["end_abs_sum"]) / (2 * n),
        "mean_alignment_score": acc["score_sum"] / n,
    }
    for tol in tolerances:
        ms = int(round(tol * 1000))
        out[f"boundary_within_{ms}ms"] = acc[f"boundary_within_{ms}ms_hits"] / (2 * n)
        out[f"segment_within_{ms}ms"] = acc[f"segment_within_{ms}ms_hits"] / n
    return out


def evaluate_model(model_name, model_path, items, args, tolerances, MyEncoderASR):
    print(f"\n{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"{'=' * 80}")
    asr_model = MyEncoderASR.from_hparams(source=model_path, hparams_file=args.hparams_file)
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    acc = new_acc(tolerances)
    rows = []

    for batch_index, batch_items in enumerate(make_batches(items, args.batch_size), start=1):
        print(f"  batch {batch_index}: {len(batch_items)} utts")
        try:
            waveforms = [load_audio(item["wav"]) for item in batch_items]
            batch, wav_lens, sample_lengths = pad_waveforms(waveforms)
            ctc_batch = asr_model.encode_batch(batch, wav_lens)
        except Exception as exc:
            print(f"    batch failed before FA: {exc}")
            acc["failures"] += len(batch_items)
            continue

        max_frames = ctc_batch.shape[1]
        for item_index, item in enumerate(batch_items):
            wav_path = item["wav"]
            try:
                phones, starts, ends, raw_phones = read_timit_phn(Path(item["phn"]), sample_rate=args.sample_rate)
                target_ids, kept_phones, dropped = encode_phones(asr_model.tokenizer, phones, args.unk_policy)
                if dropped and args.unk_policy == "drop":
                    starts = [s for s, p in zip(starts, phones) if p in asr_model.tokenizer.lab2ind]
                    ends = [e for e, p in zip(ends, phones) if p in asr_model.tokenizer.lab2ind]

                rel_len = float(wav_lens[item_index].item())
                frame_len = max(1, int(round(max_frames * rel_len)))
                ctc_p = ctc_batch[item_index : item_index + 1, :frame_len, :]
                duration = float(sample_lengths[item_index].item()) / float(args.sample_rate)

                dense_alignment, frame_scores = forced_align_ids(ctc_p, target_ids, blank_id)
                segments = dense_path_to_target_segments(dense_alignment, frame_scores, target_ids, blank_id, duration)
                metrics = timestamp_metrics(segments, starts, ends, tolerances)
                add_metrics(acc, metrics, tolerances)

                rows.append(
                    {
                        "model": model_name,
                        "utt": item["utt"],
                        "speaker": item["speaker"],
                        "dialect": item["dialect"],
                        "wav": wav_path,
                        "text": item["text"],
                        "phones": " ".join(kept_phones),
                        "raw_phones": " ".join(raw_phones),
                        "dropped_phones": " ".join(dropped),
                        "prediction": greedy_decode(ctc_p[0], asr_model.tokenizer, blank_id),
                        **metrics,
                    }
                )
            except Exception as exc:
                print(f"    failed {Path(wav_path).name}: {exc}")
                acc["failures"] += 1
                rows.append(
                    {
                        "model": model_name,
                        "utt": item["utt"],
                        "speaker": item["speaker"],
                        "dialect": item["dialect"],
                        "wav": wav_path,
                        "error": str(exc),
                    }
                )

    return finish_acc(acc, tolerances), rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timit-root", default="/home/m64000/work/dataset/TIMIT/timit")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "fa_research" / "results" / "timit_batch_fa"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--hparams-file", default="inference.yaml")
    parser.add_argument("--unk-policy", choices=["error", "drop", "sil"], default="sil")
    parser.add_argument("--tolerances", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    return parser.parse_args()


def main():
    args = parse_args()
    timit_root = Path(args.timit_root)
    if args.split == "all":
        items = collect_items(timit_root, "train") + collect_items(timit_root, "test")
    else:
        items = collect_items(timit_root, args.split)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No TIMIT wav/phn pairs found under {timit_root} split={args.split}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = [float(t) for t in args.tolerances]
    MyEncoderASR = load_my_encoder_asr()

    all_rows = []
    summary = {}
    for model_name, model_path in MODELS_CONFIG.items():
        model_summary, rows = evaluate_model(model_name, model_path, items, args, tolerances, MyEncoderASR)
        summary[model_name] = model_summary
        all_rows.extend(rows)

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_utterance.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in all_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-utterance CSV: {csv_path}")


if __name__ == "__main__":
    main()
