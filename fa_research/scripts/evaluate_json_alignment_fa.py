#!/usr/bin/env python3
"""Forced-alignment timestamp evaluation for prepared alignment JSON sets.

This evaluates the packaged LibriSpeech-small acoustic models on the 39-phone
TIMIT/Buckeye JSON alignments and writes both metrics and plots.
"""

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
import matplotlib
import numpy as np
import torch
from torchaudio.functional import forced_align

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

DEFAULT_DATASETS = {
    "l2arctic": REPO_ROOT / "data" / "test_times.json",
    "timit_dev": REPO_ROOT / "data" / "timit_alignments_39phone" / "json" / "dev.json",
    "timit_test": REPO_ROOT / "data" / "timit_alignments_39phone" / "json" / "test.json",
    "buckeye": REPO_ROOT / "data" / "buckeye_alignments_39phone" / "json" / "train_dev_test.json",
}

DEFAULT_MODELS = {
    "L2ARCTIC_CTC": REPO_ROOT / "pretrained_models" / "L2arctic_acou_model" / "CTCLoss_MLP",
    "L2ARCTIC_OTTC": REPO_ROOT / "pretrained_models" / "L2arctic_acou_model" / "l2arctic_OTTC",
    "L2ARCTIC_CRCTC": REPO_ROOT / "pretrained_models" / "L2arctic_acou_model" / "crctc",
    "L2ARCTIC_CROTTC": REPO_ROOT / "pretrained_models" / "L2arctic_acou_model" / "crottc",
    "LIBRISPEECH_CTC": REPO_ROOT / "pretrained_models" / "librispeech_small_acou_model" / "ctc",
    "LIBRISPEECH_OTTC": REPO_ROOT / "pretrained_models" / "librispeech_small_acou_model" / "ottc",
    "LIBRISPEECH_CRCTC": REPO_ROOT / "pretrained_models" / "librispeech_small_acou_model" / "crctc",
    "LIBRISPEECH_CROTTC": REPO_ROOT / "pretrained_models" / "librispeech_small_acou_model" / "crottc",
}


def load_my_encoder_asr():
    local_path = REPO_ROOT / "trainer" / "MyEncoderASR.py"
    spec = importlib.util.spec_from_file_location("MyEncoderASR", local_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyEncoderASR


def extract_phone_timeline(value: dict, l2arctic_target: str):
    if value.get("phonemes"):
        phones, starts, ends = [], [], []
        for seg in value["phonemes"]:
            phone = str(seg.get("phoneme", "")).strip().lower()
            if not phone:
                continue
            phones.append(phone)
            starts.append(float(seg["start"]))
            ends.append(float(seg["end"]))
        return phones, starts, ends, "phonemes"

    if l2arctic_target == "canonical":
        phones = value["canonical_aligned"].split()
        starts = value["canonical_starts"]
        ends = value["canonical_ends"]
        return phones, starts, ends, "canonical"

    phones = value.get("perceived_train_target", value.get("perceived_aligned", "")).split()
    starts = value["target_starts"]
    ends = value["target_ends"]
    return phones, starts, ends, "perceived"


def load_dataset(name: str, json_path: Path, limit: int = 0, l2arctic_target: str = "perceived") -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    iterator = data.items() if isinstance(data, dict) else enumerate(data)
    for key, value in iterator:
        wav = value.get("wav") or str(key)
        phones, starts, ends, target = extract_phone_timeline(value, l2arctic_target)

        if not phones:
            continue
        if not (len(phones) == len(starts) == len(ends)):
            raise ValueError(
                f"{json_path} item {key} has mismatched target lengths: "
                f"phones={len(phones)}, starts={len(starts)}, ends={len(ends)}"
            )

        utt = Path(wav).stem
        items.append(
            {
                "dataset": name,
                "target": target,
                "utt": value.get("utt", utt),
                "spk_id": value.get("spk_id", ""),
                "subset": value.get("subset", ""),
                "sex": value.get("sex", ""),
                "wav": wav,
                "duration": float(value.get("duration", 0.0) or 0.0),
                "text": value.get("wrd", ""),
                "phones": phones,
                "starts": starts,
                "ends": ends,
                "canonical_aligned": value.get("canonical_aligned", ""),
            }
        )

    items.sort(key=lambda item: (item["dataset"], item["utt"], item["wav"]))
    if limit > 0:
        items = items[:limit]
    return items


def filter_items_by_utt(items: list[dict], only_utt: list[str]) -> list[dict]:
    if not only_utt:
        return items
    requested = set(only_utt)
    return [item for item in items if item["utt"] in requested or Path(item["wav"]).stem in requested]


def load_audio(path: str, sample_rate: int):
    wav, _ = librosa.load(path, sr=sample_rate)
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


def alignment_chunks(labels: list[int], probs: list[float], blank_id: int) -> list[tuple[int, int, int, float]]:
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
        chunks.append((active_token, active_start, len(labels), float(np.mean(probs[active_start : len(labels)]))))
    return chunks


def target_prefix_coverage(chunks: list[tuple[int, int, int, float]], target_ids: list[int]) -> tuple[int, int]:
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and chunks[cursor][0] != target_id:
            cursor += 1
        if cursor >= len(chunks):
            return target_index, cursor
        cursor += 1
    return len(target_ids), cursor


def native_k2_forced_align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int):
    """Run k2 CTC forced alignment and return one label per acoustic frame."""
    try:
        import k2
    except Exception as exc:
        raise RuntimeError(
            "native-k2 alignment requested, but k2 could not be imported. "
            "Please run this on the environment where k2 is installed."
        ) from exc

    target_ids = list(target_ids)
    # This cluster's k2 build is CPU-only. Keep model inference on CUDA if
    # requested, but run the k2 graph/intersection itself on CPU.
    k2_ctc_p = ctc_p.detach().to("cpu").float().contiguous()
    k2_device = torch.device("cpu")

    supervision_segments = torch.tensor([[0, 0, k2_ctc_p.shape[-2]]], dtype=torch.int32, device=k2_device)
    dense_fsa_vec = k2.DenseFsaVec(k2_ctc_p, supervision_segments)
    num_frames = int(k2_ctc_p.shape[-2])
    graph = k2.arc_sort(k2.ctc_graph([target_ids], modified=False, device=k2_device))

    attempts = [
        (20.0, 8.0, 30, 10000, False),
        (50.0, 20.0, 30, 20000, False),
        (100.0, 40.0, 30, 50000, True),
    ]
    last_error = None
    for search_beam, output_beam, min_active, max_active, allow_partial in attempts:
        try:
            lattice = k2.intersect_dense_pruned(
                graph,
                dense_fsa_vec,
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active,
                max_active_states=max_active,
                frame_idx_name="frame_idx",
                allow_partial=allow_partial,
            )
            best_path = k2.shortest_path(lattice, use_double_scores=True)
            labels = best_path.labels.detach().long()
            frame_idx = best_path.frame_idx.detach().long()
            scores = best_path.scores.detach().float()
            valid = (labels != -1) & (frame_idx >= 0) & (frame_idx < num_frames)
            if not bool(valid.any()):
                last_error = (
                    f"empty best path with search_beam={search_beam}, "
                    f"output_beam={output_beam}, allow_partial={allow_partial}"
                )
                continue

            dense_labels = torch.full((num_frames,), int(blank_id), dtype=torch.long)
            dense_scores = torch.ones((num_frames,), dtype=torch.float32)
            dense_labels[frame_idx[valid]] = labels[valid]
            dense_scores[frame_idx[valid]] = scores[valid].exp()
            chunks = alignment_chunks(dense_labels.tolist(), dense_scores.tolist(), blank_id)
            matched_prefix, used_chunks = target_prefix_coverage(chunks, target_ids)
            if matched_prefix != len(target_ids):
                last_error = (
                    "partial best path "
                    f"matched_prefix={matched_prefix}/{len(target_ids)}, "
                    f"chunks={len(chunks)}, used_chunks={used_chunks}, "
                    f"search_beam={search_beam}, output_beam={output_beam}, "
                    f"allow_partial={allow_partial}"
                )
                continue
            return dense_labels, dense_scores
        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc!r} with search_beam={search_beam}, "
                f"output_beam={output_beam}, allow_partial={allow_partial}"
            )

    raise ValueError(
        "native-k2 returned no usable best-path alignment after beam retries "
        f"(frames={num_frames}, targets={len(target_ids)}, last_error={last_error})"
    )


def align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int, backend: str):
    if backend == "native-k2":
        return native_k2_forced_align_ids(ctc_p, target_ids, blank_id)
    if backend == "torchaudio":
        return forced_align_ids(ctc_p, target_ids, blank_id)
    raise ValueError(f"Unsupported FA backend: {backend}")


def dense_path_to_target_segments(
    dense_alignment: torch.Tensor,
    frame_scores: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    duration: float,
    blank_policy: str = "drop",
):
    labels = dense_alignment.detach().cpu().tolist()
    probs = frame_scores.detach().cpu().tolist()
    chunks = alignment_chunks(labels, probs, blank_id)

    segments = []
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and chunks[cursor][0] != target_id:
            cursor += 1
        if cursor >= len(chunks):
            raise ValueError(
                f"could not recover segment {target_index}/{len(target_ids)} "
                f"(target_id={target_id}) from {len(chunks)} chunks"
            )
        token_id, start_frame, end_frame, mean_score = chunks[cursor]
        segments.append(
            {
                "token_id": int(token_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "mean_score": mean_score,
            }
        )
        cursor += 1

    apply_blank_policy_to_segments(segments, len(labels), blank_policy)

    seconds_per_frame = duration / float(len(labels))
    for segment in segments:
        segment["start"] = segment["start_frame"] * seconds_per_frame
        segment["end"] = segment["end_frame"] * seconds_per_frame
    return segments


def apply_blank_policy_to_segments(segments: list[dict], num_frames: int, blank_policy: str) -> None:
    if blank_policy == "drop" or not segments:
        return

    if blank_policy not in {"previous", "next", "split"}:
        raise ValueError(f"Unknown blank_policy: {blank_policy}")

    original_starts = [segment["start_frame"] for segment in segments]
    original_ends = [segment["end_frame"] for segment in segments]

    if blank_policy == "previous":
        segments[0]["start_frame"] = 0
        for idx in range(len(segments) - 1):
            segments[idx]["end_frame"] = original_starts[idx + 1]
        segments[-1]["end_frame"] = num_frames
        return

    if blank_policy == "next":
        segments[0]["start_frame"] = 0
        for idx in range(1, len(segments)):
            segments[idx]["start_frame"] = original_ends[idx - 1]
        segments[-1]["end_frame"] = num_frames
        return

    segments[0]["start_frame"] = 0
    for idx in range(len(segments) - 1):
        midpoint = int(round((original_ends[idx] + original_starts[idx + 1]) / 2.0))
        segments[idx]["end_frame"] = midpoint
        segments[idx + 1]["start_frame"] = midpoint
    segments[-1]["end_frame"] = num_frames

def collapse_repeats(values: list[int]):
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


def evaluate_model_dataset(model_name, model_path, dataset_name, items, args, tolerances, MyEncoderASR):
    print(f"\n{'=' * 80}")
    print(f"Model: {model_name} | Dataset: {dataset_name} | Utterances: {len(items)}")
    print(f"{'=' * 80}")

    run_opts = {}
    if args.device:
        run_opts["device"] = args.device
    asr_model = MyEncoderASR.from_hparams(source=str(model_path), hparams_file=args.hparams_file, run_opts=run_opts)
    blank_id = int(getattr(asr_model.hparams, "blank_index", asr_model.tokenizer.lab2ind["<blank>"]))

    acc = new_acc(tolerances)
    rows = []
    alignments = []

    for batch_index, batch_items in enumerate(make_batches(items, args.batch_size), start=1):
        print(f"  batch {batch_index}: {len(batch_items)} utts")
        try:
            waveforms = [load_audio(item["wav"], args.sample_rate) for item in batch_items]
            batch, wav_lens, sample_lengths = pad_waveforms(waveforms)
            with torch.no_grad():
                ctc_batch = asr_model.encode_batch(batch, wav_lens)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc!r}"
            print(f"    batch failed before FA: {error}")
            acc["failures"] += len(batch_items)
            for item in batch_items:
                rows.append({"dataset": dataset_name, "model": model_name, "utt": item["utt"], "wav": item["wav"], "error": error})
            continue

        max_frames = ctc_batch.shape[1]
        for item_index, item in enumerate(batch_items):
            frame_len = None
            target_ids = None
            try:
                phones = item["phones"]
                starts = item["starts"]
                ends = item["ends"]
                target_ids, kept_phones, dropped = encode_phones(asr_model.tokenizer, phones, args.unk_policy)
                if dropped and args.unk_policy == "drop":
                    starts = [s for s, p in zip(starts, phones) if p in asr_model.tokenizer.lab2ind]
                    ends = [e for e, p in zip(ends, phones) if p in asr_model.tokenizer.lab2ind]
                if not target_ids:
                    raise ValueError("no target phones after phone encoding")

                rel_len = float(wav_lens[item_index].item())
                frame_len = max(1, int(round(max_frames * rel_len)))
                ctc_p = ctc_batch[item_index : item_index + 1, :frame_len, :]
                duration = float(sample_lengths[item_index].item()) / float(args.sample_rate)

                dense_alignment, frame_scores = align_ids(ctc_p, target_ids, blank_id, args.fa_backend)
                segments = dense_path_to_target_segments(
                    dense_alignment,
                    frame_scores,
                    target_ids,
                    blank_id,
                    duration,
                    blank_policy=args.blank_policy,
                )
                metrics = timestamp_metrics(segments, starts, ends, tolerances)
                add_metrics(acc, metrics, tolerances)

                row = {
                    "dataset": dataset_name,
                    "target": item["target"],
                    "model": model_name,
                    "utt": item["utt"],
                    "spk_id": item["spk_id"],
                    "subset": item["subset"],
                    "sex": item["sex"],
                    "wav": item["wav"],
                    "duration": duration,
                    "text": item["text"],
                    "phones": " ".join(kept_phones),
                    "dropped_phones": " ".join(dropped),
                    "prediction": greedy_decode(ctc_p[0], asr_model.tokenizer, blank_id),
                    **metrics,
                }
                rows.append(row)

                if args.save_alignments:
                    alignments.append(
                        {
                            "dataset": dataset_name,
                            "target": item["target"],
                            "model": model_name,
                            "utt": item["utt"],
                            "wav": item["wav"],
                            "phones": kept_phones,
                            "ref_starts": starts,
                            "ref_ends": ends,
                            "pred_starts": [float(seg["start"]) for seg in segments],
                            "pred_ends": [float(seg["end"]) for seg in segments],
                            "scores": [float(seg["mean_score"]) for seg in segments],
                        }
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc!r}"
                phone_context = ""
                if target_ids:
                    matched = ""
                    error_text = str(exc)
                    if "matched_prefix=" in error_text:
                        matched = error_text.split("matched_prefix=", 1)[1].split("/", 1)[0]
                    elif "segment " in error_text:
                        matched = error_text.split("segment ", 1)[1].split("/", 1)[0]
                    if matched.isdigit():
                        center = int(matched)
                        left = max(0, center - 5)
                        right = min(len(kept_phones), center + 6)
                        phone_context = f" phones[{left}:{right}]={' '.join(kept_phones[left:right])}"
                print(
                    f"    failed {Path(item['wav']).name}: {error} "
                    f"(backend={args.fa_backend}, frames={frame_len if frame_len is not None else 'NA'}, "
                    f"targets={len(target_ids) if target_ids is not None else 'NA'}){phone_context}"
                )
                acc["failures"] += 1
                rows.append({"dataset": dataset_name, "model": model_name, "utt": item["utt"], "wav": item["wav"], "error": error})
                if args.fail_fast:
                    raise

    return finish_acc(acc, tolerances), rows, alignments


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict):
    rows = []
    for dataset, model_map in summary.items():
        for model, metrics in model_map.items():
            row = {"dataset": dataset, "model": model}
            row.update(metrics)
            rows.append(row)
    write_csv(path, rows)


def write_json(path: Path, item) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_jsonl_files(paths: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        count += 1
    return count


def plot_summary(summary: dict, output_dir: Path, tolerances: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(summary.keys())
    models = list(next(iter(summary.values())).keys()) if summary else []
    colors = {
        "L2ARCTIC_CTC": "#4C78A8",
        "L2ARCTIC_OTTC": "#F58518",
        "L2ARCTIC_CRCTC": "#54A24B",
        "L2ARCTIC_CROTTC": "#B279A2",
        "LIBRISPEECH_CTC": "#72B7B2",
        "LIBRISPEECH_OTTC": "#E45756",
        "LIBRISPEECH_CRCTC": "#EECA3B",
        "LIBRISPEECH_CROTTC": "#9D755D",
    }

    if not datasets or not models:
        return

    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for idx, model in enumerate(models):
        values = [summary[dataset].get(model, {}).get("boundary_mae", np.nan) * 1000 for dataset in datasets]
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, values, width, label=model, color=colors.get(model))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Boundary MAE (ms)")
    ax.set_title("Forced-alignment boundary error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_mae_by_dataset_model.png", dpi=220)
    fig.savefig(output_dir / "boundary_mae_by_dataset_model.svg")
    plt.close(fig)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        xs = [int(round(tol * 1000)) for tol in tolerances]
        for model in models:
            ys = [summary[dataset].get(model, {}).get(f"boundary_within_{ms}ms", np.nan) * 100 for ms in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=model, color=colors.get(model))
        ax.set_xlabel("Tolerance (ms)")
        ax.set_ylabel("Boundary hit rate (%)")
        ax.set_title(f"{dataset}: boundary tolerance curve")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_boundary_tolerance_curve.png", dpi=220)
        fig.savefig(output_dir / f"{dataset}_boundary_tolerance_curve.svg")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.6))
    final_ms = int(round(max(tolerances) * 1000))
    for idx, model in enumerate(models):
        values = [summary[dataset].get(model, {}).get(f"segment_within_{final_ms}ms", np.nan) * 100 for dataset in datasets]
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, values, width, label=model, color=colors.get(model))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"Segments within {final_ms} ms (%)")
    ax.set_title("Whole-segment tolerance hit rate")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"segment_within_{final_ms}ms.png", dpi=220)
    fig.savefig(output_dir / f"segment_within_{final_ms}ms.svg")
    plt.close(fig)


def parse_named_paths(values: list[str], defaults: dict[str, Path]) -> dict[str, Path]:
    if not values:
        return defaults
    selected = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            selected[name] = Path(path)
        else:
            if value not in defaults:
                raise ValueError(f"unknown key {value}; choose one of {sorted(defaults)} or pass name=/path")
            selected[value] = defaults[value]
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS.keys()))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "fa_research" / "results" / "cross_domain_json_fa"))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "fa_research" / "figures" / "cross_domain_json_fa"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--hparams-file", default="inference.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--unk-policy", choices=["error", "drop", "sil"], default="error")
    parser.add_argument("--l2arctic-target", choices=["perceived", "canonical"], default="perceived")
    parser.add_argument("--only-utt", nargs="+", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fa-backend", choices=["native-k2", "torchaudio"], default="native-k2")
    parser.add_argument("--blank-policy", choices=["drop", "previous", "next", "split"], default=None)
    parser.add_argument("--tolerances", nargs="+", type=float, default=[0.01, 0.02, 0.025, 0.03, 0.04, 0.05])
    parser.add_argument("--save-alignments", action="store_true")
    parser.add_argument("--alignment-shards-dir", default="alignment_shards")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = parse_named_paths(args.datasets, DEFAULT_DATASETS)
    models = parse_named_paths(args.models, DEFAULT_MODELS)
    tolerances = [float(t) for t in args.tolerances]
    if args.blank_policy is None:
        args.blank_policy = "previous" if args.fa_backend == "native-k2" else "drop"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)

    print("Datasets:")
    dataset_items = {}
    for name, path in datasets.items():
        items = load_dataset(name, path, args.limit, l2arctic_target=args.l2arctic_target)
        items = filter_items_by_utt(items, args.only_utt)
        if not items:
            raise SystemExit(f"No usable items found in {path}")
        dataset_items[name] = items
        print(f"  {name}: {len(items)} utts from {path}")

    print("Models:")
    for name, path in models.items():
        print(f"  {name}: {path}")
    print(f"FA backend: {args.fa_backend}, blank_policy: {args.blank_policy}")

    MyEncoderASR = load_my_encoder_asr()
    summary = {dataset: {} for dataset in datasets}
    all_rows = []
    alignment_shards = []

    for dataset_name, items in dataset_items.items():
        for model_name, model_path in models.items():
            model_summary, rows, alignments = evaluate_model_dataset(
                model_name, model_path, dataset_name, items, args, tolerances, MyEncoderASR
            )
            summary[dataset_name][model_name] = model_summary
            all_rows.extend(rows)
            if args.save_alignments:
                shard_dir = output_dir / args.alignment_shards_dir / dataset_name / model_name
                alignments_path = shard_dir / "alignments.jsonl"
                write_jsonl(alignments_path, alignments)
                write_csv(shard_dir / "per_utterance.csv", rows)
                write_json(shard_dir / "summary.json", model_summary)
                alignment_shards.append(alignments_path)
                print(f"Saved alignment shard: {alignments_path}")

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    per_utt_csv = output_dir / "per_utterance.csv"
    write_json(summary_json, summary)
    write_summary_csv(summary_csv, summary)
    write_csv(per_utt_csv, all_rows)

    if args.save_alignments:
        alignments_path = output_dir / "alignments.jsonl"
        merged_count = merge_jsonl_files(alignment_shards, alignments_path)
        print(f"Merged {merged_count} alignments: {alignments_path}")

    if not args.skip_plots:
        plot_summary(summary, figures_dir, tolerances)
        print(f"Saved figures: {figures_dir}")

    print(f"Saved summary: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved per-utterance CSV: {per_utt_csv}")


if __name__ == "__main__":
    main()
