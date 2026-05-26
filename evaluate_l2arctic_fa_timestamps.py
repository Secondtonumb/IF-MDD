#!/usr/bin/env python3
"""
Evaluate forced-alignment timestamps on L2-ARCTIC test_times.json.

For each utterance and each of the 4 L2-ARCTIC models, this script runs FA for:
  - canonical_aligned       vs canonical_starts/canonical_ends
  - perceived_train_target  vs target_starts/target_ends

It writes aggregate MSE/tolerance metrics and per-utterance metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from torchaudio.functional import forced_align

warnings.filterwarnings("ignore")

speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


SAMPLE_RATE = 16000

MODELS_CONFIG = {
    "CTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/CTCLoss_MLP",
    "OTTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/l2arctic_OTTC",
    "CRCTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/crctc",
    "CROTTC": "/home/m64000/work/IF-MDD/pretrained_models/L2arctic_acou_model/crottc",
}


ERROR_FIELDNAMES = [
    "model",
    "variant",
    "loss",
    "context_phone_mode",
    "utt",
    "wav",
    "alignment_type",
    "primary_backend",
    "fallback_backend",
    "status",
    "error_type",
    "error_message",
    "frames",
    "targets",
]


def load_my_encoder_asr():
    """Load MyEncoderASR from the local repo first, then HF as fallback."""
    local_path = Path(__file__).parent / "trainer" / "MyEncoderASR.py"
    if local_path.exists():
        path = local_path
    elif hf_hub_download is not None:
        path = Path(hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py"))
    else:
        raise RuntimeError("Cannot load MyEncoderASR: local file missing and HF fallback unavailable")

    spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyEncoderASR


def load_audio(wav_path: str) -> tuple[torch.Tensor, float]:
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    duration = len(wav) / float(SAMPLE_RATE)
    return torch.from_numpy(wav).float(), duration


def encode_phones(tokenizer, phones: Iterable[str]) -> list[int]:
    ids = []
    missing = []
    for phone in phones:
        if phone in tokenizer.lab2ind:
            ids.append(tokenizer.lab2ind[phone])
        else:
            missing.append(phone)
    if missing:
        raise ValueError(f"phones missing from tokenizer: {sorted(set(missing))}")
    return ids


def forced_align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int):
    """Run torchaudio FA and return dense label path plus per-frame probabilities."""
    target_ids = list(target_ids)
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
    blank_policy: str = "drop",
) -> list[dict]:
    """
    Recover one segment per target token from the dense CTC path.

    This intentionally does not use merge_tokens, because adjacent repeated
    phones such as "sil sil" must remain separate for timestamp comparison.
    """
    labels = dense_alignment.detach().cpu().tolist()
    probs = frame_scores.detach().cpu().tolist()
    chunks: list[tuple[int, int, int, float]] = []
    active_token = None
    active_start = None

    for frame_idx, token_id in enumerate(labels):
        if token_id == blank_id:
            if active_token is not None:
                mean_score = float(np.mean(probs[active_start:frame_idx]))
                chunks.append((active_token, active_start, frame_idx, mean_score))
                active_token = None
                active_start = None
            continue

        if active_token is None:
            active_token = token_id
            active_start = frame_idx
        elif token_id != active_token:
            mean_score = float(np.mean(probs[active_start:frame_idx]))
            chunks.append((active_token, active_start, frame_idx, mean_score))
            active_token = token_id
            active_start = frame_idx

    if active_token is not None:
        mean_score = float(np.mean(probs[active_start:len(labels)]))
        chunks.append((active_token, active_start, len(labels), mean_score))

    segments = []
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and chunks[cursor][0] != target_id:
            cursor += 1
        if cursor >= len(chunks):
            raise ValueError(
                f"could not recover target segment {target_index}/{len(target_ids)} "
                f"from {len(chunks)} nonblank chunks"
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
    """Attach blank gaps to neighboring phone segments in-place."""
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

    # split: put the boundary in the middle of each blank gap.
    segments[0]["start_frame"] = 0
    for idx in range(len(segments) - 1):
        midpoint = int(round((original_ends[idx] + original_starts[idx + 1]) / 2.0))
        segments[idx]["end_frame"] = midpoint
        segments[idx + 1]["start_frame"] = midpoint
    segments[-1]["end_frame"] = num_frames


def forced_align_phones(
    asr_model,
    ctc_p: torch.Tensor,
    phones: list[str],
    duration: float,
    blank_policy: str = "drop",
):
    target_ids = encode_phones(asr_model.tokenizer, phones)
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    dense_alignment, frame_scores = forced_align_ids(ctc_p, target_ids, blank_id)
    return dense_path_to_target_segments(
        dense_alignment=dense_alignment,
        frame_scores=frame_scores,
        target_ids=target_ids,
        blank_id=blank_id,
        duration=duration,
        blank_policy=blank_policy,
    )


def forced_align_token_ids(
    asr_model,
    ctc_p: torch.Tensor,
    token_ids: list[int],
    duration: float,
    blank_policy: str = "drop",
):
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    dense_alignment, frame_scores = forced_align_ids(ctc_p, token_ids, blank_id)
    return dense_path_to_target_segments(
        dense_alignment=dense_alignment,
        frame_scores=frame_scores,
        target_ids=token_ids,
        blank_id=blank_id,
        duration=duration,
        blank_policy=blank_policy,
    )


def native_k2_forced_align_ids(ctc_p: torch.Tensor, target_ids: list[int], blank_id: int):
    """Run native K2 CTC FA and return a dense frame-level label path."""
    try:
        import k2
    except Exception as exc:
        raise RuntimeError(
            "Native K2 alignment backend requested, but k2 could not be imported. "
            "Please run this in the environment where k2 is installed."
        ) from exc

    target_ids = list(target_ids)
    # The available k2 build may be CPU-only. Keep model inference on the
    # requested device, but run native k2 graph/intersection on CPU.
    k2_ctc_p = ctc_p.detach().to("cpu").float().contiguous()
    k2_device = torch.device("cpu")

    graph = k2.ctc_graph([target_ids], modified=False, device=k2_device)
    graph = k2.arc_sort(graph)
    supervision_segments = torch.tensor(
        [[0, 0, k2_ctc_p.shape[-2]]],
        dtype=torch.int32,
        device=k2_device,
    )
    dense_fsa_vec = k2.DenseFsaVec(k2_ctc_p, supervision_segments)
    lattice = k2.intersect_dense_pruned(
        graph,
        dense_fsa_vec,
        search_beam=20.0,
        output_beam=8.0,
        min_active_states=30,
        max_active_states=10000,
        frame_idx_name="frame_idx",
    )
    best_path = k2.shortest_path(lattice, use_double_scores=True)

    labels = best_path.labels.detach().long()
    frame_idx = best_path.frame_idx.detach().long()
    scores = best_path.scores.detach().float()

    num_frames = int(k2_ctc_p.shape[-2])
    dense_labels = torch.full((num_frames,), int(blank_id), dtype=torch.long)
    dense_scores = torch.ones((num_frames,), dtype=torch.float32)

    valid = (labels != -1) & (frame_idx >= 0) & (frame_idx < num_frames)
    if not bool(valid.any()):
        raise ValueError("native K2 returned an empty best-path alignment")

    dense_labels[frame_idx[valid]] = labels[valid]
    dense_scores[frame_idx[valid]] = scores[valid].exp()
    return dense_labels, dense_scores


def native_k2_align_token_ids(
    asr_model,
    ctc_p: torch.Tensor,
    token_ids: list[int],
    duration: float,
    blank_policy: str = "drop",
):
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    dense_alignment, frame_scores = native_k2_forced_align_ids(ctc_p, token_ids, blank_id)
    return dense_path_to_target_segments(
        dense_alignment=dense_alignment,
        frame_scores=frame_scores,
        target_ids=token_ids,
        blank_id=blank_id,
        duration=duration,
        blank_policy=blank_policy,
    )


def native_k2_align_phones(
    asr_model,
    ctc_p: torch.Tensor,
    phones: list[str],
    duration: float,
    blank_policy: str = "drop",
):
    target_ids = encode_phones(asr_model.tokenizer, phones)
    return native_k2_align_token_ids(
        asr_model=asr_model,
        ctc_p=ctc_p,
        token_ids=target_ids,
        duration=duration,
        blank_policy=blank_policy,
    )


def align_token_ids_by_backend(
    *,
    backend: str,
    asr_model,
    k2_aligner,
    wav_path: str,
    ctc_p: torch.Tensor,
    token_ids: list[int],
    duration: float,
    blank_policy: str,
) -> list[dict]:
    if backend == "native-k2":
        return native_k2_align_token_ids(
            asr_model,
            ctc_p,
            token_ids,
            duration,
            blank_policy=blank_policy,
        )
    if backend == "k2":
        return k2_align_token_ids(
            k2_aligner,
            asr_model,
            wav_path,
            token_ids,
            duration,
            blank_policy=blank_policy,
        )
    if backend == "torchaudio":
        return forced_align_token_ids(
            asr_model,
            ctc_p,
            token_ids,
            duration,
            blank_policy=blank_policy,
        )
    raise ValueError(f"Unsupported FA backend: {backend}")


def make_error_event(
    *,
    model: str,
    utt: str,
    wav: str,
    alignment_type: str,
    primary_backend: str,
    fallback_backend: str,
    status: str,
    error: Exception,
    ctc_p: torch.Tensor | None = None,
    target_count: int | None = None,
    variant: str = "",
    loss: str = "",
    context_phone_mode: str = "",
) -> dict:
    return {
        "model": model,
        "variant": variant,
        "loss": loss,
        "context_phone_mode": context_phone_mode,
        "utt": utt,
        "wav": wav,
        "alignment_type": alignment_type,
        "primary_backend": primary_backend,
        "fallback_backend": fallback_backend,
        "status": status,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "frames": int(ctc_p.shape[-2]) if ctc_p is not None else "",
        "targets": int(target_count) if target_count is not None else "",
    }


def align_with_fallback(
    *,
    asr_model,
    k2_aligner,
    wav_path: str,
    ctc_p: torch.Tensor,
    token_ids: list[int],
    duration: float,
    primary_backend: str,
    blank_policy: str,
    fallback_backend: str | None,
    disable_fallback: bool,
    error_events: list[dict],
    error_context: dict,
) -> tuple[list[dict] | None, dict]:
    try:
        segments = align_token_ids_by_backend(
            backend=primary_backend,
            asr_model=asr_model,
            k2_aligner=k2_aligner,
            wav_path=wav_path,
            ctc_p=ctc_p,
            token_ids=token_ids,
            duration=duration,
            blank_policy=blank_policy,
        )
        return segments, {"fa_backend_used": primary_backend}
    except Exception as primary_error:
        can_fallback = (
            not disable_fallback
            and fallback_backend
            and primary_backend in {"native-k2", "k2"}
            and fallback_backend != primary_backend
        )
        if can_fallback:
            try:
                segments = align_token_ids_by_backend(
                    backend=fallback_backend,
                    asr_model=asr_model,
                    k2_aligner=k2_aligner,
                    wav_path=wav_path,
                    ctc_p=ctc_p,
                    token_ids=token_ids,
                    duration=duration,
                    blank_policy=blank_policy,
                )
                error_events.append(
                    make_error_event(
                        **error_context,
                        primary_backend=primary_backend,
                        fallback_backend=fallback_backend,
                        status="primary_failed_fallback_succeeded",
                        error=primary_error,
                        ctc_p=ctc_p,
                        target_count=len(token_ids),
                    )
                )
                return segments, {
                    "fa_backend_used": fallback_backend,
                    "fa_backend_primary": primary_backend,
                    "fallback_reason": str(primary_error),
                }
            except Exception as fallback_error:
                error_events.append(
                    make_error_event(
                        **error_context,
                        primary_backend=primary_backend,
                        fallback_backend=fallback_backend,
                        status="all_failed",
                        error=fallback_error,
                        ctc_p=ctc_p,
                        target_count=len(token_ids),
                    )
                )
                return None, {}

        error_events.append(
            make_error_event(
                **error_context,
                primary_backend=primary_backend,
                fallback_backend="",
                status="all_failed",
                error=primary_error,
                ctc_p=ctc_p,
                target_count=len(token_ids),
            )
        )
        return None, {}


def error_target_name(alignment_type: str) -> str:
    if alignment_type == "canonical_fa":
        return "canonical"
    if alignment_type == "perceived_fa":
        return "perceived"
    return "inference"


def write_error_reports(error_events: list[dict], error_dir: Path) -> None:
    error_dir.mkdir(parents=True, exist_ok=True)
    events_path = error_dir / "error_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(error_events)

    summary = {}
    for event in error_events:
        target = error_target_name(event["alignment_type"])
        key = (event["model"], target)
        if key not in summary:
            summary[key] = {
                "model": event["model"],
                "target": target,
                "variant": event.get("variant", ""),
                "loss": event.get("loss", ""),
                "context_phone_mode": event.get("context_phone_mode", ""),
                "primary_failures": 0,
                "fallback_succeeded": 0,
                "fallback_failed": 0,
                "skipped_alignments": 0,
            }
        row = summary[key]
        row["primary_failures"] += 1
        if event["status"] == "primary_failed_fallback_succeeded":
            row["fallback_succeeded"] += 1
        elif event["status"] == "all_failed":
            row["fallback_failed"] += 1
            row["skipped_alignments"] += 1

    summary_rows = [summary[key] for key in sorted(summary)]
    summary_path = error_dir / "error_summary_by_model_target.csv"
    fieldnames = [
        "model",
        "target",
        "variant",
        "loss",
        "context_phone_mode",
        "primary_failures",
        "fallback_succeeded",
        "fallback_failed",
        "skipped_alignments",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_summary = {
        "total_events": len(error_events),
        "total_primary_failures": len(error_events),
        "total_fallback_succeeded": sum(
            1 for event in error_events if event["status"] == "primary_failed_fallback_succeeded"
        ),
        "total_fallback_failed": sum(1 for event in error_events if event["status"] == "all_failed"),
        "total_skipped_alignments": sum(1 for event in error_events if event["status"] == "all_failed"),
        "by_model_target": summary_rows,
    }
    with (error_dir / "error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)


def make_k2_aligner(asr_model):
    """Create the SpeechBrain/K2 CTC aligner lazily, so torchaudio runs do not need k2."""
    try:
        from speechbrain.integrations.k2_fsa.align import CTCAligner
    except Exception as exc:
        raise RuntimeError(
            "K2 alignment backend requested, but SpeechBrain's K2 CTCAligner "
            "could not be imported. Please run this in the environment where k2 "
            "is installed."
        ) from exc

    return CTCAligner(model=asr_model, tokenizer=asr_model.tokenizer, device=asr_model.device)


def normalize_k2_alignment(alignment, tokenizer=None) -> list[int]:
    """Normalize common CTCAligner outputs to a flat frame-level token-id path."""
    if isinstance(alignment, dict):
        for key in ("alignment", "alignments", "tokens", "token_ids"):
            if key in alignment:
                alignment = alignment[key]
                break

    if isinstance(alignment, torch.Tensor):
        alignment = alignment.detach().cpu().reshape(-1).tolist()
    elif isinstance(alignment, np.ndarray):
        alignment = alignment.reshape(-1).tolist()

    if isinstance(alignment, (list, tuple)) and len(alignment) == 1:
        first = alignment[0]
        if isinstance(first, (list, tuple, torch.Tensor, np.ndarray)):
            return normalize_k2_alignment(first, tokenizer=tokenizer)

    if not isinstance(alignment, (list, tuple)):
        raise ValueError(f"unsupported K2 alignment output type: {type(alignment).__name__}")

    token_ids = []
    for item in alignment:
        if isinstance(item, torch.Tensor):
            item = item.item()
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            item = item[-1]
        if isinstance(item, dict):
            for key in ("token", "token_id", "id", "label"):
                if key in item:
                    item = item[key]
                    break
        if isinstance(item, str):
            if tokenizer is None or item not in tokenizer.lab2ind:
                raise ValueError(f"K2 alignment returned unknown token label: {item}")
            item = tokenizer.lab2ind[item]
        token_ids.append(int(item))
    return token_ids


def k2_align_token_ids(
    k2_aligner,
    asr_model,
    wav_path: str,
    token_ids: list[int],
    duration: float,
    blank_policy: str,
):
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    transcript = token_ids_to_phones(asr_model.tokenizer, [t for t in token_ids if t != blank_id])
    alignment = k2_aligner.align_audio_to_tokens(audio_file=str(wav_path), transcript=transcript)
    dense_alignment = torch.tensor(normalize_k2_alignment(alignment, asr_model.tokenizer), dtype=torch.long)
    frame_scores = torch.ones_like(dense_alignment, dtype=torch.float32)
    return dense_path_to_target_segments(
        dense_alignment=dense_alignment,
        frame_scores=frame_scores,
        target_ids=token_ids,
        blank_id=blank_id,
        duration=duration,
        blank_policy=blank_policy,
    )


def k2_align_phones(
    k2_aligner,
    asr_model,
    wav_path: str,
    phones: list[str],
    duration: float,
    blank_policy: str,
):
    target_ids = encode_phones(asr_model.tokenizer, phones)
    return k2_align_token_ids(
        k2_aligner=k2_aligner,
        asr_model=asr_model,
        wav_path=wav_path,
        token_ids=target_ids,
        duration=duration,
        blank_policy=blank_policy,
    )


def collapse_repeated_ids(token_ids: list[int]) -> list[int]:
    collapsed = []
    last = None
    for token_id in token_ids:
        if token_id != last:
            collapsed.append(token_id)
            last = token_id
    return collapsed


def greedy_inference_token_ids(asr_model, ctc_p: torch.Tensor) -> list[int]:
    blank_id = asr_model.tokenizer.lab2ind["<blank>"]
    dense_ids = ctc_p[0].argmax(dim=-1).detach().cpu().tolist()
    return [token_id for token_id in collapse_repeated_ids(dense_ids) if token_id != blank_id]


def token_ids_to_phones(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.ind2lab.get(int(token_id), str(int(token_id))) for token_id in token_ids]


def build_alignment_record(
    model_name: str,
    alignment_type: str,
    wav_path: str,
    phones: list[str],
    segments: list[dict],
    ref_starts=None,
    ref_ends=None,
) -> dict:
    pred_starts = [round(float(segment["start"]), 6) for segment in segments]
    pred_ends = [round(float(segment["end"]), 6) for segment in segments]
    record = {
        "model": model_name,
        "alignment_type": alignment_type,
        "utt": Path(wav_path).stem,
        "wav": wav_path,
        "phones": phones,
        "pred_starts": pred_starts,
        "pred_ends": pred_ends,
        "start_frames": [int(segment["start_frame"]) for segment in segments],
        "end_frames": [int(segment["end_frame"]) for segment in segments],
        "mean_scores": [round(float(segment["mean_score"]), 6) for segment in segments],
    }
    if ref_starts is not None and ref_ends is not None:
        record["gt_starts"] = [round(float(x), 6) for x in ref_starts]
        record["gt_ends"] = [round(float(x), 6) for x in ref_ends]
    return record


def timestamp_metrics(segments: list[dict], ref_starts, ref_ends, tolerances: list[float]):
    pred_starts = np.array([s["start"] for s in segments], dtype=np.float64)
    pred_ends = np.array([s["end"] for s in segments], dtype=np.float64)
    ref_starts = np.array(ref_starts, dtype=np.float64)
    ref_ends = np.array(ref_ends, dtype=np.float64)

    if not (len(pred_starts) == len(pred_ends) == len(ref_starts) == len(ref_ends)):
        raise ValueError(
            "timestamp length mismatch: "
            f"pred={len(pred_starts)}/{len(pred_ends)}, ref={len(ref_starts)}/{len(ref_ends)}"
        )

    start_err = pred_starts - ref_starts
    end_err = pred_ends - ref_ends
    boundary_err = np.concatenate([start_err, end_err])
    boundary_abs = np.abs(boundary_err)
    segment_abs_max = np.maximum(np.abs(start_err), np.abs(end_err))

    metrics = {
        "count": int(len(ref_starts)),
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


def new_accumulator(tolerances: list[float]):
    acc = {
        "utterances": 0,
        "failures": 0,
        "count": 0,
        "start_sq_sum": 0.0,
        "end_sq_sum": 0.0,
        "start_abs_sum": 0.0,
        "end_abs_sum": 0.0,
        "score_sum": 0.0,
    }
    for tol in tolerances:
        ms = int(round(tol * 1000))
        acc[f"boundary_within_{ms}ms_hits"] = 0.0
        acc[f"segment_within_{ms}ms_hits"] = 0.0
    return acc


def add_to_accumulator(acc: dict, metrics: dict, tolerances: list[float]):
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


def finalize_accumulator(acc: dict, tolerances: list[float]):
    n = acc["count"]
    if n == 0:
        return {
            "utterances": acc["utterances"],
            "failures": acc["failures"],
            "count": 0,
        }

    out = {
        "utterances": acc["utterances"],
        "failures": acc["failures"],
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


def iter_targets(entry: dict):
    yield (
        "canonical",
        entry["canonical_aligned"].split(),
        entry["canonical_starts"],
        entry["canonical_ends"],
    )
    yield (
        "perceived",
        entry.get("perceived_train_target", entry["perceived_aligned"]).split(),
        entry["target_starts"],
        entry["target_ends"],
    )


def evaluate(args):
    data = json.load(open(args.test_json, "r", encoding="utf-8"))
    items = list(data.items())
    if args.limit > 0:
        items = items[: args.limit]

    tolerances = [float(t) for t in args.tolerances]
    blank_policy = args.blank_policy or ("previous" if args.fa_backend in {"k2", "native-k2"} else "drop")
    MyEncoderASR = load_my_encoder_asr()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir = Path(args.error_dir) if args.error_dir else output_dir / "alignment_errors"

    rows = []
    summary = {}
    error_events = []
    alignments_path = output_dir / "alignments.jsonl"
    fallback_backend = None if args.disable_fa_fallback else args.fallback_fa_backend
    print(f"FA backend: {args.fa_backend}, blank_policy: {blank_policy}")
    print(f"Fallback FA backend: {fallback_backend or 'disabled'}")

    with alignments_path.open("w", encoding="utf-8") as align_f:
        for model_name, model_path in MODELS_CONFIG.items():
            print(f"\n{'=' * 80}")
            print(f"Model: {model_name}")
            print(f"{'=' * 80}")

            asr_model = MyEncoderASR.from_hparams(source=model_path, hparams_file="inference.yaml")
            k2_aligner = make_k2_aligner(asr_model) if args.fa_backend == "k2" else None
            accumulators = {
                "canonical": new_accumulator(tolerances),
                "perceived": new_accumulator(tolerances),
            }

            for index, (utt_key, entry) in enumerate(items, start=1):
                wav_path = entry.get("wav", utt_key)
                if index == 1 or index % args.log_every == 0:
                    print(f"  [{index}/{len(items)}] {Path(wav_path).name}")

                try:
                    waveform, wav_duration = load_audio(wav_path)
                    duration = float(entry.get("duration") or wav_duration)
                    ctc_p = asr_model.encode_batch(waveform.unsqueeze(0), torch.tensor([1.0]))
                except Exception as exc:
                    print(f"    setup failed: {Path(wav_path).name}: {type(exc).__name__}: {exc}")
                    for target_name, _, _, _ in iter_targets(entry):
                        alignment_type = "canonical_fa" if target_name == "canonical" else "perceived_fa"
                        error_events.append(
                            make_error_event(
                                model=model_name,
                                utt=Path(wav_path).stem,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=args.fa_backend,
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                            )
                        )
                        accumulators[target_name]["failures"] += 1
                        rows.append(
                            {
                                "model": model_name,
                                "target": target_name,
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                "error": str(exc),
                            }
                        )
                    continue

                for target_name, phones, starts, ends in iter_targets(entry):
                    alignment_type = "canonical_fa" if target_name == "canonical" else "perceived_fa"
                    try:
                        target_ids = encode_phones(asr_model.tokenizer, phones)
                    except Exception as exc:
                        print(f"    encode failed {Path(wav_path).name} [{alignment_type}]: {exc}")
                        error_events.append(
                            make_error_event(
                                model=model_name,
                                utt=Path(wav_path).stem,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=args.fa_backend,
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                                target_count=len(phones),
                            )
                        )
                        accumulators[target_name]["failures"] += 1
                        rows.append(
                            {
                                "model": model_name,
                                "target": target_name,
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                "error": str(exc),
                            }
                        )
                        continue

                    segments, backend_meta = align_with_fallback(
                        asr_model=asr_model,
                        k2_aligner=k2_aligner,
                        wav_path=wav_path,
                        ctc_p=ctc_p,
                        token_ids=target_ids,
                        duration=duration,
                        primary_backend=args.fa_backend,
                        blank_policy=blank_policy,
                        fallback_backend=fallback_backend,
                        disable_fallback=args.disable_fa_fallback,
                        error_events=error_events,
                        error_context={
                            "model": model_name,
                            "utt": Path(wav_path).stem,
                            "wav": wav_path,
                            "alignment_type": alignment_type,
                        },
                    )
                    if segments is None:
                        print(f"    skipped {Path(wav_path).name} [{alignment_type}]: all FA backends failed")
                        accumulators[target_name]["failures"] += 1
                        rows.append(
                            {
                                "model": model_name,
                                "target": target_name,
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                "error": "all FA backends failed",
                            }
                        )
                        continue

                    try:
                        metrics = timestamp_metrics(segments, starts, ends, tolerances)
                        add_to_accumulator(accumulators[target_name], metrics, tolerances)

                        rows.append(
                            {
                                "model": model_name,
                                "target": target_name,
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                **metrics,
                            }
                        )

                        record = build_alignment_record(
                            model_name=model_name,
                            alignment_type=alignment_type,
                            wav_path=wav_path,
                            phones=phones,
                            segments=segments,
                            ref_starts=starts,
                            ref_ends=ends,
                        )
                        record.update(backend_meta)
                        align_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        print(f"    metrics/write failed {Path(wav_path).name} [{alignment_type}]: {exc}")
                        error_events.append(
                            make_error_event(
                                model=model_name,
                                utt=Path(wav_path).stem,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=backend_meta.get("fa_backend_used", args.fa_backend),
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                                target_count=len(target_ids),
                            )
                        )
                        accumulators[target_name]["failures"] += 1
                        rows.append(
                            {
                                "model": model_name,
                                "target": target_name,
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                "error": str(exc),
                            }
                        )

                try:
                    inference_ids = greedy_inference_token_ids(asr_model, ctc_p)
                except Exception as exc:
                    print(f"    inference decode failed {Path(wav_path).name}: {exc}")
                    error_events.append(
                        make_error_event(
                            model=model_name,
                            utt=Path(wav_path).stem,
                            wav=wav_path,
                            alignment_type="inference_alignment",
                            primary_backend=args.fa_backend,
                            fallback_backend="",
                            status="all_failed",
                            error=exc,
                            ctc_p=ctc_p,
                        )
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "target": "inference_failed",
                            "utt": Path(wav_path).stem,
                            "wav": wav_path,
                            "error": str(exc),
                        }
                    )
                    continue

                if inference_ids:
                    inference_segments, backend_meta = align_with_fallback(
                        asr_model=asr_model,
                        k2_aligner=k2_aligner,
                        wav_path=wav_path,
                        ctc_p=ctc_p,
                        token_ids=inference_ids,
                        duration=duration,
                        primary_backend=args.fa_backend,
                        blank_policy=blank_policy,
                        fallback_backend=fallback_backend,
                        disable_fallback=args.disable_fa_fallback,
                        error_events=error_events,
                        error_context={
                            "model": model_name,
                            "utt": Path(wav_path).stem,
                            "wav": wav_path,
                            "alignment_type": "inference_alignment",
                        },
                    )
                    if inference_segments is None:
                        print(f"    skipped {Path(wav_path).name} [inference_alignment]: all FA backends failed")
                        rows.append(
                            {
                                "model": model_name,
                                "target": "inference_failed",
                                "utt": Path(wav_path).stem,
                                "wav": wav_path,
                                "error": "all FA backends failed",
                            }
                        )
                        continue
                    inference_phones = token_ids_to_phones(asr_model.tokenizer, inference_ids)
                    record = build_alignment_record(
                        model_name=model_name,
                        alignment_type="inference_alignment",
                        wav_path=wav_path,
                        phones=inference_phones,
                        segments=inference_segments,
                    )
                    record.update(backend_meta)
                    align_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary[model_name] = {
                name: finalize_accumulator(acc, tolerances)
                for name, acc in accumulators.items()
            }

    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_utterance.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row})
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_error_reports(error_events, error_dir)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-utterance metrics: {rows_path}")
    print(f"Saved alignment records: {alignments_path}")
    print(f"Saved FA error reports: {error_dir}")
    print("\nSummary:")
    for model_name, model_summary in summary.items():
        print(f"\n{model_name}")
        for target_name, metrics in model_summary.items():
            if metrics.get("count", 0) == 0:
                print(f"  {target_name}: no valid alignments, failures={metrics.get('failures', 0)}")
                continue
            tol_text = ", ".join(
                f"seg@{int(round(t * 1000))}ms={metrics[f'segment_within_{int(round(t * 1000))}ms']:.3f}"
                for t in tolerances
            )
            print(
                f"  {target_name}: boundary_mse={metrics['boundary_mse']:.6f}, "
                f"boundary_mae={metrics['boundary_mae']:.4f}s, {tol_text}, "
                f"segments={metrics['count']}, failures={metrics['failures']}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="/home/m64000/work/IF-MDD/data/test_times.json")
    parser.add_argument("--output-dir", default="l2arctic_fa_timestamp_metrics")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--fa-backend",
        choices=["native-k2", "k2", "torchaudio"],
        default="native-k2",
        help=(
            "Forced-alignment backend. The default 'native-k2' uses k2.ctc_graph/"
            "DenseFsaVec directly and avoids torchaudio's CTC aligner. 'k2' uses "
            "SpeechBrain's k2_fsa CTCAligner when available."
        ),
    )
    parser.add_argument(
        "--blank-policy",
        choices=["drop", "previous", "next", "split"],
        default=None,
        help=(
            "How to attach CTC blank frames to phone segments. "
            "If omitted, torchaudio uses 'drop' and k2 uses 'previous'. "
            "'drop' reproduces the old torchaudio behavior; 'previous' assigns blank gaps "
            "to the preceding phone; 'next' assigns them to the following phone; "
            "'split' divides each blank gap at its midpoint."
        ),
    )
    parser.add_argument(
        "--fallback-fa-backend",
        choices=["torchaudio"],
        default="torchaudio",
        help="Fallback backend used when native-k2/k2 fails for one alignment.",
    )
    parser.add_argument(
        "--disable-fa-fallback",
        action="store_true",
        help="Disable per-alignment fallback and skip failed native-k2/k2 alignments.",
    )
    parser.add_argument(
        "--error-dir",
        default=None,
        help="Directory for FA error_events.csv and summary files. Defaults to <output-dir>/alignment_errors.",
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[0.02, 0.05, 0.10],
        help="Tolerance margins in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
