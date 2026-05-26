#!/usr/bin/env python3
"""Run L2-Arctic timestamp FA for exported context-phone pretrained models.

This is the context-phone companion to evaluate_l2arctic_fa_timestamps.py.  It
reads the exported best_models.csv, loads each pretrained model, and writes
three alignment streams:

  - canonical_fa: forced alignment against canonical_aligned phones
  - perceived_fa: forced alignment against perceived_train_target phones
  - inference_alignment: greedy CTC phones, then forced-aligned to the audio

For context-phone models, canonical/perceived monophone sequences are converted
to the model's label space before alignment, then collapsed back to monophones
in alignments.jsonl so run_test-style timestamp metrics stay comparable.  The
actual CTC/context labels are kept in ``model_labels`` for visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_l2arctic_fa_timestamps import (  # noqa: E402
    add_to_accumulator,
    align_with_fallback,
    build_alignment_record,
    finalize_accumulator,
    forced_align_token_ids,
    greedy_inference_token_ids,
    k2_align_token_ids,
    load_audio,
    load_my_encoder_asr,
    make_error_event,
    make_k2_aligner,
    native_k2_align_token_ids,
    new_accumulator,
    timestamp_metrics,
    token_ids_to_phones,
    write_error_reports,
)
from utils.context_phone_codec import (  # noqa: E402
    decode_context_tokens,
    make_context_tokens,
    normalize_context_mode,
)


CONTEXT_TEST_JSON_BY_MODE = {
    "diphone": REPO_ROOT / "data/context_phone/l2arctic_diphone/test.json",
    "triphone": REPO_ROOT / "data/context_phone/l2arctic_triphone/test.json",
    "between_word_triphone": REPO_ROOT / "data/context_phone/l2arctic_between_word_triphone/test.json",
    "word_position_uniphone": REPO_ROOT / "data/context_phone/l2arctic_word_position_uniphone/test.json",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    variant: str
    mode: str
    loss: str
    source: Path


@dataclass(frozen=True)
class TargetSequence:
    name: str
    model_labels: list[str]
    center_phones: list[str]
    starts: list[float]
    ends: list[float]


def read_models_csv(path: Path, selected: set[str] | None = None) -> list[ModelConfig]:
    models = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            variant = row["variant"].strip()
            loss = row["loss"].strip()
            name = f"{variant}_{loss}"
            if selected and name not in selected and variant not in selected and loss not in selected:
                continue
            models.append(
                ModelConfig(
                    name=name,
                    variant=variant,
                    mode=normalize_context_mode(row["mode"]),
                    loss=loss,
                    source=Path(row["pretrained_dir"]),
                )
            )
    if not models:
        raise ValueError(f"No models selected from {path}")
    return models


def load_context_records() -> dict[str, dict[str, dict]]:
    by_mode: dict[str, dict[str, dict]] = {}
    for mode, path in CONTEXT_TEST_JSON_BY_MODE.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = {}
        for key, value in data.items():
            records[str(key)] = value
            if value.get("wav"):
                records[str(value["wav"])] = value
        by_mode[mode] = records
    return by_mode


def context_labels_for_entry(
    *,
    mode: str,
    entry_key: str,
    entry: dict,
    target_name: str,
    raw_phones: list[str],
    context_records: dict[str, dict[str, dict]],
) -> list[str]:
    if mode == "mono":
        return raw_phones

    record = context_records.get(mode, {}).get(entry_key) or context_records.get(mode, {}).get(
        str(entry.get("wav", ""))
    )
    if record and record.get(target_name):
        labels = record[target_name].split()
        if len(labels) == len(raw_phones):
            return labels
        raise ValueError(
            f"{mode}/{target_name} length mismatch for {entry_key}: "
            f"context={len(labels)}, raw={len(raw_phones)}"
        )

    if mode in {"diphone", "triphone"}:
        return make_context_tokens(raw_phones, mode)

    raise ValueError(
        f"Missing interval-aware context labels for mode={mode}, target={target_name}, utt={entry_key}"
    )


def build_targets(
    *,
    mode: str,
    entry_key: str,
    entry: dict,
    context_records: dict[str, dict[str, dict]],
) -> list[TargetSequence]:
    canonical_phones = entry["canonical_aligned"].split()
    perceived_phones = entry.get("perceived_train_target", entry["perceived_aligned"]).split()

    targets = [
        (
            "canonical",
            "canonical_aligned",
            canonical_phones,
            [float(x) for x in entry["canonical_starts"]],
            [float(x) for x in entry["canonical_ends"]],
        ),
        (
            "perceived",
            "perceived_train_target",
            perceived_phones,
            [float(x) for x in entry["target_starts"]],
            [float(x) for x in entry["target_ends"]],
        ),
    ]

    out = []
    for display_name, json_field, phones, starts, ends in targets:
        labels = context_labels_for_entry(
            mode=mode,
            entry_key=entry_key,
            entry=entry,
            target_name=json_field,
            raw_phones=phones,
            context_records=context_records,
        )
        centers = decode_context_tokens(labels, mode, drop_special=False)
        if not (len(labels) == len(centers) == len(starts) == len(ends)):
            raise ValueError(
                f"{mode}/{display_name} timestamp length mismatch for {entry_key}: "
                f"labels={len(labels)}, centers={len(centers)}, starts={len(starts)}, ends={len(ends)}"
            )
        out.append(TargetSequence(display_name, labels, centers, starts, ends))
    return out


def encode_labels(tokenizer, labels: list[str]) -> list[int]:
    missing = [label for label in labels if label not in tokenizer.lab2ind]
    if missing:
        raise ValueError(f"labels missing from tokenizer: {sorted(set(missing))[:20]}")
    return [int(tokenizer.lab2ind[label]) for label in labels]


def filter_inference_ids(tokenizer, token_ids: list[int]) -> list[int]:
    special_ids = set()
    for label in ("<blank>", "<bos>", "<eos>"):
        if label in tokenizer.lab2ind:
            special_ids.add(int(tokenizer.lab2ind[label]))
    return [int(token_id) for token_id in token_ids if int(token_id) not in special_ids]


def align_token_ids(
    *,
    args,
    asr_model,
    k2_aligner,
    wav_path: str,
    ctc_p: torch.Tensor,
    token_ids: list[int],
    duration: float,
):
    if args.fa_backend == "native-k2":
        return native_k2_align_token_ids(
            asr_model,
            ctc_p,
            token_ids,
            duration,
            blank_policy=args.blank_policy,
        )
    if args.fa_backend == "k2":
        return k2_align_token_ids(
            k2_aligner,
            asr_model,
            wav_path,
            token_ids,
            duration,
            blank_policy=args.blank_policy,
        )
    return forced_align_token_ids(
        asr_model,
        ctc_p,
        token_ids,
        duration,
        blank_policy=args.blank_policy,
    )


def write_alignment_record(
    align_f,
    *,
    model: ModelConfig,
    alignment_type: str,
    wav_path: str,
    phones: list[str],
    segments: list[dict],
    model_labels: list[str],
    ref_starts=None,
    ref_ends=None,
    backend_meta: dict | None = None,
) -> None:
    record = build_alignment_record(
        model_name=model.name,
        alignment_type=alignment_type,
        wav_path=wav_path,
        phones=phones,
        segments=segments,
        ref_starts=ref_starts,
        ref_ends=ref_ends,
    )
    record["variant"] = model.variant
    record["loss"] = model.loss
    record["context_phone_mode"] = model.mode
    record["model_labels"] = model_labels
    if backend_meta:
        record.update(backend_meta)
    align_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(args: argparse.Namespace) -> None:
    data = json.loads(args.test_json.read_text(encoding="utf-8"))
    items = list(data.items())
    if args.limit > 0:
        items = items[: args.limit]

    selected = set(args.models) if args.models else None
    models = read_models_csv(args.models_csv, selected=selected)
    context_records = load_context_records()
    tolerances = [float(t) for t in args.tolerances]

    if args.blank_policy is None:
        args.blank_policy = "previous" if args.fa_backend in {"k2", "native-k2"} else "drop"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    alignments_path = output_dir / "alignments.jsonl"
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_utterance.csv"
    error_dir = args.error_dir if args.error_dir else output_dir / "alignment_errors"
    fallback_backend = None if args.disable_fa_fallback else args.fallback_fa_backend

    MyEncoderASR = load_my_encoder_asr()
    rows = []
    summary = {}
    error_events = []

    print(f"Models: {len(models)}")
    print(f"FA backend: {args.fa_backend}, blank_policy: {args.blank_policy}, device={args.device}")
    print(f"Fallback FA backend: {fallback_backend or 'disabled'}")
    print(f"Output: {output_dir}")

    with alignments_path.open("w", encoding="utf-8") as align_f:
        for model in models:
            print(f"\n{'=' * 80}")
            print(f"Model: {model.name} | mode={model.mode} | source={model.source}")
            print(f"{'=' * 80}")
            asr_model = MyEncoderASR.from_hparams(
                source=str(model.source),
                hparams_file="inference.yaml",
                run_opts={"device": args.device},
            )
            k2_aligner = make_k2_aligner(asr_model) if args.fa_backend == "k2" else None
            accumulators = {
                "canonical": new_accumulator(tolerances),
                "perceived": new_accumulator(tolerances),
            }
            inference_utterances = 0
            inference_failures = 0

            for index, (entry_key, entry) in enumerate(items, start=1):
                wav_path = str(entry.get("wav", entry_key))
                utt = Path(wav_path).stem
                if index == 1 or index % args.log_every == 0:
                    print(f"  [{index}/{len(items)}] {Path(wav_path).name}")

                try:
                    waveform, wav_duration = load_audio(wav_path)
                    duration = float(entry.get("duration") or wav_duration)
                    with torch.no_grad():
                        ctc_p = asr_model.encode_batch(
                            waveform.unsqueeze(0),
                            torch.tensor([1.0], device=asr_model.device),
                        )
                except Exception as exc:
                    print(f"    setup failed {Path(wav_path).name}: {type(exc).__name__}: {exc}")
                    for target_name in ("canonical", "perceived"):
                        alignment_type = "canonical_fa" if target_name == "canonical" else "perceived_fa"
                        error_events.append(
                            make_error_event(
                                model=model.name,
                                variant=model.variant,
                                loss=model.loss,
                                context_phone_mode=model.mode,
                                utt=utt,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=args.fa_backend,
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                            )
                        )
                        accumulators[target_name]["failures"] += 1
                    error_events.append(
                        make_error_event(
                            model=model.name,
                            variant=model.variant,
                            loss=model.loss,
                            context_phone_mode=model.mode,
                            utt=utt,
                            wav=wav_path,
                            alignment_type="inference_alignment",
                            primary_backend=args.fa_backend,
                            fallback_backend="",
                            status="all_failed",
                            error=exc,
                        )
                    )
                    inference_failures += 1
                    rows.append(
                        {
                            "model": model.name,
                            "variant": model.variant,
                            "loss": model.loss,
                            "context_phone_mode": model.mode,
                            "target": "setup_failed",
                            "utt": utt,
                            "wav": wav_path,
                            "error": str(exc),
                        }
                    )
                    continue

                try:
                    targets = build_targets(
                        mode=model.mode,
                        entry_key=entry_key,
                        entry=entry,
                        context_records=context_records,
                    )
                except Exception as exc:
                    print(f"    target build failed {Path(wav_path).name}: {type(exc).__name__}: {exc}")
                    for target_name in ("canonical", "perceived"):
                        alignment_type = "canonical_fa" if target_name == "canonical" else "perceived_fa"
                        error_events.append(
                            make_error_event(
                                model=model.name,
                                variant=model.variant,
                                loss=model.loss,
                                context_phone_mode=model.mode,
                                utt=utt,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=args.fa_backend,
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                            )
                        )
                        accumulators[target_name]["failures"] += 1
                    rows.append(
                        {
                            "model": model.name,
                            "variant": model.variant,
                            "loss": model.loss,
                            "context_phone_mode": model.mode,
                            "target": "target_build_failed",
                            "utt": utt,
                            "wav": wav_path,
                            "error": str(exc),
                        }
                    )
                    continue

                for target in targets:
                    alignment_type = "canonical_fa" if target.name == "canonical" else "perceived_fa"
                    try:
                        token_ids = encode_labels(asr_model.tokenizer, target.model_labels)
                    except Exception as exc:
                        print(f"    encode failed {Path(wav_path).name} [{alignment_type}]: {exc}")
                        error_events.append(
                            make_error_event(
                                model=model.name,
                                variant=model.variant,
                                loss=model.loss,
                                context_phone_mode=model.mode,
                                utt=utt,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=args.fa_backend,
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                                target_count=len(target.model_labels),
                            )
                        )
                        accumulators[target.name]["failures"] += 1
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": target.name,
                                "utt": utt,
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
                        token_ids=token_ids,
                        duration=duration,
                        primary_backend=args.fa_backend,
                        blank_policy=args.blank_policy,
                        fallback_backend=fallback_backend,
                        disable_fallback=args.disable_fa_fallback,
                        error_events=error_events,
                        error_context={
                            "model": model.name,
                            "variant": model.variant,
                            "loss": model.loss,
                            "context_phone_mode": model.mode,
                            "utt": utt,
                            "wav": wav_path,
                            "alignment_type": alignment_type,
                        },
                    )
                    if segments is None:
                        print(f"    skipped {Path(wav_path).name} [{alignment_type}]: all FA backends failed")
                        accumulators[target.name]["failures"] += 1
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": target.name,
                                "utt": utt,
                                "wav": wav_path,
                                "error": "all FA backends failed",
                            }
                        )
                        continue

                    try:
                        metrics = timestamp_metrics(segments, target.starts, target.ends, tolerances)
                        add_to_accumulator(accumulators[target.name], metrics, tolerances)
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": target.name,
                                "utt": utt,
                                "wav": wav_path,
                                **metrics,
                            }
                        )
                        write_alignment_record(
                            align_f,
                            model=model,
                            alignment_type=alignment_type,
                            wav_path=wav_path,
                            phones=target.center_phones,
                            segments=segments,
                            model_labels=target.model_labels,
                            ref_starts=target.starts,
                            ref_ends=target.ends,
                            backend_meta=backend_meta,
                        )
                    except Exception as exc:
                        print(f"    metrics/write failed {Path(wav_path).name} [{alignment_type}]: {exc}")
                        error_events.append(
                            make_error_event(
                                model=model.name,
                                variant=model.variant,
                                loss=model.loss,
                                context_phone_mode=model.mode,
                                utt=utt,
                                wav=wav_path,
                                alignment_type=alignment_type,
                                primary_backend=backend_meta.get("fa_backend_used", args.fa_backend),
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                                target_count=len(token_ids),
                            )
                        )
                        accumulators[target.name]["failures"] += 1
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": target.name,
                                "utt": utt,
                                "wav": wav_path,
                                "error": str(exc),
                            }
                        )

                try:
                    inference_ids = filter_inference_ids(
                        asr_model.tokenizer,
                        greedy_inference_token_ids(asr_model, ctc_p),
                    )
                except Exception as exc:
                    print(f"    inference decode failed {Path(wav_path).name}: {exc}")
                    error_events.append(
                        make_error_event(
                            model=model.name,
                            variant=model.variant,
                            loss=model.loss,
                            context_phone_mode=model.mode,
                            utt=utt,
                            wav=wav_path,
                            alignment_type="inference_alignment",
                            primary_backend=args.fa_backend,
                            fallback_backend="",
                            status="all_failed",
                            error=exc,
                            ctc_p=ctc_p,
                        )
                    )
                    inference_failures += 1
                    rows.append(
                        {
                            "model": model.name,
                            "variant": model.variant,
                            "loss": model.loss,
                            "context_phone_mode": model.mode,
                            "target": "inference_failed",
                            "utt": utt,
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
                        blank_policy=args.blank_policy,
                        fallback_backend=fallback_backend,
                        disable_fallback=args.disable_fa_fallback,
                        error_events=error_events,
                        error_context={
                            "model": model.name,
                            "variant": model.variant,
                            "loss": model.loss,
                            "context_phone_mode": model.mode,
                            "utt": utt,
                            "wav": wav_path,
                            "alignment_type": "inference_alignment",
                        },
                    )
                    if inference_segments is None:
                        print(f"    skipped {Path(wav_path).name} [inference_alignment]: all FA backends failed")
                        inference_failures += 1
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": "inference_failed",
                                "utt": utt,
                                "wav": wav_path,
                                "error": "all FA backends failed",
                            }
                        )
                        continue

                    try:
                        inference_labels = token_ids_to_phones(asr_model.tokenizer, inference_ids)
                        inference_phones = decode_context_tokens(
                            inference_labels,
                            model.mode,
                            drop_special=True,
                        )
                        write_alignment_record(
                            align_f,
                            model=model,
                            alignment_type="inference_alignment",
                            wav_path=wav_path,
                            phones=inference_phones,
                            segments=inference_segments,
                            model_labels=inference_labels,
                            backend_meta=backend_meta,
                        )
                        inference_utterances += 1
                    except Exception as exc:
                        print(f"    inference write failed {Path(wav_path).name}: {exc}")
                        error_events.append(
                            make_error_event(
                                model=model.name,
                                variant=model.variant,
                                loss=model.loss,
                                context_phone_mode=model.mode,
                                utt=utt,
                                wav=wav_path,
                                alignment_type="inference_alignment",
                                primary_backend=backend_meta.get("fa_backend_used", args.fa_backend),
                                fallback_backend="",
                                status="all_failed",
                                error=exc,
                                ctc_p=ctc_p,
                                target_count=len(inference_ids),
                            )
                        )
                        inference_failures += 1
                        rows.append(
                            {
                                "model": model.name,
                                "variant": model.variant,
                                "loss": model.loss,
                                "context_phone_mode": model.mode,
                                "target": "inference_failed",
                                "utt": utt,
                                "wav": wav_path,
                                "error": str(exc),
                            }
                        )

            summary[model.name] = {
                target_name: finalize_accumulator(acc, tolerances)
                for target_name, acc in accumulators.items()
            }
            summary[model.name]["inference_alignment"] = {
                "utterances": inference_utterances,
                "failures": inference_failures,
                "note": "No timestamp reference is available for free decoded inference phones.",
            }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row})
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_error_reports(error_events, error_dir)

    print(f"\nSaved alignment records: {alignments_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-utterance metrics: {rows_path}")
    print(f"Saved FA error reports: {error_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", type=Path, default=REPO_ROOT / "data/test_times.json")
    parser.add_argument(
        "--models-csv",
        type=Path,
        default=REPO_ROOT
        / "pretrained_models/l2arctic_context_phone_acou_model_test_selected/best_models.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "fa_research/results/l2arctic_context_phone_test_selected_fa",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Optional names like dp_ctc or variants/losses.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--fa-backend",
        choices=["native-k2", "k2", "torchaudio"],
        default="native-k2",
    )
    parser.add_argument(
        "--blank-policy",
        choices=["drop", "previous", "next", "split"],
        default=None,
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
        type=Path,
        default=None,
        help="Directory for FA error_events.csv and summary files. Defaults to <output-dir>/alignment_errors.",
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.025, 0.03, 0.04, 0.05],
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
