#!/usr/bin/env python3
"""Dependency-free HTML visualization for L2-ARCTIC alignment examples."""

from __future__ import annotations

import argparse
import array
import base64
import html
import json
import mimetypes
import wave
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = ["CTC", "OTTC", "CRCTC", "CROTTC"]
CONTEXT_VARIANT_ORDER = ["dp", "tri", "bwt", "wpu"]
CONTEXT_LOSS_ORDER = ["ctc", "crctc", "ottc", "crottc"]
CONTEXT_MODE_ORDER = [
    "uniform",
    "diphone",
    "triphone",
    "between_word_triphone",
    "word_position_uniphone",
]
CONTEXT_MODE_TITLES = {
    "uniform": "Uniform",
    "diphone": "Diphone",
    "triphone": "Triphone",
    "between_word_triphone": "BWT",
    "word_position_uniphone": "WPU",
}
CONTEXT_MODE_SLUGS = {
    "uniform": "uniform",
    "diphone": "diphone",
    "triphone": "triphone",
    "between_word_triphone": "bwt",
    "word_position_uniphone": "wpu",
}
SILENCE = {"SIL", "sil", "SP", "sp", "SPN", "spn", ""}
SCORE_TOLERANCES_MS = [25, 50]
LANE_ORDER = ["canonical_fa", "perceived_fa"]
TAB_LANES = ["canonical_fa", "perceived_fa", "inference_alignment"]
LANE_TITLES = {
    "canonical_fa": "Canonical FA",
    "perceived_fa": "Perceived FA",
    "inference_alignment": "Inference Alignment",
}


def load_alignments(paths: list[Path]):
    grouped = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                grouped[record["utt"]][record["model"]][record["alignment_type"]] = record
    return grouped


def edit_statuses(reference: list[str], hypothesis: list[str]) -> list[str]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [(dp[i - 1][j - 1], "eq")]
            else:
                candidates = [(dp[i - 1][j - 1] + 1, "sub")]
            candidates.extend([(dp[i - 1][j] + 1, "del"), (dp[i][j - 1] + 1, "ins")])
            dp[i][j], back[i][j] = min(candidates, key=lambda item: item[0])

    statuses = ["unknown"] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "eq":
            statuses[i - 1] = "correct"
            i -= 1
            j -= 1
        elif op == "sub":
            statuses[i - 1] = "substitution"
            i -= 1
            j -= 1
        elif op == "del":
            statuses[i - 1] = "deletion"
            i -= 1
        elif op == "ins":
            j -= 1
        else:
            break
    return statuses


def record_duration(records_for_utt: dict) -> float:
    duration = 0.0
    for model_records in records_for_utt.values():
        for record in model_records.values():
            ends = record.get("gt_ends") or record.get("pred_ends") or []
            if ends:
                duration = max(duration, max(float(x) for x in ends))
    return duration or 1.0


def record_wav_path(records_for_utt: dict) -> str | None:
    for model_records in records_for_utt.values():
        for record in model_records.values():
            if record.get("wav"):
                return record["wav"]
    return None


def audio_source(records_for_utt: dict, audio_mode: str = "data") -> str:
    wav_path = record_wav_path(records_for_utt)
    if not wav_path:
        return ""
    if audio_mode == "data":
        try:
            path = Path(wav_path)
            mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            pass
    try:
        return Path(wav_path).resolve().as_uri()
    except Exception:
        return wav_path


def format_seq(phones: list[str]) -> str:
    return html.escape(" ".join(phones) if phones else "(empty)")


def display_phones(record: dict) -> list[str]:
    """Labels to draw on the plot; context-phone runs keep monophones in phones."""
    labels = record.get("display_phones") or record.get("model_labels") or record.get("phones") or []
    return [str(label) for label in labels]


def model_sort_key(model: str):
    if model in MODEL_ORDER:
        return (0, MODEL_ORDER.index(model), model)
    variant, sep, loss = model.rpartition("_")
    if sep and variant in CONTEXT_VARIANT_ORDER and loss in CONTEXT_LOSS_ORDER:
        return (
            1,
            CONTEXT_VARIANT_ORDER.index(variant),
            CONTEXT_LOSS_ORDER.index(loss),
            model,
        )
    return (2, model)


def model_order(records_for_utt: dict) -> list[str]:
    return sorted(records_for_utt, key=model_sort_key)


def normalize_mode(mode: str | None) -> str:
    if not mode or mode in {"mono", "uniphone"}:
        return "uniform"
    return mode


def mode_sort_key(mode: str):
    if mode in CONTEXT_MODE_ORDER:
        return (CONTEXT_MODE_ORDER.index(mode), mode)
    return (len(CONTEXT_MODE_ORDER), mode)


def model_mode(model_records: dict) -> str:
    for record in model_records.values():
        return normalize_mode(record.get("context_phone_mode"))
    return "uniform"


def model_groups(records_for_utt: dict) -> list[tuple[str, list[str]]]:
    grouped = defaultdict(list)
    for model in model_order(records_for_utt):
        grouped[model_mode(records_for_utt.get(model, {}))].append(model)
    return [(mode, grouped[mode]) for mode in sorted(grouped, key=mode_sort_key)]


def mode_slug(mode: str) -> str:
    return CONTEXT_MODE_SLUGS.get(mode, mode.replace("_", "-"))


def is_silence(label: str) -> bool:
    return label in SILENCE or label.upper() in SILENCE


def record_alignment_metrics(record: dict) -> dict:
    mean_scores = [float(x) for x in record.get("mean_scores", [])]
    metrics = {
        "mean_score": sum(mean_scores) / len(mean_scores) if mean_scores else None,
        "count": len(record.get("pred_starts", [])),
    }
    if not (record.get("gt_starts") and record.get("gt_ends")):
        return metrics

    ref = []
    pred = []
    for phone, rs, re, ps, pe in zip(
        record.get("phones", []),
        record.get("gt_starts", []),
        record.get("gt_ends", []),
        record.get("pred_starts", []),
        record.get("pred_ends", []),
    ):
        if is_silence(str(phone)):
            continue
        ref.append((float(rs), float(re)))
        pred.append((float(ps), float(pe)))

    if not ref or len(ref) != len(pred):
        return metrics

    tse = []
    hits = {tau: 0 for tau in SCORE_TOLERANCES_MS}
    for (rs, re), (ps, pe) in zip(ref, pred):
        start_err = abs(rs - ps)
        end_err = abs(re - pe)
        tse.append(start_err + end_err)
        for tau in SCORE_TOLERANCES_MS:
            tau_sec = tau / 1000.0
            if start_err <= tau_sec and end_err <= tau_sec:
                hits[tau] += 1

    metrics["phone_count"] = len(ref)
    metrics["phone_tse_ms"] = sum(tse) / len(tse) * 1000.0
    for tau in SCORE_TOLERANCES_MS:
        metrics[f"phone_acc_tau_{tau}"] = hits[tau] / len(ref) * 100.0
    return metrics


def render_metrics(record: dict) -> str:
    metrics = record_alignment_metrics(record)
    pieces = []
    if "phone_tse_ms" in metrics:
        pieces.append(f'TSE <strong>{metrics["phone_tse_ms"]:.1f} ms</strong>')
        for tau in SCORE_TOLERANCES_MS:
            key = f"phone_acc_tau_{tau}"
            if key in metrics:
                pieces.append(f'ACC@{tau} <strong>{metrics[key]:.1f}%</strong>')
    else:
        pieces.append("TSE/ACC <strong>n/a</strong>")
    if metrics.get("mean_score") is not None:
        pieces.append(f'Score <strong>{metrics["mean_score"]:.3f}</strong>')
    if metrics.get("phone_count"):
        pieces.append(f'N <strong>{int(metrics["phone_count"])}</strong>')
    return '<div class="metrics">' + " · ".join(pieces) + "</div>"


def tick_marks(duration: float):
    if duration <= 3:
        step = 0.25
    elif duration <= 8:
        step = 0.5
    else:
        step = 1.0
    ticks = []
    value = 0.0
    while value <= duration + 1e-9:
        ticks.append(round(value, 2))
        value += step
    return ticks


def render_timeline_axis(duration: float) -> str:
    pieces = ['<div class="mini-axis">']
    for tick in tick_marks(duration):
        left = 100.0 * tick / duration
        pieces.append(
            f'<div class="mini-tick" style="left:{left:.4f}%">'
            f'<div class="mini-tick-line"></div><div class="mini-tick-label">{tick:g}</div></div>'
        )
    pieces.append("</div>")
    return "\n".join(pieces)


def read_waveform_envelope(wav_path: str, target_points: int = 1400):
    """Read a PCM wav and return min/max envelope points for SVG rendering."""
    try:
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            raw = wf.readframes(frame_count)
    except Exception:
        return None

    if sample_width == 1:
        samples = array.array("B", raw)
        values = [sample - 128 for sample in samples[::channels]]
        max_abs = 128.0
    elif sample_width == 2:
        samples = array.array("h", raw)
        values = samples[::channels]
        max_abs = 32768.0
    elif sample_width == 4:
        samples = array.array("i", raw)
        values = samples[::channels]
        max_abs = 2147483648.0
    else:
        return None

    if not values:
        return None

    point_count = min(target_points, len(values))
    chunk = max(1, len(values) // point_count)
    envelope = []
    for idx in range(0, len(values), chunk):
        segment = values[idx : idx + chunk]
        if not segment:
            continue
        start_time = idx / float(sample_rate)
        min_val = max(-1.0, min(segment) / max_abs)
        max_val = min(1.0, max(segment) / max_abs)
        envelope.append((start_time, min_val, max_val))
    duration = len(values) / float(sample_rate)
    return duration, envelope


def render_waveform(records_for_utt: dict, duration: float, lane_type: str) -> str:
    wav_path = record_wav_path(records_for_utt)
    if not wav_path:
        panel = '<div class="waveform-panel">No waveform path found.</div>'
        return render_waveform_row(panel)

    data = read_waveform_envelope(wav_path)
    if data is None:
        panel = (
            '<div class="waveform-panel">'
            f'Waveform unavailable for <code>{html.escape(wav_path)}</code>.'
            '</div>'
        )
        return render_waveform_row(panel)

    wav_duration, envelope = data
    duration = max(duration, wav_duration)
    return render_waveform_row(render_waveform_panel(envelope, duration, lane_type))


def render_waveform_panel(envelope, duration: float, lane_type: str) -> str:
    width = 1200
    height = 150
    mid = height / 2
    amp = height * 0.42
    lines = []
    for time_s, min_val, max_val in envelope:
        x = 100.0 * time_s / duration
        y1 = mid - max_val * amp
        y2 = mid - min_val * amp
        lines.append(
            f'<line x1="{x:.4f}%" y1="{y1:.2f}" x2="{x:.4f}%" y2="{y2:.2f}" />'
        )

    tick_lines = []
    for tick in tick_marks(duration):
        x = 100.0 * tick / duration
        tick_lines.append(
            f'<line class="wave-tick" x1="{x:.4f}%" y1="0" x2="{x:.4f}%" y2="{height}" />'
        )

    return (
        '<div class="waveform-panel">'
        '<div class="waveform-box">'
        f'<svg class="waveform" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<line class="wave-mid" x1="0" y1="{mid:.2f}" x2="{width}" y2="{mid:.2f}" />'
        + "\n".join(tick_lines)
        + '<g class="wave-lines">'
        + "\n".join(lines)
        + '</g></svg>'
        f'<div class="wave-selection" data-wave-lane="{lane_type}"></div>'
        '</div>'
        '<div class="wave-axis">'
        + render_timeline_axis(duration)
        + '</div>'
        '</div>'
    )


def render_waveform_row(panel: str) -> str:
    return (
        '<div class="waveform-row">'
        '<div class="waveform-title">Waveform</div>'
        f'<div>{panel}</div>'
        '</div>'
    )


def render_lane(
    record: dict,
    inference_record: dict | None,
    duration: float,
    lane_type: str,
    is_inference: bool = False,
) -> str:
    phones = display_phones(record)
    pred_starts = [float(x) for x in record.get("pred_starts", [])]
    pred_ends = [float(x) for x in record.get("pred_ends", [])]
    gt_starts = [float(x) for x in record.get("gt_starts", [])]
    gt_ends = [float(x) for x in record.get("gt_ends", [])]
    scores = [float(x) for x in record.get("mean_scores", [])]
    if is_inference:
        statuses = ["prediction"] * len(phones)
    else:
        inference_phones = display_phones(inference_record) if inference_record else []
        statuses = edit_statuses(phones, inference_phones)

    pieces = ['<div class="timeline">']
    for idx, phone in enumerate(phones):
        if idx >= len(pred_starts) or idx >= len(pred_ends):
            continue
        start = pred_starts[idx]
        end = pred_ends[idx]
        status = statuses[idx] if idx < len(statuses) else "unknown"
        left = 100.0 * start / duration
        width = max(0.15, 100.0 * (end - start) / duration)
        gt_text = ""
        if idx < len(gt_starts) and idx < len(gt_ends):
            gt_text = f" | gt={gt_starts[idx]:.3f}-{gt_ends[idx]:.3f}s"
        score_text = f" | score={scores[idx]:.3f}" if idx < len(scores) else ""
        tooltip = html.escape(f"{phone}: pred={start:.3f}-{end:.3f}s{gt_text}{score_text} | {status}")
        label = html.escape(phone)
        pieces.append(
            f'<div class="seg {status}" style="left:{left:.4f}%;width:{width:.4f}%;" '
            f'title="{tooltip}" data-start="{start:.6f}" data-end="{end:.6f}" '
            f'data-label="{label}" data-lane="{lane_type}" role="button" tabindex="0"><span>{label}</span></div>'
        )
    pieces.append("</div>")
    pieces.append(render_timeline_axis(duration))
    return "\n".join(pieces)


def render_ground_truth_lane(record: dict, duration: float, lane_type: str) -> str:
    phones = display_phones(record)
    starts = [float(x) for x in record.get("gt_starts", [])]
    ends = [float(x) for x in record.get("gt_ends", [])]

    pieces = ['<div class="timeline gt-timeline">']
    for idx, phone in enumerate(phones):
        if idx >= len(starts) or idx >= len(ends):
            continue
        start = starts[idx]
        end = ends[idx]
        left = 100.0 * start / duration
        width = max(0.15, 100.0 * (end - start) / duration)
        tooltip = html.escape(f"{phone}: gt={start:.3f}-{end:.3f}s")
        label = html.escape(phone)
        pieces.append(
            f'<div class="seg ground-truth" style="left:{left:.4f}%;width:{width:.4f}%;" '
            f'title="{tooltip}" data-start="{start:.6f}" data-end="{end:.6f}" '
            f'data-label="{label}" data-lane="{lane_type}" role="button" tabindex="0"><span>{label}</span></div>'
        )
    pieces.append("</div>")
    pieces.append(render_timeline_axis(duration))
    return "\n".join(pieces)


def render_cell(model_records: dict, lane_type: str, duration: float) -> str:
    record = model_records.get(lane_type)
    if not record:
        return '<div class="missing">missing</div>'

    inference_record = model_records.get("inference_alignment")
    is_inference = lane_type == "inference_alignment"
    title = LANE_TITLES[lane_type]
    seq = format_seq(display_phones(record))
    return (
        f'<div class="cell-title">{html.escape(title)}</div>'
        f'{render_metrics(record)}'
        f'<div class="seq">{seq}</div>'
        f'{render_lane(record, inference_record, duration, lane_type=lane_type, is_inference=is_inference)}'
    )


def first_record(model_records_list: list[dict], lane_type: str) -> dict | None:
    for model_records in model_records_list:
        record = model_records.get(lane_type)
        if record and record.get("gt_starts") and record.get("gt_ends"):
            return record
    return None


def render_ground_truth_cell(record: dict | None, lane_type: str, duration: float) -> str:
    if not record:
        return '<div class="missing">missing ground truth</div>'
    title = "Canonical Ground Truth" if lane_type == "canonical_fa" else "Perceived Ground Truth"
    return (
        f'<div class="cell-title">{title}</div>'
        f'<div class="seq">{format_seq(display_phones(record))}</div>'
        f'{render_ground_truth_lane(record, duration, lane_type=lane_type)}'
    )


def render_reference_stack_cell(
    canonical_record: dict | None,
    perceived_record: dict | None,
    duration: float,
) -> str:
    pieces = []
    if canonical_record:
        pieces.append(
            '<div class="reference-stack-item">'
            '<div class="cell-title">Canonical Ground Truth</div>'
            f'<div class="seq">{format_seq(display_phones(canonical_record))}</div>'
            f'{render_ground_truth_lane(canonical_record, duration, lane_type="inference_alignment")}'
            '</div>'
        )
    if perceived_record:
        pieces.append(
            '<div class="reference-stack-item">'
            '<div class="cell-title">Perceived Ground Truth</div>'
            f'<div class="seq">{format_seq(display_phones(perceived_record))}</div>'
            f'{render_ground_truth_lane(perceived_record, duration, lane_type="inference_alignment")}'
            '</div>'
        )
    if not pieces:
        return '<div class="missing">missing ground truth</div>'
    return '<div class="reference-stack">' + "\n".join(pieces) + "</div>"


def render_ground_truth_row(
    mode: str,
    model_records_list: list[dict],
    lane_type: str,
    duration: float,
) -> str:
    canonical_record = first_record(model_records_list, "canonical_fa")
    perceived_record = first_record(model_records_list, "perceived_fa")
    mode_title = CONTEXT_MODE_TITLES.get(mode, mode.replace("_", " ").title())
    if lane_type == "canonical_fa":
        cell = render_ground_truth_cell(canonical_record, "canonical_fa", duration)
    elif lane_type == "perceived_fa":
        cell = render_ground_truth_cell(perceived_record, "perceived_fa", duration)
    else:
        cell = render_reference_stack_cell(canonical_record, perceived_record, duration)
    return (
        '<div class="model-row ground-truth-row">'
        f'<div class="model-name ground-truth-name"><span>Ground Truth</span><small>{html.escape(mode_title)}</small></div>'
        f'<div class="align-cell ground-truth-cell">{cell}</div>'
        '</div>'
    )


def render_tab_rows(
    records_for_utt: dict,
    lane_type: str,
    mode: str,
    models: list[str],
    duration: float,
) -> str:
    rows = []
    model_records_list = [records_for_utt.get(model, {}) for model in models]
    rows.append(render_ground_truth_row(mode, model_records_list, lane_type, duration))
    for model in models:
        model_records = records_for_utt.get(model, {})
        cell_class = "align-cell inference-cell" if lane_type == "inference_alignment" else "align-cell"
        rows.append(
            '<div class="model-row">'
            f'<div class="model-name">{html.escape(model)}</div>'
            f'<div class="{cell_class}">{render_cell(model_records, lane_type, duration)}</div>'
            '</div>'
        )
    return "\n".join(rows)


def render_tab_pane(
    records_for_utt: dict,
    lane_type: str,
    mode: str,
    models: list[str],
    duration: float,
    active: bool,
) -> str:
    active_class = " active" if active else ""
    lane_title = html.escape(LANE_TITLES[lane_type])
    mode_title = html.escape(CONTEXT_MODE_TITLES.get(mode, mode.replace("_", " ").title()))
    slug = mode_slug(mode)
    return (
        f'<section class="tab-pane{active_class}" data-view-pane="{lane_type}" data-method-pane="{slug}">'
        f'{render_waveform(records_for_utt, duration, lane_type)}'
        '<div class="tab-grid">'
        '<div class="header-row">'
        '<div></div>'
        f'<div class="header-cell">{lane_title} · {mode_title}</div>'
        '</div>'
        f'{render_tab_rows(records_for_utt, lane_type, mode, models, duration)}'
        '</div>'
        '</section>'
    )


def render_tabs(records_for_utt: dict, duration: float) -> str:
    groups = model_groups(records_for_utt)
    view_buttons = []
    method_buttons = []
    panes = []
    for index, lane_type in enumerate(TAB_LANES):
        active = index == 0
        active_class = " active" if active else ""
        selected = "true" if active else "false"
        title = html.escape(LANE_TITLES[lane_type])
        view_buttons.append(
            f'<button class="tab-button view-tab{active_class}" type="button" role="tab" '
            f'aria-selected="{selected}" data-view-target="{lane_type}">{title}</button>'
        )

    for index, (mode, _) in enumerate(groups):
        active = index == 0
        active_class = " active" if active else ""
        selected = "true" if active else "false"
        title = html.escape(CONTEXT_MODE_TITLES.get(mode, mode.replace("_", " ").title()))
        method_buttons.append(
            f'<button class="tab-button method-tab{active_class}" type="button" role="tab" '
            f'aria-selected="{selected}" data-method-target="{mode_slug(mode)}">{title}</button>'
        )

    for view_index, lane_type in enumerate(TAB_LANES):
        for method_index, (mode, models) in enumerate(groups):
            panes.append(
                render_tab_pane(
                    records_for_utt,
                    lane_type,
                    mode,
                    models,
                    duration,
                    active=view_index == 0 and method_index == 0,
                )
            )
    return (
        '<div class="tab-shell">'
        '<div class="tab-axis"><div class="tab-axis-label">View</div><div class="tab-list" role="tablist">'
        + "\n".join(view_buttons)
        + '</div></div>'
        '<div class="tab-axis"><div class="tab-axis-label">Method</div><div class="tab-list method-list" role="tablist">'
        + "\n".join(method_buttons)
        + '</div></div>'
        '<div class="tab-panes">'
        + "\n".join(panes)
        + '</div>'
        + '</div>'
    )


def render_utterance(utt: str, records_for_utt: dict, audio_mode: str = "data") -> str:
    duration = record_duration(records_for_utt)

    return TEMPLATE.format(
        title=html.escape(utt),
        utt=html.escape(utt),
        audio_src=html.escape(audio_source(records_for_utt, audio_mode=audio_mode), quote=True),
        tabs=render_tabs(records_for_utt, duration),
    )


TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title} alignment comparison</title>
<style>
body {{
  margin: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #202124;
  background: #fafafa;
}}
h1 {{ margin: 0 0 12px; font-size: 22px; }}
.legend {{ display: flex; gap: 18px; margin: 12px 0 20px; font-size: 13px; }}
.chip {{ display: inline-block; width: 16px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.correct-chip {{ background: #62b36f; }}
.substitution-chip {{ background: #f28e2b; }}
.deletion-chip {{ background: #d64f4f; }}
.prediction-chip {{ background: #4c78a8; }}
.ground-truth-chip {{ background: #6f6a55; }}
.tab-shell {{
  min-width: 1320px;
}}
.tab-axis {{
  display: grid;
  grid-template-columns: 118px 1fr;
  gap: 12px;
  align-items: end;
}}
.tab-axis-label {{
  text-align: right;
  font-size: 12px;
  font-weight: 800;
  color: #3d4249;
  padding: 0 4px 8px 0;
}}
.tab-list {{
  display: flex;
  gap: 8px;
  align-items: center;
  margin: 4px 0 10px;
  border-bottom: 2px solid #202124;
}}
.method-list {{
  border-bottom-color: #59636e;
}}
.tab-button {{
  appearance: none;
  border: 1px solid #cfd6df;
  border-bottom: 0;
  background: #eef2f7;
  color: #28313c;
  padding: 9px 14px;
  border-radius: 8px 8px 0 0;
  font-size: 13px;
  font-weight: 750;
  cursor: pointer;
}}
.tab-button.active {{
  background: white;
  color: #111;
  border-color: #202124;
}}
.tab-pane {{
  display: none;
}}
.tab-pane.active {{
  display: block;
}}
.tab-panes {{
  margin-top: 4px;
}}
.waveform-row {{
  min-width: 1320px;
  display: grid;
  grid-template-columns: 118px minmax(980px, 1fr);
  gap: 12px;
  align-items: stretch;
  margin: 0 0 12px;
}}
.waveform-title {{
  font-size: 13px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 4px;
}}
.waveform-panel {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 9px 10px 10px;
  box-sizing: border-box;
}}
.waveform-box {{
  position: relative;
  height: 118px;
}}
.waveform {{
  width: 100%;
  height: 118px;
  display: block;
  background: #f4f6f8;
  border: 1px solid #e1e5ea;
  border-radius: 6px;
}}
.wave-selection {{
  display: none;
  position: absolute;
  top: 0;
  height: 118px;
  left: 0;
  width: 0;
  background: rgba(255, 200, 64, 0.55);
  border-left: 3px solid #9a6400;
  border-right: 3px solid #9a6400;
  box-sizing: border-box;
  pointer-events: none;
  z-index: 20;
}}
.wave-selection.active {{
  display: block;
}}
.wave-lines line {{
  stroke: #2f6f9f;
  stroke-width: 0.8;
  opacity: 0.82;
}}
.wave-mid {{
  stroke: #59636e;
  stroke-width: 0.8;
  opacity: 0.45;
}}
.wave-tick {{
  stroke: #9aa5b1;
  stroke-width: 0.7;
  opacity: 0.35;
}}
.wave-axis {{
  margin: 6px 0 0;
}}
.grid {{
  min-width: 1320px;
}}
.tab-grid {{
  min-width: 1320px;
}}
.header-row {{
  display: grid;
  grid-template-columns: 118px minmax(980px, 1fr);
  gap: 12px;
  align-items: end;
  margin: 8px 0 6px;
}}
.header-cell {{
  font-size: 14px;
  font-weight: 750;
  padding: 0 2px 6px;
  border-bottom: 2px solid #202124;
}}
.model-row {{
  display: grid;
  grid-template-columns: 118px minmax(980px, 1fr);
  gap: 12px;
  align-items: stretch;
  margin: 14px 0;
}}
.model-name {{
  font-size: 15px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 4px;
}}
.ground-truth-row {{
  margin-top: 18px;
}}
.ground-truth-name {{
  flex-direction: column;
  align-items: flex-end;
  gap: 3px;
}}
.ground-truth-name small {{
  display: block;
  font-size: 10px;
  font-weight: 650;
  color: #5f6368;
  line-height: 1.15;
  text-align: right;
}}
.align-cell {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 9px 10px 12px;
  min-height: 128px;
  box-sizing: border-box;
}}
.inference-cell {{
  border-color: #b7c9e8;
  background: #fbfdff;
}}
.ground-truth-cell {{
  border-color: #c8c0a3;
  background: #fffdf7;
}}
.gt-note {{
  min-height: 128px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6d6a62;
  font-size: 12px;
  font-weight: 650;
}}
.reference-stack {{
  display: grid;
  gap: 12px;
}}
.reference-stack-item + .reference-stack-item {{
  border-top: 1px solid #d9d1b6;
  padding-top: 10px;
}}
.cell-title {{
  font-size: 12px;
  font-weight: 750;
  margin-bottom: 5px;
}}
.metrics {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  min-height: 20px;
  margin: 0 0 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: #30363d;
}}
.metrics strong {{
  font-weight: 800;
}}
.seq {{
  min-height: 34px;
  max-height: 48px;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  line-height: 1.25;
  color: #41464d;
  margin-bottom: 7px;
}}
.timeline {{
  position: relative;
  height: 44px;
  background: #f0f2f5;
  border: 1px solid #d8dde6;
  border-radius: 6px;
  overflow: hidden;
}}
.seg {{
  position: absolute;
  top: 7px;
  height: 32px;
  border-right: 1px solid rgba(255,255,255,0.75);
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.08s ease, outline 0.08s ease, filter 0.08s ease;
}}
.seg:hover {{
  filter: brightness(1.08);
  outline: 2px solid rgba(0, 0, 0, 0.35);
  z-index: 5;
}}
.seg.active {{
  outline: 3px solid #111;
  transform: translateY(-2px);
  z-index: 10;
}}
.seg span {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  white-space: nowrap;
  color: #111;
}}
.correct {{ background: #62b36f; }}
.substitution {{ background: #f28e2b; }}
.deletion {{ background: #d64f4f; }}
.unknown {{ background: #bab0ac; }}
.prediction {{ background: #4c78a8; }}
.ground-truth {{ background: #6f6a55; }}
.gt-timeline {{
  background: #f7f3e6;
}}
.mini-axis {{
  position: relative;
  height: 24px;
  margin-top: 5px;
  border-top: 1px solid #333;
}}
.mini-tick {{ position: absolute; top: -1px; transform: translateX(-50%); }}
.mini-tick-line {{ width: 1px; height: 7px; background: #333; margin: 0 auto; }}
.mini-tick-label {{ margin-top: 2px; font-size: 9px; color: #333; }}
.missing {{ color: #8a1f11; margin: 8px 0; }}
.player {{
  min-width: 1320px;
  display: grid;
  grid-template-columns: 118px 1fr;
  gap: 12px;
  align-items: center;
  margin: 0 0 14px;
}}
.player-label {{
  text-align: right;
  font-weight: 800;
  padding-right: 4px;
}}
.player-panel {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 8px 10px;
  display: grid;
  grid-template-columns: minmax(320px, 520px) 1fr;
  gap: 12px;
  align-items: center;
}}
.now-playing {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: #41464d;
}}
</style>
</head>
<body>
<h1>L2-ARCTIC Alignment Comparison: {utt}</h1>
<div class="legend">
  <div><span class="chip correct-chip"></span>matched by inference</div>
  <div><span class="chip substitution-chip"></span>wrong phone</div>
  <div><span class="chip deletion-chip"></span>not recognized</div>
  <div><span class="chip prediction-chip"></span>inference segmentation</div>
  <div><span class="chip ground-truth-chip"></span>ground truth timestamps</div>
</div>
<div class="player">
  <div class="player-label">Audio</div>
  <div class="player-panel">
    <audio id="audio-player" controls preload="auto" src="{audio_src}"></audio>
    <div id="now-playing" class="now-playing">Click any segment to play that interval.</div>
  </div>
</div>
{tabs}
<script>
(function () {{
  const audio = document.getElementById("audio-player");
  const nowPlaying = document.getElementById("now-playing");
  let activeSegment = null;
  let activeSelection = null;
  let stopAt = null;
  let rafId = null;
  let activeView = document.querySelector(".view-tab.active")?.dataset.viewTarget || "canonical_fa";
  let activeMethod = document.querySelector(".method-tab.active")?.dataset.methodTarget || "uniform";

  function updatePanes() {{
    document.querySelectorAll(".tab-pane").forEach((pane) => {{
      pane.classList.toggle(
        "active",
        pane.dataset.viewPane === activeView && pane.dataset.methodPane === activeMethod
      );
    }});
  }}

  function activateTab(button) {{
    const viewTarget = button.dataset.viewTarget;
    const methodTarget = button.dataset.methodTarget;
    if (!viewTarget && !methodTarget) return;
    if (rafId) cancelAnimationFrame(rafId);
    if (audio) audio.pause();
    stopAt = null;
    clearActive();
    if (viewTarget) activeView = viewTarget;
    if (methodTarget) activeMethod = methodTarget;

    document.querySelectorAll(".view-tab").forEach((item) => {{
      const isActive = item.dataset.viewTarget === activeView;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", isActive ? "true" : "false");
    }});
    document.querySelectorAll(".method-tab").forEach((item) => {{
      const isActive = item.dataset.methodTarget === activeMethod;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", isActive ? "true" : "false");
    }});
    updatePanes();
  }}

  function clearActive() {{
    if (activeSegment) {{
      activeSegment.classList.remove("active");
      activeSegment = null;
    }}
    if (activeSelection) {{
      activeSelection.classList.remove("active");
      activeSelection = null;
    }}
  }}

  function showWaveSelection(segment) {{
    const lane = segment.dataset.lane;
    const pane = segment.closest(".tab-pane") || document;
    const selection = pane.querySelector(`.wave-selection[data-wave-lane="${{lane}}"]`);
    if (!selection) return;
    selection.style.left = segment.style.left;
    selection.style.width = segment.style.width;
    selection.classList.add("active");
    activeSelection = selection;
  }}

  function monitorStop() {{
    if (!audio || stopAt === null) return;
    if (audio.currentTime >= stopAt) {{
      audio.pause();
      stopAt = null;
      clearActive();
      return;
    }}
    rafId = requestAnimationFrame(monitorStop);
  }}

  async function playSegment(segment) {{
      if (!audio || !audio.getAttribute("src")) {{
        nowPlaying.textContent = "No audio source available for this utterance.";
        return;
      }}

      if (rafId) cancelAnimationFrame(rafId);
      audio.pause();
      clearActive();
      activeSegment = segment;
      activeSegment.classList.add("active");
      showWaveSelection(segment);

      const start = Number(segment.dataset.start);
      const end = Number(segment.dataset.end);
      const label = segment.dataset.label || "";
      stopAt = end;
      nowPlaying.textContent = `Playing ${{label}}: ${{start.toFixed(3)}}s - ${{end.toFixed(3)}}s`;

      try {{
        audio.currentTime = Math.max(0, start);
        await audio.play();
        rafId = requestAnimationFrame(monitorStop);
      }} catch (error) {{
        nowPlaying.textContent = `Could not autoplay ${{label}}. Press play once, then click the segment again. (${{error.name || "playback error"}})`;
      }}
  }}

  document.addEventListener("click", (event) => {{
    const tabButton = event.target.closest(".tab-button");
    if (tabButton) {{
      event.preventDefault();
      activateTab(tabButton);
      return;
    }}
    const segment = event.target.closest(".seg");
    if (!segment) return;
    event.preventDefault();
    playSegment(segment);
  }});

  document.addEventListener("keydown", (event) => {{
    if (event.key !== "Enter" && event.key !== " ") return;
    const segment = event.target.closest(".seg");
    if (!segment) return;
    event.preventDefault();
    playSegment(segment);
  }});

  if (nowPlaying) {{
    nowPlaying.textContent = "Interactive playback ready. Click any segment to highlight and play it.";
  }}

  if (audio) {{
    audio.addEventListener("pause", () => {{
      if (stopAt === null) clearActive();
    }});

    audio.addEventListener("error", () => {{
      nowPlaying.textContent = "Audio failed to load. Regenerate with --audio-mode data, or open the HTML in a browser that can access the wav path.";
    }});
  }}
}})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignments",
        nargs="+",
        default=["l2arctic_fa_timestamp_metrics/alignments.jsonl"],
        help="One or more alignment JSONL files. Later files override duplicate model/utt/type records.",
    )
    parser.add_argument("--output-dir", default="l2arctic_fa_timestamp_metrics/example_html")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--utt", nargs="*", default=None)
    parser.add_argument(
        "--audio-mode",
        choices=["data", "file"],
        default="data",
        help="Use embedded base64 audio ('data') or link to the wav path ('file').",
    )
    args = parser.parse_args()

    grouped = load_alignments([Path(path) for path in args.alignments])
    utts = sorted(grouped)
    if args.utt:
        wanted = set(args.utt)
        utts = [utt for utt in utts if utt in wanted]
    if args.limit > 0:
        utts = utts[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, utt in enumerate(utts, start=1):
        out_path = output_dir / f"{utt}_alignment_comparison.html"
        out_path.write_text(
            render_utterance(utt, grouped[utt], audio_mode=args.audio_mode),
            encoding="utf-8",
        )
        print(f"[{idx}/{len(utts)}] saved {out_path}")


if __name__ == "__main__":
    main()
