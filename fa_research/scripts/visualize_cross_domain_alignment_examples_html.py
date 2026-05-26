#!/usr/bin/env python3
"""Dependency-free HTML examples for cross-domain JSON FA alignments."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_ORDER = [
    "L2ARCTIC_CTC",
    "L2ARCTIC_CRCTC",
    "L2ARCTIC_OTTC",
    "L2ARCTIC_CROTTC",
    "LIBRISPEECH_CTC",
    "LIBRISPEECH_CRCTC",
    "LIBRISPEECH_OTTC",
    "LIBRISPEECH_CROTTC",
]


def load_alignments(path: Path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            grouped[record["utt"]].append(record)
    return grouped


def record_duration(records: list[dict]) -> float:
    duration = 0.0
    for record in records:
        for key in ("ref_ends", "pred_ends"):
            values = record.get(key) or []
            if values:
                duration = max(duration, max(float(x) for x in values))
    return duration or 1.0


def record_wav_path(records: list[dict]) -> str | None:
    for record in records:
        if record.get("wav"):
            return record["wav"]
    return None


def audio_source(records: list[dict], audio_mode: str) -> str:
    wav_path = record_wav_path(records)
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


def read_audio_samples(wav_path: str):
    try:
        import librosa

        samples, sample_rate = librosa.load(wav_path, sr=None, mono=True)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return None
        duration = samples.size / float(sample_rate)
        return sample_rate, duration, np.clip(samples, -1.0, 1.0)
    except Exception:
        pass

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
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        max_abs = 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        max_abs = 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        max_abs = 2147483648.0
    else:
        return None

    if samples.size == 0:
        return None

    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    samples = np.clip(samples / max_abs, -1.0, 1.0)
    duration = samples.size / float(sample_rate)
    return sample_rate, duration, samples


def spectrogram_matrix(samples: np.ndarray, sample_rate: int, target_frames: int = 300, freq_bins: int = 72):
    n_fft = 512
    hop = max(80, int(max(1, samples.size - n_fft) / max(1, target_frames - 1)))
    try:
        import librosa

        spec = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop, window="hann", center=True))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        spec = (spec + 80.0) / 80.0
        spec = np.clip(spec, 0.0, 1.0)
    except Exception:
        if samples.size < n_fft:
            samples = np.pad(samples, (0, n_fft - samples.size))
        starts = list(range(0, max(1, samples.size - n_fft + 1), hop))
        if not starts:
            starts = [0]
        window = np.hanning(n_fft).astype(np.float32)
        columns = []
        for start in starts[:target_frames]:
            frame = samples[start : start + n_fft]
            if frame.size < n_fft:
                frame = np.pad(frame, (0, n_fft - frame.size))
            mag = np.abs(np.fft.rfft(frame * window))
            columns.append(mag)
        spec = np.stack(columns, axis=1)
        spec = np.log1p(spec)
        lo = float(np.percentile(spec, 8))
        hi = float(np.percentile(spec, 99))
        if hi <= lo:
            hi = lo + 1e-6
        spec = np.clip((spec - lo) / (hi - lo), 0.0, 1.0)

    if spec.shape[1] > target_frames:
        idx = np.linspace(0, spec.shape[1] - 1, target_frames).astype(int)
        spec = spec[:, idx]

    # Collapse FFT bins to a compact display height.
    if spec.shape[0] > freq_bins:
        edges = np.linspace(0, spec.shape[0], freq_bins + 1).astype(int)
        reduced = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            hi = max(hi, lo + 1)
            reduced.append(spec[lo:hi].mean(axis=0))
        spec = np.stack(reduced, axis=0)

    return spec


def heat_color(value: float) -> str:
    # Compact blue-to-yellow palette, tuned for small SVG spectrogram cells.
    stops = [
        (18, 32, 73),
        (33, 102, 172),
        (67, 147, 195),
        (171, 217, 233),
        (255, 232, 120),
    ]
    value = max(0.0, min(1.0, value))
    pos = value * (len(stops) - 1)
    idx = min(len(stops) - 2, int(pos))
    frac = pos - idx
    a = stops[idx]
    b = stops[idx + 1]
    rgb = [round(a[i] + (b[i] - a[i]) * frac) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_waveform(records: list[dict], duration: float) -> str:
    wav_path = record_wav_path(records)
    if not wav_path:
        return '<div class="wave-panel">No spectrogram path found.</div>'

    data = read_audio_samples(wav_path)
    if data is None:
        return f'<div class="wave-panel">Spectrogram unavailable for <code>{html.escape(wav_path)}</code>.</div>'

    sample_rate, wav_duration, samples = data
    duration = max(duration, wav_duration)
    width = 1200
    height = 150
    spec = spectrogram_matrix(samples, sample_rate)
    bins, frames = spec.shape
    cell_w = width / frames
    cell_h = height / bins
    rects = []
    for frame_idx in range(frames):
        x = frame_idx * cell_w
        for bin_idx in range(bins):
            y = height - (bin_idx + 1) * cell_h
            color = heat_color(float(spec[bin_idx, frame_idx]))
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.25:.2f}" '
                f'height="{cell_h + 0.25:.2f}" fill="{color}"/>'
            )

    tick_lines = []
    for tick in tick_marks(duration):
        x = 100.0 * tick / duration
        tick_lines.append(f'<line class="wave-tick" x1="{x:.4f}%" y1="0" x2="{x:.4f}%" y2="{height}" />')

    return (
        '<div class="wave-panel">'
        '<div class="wave-box">'
        f'<svg class="spectrogram" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        '<g class="spec-cells">'
        + "\n".join(rects)
        + '</g>'
        + "\n".join(tick_lines)
        + '</svg>'
        '<div id="wave-selection" class="wave-selection"></div>'
        '</div>'
        + render_axis(duration)
        + '</div>'
    )


def tick_marks(duration: float):
    step = 0.25 if duration <= 3 else 0.5 if duration <= 8 else 1.0
    ticks = []
    value = 0.0
    while value <= duration + 1e-9:
        ticks.append(round(value, 2))
        value += step
    return ticks


def render_axis(duration: float) -> str:
    parts = ['<div class="axis">']
    for tick in tick_marks(duration):
        left = 100 * tick / duration
        parts.append(
            f'<div class="tick" style="left:{left:.4f}%">'
            f'<div class="tick-line"></div><div class="tick-label">{tick:g}</div></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def display_phones(record: dict) -> list[str]:
    labels = record.get("display_phones") or record.get("model_labels") or record.get("phones") or []
    return [str(label) for label in labels]


def render_segments(record: dict, starts_key: str, ends_key: str, duration: float, lane: str) -> str:
    phones = display_phones(record)
    starts = [float(x) for x in record.get(starts_key, [])]
    ends = [float(x) for x in record.get(ends_key, [])]
    scores = [float(x) for x in record.get("scores", [])]
    pieces = [f'<div class="timeline {lane}">']
    for idx, phone in enumerate(phones):
        if idx >= len(starts) or idx >= len(ends):
            continue
        start = starts[idx]
        end = ends[idx]
        left = 100.0 * start / duration
        width = max(0.12, 100.0 * max(0.0, end - start) / duration)
        score = f" score={scores[idx]:.3f}" if lane == "pred" and idx < len(scores) else ""
        tooltip = html.escape(f"{phone}: {start:.3f}-{end:.3f}s{score}")
        label = html.escape(str(phone))
        pieces.append(
            f'<div class="seg" style="left:{left:.4f}%;width:{width:.4f}%;" '
            f'title="{tooltip}" data-start="{start:.6f}" data-end="{end:.6f}" '
            f'data-label="{label}" role="button" tabindex="0"><span>{label}</span></div>'
        )
    pieces.append("</div>")
    return "\n".join(pieces)


def model_sort_key(record: dict):
    model = record.get("model", "")
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def render_record(record: dict, duration: float) -> str:
    model = html.escape(record.get("model", ""))
    dataset = html.escape(record.get("dataset", ""))
    target = html.escape(record.get("target", ""))
    phones = html.escape(" ".join(display_phones(record)))
    return f"""
<section class="model-card">
  <h2>{model}</h2>
  <div class="meta">{dataset} / {target} / {html.escape(record.get("wav", ""))}</div>
  <div class="phones">{phones}</div>
  <div class="lane-label">Reference</div>
  {render_segments(record, "ref_starts", "ref_ends", duration, "ref")}
  {render_axis(duration)}
  <div class="lane-label pred-label">Prediction</div>
  {render_segments(record, "pred_starts", "pred_ends", duration, "pred")}
  {render_axis(duration)}
</section>
"""


def render_utterance(utt: str, records: list[dict], audio_mode: str) -> str:
    records = sorted(records, key=model_sort_key)
    duration = record_duration(records)
    rows = "\n".join(render_record(record, duration) for record in records)
    audio_src = html.escape(audio_source(records, audio_mode), quote=True)
    return TEMPLATE.format(
        title=html.escape(utt),
        utt=html.escape(utt),
        audio_src=audio_src,
        duration=f"{duration:.3f}",
        waveform=render_waveform(records, duration),
        rows=rows,
    )


TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title} alignment examples</title>
<style>
body {{
  margin: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f8fa;
  color: #202124;
}}
h1 {{ margin: 0 0 8px; font-size: 23px; }}
.subtitle {{ margin: 0 0 16px; color: #5f6368; }}
.player {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 16px;
  display: grid;
  grid-template-columns: minmax(320px, 520px) 1fr;
  gap: 14px;
  align-items: center;
}}
.wave-panel {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 16px;
  min-width: 980px;
}}
.wave-box {{
  position: relative;
  height: 126px;
}}
.spectrogram {{
  width: 100%;
  height: 126px;
  display: block;
  background: #f4f6f8;
  border: 1px solid #e1e5ea;
  border-radius: 6px;
}}
.wave-tick {{
  stroke: #ffffff;
  stroke-width: 0.7;
  opacity: 0.28;
}}
.wave-selection {{
  display: none;
  position: absolute;
  top: 0;
  height: 126px;
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
.now-playing {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: #41464d;
}}
.model-card {{
  background: white;
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 13px 14px 16px;
  margin: 14px 0;
  min-width: 980px;
}}
h2 {{ margin: 0 0 4px; font-size: 17px; }}
.meta {{ color: #68707a; font-size: 12px; margin-bottom: 8px; }}
.phones {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  line-height: 1.35;
  color: #3d4249;
  max-height: 44px;
  overflow: auto;
  margin-bottom: 10px;
}}
.lane-label {{
  font-size: 12px;
  font-weight: 750;
  margin: 8px 0 5px;
}}
.pred-label {{ margin-top: 12px; }}
.timeline {{
  position: relative;
  height: 42px;
  border: 1px solid #d8dde6;
  border-radius: 6px;
  overflow: hidden;
  background: #f0f2f5;
}}
.timeline.ref {{ background: #eef6ff; }}
.timeline.pred {{ background: #fff5e6; }}
.seg {{
  position: absolute;
  top: 6px;
  height: 30px;
  box-sizing: border-box;
  border-right: 1px solid rgba(255,255,255,0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  cursor: pointer;
}}
.ref .seg {{ background: #4c78a8; }}
.pred .seg {{ background: #f28e2b; }}
.seg:hover, .seg.active {{
  outline: 3px solid #111;
  z-index: 8;
  filter: brightness(1.08);
}}
.seg span {{
  color: #111;
  font-size: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: nowrap;
}}
.axis {{
  position: relative;
  height: 23px;
  border-top: 1px solid #333;
  margin: 5px 0 0;
}}
.tick {{ position: absolute; top: -1px; transform: translateX(-50%); }}
.tick-line {{ width: 1px; height: 7px; background: #333; margin: 0 auto; }}
.tick-label {{ margin-top: 2px; font-size: 9px; color: #333; }}
</style>
</head>
<body>
<h1>Alignment Example: {utt}</h1>
<div class="subtitle">Duration: {duration}s. Reference and prediction are drawn with the same phone sequence.</div>
<div class="player">
  <audio id="audio-player" controls preload="auto" src="{audio_src}"></audio>
  <div id="now-playing" class="now-playing">Click a segment to play that interval.</div>
</div>
{waveform}
{rows}
<script>
(function () {{
  const audio = document.getElementById("audio-player");
  const nowPlaying = document.getElementById("now-playing");
  let active = null;
  let stopAt = null;
  let rafId = null;

  function clearActive() {{
    if (active) active.classList.remove("active");
    active = null;
  }}

  function showWaveSelection(segment) {{
    const selection = document.getElementById("wave-selection");
    if (!selection) return;
    selection.style.left = segment.style.left;
    selection.style.width = segment.style.width;
    selection.classList.add("active");
  }}

  function monitorStop() {{
    if (!audio || stopAt === null) return;
    if (audio.currentTime >= stopAt) {{
      audio.pause();
      stopAt = null;
      return;
    }}
    rafId = requestAnimationFrame(monitorStop);
  }}

  async function playSegment(segment) {{
    if (!audio || !audio.getAttribute("src")) {{
      nowPlaying.textContent = "No audio source available.";
      return;
    }}
    if (rafId) cancelAnimationFrame(rafId);
    audio.pause();
    clearActive();
    active = segment;
    active.classList.add("active");
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
      nowPlaying.textContent = `Could not autoplay. Press play once, then click again. (${{error.name || "playback error"}})`;
    }}
  }}

  document.addEventListener("click", (event) => {{
    const segment = event.target.closest(".seg");
    if (!segment) return;
    event.preventDefault();
    playSegment(segment);
  }});
}})();
</script>
</body>
</html>
"""


def choose_utts(grouped: dict, requested: list[str] | None, limit: int) -> list[str]:
    if requested:
        return [utt for utt in requested if utt in grouped]
    utts = sorted(grouped, key=lambda utt: (-len(grouped[utt]), utt))
    return utts[:limit] if limit > 0 else utts[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", default="fa_research/results/cross_domain_json_fa_k2/alignments.jsonl")
    parser.add_argument("--output-dir", default="fa_research/figures/cross_domain_json_fa_k2/example_html")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--utt", nargs="*", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--audio-mode", choices=["data", "file"], default="data")
    parser.add_argument(
        "--embed-audio",
        action="store_true",
        help="Alias for --audio-mode data: embed wav bytes directly into each HTML file.",
    )
    args = parser.parse_args()
    if args.embed_audio:
        args.audio_mode = "data"

    grouped_all = load_alignments(Path(args.alignments))
    grouped = defaultdict(list)
    wanted_models = set(args.model or [])
    for utt, records in grouped_all.items():
        for record in records:
            if args.dataset and record.get("dataset") != args.dataset:
                continue
            if wanted_models and record.get("model") not in wanted_models:
                continue
            grouped[utt].append(record)

    utts = choose_utts(grouped, args.utt, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, utt in enumerate(utts, start=1):
        out_path = output_dir / f"{utt}_cross_domain_alignment.html"
        out_path.write_text(render_utterance(utt, grouped[utt], args.audio_mode), encoding="utf-8")
        print(f"[{idx}/{len(utts)}] saved {out_path}")


if __name__ == "__main__":
    main()
