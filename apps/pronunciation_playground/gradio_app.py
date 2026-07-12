"""Gradio entrypoint for the IF-MDD Hugging Face Space."""

from __future__ import annotations

import json
from html import escape
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:  # Local unit-test environments do not need the HF shim.
    class _SpacesFallback:
        @staticmethod
        def GPU(function):
            return function

    spaces = _SpacesFallback()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.pronunciation_playground.model_matrix import SYSTEMS, build_matrix, option_key
from apps.pronunciation_playground.reference_builder import build_reference
from apps.pronunciation_playground.service import AssessmentEngine
from apps.pronunciation_playground.visualization import plot_assessment


_MODEL_ROOT_CANDIDATES = [
    Path(os.getenv("OTTC_MODEL_ROOT", "")) if os.getenv("OTTC_MODEL_ROOT") else None,
    Path("/data/models"),
    Path("/data"),
]
_BUCKET_REGISTRY = next(
    (root / "bucket_model_registry.json" for root in _MODEL_ROOT_CANDIDATES if root and (root / "bucket_model_registry.json").exists()),
    None,
)
REGISTRY = _BUCKET_REGISTRY or ROOT / "scripts/public_space_model_registry.json"
MATRIX = build_matrix(ROOT)
ENGINE = AssessmentEngine(
    registry=REGISTRY,
    device=os.getenv("PRONUNCIATION_DEVICE", "cpu"),
    default_model="ipa_ottc",
)

CSS = """
body, .gradio-container { background: #ffffff !important; color: #494e52 !important; font-family: 'Trebuchet MS', Helvetica, sans-serif !important; }
.gradio-container { max-width: none !important; width: 100% !important; margin: 0 !important; padding: 0 28px 28px !important; }
.hero { padding: 28px 6px 20px; border-bottom: 1px solid #f2f3f3; }
.hero h1 { color: #333; font-size: 30px; margin: 0 0 6px; }
.hero p { color: #7a8288; margin: 0; font-size: 15px; }
.section-card { border: 1px solid #e3e3e3 !important; border-radius: 4px !important; background: #fff !important; padding: 18px !important; box-shadow: 0 1px 3px rgba(189,193,196,.18); }
.section-title { color: #224b8d; font-weight: 700; font-size: 17px; margin-bottom: 8px; }
.status-card { background: #e9edf4; border-left: 4px solid #224b8d; padding: 12px 16px; border-radius: 3px; }
.workspace-row { align-items: stretch !important; min-height: calc(100vh - 150px); }
.workspace-row > .gr-column { min-height: calc(100vh - 150px) !important; height: auto !important; }
.workspace-row > .gr-column > .section-card { height: 100% !important; box-sizing: border-box; }
.left-stack { gap: 18px !important; }
.left-stack > .section-card { box-sizing: border-box; }
.left-stack > .section-card:last-child { flex: 1 1 auto !important; }
.assessment { height: 100%; overflow: auto; }
.assessment-head { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 16px; }
.metric { flex: 1 1 120px; min-width: 110px; border: 1px solid #e1e5ea; border-radius: 5px; padding: 12px 14px; background: #fafbfc; }
.metric-label { color: #7a8288; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.metric-value { color: #224b8d; font-size: 24px; font-weight: 700; margin-top: 3px; }
.assessment h3 { color: #333; font-size: 15px; margin: 16px 0 8px; }
.phone-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.phone-table th { color: #7a8288; font-weight: 600; text-align: left; border-bottom: 1px solid #dfe3e7; padding: 7px 6px; }
.phone-table td { border-bottom: 1px solid #edf0f2; padding: 7px 6px; }
.phone-table .ok { color: #16803c; }
.phone-table .bad { color: #b42318; font-weight: 600; }
.feedback { margin: 0; padding-left: 20px; color: #555; }
.muted { color: #7a8288; font-size: 13px; }
button.primary { background: #224b8d !important; border-color: #224b8d !important; }
button.primary:hover { background: #1b3c71 !important; }
button.secondary { color: #224b8d !important; border: 1px solid #bdc1c4 !important; background: #fff !important; }
footer { display: none !important; }
"""


def _choice(system: str, label_space: str, model_type: str) -> str:
    return option_key(system, label_space, model_type)


def describe(system: str, label_space: str, model_type: str) -> str:
    item = MATRIX[_choice(system, label_space, model_type)]
    state = "可运行" if item.status == "ready" else "待部署打包"
    return f"**{item.display}** · {state}\n\n{item.note}"


def render_assessment(result: dict) -> str:
    """Turn the machine-readable response into a readable pronunciation report."""
    model = result.get("model", {})
    score = float(result.get("utterance_score", 0.0))
    rows = []
    for index, phone in enumerate(result.get("phones", []), 1):
        error = phone.get("error")
        state = "✓" if not error else escape(str(error))
        state_class = "ok" if not error else "bad"
        rows.append(
            "<tr>"
            f"<td>{index}</td><td><b>{escape(str(phone.get('canonical', '')))}</b></td>"
            f"<td>{escape(str(phone.get('predicted') or '—'))}</td>"
            f"<td>{float(phone.get('score', 0.0)):.1f}</td>"
            f"<td class='{state_class}'>{state}</td>"
            "</tr>"
        )
    table = (
        "<table class='phone-table'><thead><tr><th>#</th><th>Reference</th>"
        "<th>Recognized</th><th>Score</th><th>Status</th></tr></thead><tbody>"
        + ("".join(rows) or "<tr><td colspan='5'>No phone-level result.</td></tr>")
        + "</tbody></table>"
    )
    feedback = result.get("feedback", [])
    feedback_html = (
        "<ul class='feedback'>" + "".join(f"<li>{escape(str(item))}</li>" for item in feedback) + "</ul>"
        if feedback else "<p class='muted'>No pronunciation issues were highlighted.</p>"
    )
    model_name = escape(str(model.get("display", model.get("id", ""))))
    transcript = escape(str(result.get("transcript") or "—"))
    return (
        "<div class='assessment'><div class='assessment-head'>"
        f"<div class='metric'><div class='metric-label'>Overall score</div><div class='metric-value'>{score:.1f}</div></div>"
        f"<div class='metric'><div class='metric-label'>Phone errors</div><div class='metric-value'>{int(result.get('error_count', 0))}</div></div>"
        f"<div class='metric'><div class='metric-label'>Duration</div><div class='metric-value'>{float(result.get('duration_s', 0.0)):.2f}s</div></div>"
        "</div>"
        f"<p class='muted'><b>Model:</b> {model_name} &nbsp;·&nbsp; <b>Transcript:</b> {transcript}</p>"
        + "<h3>Phone-by-phone assessment</h3>" + table
        + "<h3>Feedback</h3>" + feedback_html + "</div>"
    )


def prepare_reference(audio, reference_source: str, transcript: str, label_space: str, whisper_model: str):
    """Generate editable reference phones from transcript or Whisper words."""
    try:
        result = build_reference(
            audio=Path(audio) if audio else None,
            source=reference_source,
            transcript=transcript or "",
            label_space=label_space,
            whisper_model=whisper_model,
            device=os.getenv("PRONUNCIATION_DEVICE", "cpu"),
        )
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    status = f"Reference source: `{result.source}` · {len(result.phones)} phones · {len(result.words)} words"
    return result.transcript, " ".join(result.phones), result.words, status


@spaces.GPU
def analyze(audio, reference_phones: str, transcript: str, words, system: str, label_space: str, model_type: str):
    item = MATRIX[_choice(system, label_space, model_type)]
    if item.status != "ready" or not item.model_id:
        raise gr.Error(f"{item.display} 当前还没有可部署的 inference bundle：{item.note}")
    if not audio:
        raise gr.Error("请先上传或录制音频。")
    phones = [token for token in str(reference_phones or "").replace(",", " ").split() if token]
    if not phones:
        raise gr.Error("请提供参考音素序列，例如：dh ih s ih z.")
    result = ENGINE.assess(
        wav_path=Path(audio),
        canonical_phones=phones,
        transcript=transcript or "",
        words=words if isinstance(words, list) else None,
        model_id=item.model_id,
    )
    figure = plot_assessment(Path(audio), result)
    return render_assessment(result), json.dumps(result, ensure_ascii=False, indent=2), figure


@spaces.GPU
def auto_pipeline(audio, reference_source: str, transcript: str, system: str, label_space: str, model_type: str, whisper_model: str):
    """Run ASR/G2P and acoustic encoding concurrently after recording stops."""
    item = MATRIX[_choice(system, label_space, model_type)]
    if item.status != "ready" or not item.model_id:
        raise gr.Error(f"{item.display} 当前还没有可部署的 inference bundle：{item.note}")
    if not audio:
        raise gr.Error("请先录音。")
    if reference_source == "manual":
        raise gr.Error("自动录音流程需要 Whisper 或 Transcript G2P；手动 phoneme 请点击 Analyze。")
    with ThreadPoolExecutor(max_workers=2) as pool:
        reference_future = pool.submit(
            build_reference,
            audio=Path(audio), source=reference_source, transcript=transcript or "",
            label_space=label_space, whisper_model=whisper_model,
            device=os.getenv("PRONUNCIATION_DEVICE", "cpu"),
        )
        acoustic_future = pool.submit(ENGINE.encode_audio, wav_path=Path(audio), model_id=item.model_id)
        reference_result = reference_future.result()
        encoded = acoustic_future.result()
    if not reference_result.phones:
        raise gr.Error("没有生成 canonical phones，请检查 transcript 或 Whisper 输出。")
    result = ENGINE.assess_encoded(
        encoded, canonical_phones=reference_result.phones,
        transcript=reference_result.transcript, words=reference_result.words,
        model_id=item.model_id,
    )
    figure = plot_assessment(Path(audio), result)
    status = f"自动完成：{reference_result.source} · {len(reference_result.words)} words · {len(reference_result.phones)} phones"
    return (
        reference_result.transcript, " ".join(reference_result.phones), reference_result.words,
        status, render_assessment(result), json.dumps(result, ensure_ascii=False, indent=2), figure,
    )


with gr.Blocks(title="IF-MDD Pronunciation Space", css=CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown("<div class='hero'><h1>IF-MDD Pronunciation Assessment</h1><p>Frame-level acoustic evidence for L2 speech, with CTC, OTTC and subphonetic topology.</p></div>")
    with gr.Row(equal_height=True, elem_classes="workspace-row"):
        with gr.Column(scale=1.0, min_width=520, elem_classes="left-stack"):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown("Model configuration", elem_classes="section-title")
                system = gr.Dropdown(list(SYSTEMS), value="mdd_specific", label="System")
                label_space = gr.Dropdown(["ipa", "arpabet"], value="arpabet", label="Label space")
                model_type = gr.Dropdown(["ctc", "ottc", "subphonetic_ottc"], value="ottc", label="Model")
                status = gr.Markdown(describe("mdd_specific", "arpabet", "ottc"), elem_classes="status-card")

            with gr.Group(elem_classes="section-card"):
                gr.Markdown("Record and reference", elem_classes="section-title")
                audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Audio")
                reference_source = gr.Radio(
                    choices=[("Whisper → words → G2P", "whisper_g2p"), ("Transcript → G2P", "transcript_g2p"), ("Manual phones", "manual")],
                    value="whisper_g2p", label="Reference source",
                )
                whisper_model = gr.Dropdown(["base.en", "small.en", "medium.en"], value="small.en", label="Whisper model")
                transcript = gr.Textbox(label="Transcript / recognized words", placeholder="After recording, Whisper will fill this automatically.")
                build_reference_button = gr.Button("Generate reference phones", variant="secondary")
                reference = gr.Textbox(label="Reference phones (editable)", placeholder="ARPAbet: DH IH S IH Z · IPA: ð ɪ s ɪ z")
                word_metadata = gr.JSON(value=[], label="Recognized word metadata", visible=False)
                reference_status = gr.Markdown("录音停止后会自动执行 Whisper/G2P 和 acoustic inference。", elem_classes="status-card")
                run = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1.0, min_width=520, elem_classes="section-card"):
            gr.Markdown("Segmentation and acoustic evidence", elem_classes="section-title")
            plot = gr.Plot(label="Spectrogram / segmentation")
            result = gr.HTML("<p class='muted'>Record audio and run an analysis to see the pronunciation report.</p>", label="Assessment")
            with gr.Accordion("Raw JSON", open=False):
                raw = gr.Code(language="json")

    system.change(describe, [system, label_space, model_type], status)
    label_space.change(describe, [system, label_space, model_type], status)
    model_type.change(describe, [system, label_space, model_type], status)
    build_reference_button.click(
        prepare_reference,
        [audio, reference_source, transcript, label_space, whisper_model],
        [transcript, reference, word_metadata, reference_status],
    )
    run.click(analyze, [audio, reference, transcript, word_metadata, system, label_space, model_type], [result, raw, plot])
    audio.stop_recording(
        auto_pipeline,
        [audio, reference_source, transcript, system, label_space, model_type, whisper_model],
        [transcript, reference, word_metadata, reference_status, result, raw, plot],
    )
    # Uploaded files should follow the same automatic path as microphone
    # recordings; no Analyze click is needed in either case.
    audio.upload(
        auto_pipeline,
        [audio, reference_source, transcript, system, label_space, model_type, whisper_model],
        [transcript, reference, word_metadata, reference_status, result, raw, plot],
    )


if __name__ == "__main__":
    demo.launch()
