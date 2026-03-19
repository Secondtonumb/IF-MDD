import sys
from pathlib import Path

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR
from trainer.MyEncoderASR import MyCTCPrefixBeamSearcher
from trainer.MyEncoderASR import MyCTCBeamSearcher

from speechbrain.decoders.ctc import TorchAudioCTCPrefixBeamSearcher, CTCPrefixBeamSearcher, CTCBeamSearcher

from trainer.MyEncoderASR import plot_alignments
import torch
import numpy as np
import tgt
import itertools

# file_id = "arctic_b0503"
# file_id= "DOS_F01_S6_001"
# file_id = "0002-0000"
# file_path = "/home/m64000/work/dataset/data_iqra/test/wav/00000_00001.wav"
file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_94.wav"

json_path = "/home/m64000/work/dataset/data_iqra_extra_is26/iqra_extra_is26_test_aligned.json"
with open(json_path, 'r') as f:
    import json
    data = json.load(f)

canonical_label = data[file_path]['canonical_aligned']
perceived_label = data[file_path]['perceived_aligned']
perceived_train_target = data[file_path]['perceived_train_target']
import pdb; pdb.set_trace()
# file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_225.wav"
file_id = Path(file_path).stem

# # Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Simple Transcribe
# asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD", hparams_file="inference.yaml")
# asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTCwithLP", hparams_file="inference.yaml")
# K3 CROTTC
asr_model_path  = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/crottc_k3_RoPE_TTS_FT.ckpt"
# # K7 CROTTC
# # asr_model_path  = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/ottc_k7_RoPE_TTS_FT"
# # CTCLP
# # asr_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_ctclp/save/CKPT+best_per_099_7.8231.ckpt"
# # CTC
# asr_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CTC"
# # asr_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CTC_best_k3"
# # OTTC
# asr_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/OTTC"
# # CRCTC
# asr_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CRCTC"
asr_model = MyEncoderASR.from_hparams(source=asr_model_path, hparams_file="inference.yaml")
x = asr_model.transcribe_file(file_path)
x_id = asr_model.tokenizer.encode_sequence(x.split(' '))

canonical_label_ids = asr_model.tokenizer.encode_sequence(canonical_label.strip().split())
perceived_label_ids = asr_model.tokenizer.encode_sequence(perceived_label.strip().split())
perceived_train_target_ids = asr_model.tokenizer.encode_sequence(perceived_train_target.strip().split())

# print(x)
# sil f ao r dh ah t w eh n t iy th t ay m sil dh ae t sil iy v n ih n sil d iy t uw m eh n sh uw k sil hh ae n s sil

# Get CTC Probabililty
# waveform = asr_model.load_audio(f"examples/{file_id}.wav")
try:
    waveform, sample_rate = asr_model.load_audio(f"{file_path}")
except:
    waveform = asr_model.load_audio(f"{file_path}")
batch = waveform.unsqueeze(0)
rel_length = torch.tensor([1.0])

ctc_p = asr_model.encode_batch(batch, rel_length)
# print(ctc_p.shape)
# torch.Size([1, 221, 44])

# CTC Tokens
ctc_id = ctc_p.argmax(-1)

# Get verbose CTC output
# import pdb; pdb.set_trace()
searcher = MyCTCPrefixBeamSearcher(
    tokens=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    sil_index=asr_model.tokenizer.lab2ind["<sil>"],
)

# import pdb; pdb.set_trace()
hyps = searcher(ctc_p, rel_length)

s = TorchAudioCTCPrefixBeamSearcher(
    tokens=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    sil_index=asr_model.tokenizer.lab2ind["<sil>"],
)

hyps_ = s(ctc_p, rel_length)
from torchaudio.functional import forced_align, merge_tokens

forced_alignments, scores = forced_align(
    log_probs=ctc_p,
    targets=torch.tensor([x_id], dtype=torch.int32, device=ctc_p.device),
    input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32, device=ctc_p.device),
    target_lengths=torch.tensor([len(x_id)], dtype=torch.int32, device=ctc_p.device),
    blank=asr_model.tokenizer.lab2ind["<blank>"]
)
# import pdb; import pdb; pdb.set_trace()

forced_alignments = forced_alignments[0]
scores = scores[0].exp()
aligned_tokens_gt = merge_tokens(forced_alignments, scores)

from utils.align import monitor_alignment, plot_scores, plot_alignment_comparison

# Create spectrogram from CTC log probabilities
import matplotlib.pyplot as plt
import numpy as np

# ===== VISUALIZATION MODE =====
# Set to True for overlaid view (log probability + spectrogram)
# Set to False for side-by-side view (separate subplots)
OVERLAY_MODE = True  # Change this to switch between modes
# ==========================

if OVERLAY_MODE:
    # Overlaid mode: single axis with both spectrogram and log probability
    fig, ax2 = plt.subplots(figsize=(14, 5))
    ax1 = None
else:
    # Side-by-side mode: two subplots stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)

# Top: CTC scores with alignment (only if not in overlay mode)
if not OVERLAY_MODE:
    span_xs, span_hs = [], []
    ax1.axvspan(aligned_tokens_gt[0].start - 0.05, aligned_tokens_gt[-1].end + 0.05, 
                facecolor="paleturquoise", edgecolor="none", zorder=-1)
    for span in aligned_tokens_gt:
        for t in range(span.start, span.end):
            span_xs.append(t + 0.5)
            span_hs.append(scores[t, span.token].item() if scores.ndim > 1 else scores[t].item())
        # Calculate span center and max height for label placement
        span_center = (span.start + span.end) / 2
        span_max_height = max(scores[t, span.token].item() if scores.ndim > 1 else scores[t].item() 
                              for t in range(span.start, span.end))
        ax1.annotate(asr_model.tokenizer.decode_ndim(span.token), 
                    (span_center, span_max_height / 2),
                    ha='center', va='center', fontsize=9, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        ax1.axvspan(span.start - 0.05, span.end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
        # Add vertical dashed line at segment boundaries
        ax1.axvline(span.start, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax1.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral")
    ax1.set_title("Frame-level CTC scores and phoneme segments")
    ax1.set_ylabel('Confidence Score')
    ax1.set_ylim(-0.1, None)
    ax1.set_xlim(-1, len(ctc_p[0]))
    ax1.grid(True, axis="y")
    ax1.axhline(0, color="black")
    ax1.tick_params(labelbottom=False)  # Hide x-axis labels for top plot

# Bottom: Mel Spectrogram with alignment boundaries
import librosa
import librosa.display

# Compute mel spectrogram
sample_rate = 16000  # Assuming 16kHz sample rate
mel_spec = librosa.feature.melspectrogram(y=waveform.cpu().numpy(), sr=sample_rate, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Get CTC log probability and normalize
log_probs_np = ctc_p[0].detach().cpu().numpy()  # (time_steps, vocab_size)
# Max pool over vocabulary dimension to get per-frame confidence
ctc_confidence = np.max(log_probs_np, axis=1, keepdims=True)  # (time_steps, 1)

# Resample mel spectrogram to match CTC frame count if needed
from scipy.interpolate import interp1d
ctc_n_frames = len(ctc_p[0])
mel_n_frames = mel_spec_db.shape[1]

# Interpolate mel spectrogram to CTC frame resolution
if mel_n_frames != ctc_n_frames:
    mel_spec_db_resampled = np.zeros((mel_spec_db.shape[0], ctc_n_frames))
    for i in range(mel_spec_db.shape[0]):
        f = interp1d(np.linspace(0, 1, mel_n_frames), mel_spec_db[i, :], kind='linear', fill_value='extrapolate')
        mel_spec_db_resampled[i, :] = f(np.linspace(0, 1, ctc_n_frames))
    mel_spec_db = mel_spec_db_resampled

# Resample CTC confidence to match mel spectrogram dimensions
ctc_confidence_resampled = np.zeros((ctc_n_frames,))
for i in range(ctc_n_frames):
    if i < len(ctc_confidence):
        ctc_confidence_resampled[i] = ctc_confidence[i, 0]
    else:
        ctc_confidence_resampled[i] = ctc_confidence[-1, 0]
ctc_confidence = ctc_confidence_resampled.reshape(-1, 1)

# Repeat to match mel spectrogram height for overlay
ctc_confidence_expanded = np.repeat(ctc_confidence, 80, axis=1)  # (time_steps, 80)
ctc_confidence_db = 20 * np.log10(np.maximum(ctc_confidence_expanded, 1e-10))

# Normalize both to similar scale for visualization
mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10) * 100
ctc_confidence_db_norm = (ctc_confidence_db - ctc_confidence_db.min()) / (np.maximum(ctc_confidence_db.max() - ctc_confidence_db.min(), 1e-10)) * 100

# Adjust transparency based on mode
if OVERLAY_MODE:
    mel_alpha = 1.0  # Spectrogram more visible
    im1 = ax2.imshow(mel_spec_db_norm, aspect='auto', origin='lower', cmap='viridis', 
                      extent=[0, len(ctc_p[0]), 0, 80], alpha=mel_alpha, zorder=1)
    title = 'Mel Spectrogram + CTC Scores (Probability) - Overlaid'
    
    # Draw probability bars based on aligned tokens
    span_xs, span_hs = [], []
    max_score = 0
    for span in aligned_tokens_gt:
        for t in range(span.start, span.end):
            span_xs.append(t + 0.5)
            score_val = scores[t, span.token].item() if scores.ndim > 1 else scores[t].item()
            span_hs.append(score_val)
            max_score = max(max_score, score_val)
    
    # Scale probability bars to fit within mel spectrogram height (0-80)
    if max_score > 0:
        span_hs_scaled = [h / max_score * 80 for h in span_hs]  # Scale to 0-40 for visibility
    else:
        span_hs_scaled = span_hs
    
    # Plot probability bars with semi-transparency and hatch pattern
    ax2.bar(span_xs, span_hs_scaled, 
        color="white",       # 填充色用白色，在深色语谱图上最透明
        edgecolor="gold",  # 边框用对比色（橙色/金色），强调识别重点
        alpha=0.5,           # 全局透明度设低
        linewidth=0.8,       # 细边框
        hatch='/',         # 增加斜纹，有助于在黑白打印或复杂背景下分辨
        zorder=3)            # 确保在语谱图（zorder通常为1）之上
    
    # Add colorbar for probability scale
    # cbar = plt.colorbar(im1, ax=ax2, label='CTC Score (Probability 0~1)', pad=0.02)
    # add probablity y axis label on the right side
    ax2_secondary = ax2.twinx()
    ax2_secondary.set_ylabel('CTC Score (Probability 0~1)')
    # make span_hs_scaled range from 0 to 1
    # ax2_secondary.set_yticks([0, 10, 20, 30, 40])
    ax2_secondary.set_yticks([0, 20, 40, 60, 80])
    ax2_secondary.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
    
    ax2_secondary.tick_params(axis='y') 
else:
    # Side-by-side mode: both fully opaque
    im1 = ax2.imshow(mel_spec_db_norm, aspect='auto', origin='lower', cmap='viridis', 
                      extent=[0, len(ctc_p[0]), 0, 80], alpha=0.8, zorder=1)
    im2 = ax2.imshow(ctc_confidence_db_norm, aspect='auto', origin='lower', cmap='hot', 
                      extent=[0, len(ctc_p[0]), 0, 80], alpha=0.5, zorder=2)
    title = 'Mel Spectrogram (Viridis) + CTC Confidence (Hot)'

ax2.set_ylabel('Mel Frequency Bin')
ax2.set_xlabel('Frame (Timeline)')
ax2.set_title(title, pad=-30 if not OVERLAY_MODE else 20, loc='center')
ax2.set_xlim(-1, len(ctc_p[0]))

# Draw alignment boundaries and segments
for i, span in enumerate(aligned_tokens_gt):
    # Add semi-transparent background for segments (alternating colors)
    if i % 2 == 0:
        ax2.axvspan(span.start - 0.05, span.end + 0.05, facecolor="cyan", alpha=0.1, zorder=0)
    else:
        ax2.axvspan(span.start - 0.05, span.end + 0.05, facecolor="yellow", alpha=0.1, zorder=0)
    
    # Add boundary lines
    # ax2.axvline(span.start, color='black', linestyle='--', alpha=0.9, linewidth=1.5)
    ax2.text(span.start + (span.end - span.start) / 2, 75, 
             asr_model.tokenizer.decode_ndim(span.token), 
             ha='center', color='white', fontsize=8, weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

# Align x-axis (only if ax1 exists)
if ax1 is not None:
    ax1.set_xlim(-1, len(ctc_p[0]))
else:
    ax2.set_xlim(-1, len(ctc_p[0]))

fig.suptitle(f"GT Alignment with Spectrogram for {file_id}", fontsize=14, weight='bold')
plt.tight_layout()

asr_model_name = Path(asr_model_path).stem
asr_model_parent = Path(asr_model_path).parent.name
output_dir = Path(f"{asr_model_parent}_{asr_model_name}_inference_outputs_{file_id}")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / f"{file_id}_gt_alignment_with_spectrogram.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved GT alignment with spectrogram to {output_dir / f'{file_id}_gt_alignment_with_spectrogram.png'}")