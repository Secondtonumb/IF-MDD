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

import pdb; pdb.set_trace()

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


forced_alignments_cano, scores_cano = forced_align(
    log_probs=ctc_p,
    targets=torch.tensor([canonical_label_ids], dtype=torch.int32, device=ctc_p.device),
    input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32, device=ctc_p.device),
    target_lengths=torch.tensor([len(canonical_label_ids)], dtype=torch.int32, device=ctc_p.device),
    blank=asr_model.tokenizer.lab2ind["<blank>"]
)

forced_alignments_perc, scores_perc = forced_align(
    log_probs=ctc_p,
    targets=torch.tensor([perceived_label_ids], dtype=torch.int32, device=ctc_p.device),
    input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32, device=ctc_p.device),
    target_lengths=torch.tensor([len(perceived_label_ids)], dtype=torch.int32, device=ctc_p.device),
    blank=asr_model.tokenizer.lab2ind["<blank>"]
)
# import pdb; import pdb; pdb.set_trace()


# Process all three alignments
alignments_info = []

# 1. Hypothesis Alignment
forced_alignments = forced_alignments[0]
scores = scores[0].exp()
aligned_tokens_hyp = merge_tokens(forced_alignments, scores)
alignments_info.append({
    "name": "Hypothesis Alignment (Model Prediction)",
    "tokens": aligned_tokens_hyp,
    "scores": scores
})

# 2. Canonical Alignment
forced_alignments_cano = forced_alignments_cano[0]
scores_cano = scores_cano[0].exp()
aligned_tokens_cano = merge_tokens(forced_alignments_cano, scores_cano)
alignments_info.append({
    "name": "Canonical Alignment (Ground Truth)",
    "tokens": aligned_tokens_cano,
    "scores": scores_cano
})

# 3. Perceived Alignment
forced_alignments_perc = forced_alignments_perc[0]
scores_perc = scores_perc[0].exp()
aligned_tokens_perc = merge_tokens(forced_alignments_perc, scores_perc)
alignments_info.append({
    "name": "Perceived Alignment (Human Transcribed)",
    "tokens": aligned_tokens_perc,
    "scores": scores_perc
})

# -----------------------------------------------------------------------------
# 4. Voting Logic (Hypothesis vs Canonical)
# -----------------------------------------------------------------------------
blank_id = asr_model.tokenizer.lab2ind["<blank>"]
T = ctc_p.shape[-2] # Num frames

# Helper to get dense frame info
def get_dense_frame_info(token_spans, scores_tensor, num_frames, blank_idx):
    frame_tokens = torch.full((num_frames,), blank_idx, dtype=torch.int32)
    frame_scores = torch.zeros(num_frames, dtype=torch.float32)
    
    for span in token_spans:
        for t in range(span.start, span.end):
            frame_tokens[t] = span.token
            if scores_tensor.ndim > 1:
                frame_scores[t] = scores_tensor[t, span.token]
            else:
                 frame_scores[t] = scores_tensor[t]
    return frame_tokens, frame_scores

# Expand sparse spans to dense arrays
dense_tokens_h, dense_scores_h = get_dense_frame_info(aligned_tokens_hyp, scores, T, blank_id)
dense_tokens_c, dense_scores_c = get_dense_frame_info(aligned_tokens_cano, scores_cano, T, blank_id)

voting_tokens = []
voting_scores = []
VOTING_THRESH = 0.7

for t in range(T):
    sh = dense_scores_h[t].item()
    sc = dense_scores_c[t].item()
    
    # Voting Logic:
    # 1. If both < threshold -> Use blank with 0 score
    # 2. Else -> Pick the one with higher probability
    
    if sh < VOTING_THRESH and sc < VOTING_THRESH:
        voting_tokens.append(blank_id)
        voting_scores.append(0.0)
    else:
        if sh >= sc:
            voting_tokens.append(dense_tokens_h[t].item())
            voting_scores.append(sh)
        else:
            voting_tokens.append(dense_tokens_c[t].item())
            voting_scores.append(sc)
p_voting = torch.tensor(voting_scores)
log_voting = torch.log(p_voting)

# Convert voting result back to Spans for plotting
from torchaudio.functional import TokenSpan
voting_spans_list = []
curr = 0
while curr < T:
    if voting_tokens[curr] == blank_id:
        curr += 1
        continue
    # Start of a non-blank segment
    tok = voting_tokens[curr]
    start = curr
    while curr < T and voting_tokens[curr] == tok:
        curr += 1
    end = curr
    # Calculate average score for this span
    span_score = sum(voting_scores[start:end]) / (end - start) if end > start else 0.0
    voting_spans_list.append(TokenSpan(tok, start=start, end=end, score=span_score))

score_tensor_voting = torch.tensor(voting_scores)

# Greedy decode the voting result sequence
# Collapse repeats then remove blanks
import itertools
unique_tokens_iter = [k for k, g in itertools.groupby(voting_tokens)]
final_tokens = [k for k in unique_tokens_iter if k != blank_id]
# import pdb; pdb.set_trace()
final_voting_text = " ".join([asr_model.tokenizer.ind2lab.get(k, "?") for k in final_tokens])

alignments_info.append({
    "name": f"Voting Result (Greedy: {final_voting_text})",
    "tokens": voting_spans_list,
    "scores": score_tensor_voting
})
# -----------------------------------------------------------------------------

# import pdb; pdb.set_trace()
from utils.align import monitor_alignment, plot_scores, plot_alignment_comparison

# Create spectrogram from CTC log probabilities
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
# from scipy.interpolate import interp1d

# Compute mel spectrogram
sample_rate = 16000
try:
    waveform_np = waveform.cpu().numpy()
except:
    waveform_np = waveform.numpy()

mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=sample_rate, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Normalize Spectrogram
mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10) * 100

# Setup 4-row Plot
fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True, constrained_layout=True)

# Loop to plot each alignment
for idx, align_data in enumerate(alignments_info):
    ax = axes[idx]
    plot_name = align_data["name"]
    aligned_tokens = align_data["tokens"]
    scores_tensor = align_data["scores"]
    
    # 1. Background Spectrogram
    ax.imshow(mel_spec_db_norm, aspect='auto', origin='lower', cmap='viridis', 
              extent=[0, len(ctc_p[0]), 0, 80], alpha=0.9)
    
    # 2. Prepare Profitability Bars & Labels
    span_xs, span_hs = [], []
    max_score_in_plot = 0
    
    for span in aligned_tokens:
        # Collect score bars
        for t in range(span.start, span.end):
            span_xs.append(t + 0.5)
            # Handle score tensor dimension safely
            if scores_tensor.ndim > 1:
                score_val = scores_tensor[t, span.token].item()
            else:
                score_val = scores_tensor[t].item()
            span_hs.append(score_val)
            max_score_in_plot = max(max_score_in_plot, score_val)

        # Draw label for the segment
        center_x = span.start + (span.end - span.start) / 2
        
        # Determine colors based on score
        # Probability < 0.7 -> Black text (White box) -> Show Prediction (Greedy)
        # Probability >= 0.7 -> White text (Black box) -> Show Reference (Target)
        if span.score < 0.7:
             text_color = 'black'
             box_color = 'white'
             
             # Calculate CTC greedy decode for this segment
             # ctc_p is log_probs [1, T, V]
             segment_probs = ctc_p[0, span.start:span.end, :]
             best_indices = torch.argmax(segment_probs, dim=-1)
             unique_indices = torch.unique_consecutive(best_indices)
             blank_idx = asr_model.tokenizer.lab2ind["<blank>"]
             pred_tokens = [i.item() for i in unique_indices if i.item() != blank_idx]
             
             if pred_tokens:
                 # Join predicted phonemes
                 label_text = " ".join([asr_model.tokenizer.ind2lab[i] for i in pred_tokens])
             else:
                 label_text = ""
        else:
             text_color = 'white'
             box_color = 'black'
             try:
                label_text = asr_model.tokenizer.decode_ndim(span.token)
             except:
                import pdb; pdb.set_trace()

        ax.text(center_x, 70,  # Position text near top of spectrogram (y=80 max)
                label_text, 
                ha='center', va='center', 
                color=text_color, fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=box_color, alpha=0.8, edgecolor='none'),
                zorder=10)
        
        # Draw vertical dash line at start of segment
        ax.axvline(x=span.start, color='white', linestyle='--', linewidth=1.5, alpha=0.7)

    # 3. Plot Probability Bars
    # Scale bars to fit 0-80 range (spectrogram height)
    # We essentially map probability 0.0-1.0 to Y-pixel 0-60 (leaving space for text)
    SCALE_FACTOR = 80
    span_hs_scaled = [h * SCALE_FACTOR for h in span_hs]
    
    # Conditional coloring for bars
    bar_colors = ['black' if h < 0.7 else 'white' for h in span_hs]

    ax.bar(span_xs, span_hs_scaled, 
           color=bar_colors, alpha=0.6, width=1.0, 
           align='center', 
           zorder=5)
    
    # Decoration
    ax.set_title(plot_name, fontsize=14, loc='left', pad=10, color='darkblue')
    ax.set_ylabel('Mel Bins / Prob', fontsize=10)
    ax.set_xlim(0, len(ctc_p[0]))
    ax.set_ylim(0, 80)
    
    # Add secondary y-axis for probability reference on the right
    ax2 = ax.twinx()
    ax2.set_ylim(0, 80) # Same scale
    # ax2.set_yticks([0, 30, 60])
    ax2.set_yticks([0, 40, 80])
    ax2.set_yticklabels(['0.0', '0.5', '1.0'])
    ax2.set_ylabel('Probability', fontsize=9)


axes[-1].set_xlabel('Time Frames', fontsize=12)
fig.suptitle(f"Alignment Comparison: {file_id}", fontsize=16, weight='bold')

# Save
asr_model_name = Path(asr_model_path).stem
asr_model_parent = Path(asr_model_path).parent.name
output_dir = Path(f"{asr_model_parent}_{asr_model_name}_inference_outputs_{file_id}")
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = output_dir / f"{file_id}_alignment_comparison_3way.png"
fig.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved 3-way alignment comparison to {output_filename}")
