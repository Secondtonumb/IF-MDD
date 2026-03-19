import sys
from pathlib import Path
import json
import csv
import itertools

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR
from trainer.MyEncoderASR import MyCTCPrefixBeamSearcher

from speechbrain.decoders.ctc import TorchAudioCTCPrefixBeamSearcher

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torchaudio.functional import forced_align, merge_tokens, TokenSpan

# ============================================================================
# Setup
# ============================================================================
json_path = "/home/m64000/work/dataset/data_iqra_extra_is26/iqra_extra_is26_test_aligned.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Load model
asr_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/crottc_k3_RoPE_TTS_FT.ckpt"
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

asr_model = MyEncoderASR.from_hparams(source=asr_model_path, hparams_file="inference.yaml")

# Output directories
output_base_dir = Path("voting_results")
output_base_dir.mkdir(exist_ok=True)
img_output_dir = output_base_dir / "images"
img_output_dir.mkdir(exist_ok=True)

csv_output_path = output_base_dir / "voting_results.csv"

# ============================================================================
# Helper Functions
# ============================================================================
def get_dense_frame_info(token_spans, scores_tensor, num_frames, blank_idx):
    """Convert sparse token spans to dense frame arrays"""
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

def voting_and_decode(aligned_tokens_hyp, scores_h, aligned_tokens_cano, scores_c, 
                      asr_model, blank_id, T, VOTING_THRESH=0.7):
    """Perform voting between hypothesis and canonical alignments"""
    
    # Get dense frame info
    dense_tokens_h, dense_scores_h = get_dense_frame_info(aligned_tokens_hyp, scores_h, T, blank_id)
    dense_tokens_c, dense_scores_c = get_dense_frame_info(aligned_tokens_cano, scores_c, T, blank_id)
    
    voting_tokens = []
    voting_scores = []
    
    for t in range(T):
        sh = dense_scores_h[t].item()
        sc = dense_scores_c[t].item()
        tok_h = dense_tokens_h[t].item()
        tok_c = dense_tokens_c[t].item()
        
        # Voting Logic
        if sh < VOTING_THRESH and sc < VOTING_THRESH:
            voting_tokens.append(blank_id)
            voting_scores.append(0.0)
        else:
            # Pick the one with higher score
            if sh >= sc:
                if tok_h in asr_model.tokenizer.ind2lab:
                    voting_tokens.append(tok_h)
                    voting_scores.append(sh)
                elif tok_c in asr_model.tokenizer.ind2lab:
                    voting_tokens.append(tok_c)
                    voting_scores.append(sc)
                else:
                    voting_tokens.append(blank_id)
                    voting_scores.append(0.0)
            else:
                if tok_c in asr_model.tokenizer.ind2lab:
                    voting_tokens.append(tok_c)
                    voting_scores.append(sc)
                elif tok_h in asr_model.tokenizer.ind2lab:
                    voting_tokens.append(tok_h)
                    voting_scores.append(sh)
                else:
                    voting_tokens.append(blank_id)
                    voting_scores.append(0.0)
    
    # Greedy decode
    unique_tokens_iter = [k for k, g in itertools.groupby(voting_tokens)]
    final_tokens = [k for k in unique_tokens_iter if k != blank_id]
    final_voting_text = " ".join([asr_model.tokenizer.ind2lab.get(k, "?") for k in final_tokens])
    
    return voting_tokens, voting_scores, final_voting_text

def create_plot(ctc_p, alignments_info, file_id, asr_model, waveform):
    """Create alignment comparison plot"""
    
    # Compute mel spectrogram
    sample_rate = 16000
    try:
        waveform_np = waveform.cpu().numpy()
    except:
        waveform_np = waveform.numpy()
    
    mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=sample_rate, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10) * 100
    
    # Setup Plot (3 rows: Hypothesis, Canonical, Voting)
    num_plots = len(alignments_info)
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots), sharex=True, constrained_layout=True)
    
    # Setup Plot (3 rows: Hypothesis, Canonical, Voting)
    num_plots = len(alignments_info)
    if num_plots == 1:
        fig, ax = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots), sharex=True, constrained_layout=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots), sharex=True, constrained_layout=True)
    
    # Plot each alignment
    for idx, align_data in enumerate(alignments_info):
        ax = axes[idx] if len(axes) > 1 else axes[0]
        plot_name = align_data["name"]
        aligned_tokens = align_data["tokens"]
        scores_tensor = align_data["scores"]
        
        # Skip if no tokens
        if not aligned_tokens:
            continue
        
        # Background spectrogram
        ax.imshow(mel_spec_db_norm, aspect='auto', origin='lower', cmap='viridis', 
                  extent=[0, len(ctc_p[0]), 0, 80], alpha=0.9)
        
        # Prepare bars and labels
        span_xs, span_hs = [], []
        
        for span in aligned_tokens:
            # Collect score bars
            for t in range(span.start, span.end):
                span_xs.append(t + 0.5)
                try:
                    if scores_tensor.ndim > 1:
                        if span.token < scores_tensor.shape[1]:
                            score_val = scores_tensor[t, span.token].item()
                        else:
                            score_val = 0.5
                    else:
                        score_val = scores_tensor[t].item()
                except Exception:
                    score_val = 0.5
                span_hs.append(score_val)
            
            # Draw label
            center_x = span.start + (span.end - span.start) / 2
            
            if span.score < 0.7:
                text_color = 'black'
                box_color = 'white'
            else:
                text_color = 'white'
                box_color = 'black'
            
            # Get label text
            try:
                if scores_tensor.ndim > 1:
                    label_text = asr_model.tokenizer.decode_ndim(span.token)
                else:
                    # For voting result (1D tensor), token is in span
                    label_text = asr_model.tokenizer.ind2lab.get(span.token, "?")
            except Exception:
                label_text = "?"
            
            ax.text(center_x, 70, label_text, 
                    ha='center', va='center', 
                    color=text_color, fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=box_color, alpha=0.8, edgecolor='none'),
                    zorder=10)
            
            ax.axvline(x=span.start, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Plot bars
        if span_xs:
            SCALE_FACTOR = 80
            span_hs_scaled = [h * SCALE_FACTOR for h in span_hs]
            bar_colors = ['black' if h < 0.7 else 'white' for h in span_hs]
            
            ax.bar(span_xs, span_hs_scaled, 
                   color=bar_colors, alpha=0.6, width=1.0, 
                   align='center', zorder=5)
        
        ax.set_title(plot_name, fontsize=14, loc='left', pad=10, color='darkblue')
        ax.set_ylabel('Mel Bins / Prob', fontsize=10)
        ax.set_xlim(0, len(ctc_p[0]))
        ax.set_ylim(0, 80)
        
        ax2 = ax.twinx()
        ax2.set_ylim(0, 80)
        ax2.set_yticks([0, 40, 80])
        ax2.set_yticklabels(['0.0', '0.5', '1.0'])
        ax2.set_ylabel('Probability', fontsize=9)
    
    axes[-1].set_xlabel('Time Frames', fontsize=12)
    fig.suptitle(f"Alignment Comparison: {file_id}", fontsize=16, weight='bold')
    
    return fig

# ============================================================================
# Main Processing Loop
# ============================================================================
csv_rows = []
blank_id = asr_model.tokenizer.lab2ind["<blank>"]

for idx, (file_path, file_data) in enumerate(data.items()):
    file_id = Path(file_path).stem
    print(f"[{idx+1}/{len(data)}] Processing {file_id}...")
    
    try:
        # Load audio and labels - keep all labels as-is
        canonical_label = file_data['canonical_aligned'].strip()
        perceived_label = file_data['perceived_aligned'].strip()
        
        if not canonical_label or not perceived_label:
            print(f"  ✗ Skip: Empty labels")
            csv_rows.append({
                "file_id": file_id,
                "voting_label": "EMPTY"
            })
            continue
        
        # Transcribe
        x = asr_model.transcribe_file(file_path)
        
        # Encode sequences - all labels are valid, don't filter
        try:
            x_id = asr_model.tokenizer.encode_sequence(x.split(' '))
            canonical_label_ids = asr_model.tokenizer.encode_sequence(canonical_label.split())
        except Exception as e:
            print(f"  ✗ Skip: Encoding error - {str(e)[:100]}")
            csv_rows.append({
                "file_id": file_id,
                "voting_label": "ENCODE_ERROR"
            })
            continue
        
        # Load audio
        try:
            waveform, _ = asr_model.load_audio(file_path)
        except:
            waveform = asr_model.load_audio(file_path)
        
        # Get CTC probabilities
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        ctc_p = asr_model.encode_batch(batch, rel_length)
        T = ctc_p.shape[-2]
        
        # Forced alignment
        forced_alignments, scores = forced_align(
            log_probs=ctc_p,
            targets=torch.tensor([x_id], dtype=torch.int32, device=ctc_p.device),
            input_lengths=torch.tensor([T], dtype=torch.int32, device=ctc_p.device),
            target_lengths=torch.tensor([len(x_id)], dtype=torch.int32, device=ctc_p.device),
            blank=blank_id
        )
        
        forced_alignments_cano, scores_cano = forced_align(
            log_probs=ctc_p,
            targets=torch.tensor([canonical_label_ids], dtype=torch.int32, device=ctc_p.device),
            input_lengths=torch.tensor([T], dtype=torch.int32, device=ctc_p.device),
            target_lengths=torch.tensor([len(canonical_label_ids)], dtype=torch.int32, device=ctc_p.device),
            blank=blank_id
        )
        
        # Process alignments
        scores = scores[0].exp()
        scores_cano = scores_cano[0].exp()
        aligned_tokens_hyp = merge_tokens(forced_alignments[0], scores)
        aligned_tokens_cano = merge_tokens(forced_alignments_cano[0], scores_cano)
        
        # Perform voting
        voting_tokens, voting_scores, final_voting_text = voting_and_decode(
            aligned_tokens_hyp, scores, aligned_tokens_cano, scores_cano,
            asr_model, blank_id, T
        )
        
        # Create voting spans for plotting
        voting_spans_list = []
        curr = 0
        while curr < T:
            if voting_tokens[curr] == blank_id:
                curr += 1
                continue
            tok = voting_tokens[curr]
            start = curr
            while curr < T and voting_tokens[curr] == tok:
                curr += 1
            end = curr
            span_score = sum(voting_scores[start:end]) / (end - start) if end > start else 0.0
            # voting_spans_list.append(TokenSpan(start, end, tok, span_score))
            voting_spans_list.append(TokenSpan(tok, start=start, end=end, score=span_score))
        
        # Prepare alignments info for plotting
        alignments_info = [
            {
                "name": "Hypothesis Alignment (Model Prediction)",
                "tokens": aligned_tokens_hyp,
                "scores": scores
            },
            {
                "name": "Canonical Alignment (Ground Truth)",
                "tokens": aligned_tokens_cano,
                "scores": scores_cano
            },
            {
                "name": f"Voting Result (Greedy: {final_voting_text})",
                "tokens": voting_spans_list,
                "scores": torch.tensor(voting_scores, dtype=torch.float32)
            }
        ]
        
        # Create and save plot
        fig = create_plot(ctc_p, alignments_info, file_id, asr_model, waveform)
        img_filename = img_output_dir / f"{file_id}_alignment_voting.png"
        fig.savefig(img_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # Record voting result
        csv_rows.append({
            "file_id": file_id,
            "voting_label": final_voting_text
        })
        
        print(f"  ✓ Saved: {img_filename}")
        print(f"  ✓ Voting Label: {final_voting_text}")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        csv_rows.append({
            "file_id": file_id,
            "voting_label": "ERROR"
        })

# ============================================================================
# Save CSV
# ============================================================================
with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file_id', 'voting_label'])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n{'='*60}")
print(f"✓ Completed! Total: {len(csv_rows)} files")
print(f"✓ Images saved to: {img_output_dir}")
print(f"✓ CSV saved to: {csv_output_path}")
print(f"{'='*60}")
