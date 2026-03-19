#!/usr/bin/env python3
"""
Compare 3 models inference: CTC, OTTC, CROTTC (Checking option without CRCTC)
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from huggingface_hub import hf_hub_download
import importlib.util
import librosa

# ===== 配置 =====
file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_94.wav"
file_id = Path(file_path).stem

# Three models path (Removed CRCTC)
models_config = {
    "CTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CTC",
    "OTTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/OTTC",
    "CROTTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/crottc_k3_RoPE_TTS_FT.ckpt"
}

SAMPLE_RATE = 16000
# ===============

print(f"Loading MyEncoderASR dynamically...")
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MyEncoderASR = module.MyEncoderASR

# 加载音频一次
print(f"Loading audio: {file_path}")
try:
    waveform_orig, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(waveform_orig).float()
except Exception as e:
    print(f"Error loading audio: {e}")
    sys.exit(1)

# 计算 mel spectrogram
print("Computing mel spectrogram...")
mel_spec = librosa.feature.melspectrogram(y=waveform_orig, sr=SAMPLE_RATE, n_mels=80, hop_length=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 创建竖长的3个子图：3行1列
num_models = len(models_config)
fig, axes = plt.subplots(num_models, 1, figsize=(14, 4 * num_models), sharex=False)
fig.patch.set_facecolor('white')

results = {}
model_names = list(models_config.keys())

for idx, (model_name, model_path) in enumerate(models_config.items()):
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{num_models}] Processing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 加载模型
        print(f"  Loading {model_name}...")
        asr_model = MyEncoderASR.from_hparams(
            source=model_path, 
            hparams_file="inference.yaml"
        )
        
        # 转录
        print(f"  Transcribing...")
        x = asr_model.transcribe_file(file_path)
        x_id = asr_model.tokenizer.encode_sequence(x.split(' '))
        print(f"  Transcription: {x}")
        
        # 获取 CTC 概率
        print(f"  Computing CTC probabilities...")
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        ctc_p = asr_model.encode_batch(batch, rel_length)
        
        # 强制对齐
        from torchaudio.functional import forced_align, merge_tokens
        print(f"  Performing forced alignment...")
        
        forced_alignments, scores = forced_align(
            log_probs=ctc_p,
            targets=torch.tensor([x_id], dtype=torch.int32, device=ctc_p.device),
            input_lengths=torch.tensor([ctc_p.shape[-2]], dtype=torch.int32, device=ctc_p.device),
            target_lengths=torch.tensor([len(x_id)], dtype=torch.int32, device=ctc_p.device),
            blank=asr_model.tokenizer.lab2ind["<blank>"]
        )
        
        forced_alignments = forced_alignments[0]
        scores = scores[0].exp()
        aligned_tokens_gt = merge_tokens(forced_alignments, scores)
        
        results[model_name] = {
            'asr_model': asr_model,
            'transcription': x,
            'ctc_p': ctc_p,
            'scores': scores,
            'aligned_tokens': aligned_tokens_gt,
            'x_id': x_id
        }
        
        # 绘制该模型结果
        ax = axes[idx]
        
        # 调整 mel spectrogram 分辨率
        ctc_n_frames = len(ctc_p[0])
        mel_n_frames = mel_spec_db.shape[1]
        
        if mel_n_frames != ctc_n_frames:
            mel_spec_db_resampled = np.zeros((mel_spec_db.shape[0], ctc_n_frames))
            for i in range(mel_spec_db.shape[0]):
                f = interp1d(
                    np.linspace(0, 1, mel_n_frames), 
                    mel_spec_db[i, :], 
                    kind='linear', 
                    fill_value='extrapolate'
                )
                mel_spec_db_resampled[i, :] = f(np.linspace(0, 1, ctc_n_frames))
            mel_spec_used = mel_spec_db_resampled
        else:
            mel_spec_used = mel_spec_db
        
        # 标准化
        mel_spec_db_norm = (mel_spec_used - mel_spec_used.min()) / (mel_spec_used.max() - mel_spec_used.min() + 1e-10) * 100
        
        # === 只显示最后40帧 ===
        CUT_LAST_FRAMES = 40
        start_frame = max(0, ctc_n_frames - CUT_LAST_FRAMES)
        end_frame = ctc_n_frames
        display_n_frames = end_frame - start_frame
        
        mel_spec_display = mel_spec_db_norm[:, start_frame:end_frame]
        
        # 绘制 mel spectrogram
        im = ax.imshow(
            mel_spec_display, 
            aspect='auto', 
            origin='lower', 
            cmap='viridis',
            extent=[0, display_n_frames, 0, 80], 
            alpha=1.0, 
            zorder=1
        )
        
        # 设置标题 - 大字体
        ax.set_title(f'{model_name}', fontsize=18, weight='bold', pad=15)
        
        # 设置轴标签 - 大字体
        ax.set_ylabel('Mel Bin', fontsize=14, weight='bold')
        ax.set_xlabel('Frame', fontsize=14, weight='bold')
        
        ax.set_xlim(0, display_n_frames)
        ax.set_ylim(0, 80)
        
        # 移除 spines（边框）
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # 移除刻度线
        ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
        
        # 增大刻度标签字体
        ax.tick_params(labelsize=12)
        
        # 绘制对齐边界和分段
        for i, span in enumerate(aligned_tokens_gt):
            # 调整坐标到裁剪窗口
            adj_start = span.start - start_frame
            adj_end = span.end - start_frame
            
            # 如果分段在显示区域之外，则跳过
            if adj_end <= 0 or adj_start >= display_n_frames:
                continue
            
            # 限制绘制范围在 (0, display_n_frames)
            draw_start = max(0, adj_start - 0.05)
            draw_end = min(display_n_frames, adj_end + 0.05)

            if i % 2 == 0:
                ax.axvspan(draw_start, draw_end, facecolor="cyan", alpha=0.15, zorder=0)
            else:
                ax.axvspan(draw_start, draw_end, facecolor="yellow", alpha=0.15, zorder=0)

            # 绘制分割线 - 虚线
            if adj_start >= 0 and adj_start <= display_n_frames:
                ax.vlines(adj_start, 0, 80, colors='white', linestyles='--', alpha=0.5, linewidth=1.0, zorder=5)
            if adj_end >= 0 and adj_end <= display_n_frames:
                ax.vlines(adj_end, 0, 80, colors='white', linestyles='--', alpha=0.5, linewidth=1.0, zorder=5)
        
        # 添加概率柱状图
        span_xs, span_hs = [], []
        max_score = 0
        for span in aligned_tokens_gt:
            for t in range(span.start, span.end):
                # 调整坐标
                t_adj = t - start_frame
                if 0 <= t_adj < display_n_frames:
                    span_xs.append(t_adj + 0.5)
                    score_val = scores[t, span.token].item() if scores.ndim > 1 else scores[t].item()
                    span_hs.append(score_val)
                    max_score = max(max_score, score_val)
        
        if max_score > 0:
            span_hs_scaled = [h / max_score * 80 for h in span_hs]
        else:
            span_hs_scaled = span_hs
        
        ax.bar(
            span_xs, 
            span_hs_scaled, 
            color="white", 
            edgecolor="gold",
            alpha=0.7, 
            linewidth=1.2, 
            hatch='/', 
            zorder=3,
            width=0.8
        )
        
        # 绘制音素标签 - 最后绘制以确保显示在最前面
        for i, span in enumerate(aligned_tokens_gt):
            # 调整坐标
            adj_start = span.start - start_frame
            adj_end = span.end - start_frame
            
            # 如果分段在显示区域之外，则跳过
            if adj_end <= 0 or adj_start >= display_n_frames:
                continue
                
            # 音素标签 - 增大字体，zorder=10 确保在最前面
            ax.text(
                adj_start + (adj_end - adj_start) / 2, 
                60,
                asr_model.tokenizer.decode_ndim(span.token),
                ha='center', 
                va='bottom',
                color='white', 
                fontsize=14, 
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='white', linewidth=1),
                zorder=10,
                clip_on=True
            )
        
        # 添加右侧概率轴 - 大字体
        ax_r = ax.twinx()
        ax_r.set_ylabel('Probability', fontsize=14, weight='bold', color='black', labelpad=10)
        ax_r.set_yticks([0, 20, 40, 60, 80])
        ax_r.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], fontsize=12, color='black', weight='bold')
        ax_r.tick_params(axis='y', labelcolor='black', labelsize=12)
        
        # 移除右侧 spines
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)
        ax_r.spines['left'].set_visible(False)
        ax_r.spines['bottom'].set_visible(False)
        
        # 设置概率轴线强调
        ax_r.spines['right'].set_visible(True)
        ax_r.spines['right'].set_linewidth(2.5)
        ax_r.spines['right'].set_color('black')
        
        print(f"  ✓ {model_name} processed successfully")
        
    except Exception as e:
        print(f"  ✗ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()

# 总标题 - 大字体
fig.suptitle(
    f"Phoneme Detection Model Comparison | Audio: {file_id}", 
    fontsize=20, 
    weight='bold', 
    y=0.98
)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图表
output_dir = Path(f"model_comparison_inference_outputs_{file_id}")
output_dir.mkdir(exist_ok=True)
save_path = output_dir / "comparison_plot_3models.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {save_path}")

plt.close()
