import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR, MyCTCPrefixBeamSearcher
from torchaudio.functional import forced_align, merge_tokens
import librosa
import librosa.display

# ===== 配置 =====
file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_94.wav"
file_id = Path(file_path).stem

# 三个模型路径
models_config = {
    "CTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CTC",
    "CRCTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CRCTC",
    "OTTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/OTTC",
    "K3_CROTTC": "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/crottc_k3_RoPE_TTS_FT.ckpt"
    
}

SAMPLE_RATE = 16000
OVERLAY_MODE = True  # True: 叠加模式，False: 并排模式
# ===============

# 动态导入 MyEncoderASR
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 加载音频一次
try:
    waveform_orig, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(waveform_orig).float()
except Exception as e:
    print(f"Error loading audio: {e}")
    sys.exit(1)

# 计算 mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=waveform_orig, sr=SAMPLE_RATE, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 创建对比图表
if OVERLAY_MODE:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
else:
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

results = {}

for idx, (model_name, model_path) in enumerate(models_config.items()):
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")
    
    try:
        # 加载模型
        print(f"Loading {model_name} from {model_path}...")
        asr_model = MyEncoderASR.from_hparams(source=model_path, hparams_file="inference.yaml")
        
        # 转录
        x = asr_model.transcribe_file(file_path)
        x_id = asr_model.tokenizer.encode_sequence(x.split(' '))
        print(f"Transcription: {x}")
        
        # 获取 CTC 概率
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        ctc_p = asr_model.encode_batch(batch, rel_length)
        
        # 强制对齐
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
        if OVERLAY_MODE:
            ax = axes[idx]
            ax2 = ax
            ax1 = None
        else:
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
        
        # 调整 mel spectrogram 分辨率
        ctc_n_frames = len(ctc_p[0])
        mel_n_frames = mel_spec_db.shape[1]
        
        if mel_n_frames != ctc_n_frames:
            mel_spec_db_resampled = np.zeros((mel_spec_db.shape[0], ctc_n_frames))
            for i in range(mel_spec_db.shape[0]):
                f = interp1d(np.linspace(0, 1, mel_n_frames), mel_spec_db[i, :], kind='linear', fill_value='extrapolate')
                mel_spec_db_resampled[i, :] = f(np.linspace(0, 1, ctc_n_frames))
            mel_spec_used = mel_spec_db_resampled
        else:
            mel_spec_used = mel_spec_db
        
        # 标准化
        mel_spec_db_norm = (mel_spec_used - mel_spec_used.min()) / (mel_spec_used.max() - mel_spec_used.min() + 1e-10) * 100
        
        # 绘制 mel spectrogram
        im = ax2.imshow(mel_spec_db_norm, aspect='auto', origin='lower', cmap='viridis', 
                       extent=[0, ctc_n_frames, 0, 80], alpha=1.0, zorder=1)
        
        ax2.set_ylabel('Mel Bin')
        ax2.set_xlabel('Frame')
        ax2.set_title(f'{model_name} - Alignment & Spectrogram', fontsize=12, weight='bold')
        ax2.set_xlim(-1, ctc_n_frames)
        
        # 绘制对齐边界和分段
        for i, span in enumerate(aligned_tokens_gt):
            if i % 2 == 0:
                ax2.axvspan(span.start - 0.05, span.end + 0.05, facecolor="cyan", alpha=0.15, zorder=0)
            else:
                ax2.axvspan(span.start - 0.05, span.end + 0.05, facecolor="yellow", alpha=0.15, zorder=0)
            
            ax2.text(span.start + (span.end - span.start) / 2, 75, 
                    asr_model.tokenizer.decode_ndim(span.token), 
                    ha='center', color='white', fontsize=7, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # 添加概率柱状图
        span_xs, span_hs = [], []
        max_score = 0
        for span in aligned_tokens_gt:
            for t in range(span.start, span.end):
                span_xs.append(t + 0.5)
                score_val = scores[t, span.token].item() if scores.ndim > 1 else scores[t].item()
                span_hs.append(score_val)
                max_score = max(max_score, score_val)
        
        if max_score > 0:
            span_hs_scaled = [h / max_score * 80 for h in span_hs]
        else:
            span_hs_scaled = span_hs
        
        ax2.bar(span_xs, span_hs_scaled, color="white", edgecolor="gold", 
               alpha=0.6, linewidth=0.8, hatch='/', zorder=3)
        
        # 添加右侧概率轴
        ax2_r = ax2.twinx()
        ax2_r.set_ylabel('CTC Score', fontsize=10)
        ax2_r.set_yticks([0, 20, 40, 60, 80])
        ax2_r.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        
        print(f"✓ {model_name} processed successfully")
        
    except Exception as e:
        print(f"✗ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()

fig.suptitle(f"Model Comparison: CTC vs OTTC vs CRCTC - {file_id}", 
            fontsize=14, weight='bold', y=0.995)
plt.tight_layout()

# 保存图表
output_dir = Path(f"model_comparison_inference_outputs_{file_id}")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / f"{file_id}_4models_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison figure to: {output_dir / f'{file_id}_4models_comparison.png'}")

plt.show()