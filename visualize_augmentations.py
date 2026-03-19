import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.patches import Rectangle

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from speechbrain.augment.time_domain import DropChunk, DropFreq, SpeedPerturb
from speechbrain.augment.freq_domain import SpectrogramDrop, Warping
from speechbrain.augment.augmenter import Augmenter

# Audio file
file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_94.wav"
file_id = Path(file_path).stem

# Load audio
waveform, sr = librosa.load(file_path, sr=16000)
waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

print(f"Original waveform shape: {waveform_tensor.shape}")
print(f"Sample rate: {sr}")

# ===== STRATEGY 1: DropFreq + DropChunk (from YAML) =====
freq_chunk_augmentation = DropFreq(
    drop_freq_low=1e-14,
    drop_freq_high=1,
    drop_freq_count_low=1,
    drop_freq_count_high=3,
    drop_freq_width=0.10,
    epsilon=1e-12
)

time_chunk_augmentation = DropChunk(
    drop_length_low=1000,
    drop_length_high=3000,
    drop_count_low=1,
    drop_count_high=3
)

# ===== STRATEGY 2: SpectrogramDrop only =====
spec_augmentation = SpectrogramDrop(
    drop_length_low=5,
    drop_length_high=27,
    drop_count_low=1,
    drop_count_high=3,
    replace='zeros'
)

# ===== STRATEGY 3: Warping only =====
timewarp_augmentation = Warping(
    warp_window=5,
    dim=1  # Time warping
)

# Apply augmentations
rel_length = torch.tensor([1.0])

# Strategy 1: DropFreq + DropChunk (waveform level)
waveform_aug1 = freq_chunk_augmentation(waveform_tensor.clone())
waveform_aug1 = time_chunk_augmentation(waveform_aug1, rel_length)

# Compute mel spectrograms
mel_spec_orig = librosa.feature.melspectrogram(y=waveform_tensor[0].numpy(), sr=sr, n_mels=80)
mel_spec_db_orig = librosa.power_to_db(mel_spec_orig, ref=np.max)

mel_spec_aug1 = librosa.feature.melspectrogram(y=waveform_aug1[0].detach().numpy(), sr=sr, n_mels=80)
mel_spec_db_aug1 = librosa.power_to_db(mel_spec_aug1, ref=np.max)

# Strategy 2: SpectrogramDrop (frequency domain)
mel_spec_tensor = torch.tensor(mel_spec_db_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 80, time)
try:
    mel_spec_aug2 = spec_augmentation(mel_spec_tensor.clone())
    # Handle different output shapes from SpectrogramDrop
    if mel_spec_aug2.dim() == 4:
        mel_spec_db_aug2 = mel_spec_aug2[0, 0].detach().numpy()
    elif mel_spec_aug2.dim() == 3:
        mel_spec_db_aug2 = mel_spec_aug2[0].detach().numpy()
    else:
        mel_spec_db_aug2 = mel_spec_aug2.detach().numpy()
    
    # Ensure shape matches original
    if mel_spec_db_aug2.shape != mel_spec_db_orig.shape:
        print(f"Warning: SpectrogramDrop output shape {mel_spec_db_aug2.shape} != original {mel_spec_db_orig.shape}")
        # If shapes don't match, use original
        mel_spec_db_aug2 = mel_spec_db_orig.copy()
except Exception as e:
    print(f"SpectrogramDrop failed: {e}")
    mel_spec_db_aug2 = mel_spec_db_orig.copy()

# Strategy 3: Warping
try:
    mel_spec_aug3 = timewarp_augmentation(mel_spec_tensor.clone())
    mel_spec_db_aug3 = mel_spec_aug3[0, 0].detach().numpy()
except Exception as e:
    print(f"Warping failed: {e}")
    mel_spec_db_aug3 = mel_spec_db_orig.copy()

# Normalize for visualization
def normalize_spec(spec):
    spec_normalized = (spec - spec.min()) / (spec.max() - spec.min() + 1e-10) * 100
    return spec_normalized

mel_spec_db_orig_norm = normalize_spec(mel_spec_db_orig)
mel_spec_db_aug1_norm = normalize_spec(mel_spec_db_aug1)
mel_spec_db_aug2_norm = normalize_spec(mel_spec_db_aug2)
mel_spec_db_aug3_norm = normalize_spec(mel_spec_db_aug3)

# Detect masked regions
# Mask 1: DropChunk/DropFreq creates very low values
mask1 = mel_spec_db_aug1_norm < 5  # Very dark regions

# Mask 2: SpectrogramDrop creates zero regions
mask2 = mel_spec_db_aug2_norm < 1  # Zero/near-zero regions

# Mask 3: Warping doesn't create binary masks, but shows continuous deformation
diff3 = np.abs(mel_spec_db_aug3_norm - mel_spec_db_orig_norm)
mask3 = diff3 > 20  # Significant differences indicate warping

# Create main comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Original
im0 = axes[0, 0].imshow(mel_spec_db_orig_norm, aspect='auto', origin='lower', cmap='viridis', alpha=1.0)
axes[0, 0].set_title('Original Spectrogram', fontsize=13, weight='bold')
axes[0, 0].set_ylabel('Mel Frequency Bin')
axes[0, 0].set_xlabel('Time Frame')
cbar0 = plt.colorbar(im0, ax=axes[0, 0], label='Power (dB)')

# Strategy 1: DropFreq + DropChunk
im1 = axes[0, 1].imshow(mel_spec_db_aug1_norm, aspect='auto', origin='lower', cmap='viridis', alpha=0.9)
# Overlay masked regions with red
axes[0, 1].contourf(mask1, levels=[0.5, 1.5], colors='red', alpha=0.4)
axes[0, 1].contour(mask1, levels=[0.5], colors='darkred', linewidths=2.5, linestyles='-')
axes[0, 1].set_title('Strategy 1: DropFreq + DropChunk\n[Red areas = Dropped chunks]', fontsize=13, weight='bold')
axes[0, 1].set_ylabel('Mel Frequency Bin')
axes[0, 1].set_xlabel('Time Frame')
cbar1 = plt.colorbar(im1, ax=axes[0, 1], label='Power (dB)')

# Strategy 2: SpectrogramDrop
im2 = axes[1, 0].imshow(mel_spec_db_aug2_norm, aspect='auto', origin='lower', cmap='viridis', alpha=0.9)
# Overlay masked regions with green
axes[1, 0].contourf(mask2, levels=[0.5, 1.5], colors='lime', alpha=0.4)
axes[1, 0].contour(mask2, levels=[0.5], colors='darkgreen', linewidths=2.5, linestyles='-')
axes[1, 0].set_title('Strategy 2: SpectrogramDrop\n[Green areas = Zeroed regions]', fontsize=13, weight='bold')
axes[1, 0].set_ylabel('Mel Frequency Bin')
axes[1, 0].set_xlabel('Time Frame')
cbar2 = plt.colorbar(im2, ax=axes[1, 0], label='Power (dB)')

# Strategy 3: Warping
im3 = axes[1, 1].imshow(mel_spec_db_aug3_norm, aspect='auto', origin='lower', cmap='viridis', alpha=0.9)
# Draw grid to show time warping deformation
time_frames = mel_spec_db_aug3.shape[1]
freq_bins = mel_spec_db_aug3.shape[0]
for i in range(0, freq_bins, 10):
    axes[1, 1].axhline(y=i, color='cyan', alpha=0.3, linewidth=1, linestyle=':')
for j in range(0, time_frames, 10):
    axes[1, 1].axvline(x=j, color='yellow', alpha=0.3, linewidth=1, linestyle=':')

# Overlay warped regions with orange contour
axes[1, 1].contour(mask3, levels=[0.5], colors='orange', linewidths=2, linestyles='--')
axes[1, 1].set_title('Strategy 3: Time Warping (warp_window=5)\n[Grid + Orange contour = Warped regions]', fontsize=13, weight='bold')
axes[1, 1].set_ylabel('Mel Frequency Bin')
axes[1, 1].set_xlabel('Time Frame')
cbar3 = plt.colorbar(im3, ax=axes[1, 1], label='Power (dB)')

fig.suptitle(f'Spectrogram Augmentation Strategies for {file_id}', fontsize=15, weight='bold')
plt.tight_layout()

# Save figure
output_dir = Path(f"augmentation_comparison_{file_id}")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / f"{file_id}_augmentation_strategies.png", dpi=300, bbox_inches='tight')
print(f"Saved augmentation strategies to {output_dir / f'{file_id}_augmentation_strategies.png'}")

plt.close(fig)

# Create detailed difference visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Difference: Strategy 1 - Original
diff1 = mel_spec_db_aug1_norm - mel_spec_db_orig_norm
im1 = axes[0].imshow(diff1, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-80, vmax=80)
axes[0].contour(mask1, levels=[0.5], colors='black', linewidths=2.5, linestyles='-')
axes[0].set_title('DropFreq + DropChunk - Original\n[Black outline = Masked regions]', fontsize=13, weight='bold')
axes[0].set_ylabel('Mel Frequency Bin')
axes[0].set_xlabel('Time Frame')
cbar1 = plt.colorbar(im1, ax=axes[0], label='Difference (dB)')

# Difference: Strategy 2 - Original
diff2 = mel_spec_db_aug2_norm - mel_spec_db_orig_norm
im2 = axes[1].imshow(diff2, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-80, vmax=80)
axes[1].contour(mask2, levels=[0.5], colors='black', linewidths=2.5, linestyles='-')
axes[1].set_title('SpectrogramDrop - Original\n[Black outline = Zeroed regions]', fontsize=13, weight='bold')
axes[1].set_ylabel('Mel Frequency Bin')
axes[1].set_xlabel('Time Frame')
cbar2 = plt.colorbar(im2, ax=axes[1], label='Difference (dB)')

# Difference: Strategy 3 - Original
diff3 = mel_spec_db_aug3_norm - mel_spec_db_orig_norm
im3 = axes[2].imshow(diff3, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-80, vmax=80)
axes[2].contour(mask3, levels=[0.5], colors='black', linewidths=2, linestyles='--')
axes[2].set_title('Time Warping - Original\n[Black contour = Warped regions]', fontsize=13, weight='bold')
axes[2].set_ylabel('Mel Frequency Bin')
axes[2].set_xlabel('Time Frame')
cbar3 = plt.colorbar(im3, ax=axes[2], label='Difference (dB)')

fig.suptitle(f'Augmentation Differences for {file_id}', fontsize=15, weight='bold')
plt.tight_layout()

fig.savefig(output_dir / f"{file_id}_augmentation_differences.png", dpi=300, bbox_inches='tight')
print(f"Saved augmentation differences to {output_dir / f'{file_id}_augmentation_differences.png'}")

plt.close(fig)

print("\n=== Augmentation Analysis Complete ===")
print(f"Output saved to: {output_dir}/")
print(f"\nFiles generated:")
print(f"  1. {file_id}_augmentation_strategies.png - Comparison of 3 strategies")
print(f"  2. {file_id}_augmentation_differences.png - Differences from original")
print(f"\nMask visualization legend:")
print(f"  - RED areas: DropFreq/DropChunk masked regions")
print(f"  - GREEN areas: SpectrogramDrop zeroed regions")
print(f"  - ORANGE/GRID: Time Warping deformed regions")
