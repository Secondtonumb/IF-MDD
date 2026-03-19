import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from speechbrain.augment.time_domain import DropChunk, DropFreq

# Audio file
file_path = "/home/m64000/work/dataset/data_iqra_extra_is26/wav/is26_sample_94.wav"
file_id = Path(file_path).stem

# Load audio
waveform, sr = librosa.load(file_path, sr=16000)
waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

print(f"Original waveform shape: {waveform_tensor.shape}")
print(f"Sample rate: {sr}")

# Define augmentation (same as YAML config)
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

# Apply augmentations and track mask regions
rel_length = torch.tensor([1.0])

# First augmentation - track mask
waveform_aug1 = freq_chunk_augmentation(waveform_tensor.clone())
waveform_aug1 = time_chunk_augmentation(waveform_aug1, rel_length)
# Create binary mask: 1 where values changed (augmented), 0 where original
mask_aug1_binary = (torch.abs(waveform_aug1 - waveform_tensor) > 1e-6).float()

# Second augmentation - track mask
waveform_aug2 = freq_chunk_augmentation(waveform_tensor.clone())
waveform_aug2 = time_chunk_augmentation(waveform_aug2, rel_length)
# Create binary mask: 1 where values changed (augmented), 0 where original
mask_aug2_binary = (torch.abs(waveform_aug2 - waveform_tensor) > 1e-6).float()

# Compute mel spectrograms for all three
mel_spec_orig = librosa.feature.melspectrogram(y=waveform_tensor[0].numpy(), sr=sr, n_mels=80)
mel_spec_db_orig = librosa.power_to_db(mel_spec_orig, ref=np.max)

mel_spec_aug1 = librosa.feature.melspectrogram(y=waveform_aug1[0].detach().numpy(), sr=sr, n_mels=80)
mel_spec_db_aug1 = librosa.power_to_db(mel_spec_aug1, ref=np.max)

mel_spec_aug2 = librosa.feature.melspectrogram(y=waveform_aug2[0].detach().numpy(), sr=sr, n_mels=80)
mel_spec_db_aug2 = librosa.power_to_db(mel_spec_aug2, ref=np.max)

# Also compute mel spectrogram of mask signals (shows rectangular regions)
mel_spec_mask_aug1 = librosa.feature.melspectrogram(y=mask_aug1_binary[0].numpy(), sr=sr, n_mels=80)
mel_spec_mask_aug1 = (mel_spec_mask_aug1 > 0).astype(float)  # Binary mask (0 or 1)

mel_spec_mask_aug2 = librosa.feature.melspectrogram(y=mask_aug2_binary[0].numpy(), sr=sr, n_mels=80)
mel_spec_mask_aug2 = (mel_spec_mask_aug2 > 0).astype(float)  # Binary mask (0 or 1)

# Normalize spectrograms
def normalize_spec(spec):
    return (spec - spec.min()) / (spec.max() - spec.min() + 1e-10) * 100

mel_spec_db_orig_norm = normalize_spec(mel_spec_db_orig)
mel_spec_db_aug1_norm = normalize_spec(mel_spec_db_aug1)
mel_spec_db_aug2_norm = normalize_spec(mel_spec_db_aug2)

# Create augmented spectrograms with rectangular mask highlights
mel_spec_aug1_marked = mel_spec_db_aug1_norm.copy()
mel_spec_aug1_marked[mel_spec_mask_aug1 > 0.5] = 100  # Set mask regions to white (rectangular)

mel_spec_aug2_marked = mel_spec_db_aug2_norm.copy()
mel_spec_aug2_marked[mel_spec_mask_aug2 > 0.5] = 100  # Set mask regions to white (rectangular)

# Output directory
output_dir = Path(f"augmentation_comparison_{file_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Create clean spectrograms (no axes, labels, or annotations)
# Original
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(mel_spec_db_orig_norm, aspect='auto', origin='lower', cmap='viridis')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
fig.savefig(output_dir / f"{file_id}_original_clean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved original spectrogram to {output_dir / f'{file_id}_original_clean.png'}")

# Augmentation 1 (with rectangular mask positions)
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(mel_spec_aug1_marked, aspect='auto', origin='lower', cmap='viridis')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
fig.savefig(output_dir / f"{file_id}_augmentation_v1_clean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved augmentation v1 to {output_dir / f'{file_id}_augmentation_v1_clean.png'}")

# Augmentation 2 (with different rectangular mask positions)
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(mel_spec_aug2_marked, aspect='auto', origin='lower', cmap='viridis')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
fig.savefig(output_dir / f"{file_id}_augmentation_v2_clean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved augmentation v2 to {output_dir / f'{file_id}_augmentation_v2_clean.png'}")

# Create comparison figure (3 images side by side)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].imshow(mel_spec_db_orig_norm, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_xticks([])
axes[0].set_yticks([])
for spine in axes[0].spines.values():
    spine.set_visible(False)

axes[1].imshow(mel_spec_aug1_marked, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_xticks([])
axes[1].set_yticks([])
for spine in axes[1].spines.values():
    spine.set_visible(False)

axes[2].imshow(mel_spec_aug2_marked, aspect='auto', origin='lower', cmap='viridis')
axes[2].set_xticks([])
axes[2].set_yticks([])
for spine in axes[2].spines.values():
    spine.set_visible(False)

plt.tight_layout()
fig.savefig(output_dir / f"{file_id}_augmentation_comparison_clean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved comparison to {output_dir / f'{file_id}_augmentation_comparison_clean.png'}")

print(f"\n=== Output Complete ===")
print(f"All images saved to: {output_dir}/")
print(f"  1. {file_id}_original_clean.png - Original spectrogram")
print(f"  2. {file_id}_augmentation_v1_clean.png - Augmentation v1 (with rectangular mask regions)")
print(f"  3. {file_id}_augmentation_v2_clean.png - Augmentation v2 (with different rectangular mask regions)")
print(f"  4. {file_id}_augmentation_comparison_clean.png - All three side by side")
print(f"\nMask regions shown in WHITE (100) represent augmented areas")
