#!/usr/bin/env python3
"""
Design label encoder with:
1. Statistics of all phones from canonical_aligned
2. Create two versions for each phone (original + variant with 0)
3. Add special tokens: sil, <bos>, <eos>
"""
import json
from collections import Counter
import os

def extract_phones_from_file(filepath):
    """Extract all phones from canonical_aligned field"""
    phones_list = []
    
    print(f"Reading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, entry in data.items():
        if "canonical_aligned" in entry:
            phones = entry["canonical_aligned"].split()
            phones_list.extend(phones)
    
    return phones_list

# Collect phones from all three files
base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
all_phones = []

for filename in ["train.json", "dev.json", "test.json"]:
    filepath = os.path.join(base_dir, filename)
    phones = extract_phones_from_file(filepath)
    all_phones.extend(phones)

# Count phone frequencies
phone_counter = Counter(all_phones)
unique_phones = sorted(set(all_phones))

print("\n" + "="*60)
print("PHONE STATISTICS")
print("="*60)
print(f"Total phones: {len(all_phones)}")
print(f"Unique phones: {len(unique_phones)}")
print(f"\nPhone frequencies:")
for phone, count in sorted(phone_counter.items(), key=lambda x: -x[1]):
    print(f"  {phone}: {count}")

# Design label encoder
print("\n" + "="*60)
print("LABEL ENCODER DESIGN")
print("="*60)

# Create vocabulary with original + variant (with 0)
vocab = []

# First add 'sil' with its variant
vocab.append("sil")
vocab.append("sil0")

# Add all other phones with their variants
for phone in unique_phones:
    if phone != "sil":
        vocab.append(phone)
        vocab.append(phone + "0")

# Add special tokens
vocab.append("<bos>")
vocab.append("<eos>")

# Create label to index mapping
label_to_idx = {label: idx for idx, label in enumerate(vocab)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print(f"\nTotal vocabulary size: {len(vocab)}")
print(f"\nVocabulary (label -> index):")
for idx, label in enumerate(vocab):
    print(f"  {idx}: {label}")

# Save encoder mappings
encoder_data = {
    "vocab": vocab,
    "label_to_idx": label_to_idx,
    "idx_to_label": idx_to_label,
    "unique_phones": unique_phones,
    "phone_stats": dict(phone_counter)
}

output_path = os.path.join(base_dir, "label_encoder.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(encoder_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Label encoder saved to {output_path}")
print(f"\nEncoder summary:")
print(f"  - Unique phones: {len(unique_phones)}")
print(f"  - Total vocab with variants: {len(vocab) - 2} (phones only)")
print(f"  - Special tokens: <bos>, <eos>")
print(f"  - Total vocabulary size: {len(vocab)}")
