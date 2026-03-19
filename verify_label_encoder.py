#!/usr/bin/env python3
"""
Verify label encoder consistency with generated data
"""
import json
import os

base_dir = "/home/m64000/work/IF-MDD/data_CRJ"

# Load encoder
with open(os.path.join(base_dir, "label_encoder.json"), 'r', encoding='utf-8') as f:
    encoder = json.load(f)

# Load one sample
with open(os.path.join(base_dir, "train.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get first entry
first_key = list(data.keys())[0]
entry = data[first_key]

print("="*70)
print("LABEL ENCODER VERIFICATION")
print("="*70)
print(f"\nSample entry: {first_key}")
print(f"\nCanonical aligned: {entry['canonical_aligned']}")
print(f"\nCanonical phones: {entry['canonical_phones']}")
print(f"\nCanonical extended: {entry['canonical_extended']}")
print(f"\nLabel indices: {entry['label_indices']}")

# Verify indices map back to correct labels
print("\n" + "-"*70)
print("Index to Label Mapping Verification:")
print("-"*70)
idx_to_label = encoder["idx_to_label"]

for idx in entry["label_indices"]:
    label = idx_to_label[str(idx)]
    print(f"  {idx:3d} → {label}")

# Statistics
print("\n" + "-"*70)
print("Statistics:")
print("-"*70)
print(f"Label indices length: {len(entry['label_indices'])}")
print(f"  - <bos>: index 160")
print(f"  - Extended phones: {len(entry['canonical_extended'])} phones")
print(f"  - <eos>: index 161")
print(f"  - Total: 1 + {len(entry['canonical_extended'])} + 1 = {len(entry['label_indices'])}")

print(f"\nVocabulary size: {len(encoder['vocab'])}")
print(f"Unique phones: {len(encoder['unique_phones'])}")
print(f"Total phones × 2 (with 0 variant): {len(encoder['unique_phones']) * 2}")

print("\n✓ Label encoder verification complete!")
