#!/usr/bin/env python3
"""
Generate label_encoder.txt with the target format:
'label' => index
...
'<blank>' => 0
'<bos>' => starting_index
'<eos>' => starting_index + 1
================
metadata
"""
import json
import os

base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
json_path = os.path.join(base_dir, "label_encoder.json")
txt_path = os.path.join(base_dir, "label_encoder.txt")

print(f"Reading {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    encoder = json.load(f)

vocab = encoder["vocab"]

# Remove special tokens from regular vocab
regular_labels = [label for label in vocab if label not in ["<bos>", "<eos>"]]

print(f"Total labels: {len(vocab)}")
print(f"Regular labels: {len(regular_labels)}")

# Build output with correct format
lines = []

# Add regular labels starting from index 1
for idx, label in enumerate(regular_labels, start=1):
    lines.append(f"'{label}' => {idx}")

# Add special tokens
lines.append(f"'<blank>' => 0")
lines.append(f"'<bos>' => {len(regular_labels) + 1}")
lines.append(f"'<eos>' => {len(regular_labels) + 2}")

# Add metadata separator and info
lines.append("================")
lines.append(f"'starting_index' => 0")
lines.append(f"'blank_label' => '<blank>'")

print(f"Writing to {txt_path}...")
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"\n✓ Generated label_encoder.txt in target format")
print(f"Total lines: {len(lines)}")
print(f"\nFirst 10 lines:")
for line in lines[:10]:
    print(f"  {line}")
print(f"\nLast 10 lines:")
for line in lines[-10:]:
    print(f"  {line}")
