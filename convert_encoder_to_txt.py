#!/usr/bin/env python3
"""
Convert label_encoder.json to txt format
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

print(f"Writing to {txt_path}...")
with open(txt_path, 'w', encoding='utf-8') as f:
    for idx, label in enumerate(vocab):
        f.write(f"{label}\n")

print(f"\n✓ Converted to TXT format")
print(f"Total lines: {len(vocab)}")
print(f"\nFirst 10 labels:")
for i, label in enumerate(vocab[:10]):
    print(f"  {i}: {label}")
print(f"\nLast 5 labels:")
for i, label in enumerate(vocab[-5:], len(vocab)-5):
    print(f"  {i}: {label}")
