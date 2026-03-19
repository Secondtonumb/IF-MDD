#!/usr/bin/env python3
"""
Replace JSON keys with corresponding wav file paths
"""
import json
import os

def replace_keys_with_wav_paths(filepath):
    """Replace dict keys with wav path values"""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = {}
    
    for key, entry in data.items():
        wav_path = entry.get("wav", "")
        
        if not wav_path:
            print(f"  ⚠ Warning: {key} has no wav path, skipping")
            continue
        
        # Use wav path as new key
        new_data[wav_path] = entry
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Converted {len(new_data)} entries")
    return len(new_data)

# Process all three files
base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
files = ["train.json", "dev.json", "test.json"]

print("="*70)
print("Replacing JSON keys with wav file paths")
print("="*70)

total = 0
for filename in files:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        count = replace_keys_with_wav_paths(filepath)
        total += count
    else:
        print(f"✗ File not found: {filepath}")

print("\n" + "="*70)
print(f"Total: {total} entries processed")
print("="*70)
