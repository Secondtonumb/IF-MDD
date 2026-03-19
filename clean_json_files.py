#!/usr/bin/env python3
"""
Clean JSON files by keeping only specified keys and updating perceived_train_target
"""
import json
import os

# Keys to keep
KEYS_TO_KEEP = {
    "wrd",
    "wrd_pinyin", 
    "wav",
    "duration",
    "spk_id",
    "canonical_aligned",
    "perceived_aligned"
}

def clean_json_file(filepath):
    """Clean a JSON file by removing unwanted keys and updating perceived_train_target"""
    print(f"Processing {filepath}...")
    
    # Read the JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each entry
    cleaned_data = {}
    for key, entry in data.items():
        cleaned_entry = {}
        
        # Keep only specified keys
        for keep_key in KEYS_TO_KEEP:
            if keep_key in entry:
                cleaned_entry[keep_key] = entry[keep_key]
        
        # Add perceived_train_target same as perceived_aligned
        if "perceived_aligned" in cleaned_entry:
            cleaned_entry["perceived_train_target"] = cleaned_entry["perceived_aligned"]
        
        cleaned_data[key] = cleaned_entry
    
    # Write back to the file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Cleaned {filepath} - {len(cleaned_data)} entries processed")

# Process all three files
base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
files = [
    os.path.join(base_dir, "train.json"),
    os.path.join(base_dir, "dev.json"),
    os.path.join(base_dir, "test.json")
]

for filepath in files:
    if os.path.exists(filepath):
        clean_json_file(filepath)
    else:
        print(f"✗ File not found: {filepath}")

print("\nAll files processed!")
