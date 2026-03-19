#!/usr/bin/env python3
"""
Generate label sequences for train/dev/test data based on label encoder
"""
import json
import os

def add_label_sequences(data_filepath, label_encoder):
    """
    Add label sequences to each entry:
    - canonical_phones: phones from canonical_aligned
    - canonical_extended: each phone + its 0 variant (doubled)
    - label_indices: indices for canonical_extended with <bos> and <eos>
    """
    print(f"Processing {data_filepath}...")
    
    with open(data_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    label_to_idx = label_encoder["label_to_idx"]
    processed_count = 0
    error_count = 0
    
    for key, entry in data.items():
        if "canonical_aligned" not in entry:
            print(f"  ⚠ Warning: {key} has no canonical_aligned")
            continue
        
        try:
            # Extract phones from canonical_aligned
            canonical_aligned_str = entry["canonical_aligned"]
            canonical_phones = canonical_aligned_str.split()
            
            # Create extended sequence: each phone followed by its 0 variant
            canonical_extended = []
            for phone in canonical_phones:
                canonical_extended.append(phone)
                canonical_extended.append(phone + "0")
            
            # Create label indices with <bos> and <eos>
            label_indices = [label_to_idx["<bos>"]]
            
            for label in canonical_extended:
                if label in label_to_idx:
                    label_indices.append(label_to_idx[label])
                else:
                    print(f"  ⚠ Warning: {key} has unknown label: {label}")
                    error_count += 1
            
            label_indices.append(label_to_idx["<eos>"])
            
            # Add to entry
            entry["canonical_phones"] = canonical_phones
            entry["canonical_extended"] = canonical_extended
            entry["label_indices"] = label_indices
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {key}: {e}")
            error_count += 1
    
    # Write back
    with open(data_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ {processed_count} entries processed, {error_count} errors")
    return processed_count, error_count

# Load label encoder
base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
encoder_path = os.path.join(base_dir, "label_encoder.json")

print(f"Loading label encoder from {encoder_path}...")
with open(encoder_path, 'r', encoding='utf-8') as f:
    label_encoder = json.load(f)

print(f"Vocabulary size: {len(label_encoder['vocab'])}\n")

# Process all three files
print("="*60)
print("Processing data files")
print("="*60)

total_processed = 0
total_errors = 0

for filename in ["train.json", "dev.json", "test.json"]:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        processed, errors = add_label_sequences(filepath, label_encoder)
        total_processed += processed
        total_errors += errors
    else:
        print(f"✗ File not found: {filepath}")

print("\n" + "="*60)
print(f"Total: {total_processed} entries processed, {total_errors} errors")
print("="*60)
