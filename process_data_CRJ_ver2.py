#!/usr/bin/env python3
"""
Complete processing pipeline for data_CRJ_ver2:
1. Clean JSON - keep only specified keys
2. Set perceived_train_target = perceived_aligned
3. Remove entries with standalone '0'
4. Generate label encoder
5. Add label sequences
6. Replace keys with wav paths
"""
import json
import os
from collections import Counter

def step1_clean_json(base_dir):
    """Step 1: Clean JSON files - keep only specified keys"""
    print("\n" + "="*70)
    print("STEP 1: Cleaning JSON files")
    print("="*70)
    
    KEYS_TO_KEEP = {
        "wrd", "wrd_pinyin", "wav", "duration", "spk_id",
        "canonical_aligned", "perceived_aligned"
    }
    
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = os.path.join(base_dir, filename)
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = {}
        for key, entry in data.items():
            cleaned_entry = {}
            for keep_key in KEYS_TO_KEEP:
                if keep_key in entry:
                    cleaned_entry[keep_key] = entry[keep_key]
            if "perceived_aligned" in cleaned_entry:
                cleaned_entry["perceived_train_target"] = cleaned_entry["perceived_aligned"]
            cleaned_data[key] = cleaned_entry
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Cleaned {len(cleaned_data)} entries")

def step2_remove_standalone_zeros(base_dir):
    """Step 2: Remove entries with standalone '0'"""
    print("\n" + "="*70)
    print("STEP 2: Removing entries with standalone '0'")
    print("="*70)
    
    def check_standalone_zero(aligned_str):
        if not aligned_str:
            return False
        return '0' in aligned_str.split()
    
    total_removed = 0
    
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = os.path.join(base_dir, filename)
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries_to_remove = []
        for key, entry in data.items():
            perceived_aligned = entry.get("perceived_aligned", "")
            canonical_aligned = entry.get("canonical_aligned", "")
            
            if check_standalone_zero(perceived_aligned) or check_standalone_zero(canonical_aligned):
                entries_to_remove.append(key)
        
        for key in entries_to_remove:
            del data[key]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Removed {len(entries_to_remove)} entries, {len(data)} remaining")
        total_removed += len(entries_to_remove)
    
    return total_removed

def step3_generate_label_encoder(base_dir):
    """Step 3: Generate label encoder"""
    print("\n" + "="*70)
    print("STEP 3: Generating label encoder")
    print("="*70)
    
    all_phones = []
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data.values():
            if "canonical_aligned" in entry:
                phones = entry["canonical_aligned"].split()
                all_phones.extend(phones)
    
    unique_phones = sorted(set(all_phones))
    phone_counter = Counter(all_phones)
    
    # Build vocabulary
    vocab = []
    vocab.append("sil")
    vocab.append("sil0")
    for phone in unique_phones:
        if phone != "sil":
            vocab.append(phone)
            vocab.append(phone + "0")
    vocab.append("<bos>")
    vocab.append("<eos>")
    
    label_to_idx = {label: idx for idx, label in enumerate(vocab)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    encoder_data = {
        "vocab": vocab,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "unique_phones": unique_phones,
        "phone_stats": dict(phone_counter)
    }
    
    encoder_path = os.path.join(base_dir, "label_encoder.json")
    with open(encoder_path, 'w', encoding='utf-8') as f:
        json.dump(encoder_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Generated label encoder")
    print(f"    - Unique phones: {len(unique_phones)}")
    print(f"    - Vocabulary size: {len(vocab)}")
    
    return encoder_data

def step4_add_label_sequences(base_dir, encoder_data):
    """Step 4: Add label sequences"""
    print("\n" + "="*70)
    print("STEP 4: Adding label sequences")
    print("="*70)
    
    label_to_idx = encoder_data["label_to_idx"]
    total_processed = 0
    
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = os.path.join(base_dir, filename)
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, entry in data.items():
            if "canonical_aligned" in entry:
                canonical_aligned_str = entry["canonical_aligned"]
                canonical_phones = canonical_aligned_str.split()
                
                canonical_extended = []
                for phone in canonical_phones:
                    canonical_extended.append(phone)
                    canonical_extended.append(phone + "0")
                
                label_indices = [label_to_idx["<bos>"]]
                for label in canonical_extended:
                    if label in label_to_idx:
                        label_indices.append(label_to_idx[label])
                label_indices.append(label_to_idx["<eos>"])
                
                entry["canonical_phones"] = canonical_phones
                entry["canonical_extended"] = canonical_extended
                entry["label_indices"] = label_indices
                total_processed += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Processed {len(data)} entries")
    
    return total_processed

def step5_convert_encoder_to_txt(base_dir):
    """Step 5: Convert label encoder to txt format"""
    print("\n" + "="*70)
    print("STEP 5: Converting label encoder to TXT format")
    print("="*70)
    
    json_path = os.path.join(base_dir, "label_encoder.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        encoder = json.load(f)
    
    vocab = encoder["vocab"]
    regular_labels = [label for label in vocab if label not in ["<bos>", "<eos>"]]
    
    txt_path = os.path.join(base_dir, "label_encoder.txt")
    lines = []
    
    for idx, label in enumerate(regular_labels, start=1):
        lines.append(f"'{label}' => {idx}")
    
    lines.append(f"'<blank>' => 0")
    lines.append(f"'<bos>' => {len(regular_labels) + 1}")
    lines.append(f"'<eos>' => {len(regular_labels) + 2}")
    lines.append("================")
    lines.append(f"'starting_index' => 0")
    lines.append(f"'blank_label' => '<blank>'")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"  ✓ Generated label_encoder.txt with {len(lines)} lines")

def step6_replace_keys_with_wav_paths(base_dir):
    """Step 6: Replace JSON keys with wav paths"""
    print("\n" + "="*70)
    print("STEP 6: Replacing JSON keys with wav paths")
    print("="*70)
    
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = os.path.join(base_dir, filename)
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_data = {}
        for key, entry in data.items():
            wav_path = entry.get("wav", "")
            if wav_path:
                new_data[wav_path] = entry
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Converted {len(new_data)} entries")

# Main execution
if __name__ == "__main__":
    base_dir = "/home/m64000/work/IF-MDD/data_CRJ_ver2"
    
    print("\n" + "="*70)
    print("COMPLETE PROCESSING PIPELINE FOR data_CRJ_ver2")
    print("="*70)
    
    # Step 1: Clean JSON
    step1_clean_json(base_dir)
    
    # Step 2: Remove standalone zeros
    step2_remove_standalone_zeros(base_dir)
    
    # Step 3: Generate label encoder
    encoder_data = step3_generate_label_encoder(base_dir)
    
    # Step 4: Add label sequences
    step4_add_label_sequences(base_dir, encoder_data)
    
    # Step 5: Convert to TXT format
    step5_convert_encoder_to_txt(base_dir)
    
    # Step 6: Replace keys with wav paths
    step6_replace_keys_with_wav_paths(base_dir)
    
    print("\n" + "="*70)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*70)
