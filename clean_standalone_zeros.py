#!/usr/bin/env python3
"""
Find and remove entries with standalone '0' in perceived_aligned or canonical_aligned
"""
import json
import os

def check_standalone_zero(aligned_str):
    """Check if there's a standalone '0' in the aligned string"""
    if not aligned_str:
        return False
    phones = aligned_str.split()
    return '0' in phones

base_dir = "/home/m64000/work/IF-MDD/data_CRJ"
files = ["train.json", "dev.json", "test.json"]

for filename in files:
    filepath = os.path.join(base_dir, filename)
    print(f"\nProcessing {filename}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problematic_entries = []
    entries_to_remove = []
    
    for key, entry in data.items():
        perceived_aligned = entry.get("perceived_aligned", "")
        canonical_aligned = entry.get("canonical_aligned", "")
        
        has_zero_perceived = check_standalone_zero(perceived_aligned)
        has_zero_canonical = check_standalone_zero(canonical_aligned)
        
        if has_zero_perceived or has_zero_canonical:
            problematic_entries.append({
                "key": key,
                "wrd": entry.get("wrd", ""),
                "perceived_aligned": perceived_aligned,
                "canonical_aligned": canonical_aligned,
                "has_zero_perceived": has_zero_perceived,
                "has_zero_canonical": has_zero_canonical
            })
            entries_to_remove.append(key)
    
    print(f"Found {len(problematic_entries)} problematic entries:")
    for entry in problematic_entries:
        print(f"\n  Key: {entry['key']}")
        print(f"  Word: {entry['wrd']}")
        if entry['has_zero_perceived']:
            print(f"  perceived_aligned: {entry['perceived_aligned']}")
        if entry['has_zero_canonical']:
            print(f"  canonical_aligned: {entry['canonical_aligned']}")
    
    # Remove problematic entries
    for key in entries_to_remove:
        del data[key]
    
    # Save cleaned data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ Removed {len(entries_to_remove)} entries")
    print(f"  Remaining entries: {len(data)}")

print("\n✓ All files cleaned!")
