#!/usr/bin/env python3
"""
Generate JSONL files with phoneme targets (canonical and perceived).
Processes JSON metadata files and creates separate JSONL datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_json_metadata(json_file: str) -> Dict:
    """Load JSON metadata file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_phoneme_jsonl(
    json_metadata_file: str,
    output_dir: str = ".",
    phoneme_type: str = "canonical"
) -> Tuple[str, int]:
    """
    Generate JSONL file with phoneme targets.
    
    Args:
        json_metadata_file: Path to JSON metadata file
        output_dir: Output directory for JSONL files
        phoneme_type: "canonical", "perceived", or "both"
    
    Returns:
        Tuple of (output_file_path, num_entries)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata = load_json_metadata(json_metadata_file)
    
    # Determine output files based on phoneme_type
    base_name = Path(json_metadata_file).stem  # e.g., "test" from "test.json"
    
    output_files = {}
    if phoneme_type in ["canonical", "both"]:
        output_files["canonical"] = os.path.join(
            output_dir, 
            f"{base_name}_phoneme_canonical.jsonl"
        )
    if phoneme_type in ["perceived", "both"]:
        output_files["perceived"] = os.path.join(
            output_dir, 
            f"{base_name}_phoneme_perceived.jsonl"
        )
    
    # Open output files
    file_handles = {}
    for key, path in output_files.items():
        file_handles[key] = open(path, 'w', encoding='utf-8')
    
    try:
        # Process each entry
        count = 0
        for wav_path, data in tqdm(metadata.items(), desc=f"Processing {base_name}"):
            if not isinstance(data, dict):
                continue
            
            # Extract key information
            key = Path(wav_path).stem  # Use wav filename as key
            spk_id = data.get("spk_id", "")
            
            # Create entries for each phoneme type
            if "canonical" in file_handles:
                canonical_phoneme = data.get("canonical_aligned", "")
                if canonical_phoneme:
                    entry = {
                        "key": f"{spk_id}_{key}_canonical_phoneme",
                        "source": wav_path,
                        "target": canonical_phoneme
                    }
                    file_handles["canonical"].write(json.dumps(entry) + '\n')
                    count += 1
            
            if "perceived" in file_handles:
                perceived_phoneme = data.get("perceived_aligned", "")
                if perceived_phoneme:
                    entry = {
                        "key": f"{spk_id}_{key}_perceived_phoneme",
                        "source": wav_path,
                        "target": perceived_phoneme
                    }
                    file_handles["perceived"].write(json.dumps(entry) + '\n')
                    count += 1
    
    finally:
        # Close all files
        for fh in file_handles.values():
            fh.close()
    
    # Print results
    result_msg = f"\nProcessed {json_metadata_file}:"
    for key, path in output_files.items():
        result_msg += f"\n  ✓ {key}: {path}"
    
    print(result_msg)
    
    return result_msg, count


def batch_generate_phoneme_jsonl(
    data_dir: str = "/home/kevingenghaopeng/MDD/IF-MDD/data",
    output_dir: str = None,
    phoneme_type: str = "both"
):
    """
    Batch generate JSONL files for multiple datasets.
    
    Args:
        data_dir: Directory containing JSON metadata files
        output_dir: Output directory for JSONL files
        phoneme_type: "canonical", "perceived", or "both"
    """
    
    if output_dir is None:
        output_dir = os.path.join(data_dir, "jsonl_datasets_phoneme")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # List of JSON files to process
    json_files = [
        "test.json",
        "train.json",
        "test_erj_spk_open_test_1.1.json",
        "train_erj_spk_open_train_1.1.json",
    ]
    
    total_entries = 0
    
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"⚠ Skipping {json_file}: not found")
            continue
        
        try:
            _, count = generate_phoneme_jsonl(
                json_path,
                output_dir=output_dir,
                phoneme_type=phoneme_type
            )
            total_entries += count
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total entries generated: {total_entries}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate JSONL files with phoneme targets"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        help="Single JSON metadata file to process"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/kevingenghaopeng/MDD/IF-MDD/data",
        help="Directory containing JSON metadata files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--phoneme_type",
        type=str,
        choices=["canonical", "perceived", "both"],
        default="both",
        help="Which phoneme types to generate"
    )
    
    args = parser.parse_args()
    
    if args.json_file:
        # Process single file
        generate_phoneme_jsonl(
            args.json_file,
            output_dir=args.output_dir or ".",
            phoneme_type=args.phoneme_type
        )
    else:
        # Batch process
        batch_generate_phoneme_jsonl(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            phoneme_type=args.phoneme_type
        )
