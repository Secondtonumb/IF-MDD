"""
Generate Phoneme Variants with Confusion Matrix

This script takes a CSV file with predicted phoneme sequences and a confusion matrix JSON,
then generates variants by replacing each phoneme with its top 5 potential confusions.

Input CSV format:
    id,Labels
    00000_00034,< i y aa k n E b a d u
    ...

Output CSV format:
    id,Labels,variant_1,variant_2,...
    
Author: Haopeng (Kevin) Geng
Year: 2026
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


def load_confusion_matrix(json_path: str) -> Dict[str, List[str]]:
    """
    Load confusion matrix from JSON file.
    Returns {phoneme: [top_5_confusions]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    confusion_map = {}
    
    for phoneme, info in data.get("confusion_matrix", {}).items():
        # Get top 5 confusions
        confusions = info.get("confusions", [])
        top_5 = [c["phoneme"] for c in confusions[:5]]
        confusion_map[phoneme] = top_5
    
    logger.info(f"Loaded confusion matrix for {len(confusion_map)} phonemes")
    return confusion_map


def generate_variants_for_position(phoneme_seq: List[str], position: int, confusion_map: Dict[str, List[str]]) -> List[List[str]]:
    """
    Generate all variants by replacing phoneme at position with its confusions.
    Returns list of sequences.
    """
    if position >= len(phoneme_seq):
        return []
    
    current_phoneme = phoneme_seq[position]
    confusions = confusion_map.get(current_phoneme, [])
    
    variants = []
    for confusion_phoneme in confusions:
        new_seq = phoneme_seq.copy()
        new_seq[position] = confusion_phoneme
        variants.append(new_seq)
    
    return variants


def generate_all_variants(phoneme_seq: List[str], confusion_map: Dict[str, List[str]], max_k: int = 5) -> List[List[str]]:
    """
    Generate all variants for a phoneme sequence.
    For each position, generate up to max_k variants by replacing with confusions.
    
    Returns list of all unique variants (excluding original).
    """
    all_variants = set()
    
    for position in range(len(phoneme_seq)):
        position_variants = generate_variants_for_position(phoneme_seq, position, confusion_map)
        
        for variant in position_variants:
            variant_str = ' '.join(variant)
            all_variants.add(variant_str)
    
    return list(all_variants)


def process_csv_file(input_csv: str, output_csv: str, confusion_json: str):
    """
    Process CSV file and generate variants with confusion matrix.
    """
    # Load confusion matrix
    logger.info(f"Loading confusion matrix from {confusion_json}")
    confusion_map = load_confusion_matrix(confusion_json)
    
    # Read input CSV
    logger.info(f"Reading input CSV from {input_csv}")
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"Loaded {len(rows)} rows from input CSV")
    
    # Process each row
    output_rows = []
    total_variants = 0
    
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing row {i + 1}/{len(rows)}")
        
        row_id = row['id']
        labels = row['Labels'].strip()
        phoneme_seq = labels.split()
        
        # Generate variants for this sequence
        variants = generate_all_variants(phoneme_seq, confusion_map)
        
        # Create output row
        output_row = {
            'id': row_id,
            'original': labels,
            'num_variants': len(variants)
        }
        
        # Add variants as separate columns
        for j, variant in enumerate(sorted(variants)):
            output_row[f'variant_{j+1}'] = variant
        
        output_rows.append(output_row)
        total_variants += len(variants)
    
    # Write output CSV
    logger.info(f"Writing output CSV to {output_csv}")
    
    # Collect all field names
    all_fieldnames = ['id', 'original', 'num_variants']
    max_variants = max(row['num_variants'] for row in output_rows)
    for i in range(1, max_variants + 1):
        all_fieldnames.append(f'variant_{i}')
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        
        # Fill missing variant columns with empty strings
        for row in output_rows:
            for field in all_fieldnames:
                if field not in row:
                    row[field] = ''
            writer.writerow(row)
    
    logger.info(f"✓ wrote {len(output_rows)} rows to output CSV")
    logger.info(f"✓ Total variants generated: {total_variants}")
    logger.info(f"✓ Average variants per sequence: {total_variants / len(output_rows):.1f}")


def process_csv_file_compact(input_csv: str, output_csv: str, confusion_json: str):
    """
    Process CSV file and generate variants with confusion matrix.
    Output format: id,original,variant_1,variant_2,...
    (All variants in a single row)
    """
    # Load confusion matrix
    logger.info(f"Loading confusion matrix from {confusion_json}")
    confusion_map = load_confusion_matrix(confusion_json)
    
    # Read input CSV
    logger.info(f"Reading input CSV from {input_csv}")
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"Loaded {len(rows)} rows from input CSV")
    
    # Process each row
    output_rows = []
    total_variants = 0
    max_variants_seen = 0
    
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing row {i + 1}/{len(rows)}")
        
        row_id = row['id']
        labels = row['Labels'].strip()
        phoneme_seq = labels.split()
        
        # Generate variants for this sequence
        variants = generate_all_variants(phoneme_seq, confusion_map)
        
        # Create output row
        output_row = {'id': row_id, 'original': labels}
        
        # Add variants as separate columns
        for j, variant in enumerate(sorted(variants)):
            output_row[f'variant_{j+1}'] = variant
        
        output_rows.append(output_row)
        total_variants += len(variants)
        max_variants_seen = max(max_variants_seen, len(variants))
    
    # Write output CSV with dynamic columns
    logger.info(f"Writing compact output CSV to {output_csv}")
    
    # Collect all field names
    all_fieldnames = ['id', 'original']
    for i in range(1, max_variants_seen + 1):
        all_fieldnames.append(f'variant_{i}')
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval='')
        writer.writeheader()
        writer.writerows(output_rows)
    
    logger.info(f"✓ Wrote {len(output_rows)} rows to output CSV")
    logger.info(f"✓ Total variants generated: {total_variants}")
    logger.info(f"✓ Average variants per sequence: {total_variants / len(output_rows):.1f}")
    logger.info(f"✓ Max variants in single sequence: {max_variants_seen}")


def main():
    # Configuration
    input_csv = "/home/m64000/work/IF-MDD/utils/IFMDD_ConPCO_inference_results_0.9999_k3_seq2seq.csv"
    confusion_json = "/home/m64000/work/IF-MDD/utils/iqra_labeled_conf_labeled_all_top_5.json"
    output_csv = "/home/m64000/work/IF-MDD/utils/IFMDD_ConPCO_with_phoneme_variants.csv"
    
    logger.info("="*80)
    logger.info("Phoneme Variant Generation with Confusion Matrix")
    logger.info("="*80)
    logger.info(f"Input CSV:       {input_csv}")
    logger.info(f"Confusion JSON:  {confusion_json}")
    logger.info(f"Output CSV:      {output_csv}")
    logger.info("")
    
    # Validate files exist
    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found: {input_csv}")
        return
    
    if not Path(confusion_json).exists():
        logger.error(f"Confusion JSON not found: {confusion_json}")
        return
    
    # Process file
    process_csv_file_compact(input_csv, output_csv, confusion_json)
    
    logger.info("")
    logger.info("="*80)
    logger.info("DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()
