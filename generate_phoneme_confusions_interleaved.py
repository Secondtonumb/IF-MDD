"""
Generate Phoneme Sequence with Interleaved Confusion Variants

This script takes a CSV file with predicted phoneme sequences and a confusion matrix JSON.
For each phoneme, it adds the top 5 potential confusion phonemes inline.

Input CSV format:
    id,Labels
    00000_00034,A B C
    ...

Output CSV format:
    id,Labels
    00000_00034,"A A1 A2 A3 A4 A5 B B1 B2 B3 B4 B5 C C1 C2 C3 C4 C5"

Where A1-A5 are the top 5 confusions for phoneme A from the confusion matrix.

Author: Haopeng (Kevin) Geng
Year: 2026
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
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


def expand_phoneme_sequence(phoneme_seq: List[str], confusion_map: Dict[str, List[str]]) -> List[str]:
    """
    Expand each phoneme with its top 5 confusions.
    
    Input:  [A, B, C]
    Output: [A, A1, A2, A3, A4, A5, B, B1, B2, B3, B4, B5, C, C1, C2, C3, C4, C5]
    
    Returns list of phonemes.
    """
    expanded = []
    
    for phoneme in phoneme_seq:
        # Add original phoneme
        expanded.append(phoneme)
        
        # Add top 5 confusions
        confusions = confusion_map.get(phoneme, [])
        expanded.extend(confusions)
    
    return expanded


def process_csv_file(input_csv: str, output_csv: str, confusion_json: str):
    """
    Process CSV file and generate sequences with interleaved confusion variants.
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
    
    for i, row in enumerate(rows):
        if (i + 1) % 200 == 0:
            logger.info(f"Processing row {i + 1}/{len(rows)}")
        
        row_id = row['id']
        labels = row['Labels'].strip()
        phoneme_seq = labels.split()
        
        # Expand sequence with confusion variants
        expanded_seq = expand_phoneme_sequence(phoneme_seq, confusion_map)
        
        # Create output row (only id and Labels)
        output_row = {
            'id': row_id,
            'Labels': ' '.join(expanded_seq)
        }
        
        output_rows.append(output_row)
    
    # Write output CSV
    logger.info(f"Writing output CSV to {output_csv}")
    
    fieldnames = ['id', 'Labels']
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    logger.info(f"✓ Wrote {len(output_rows)} rows to output CSV")


def main():
    # Configuration
    input_csv = "/home/m64000/work/IF-MDD/utils/IFMDD_ConPCO_inference_results_0.9999_k3_seq2seq.csv"
    confusion_json = "/home/m64000/work/IF-MDD/utils/iqra_labeled_conf_labeled_all_top_5.json"
    output_csv = "/home/m64000/work/IF-MDD/utils/IFMDD_ConPCO_with_phoneme_confusions_interleaved.csv"
    
    logger.info("="*80)
    logger.info("Phoneme Sequence Expansion with Confusion Matrix")
    logger.info("="*80)
    logger.info(f"Input CSV:       {input_csv}")
    logger.info(f"Confusion JSON:  {confusion_json}")
    logger.info(f"Output CSV:      {output_csv}")
    logger.info("")
    logger.info("Strategy: A B C → A A1 A2 A3 A4 A5 B B1 B2 B3 B4 B5 C C1 C2 C3 C4 C5")
    logger.info("")
    
    # Validate files exist
    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found: {input_csv}")
        return
    
    if not Path(confusion_json).exists():
        logger.error(f"Confusion JSON not found: {confusion_json}")
        return
    
    # Process file
    process_csv_file(input_csv, output_csv, confusion_json)
    
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
