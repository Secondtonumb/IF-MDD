"""
Verify SpeechOcean762 dataset preparation

This script checks the prepared JSON files and provides statistics.
"""

import json
import os
from collections import Counter
import argparse


def verify_dataset(data_dir="./data/speechocean762"):
    """
    Verify the prepared SpeechOcean762 dataset.
    
    Args:
        data_dir: Directory containing JSON files
    """
    print(f"Verifying dataset in: {data_dir}\n")
    print("=" * 80)
    
    # Check for expected files
    expected_files = [
        "train.json",
        "train-train.json", 
        "train-dev.json",
        "test.json"
    ]
    
    missing_files = []
    for file in expected_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("Please run prepare_speechocean762.py first.\n")
        return
    
    # Analyze each split
    for split_file in ["train-train.json", "train-dev.json", "test.json"]:
        filepath = os.path.join(data_dir, split_file)
        
        print(f"\n📊 Analyzing: {split_file}")
        print("-" * 80)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic statistics
        num_examples = len(data)
        print(f"Number of examples: {num_examples}")
        
        if num_examples == 0:
            print("⚠️  Empty dataset!")
            continue
        
        # Duration statistics
        durations = [v['duration'] for v in data.values()]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        total_hours = sum(durations) / 3600
        
        print(f"Total duration: {total_hours:.2f} hours")
        print(f"Average duration: {avg_duration:.2f}s")
        print(f"Min/Max duration: {min_duration:.2f}s / {max_duration:.2f}s")
        
        # Speaker statistics
        speakers = [v['spk_id'] for v in data.values()]
        unique_speakers = set(speakers)
        print(f"Number of speakers: {len(unique_speakers)}")
        
        # Phoneme statistics
        all_phonemes = []
        for v in data.values():
            canonical = v['canonical_aligned'].split()
            perceived = v['perceived_aligned'].split()
            all_phonemes.extend(canonical)
            all_phonemes.extend(perceived)
        
        phoneme_counts = Counter(all_phonemes)
        unique_phonemes = len(phoneme_counts)
        total_phonemes = sum(phoneme_counts.values())
        
        print(f"Unique phonemes: {unique_phonemes}")
        print(f"Total phoneme tokens: {total_phonemes}")
        
        # Top 10 most common phonemes
        print("\nTop 10 most common phonemes:")
        for phn, count in phoneme_counts.most_common(10):
            percentage = (count / total_phonemes) * 100
            print(f"  {phn:6s}: {count:6d} ({percentage:5.2f}%)")
        
        # Check for mispronunciations
        mispronunciation_count = 0
        for v in data.values():
            canonical = v['canonical_aligned']
            perceived = v['perceived_aligned']
            if canonical != perceived:
                mispronunciation_count += 1
        
        mispro_percentage = (mispronunciation_count / num_examples) * 100
        print(f"\nMispronunciation examples: {mispronunciation_count} ({mispro_percentage:.2f}%)")
        
        # Sample entry
        print("\n📝 Sample entry:")
        sample_key = list(data.keys())[0]
        sample = data[sample_key]
        
        print(f"  Audio: {sample['wav']}")
        print(f"  Text: {sample['wrd']}")
        print(f"  Duration: {sample['duration']:.2f}s")
        print(f"  Speaker: {sample['spk_id']}")
        print(f"  Canonical: {sample['canonical_aligned']}")
        print(f"  Perceived:  {sample['perceived_aligned']}")
        print(f"  Target:     {sample['perceived_train_target']}")
        
        # Check for accuracy scores if available
        if 'accuracy_scores' in sample:
            print(f"  Accuracy scores: {sample['accuracy_scores'][:5]}...")
    
    print("\n" + "=" * 80)
    print("✅ Dataset verification complete!")
    print("\nTo train on this dataset:")
    print("  python train.py hparams/speechocean762.yaml")


def compare_with_l2arctic(so762_dir="./data/speechocean762", 
                          l2arctic_dir="./data"):
    """
    Compare SpeechOcean762 with L2-ARCTIC dataset.
    """
    print("\n" + "=" * 80)
    print("Comparing SpeechOcean762 with L2-ARCTIC")
    print("=" * 80)
    
    datasets = {
        "SpeechOcean762": os.path.join(so762_dir, "train-train.json"),
        "L2-ARCTIC": os.path.join(l2arctic_dir, "train-train.json")
    }
    
    stats = {}
    
    for name, filepath in datasets.items():
        if not os.path.exists(filepath):
            print(f"⚠️  {name} not found at {filepath}")
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        durations = [v['duration'] for v in data.values()]
        speakers = set(v['spk_id'] for v in data.values())
        
        # Count phonemes
        all_phonemes = []
        for v in data.values():
            all_phonemes.extend(v['canonical_aligned'].split())
        
        stats[name] = {
            'examples': len(data),
            'duration': sum(durations) / 3600,
            'speakers': len(speakers),
            'phonemes': len(set(all_phonemes))
        }
    
    # Print comparison table
    print(f"\n{'Metric':<20} {'SpeechOcean762':>15} {'L2-ARCTIC':>15}")
    print("-" * 52)
    
    if 'SpeechOcean762' in stats and 'L2-ARCTIC' in stats:
        print(f"{'Examples':<20} {stats['SpeechOcean762']['examples']:>15,} {stats['L2-ARCTIC']['examples']:>15,}")
        print(f"{'Duration (hours)':<20} {stats['SpeechOcean762']['duration']:>15.2f} {stats['L2-ARCTIC']['duration']:>15.2f}")
        print(f"{'Speakers':<20} {stats['SpeechOcean762']['speakers']:>15} {stats['L2-ARCTIC']['speakers']:>15}")
        print(f"{'Unique phonemes':<20} {stats['SpeechOcean762']['phonemes']:>15} {stats['L2-ARCTIC']['phonemes']:>15}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify SpeechOcean762 dataset preparation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/speechocean762",
        help="Directory containing prepared JSON files"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with L2-ARCTIC dataset"
    )
    parser.add_argument(
        "--l2arctic_dir",
        type=str,
        default="./data",
        help="Directory containing L2-ARCTIC data"
    )
    
    args = parser.parse_args()
    
    verify_dataset(args.data_dir)
    
    if args.compare:
        compare_with_l2arctic(args.data_dir, args.l2arctic_dir)


if __name__ == "__main__":
    main()
