#!/usr/bin/env python3
"""
Test script for MFATimestampDataIOPrep

This script verifies that the MFA timestamp data loader works correctly
and displays sample data for inspection.
"""

import sys
import torch
from hyperpyyaml import load_hyperpyyaml
from utils.DataPrepIO import MFATimestampDataIOPrep

def test_mfa_dataloader():
    """Test MFA timestamp data loading."""
    
    # Minimal hyperparameters for testing
    hparams = {
        "data_folder_save": "/home/kevingenghaopeng/MDD/speechocean762",
        "train_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "valid_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "test_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "save_folder": "exp_test_mfa/save",
        "output_folder": "exp_test_mfa",
        "sorting": "ascending",
        "blank_index": 0,
        "frame_ms": 20.0,
    }
    
    print("=" * 80)
    print("Testing MFATimestampDataIOPrep")
    print("=" * 80)
    
    # Initialize data preparation
    print("\n[1/3] Initializing MFATimestampDataIOPrep...")
    data_prep = MFATimestampDataIOPrep(hparams)
    
    # Prepare datasets
    print("\n[2/3] Preparing datasets...")
    train_data, valid_data, test_data, label_encoder = data_prep.prepare()
    
    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Valid samples: {len(valid_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    print(f"✓ Vocabulary size: {len(label_encoder)}")
    
    # Test loading a sample
    print("\n[3/3] Loading and inspecting first sample...")
    sample = train_data[0]
    
    print("\n" + "=" * 80)
    print("SAMPLE INSPECTION")
    print("=" * 80)
    
    print(f"\n📁 Sample ID: {sample['id']}")
    
    print(f"\n🎵 Audio:")
    print(f"  - Signal shape: {sample['sig'].shape}")
    print(f"  - Duration: {sample['sig'].shape[0] / 16000:.2f}s")
    
    print(f"\n📝 Phoneme Sequences:")
    print(f"  - Target: {' '.join(sample['phn_list_target'])}")
    print(f"  - Canonical: {' '.join(sample['phn_list_canonical'])}")
    print(f"  - Perceived: {' '.join(sample['phn_list_perceived'])}")
    
    print(f"\n🕐 MFA Phone-level Timestamps:")
    print(f"  - Phones: {' '.join(sample['mfa_phone_list'])}")
    print(f"  - Start frames: {sample['mfa_phone_start_frames'].tolist()[:5]}... (showing first 5)")
    print(f"  - End frames: {sample['mfa_phone_end_frames'].tolist()[:5]}... (showing first 5)")
    print(f"  - Frame ranges: {sample['mfa_phone_frame_ranges'][:3]}... (showing first 3)")
    
    print(f"\n📚 MFA Word-level Timestamps:")
    print(f"  - Words: {' '.join(sample['mfa_word_list'])}")
    print(f"  - Start frames: {sample['mfa_word_start_frames'].tolist()}")
    print(f"  - End frames: {sample['mfa_word_end_frames'].tolist()}")
    print(f"  - Phone ranges: {sample['mfa_word_phone_ranges']}")
    
    print(f"\n🏷️  Mispronunciation Labels:")
    print(f"  - Labels: {sample['mispro_label'].tolist()}")
    print(f"  - Mispronounced phones: {sum(sample['mispro_label']).item()} / {len(sample['mispro_label'])}")
    
    # Verify alignment
    print(f"\n✅ Verification:")
    phone_count = len(sample['mfa_phone_list'])
    word_count = len(sample['mfa_word_list'])
    print(f"  - Phone count matches: {phone_count == len(sample['mfa_phone_start_frames']) == len(sample['mfa_phone_end_frames'])}")
    print(f"  - Word count matches: {word_count == len(sample['mfa_word_start_frames']) == len(sample['mfa_word_end_frames'])}")
    print(f"  - Word-phone mapping valid: {len(sample['mfa_word_phone_ranges']) == word_count}")
    
    # Display word-phone alignment
    print(f"\n🔗 Word-Phone Alignment:")
    for i, (word, (phone_start, phone_end)) in enumerate(zip(sample['mfa_word_list'], sample['mfa_word_phone_ranges'])):
        phones_in_word = sample['mfa_phone_list'][phone_start:phone_end]
        print(f"  {i+1}. '{word}' → {' '.join(phones_in_word)} (phones {phone_start}-{phone_end})")
    
    print("\n" + "=" * 80)
    print("✅ MFATimestampDataIOPrep test completed successfully!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_mfa_dataloader()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
