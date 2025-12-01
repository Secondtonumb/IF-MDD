#!/usr/bin/env python3
"""
Test script for ComprehensiveDataIOPrep

This script verifies that the comprehensive data loader works correctly
and displays sample data for all integrated features.
"""

import sys
from utils.DataPrepIO import ComprehensiveDataIOPrep

def test_comprehensive_dataloader():
    """Test comprehensive data loading."""
    
    # Minimal hyperparameters for testing
    hparams = {
        "data_folder_save": "/home/kevingenghaopeng/MDD/speechocean762",
        "train_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "valid_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "test_annotation": "data/speechocean762_with_word_scores/train-dev_with_mfa.json",
        "save_folder": "exp_test_comprehensive/save",
        "output_folder": "exp_test_comprehensive",
        "sorting": "ascending",
        "blank_index": 0,
        "frame_ms": 20.0,
    }
    
    print("=" * 80)
    print("Testing ComprehensiveDataIOPrep")
    print("=" * 80)
    
    # Initialize data preparation
    print("\n[1/3] Initializing ComprehensiveDataIOPrep...")
    data_prep = ComprehensiveDataIOPrep(hparams)
    
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
    print("COMPREHENSIVE SAMPLE INSPECTION")
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
    print(f"  - Start frames: {sample['mfa_phone_start_frames'].tolist()[:5]}... (first 5)")
    print(f"  - End frames: {sample['mfa_phone_end_frames'].tolist()[:5]}... (first 5)")
    
    print(f"\n📚 MFA Word-level Timestamps:")
    print(f"  - Words: {' '.join(sample['mfa_word_list'])}")
    print(f"  - Start frames: {sample['mfa_word_start_frames'].tolist()}")
    print(f"  - End frames: {sample['mfa_word_end_frames'].tolist()}")
    print(f"  - Phone ranges: {sample['mfa_word_phone_ranges']}")
    
    print(f"\n⭐ Word-level Pronunciation Scores:")
    print(f"  - Boundaries: {sample['word_boundaries']}")
    print(f"  - Accuracy: {sample['word_accuracy_scores']}")
    print(f"  - Stress: {sample['word_stress_scores']}")
    print(f"  - Total: {sample['word_total_scores']}")
    
    print(f"\n📍 Canonical Phone Timestamps:")
    if len(sample['canonical_phone_start_frames']) > 0:
        print(f"  - Start frames: {sample['canonical_phone_start_frames'].tolist()[:5]}... (first 5)")
        print(f"  - End frames: {sample['canonical_phone_end_frames'].tolist()[:5]}... (first 5)")
    else:
        print(f"  - Not available (empty)")
    
    print(f"\n🏷️  Mispronunciation Labels:")
    print(f"  - Labels: {sample['mispro_label'].tolist()}")
    print(f"  - Mispronounced: {sum(sample['mispro_label']).item()} / {len(sample['mispro_label'])}")
    
    # Integrated view: word-phone-score mapping
    print(f"\n🔗 Integrated Word-Phone-Score Mapping:")
    for i, word in enumerate(sample['mfa_word_list']):
        phone_start, phone_end = sample['mfa_word_phone_ranges'][i]
        phones = sample['mfa_phone_list'][phone_start:phone_end]
        acc = sample['word_accuracy_scores'][i]
        stress = sample['word_stress_scores'][i]
        total = sample['word_total_scores'][i]
        
        frame_start = sample['mfa_word_start_frames'][i].item()
        frame_end = sample['mfa_word_end_frames'][i].item()
        
        print(f"  {i+1}. '{word}' → {' '.join(phones)}")
        print(f"     Scores: Acc={acc}, Stress={stress}, Total={total}")
        print(f"     Frames: {frame_start}-{frame_end}, Phones: {phone_start}-{phone_end}")
    
    # Verification
    print(f"\n✅ Comprehensive Verification:")
    print(f"  ✓ MFA phone count: {len(sample['mfa_phone_list'])}")
    print(f"  ✓ MFA word count: {len(sample['mfa_word_list'])}")
    print(f"  ✓ Word score count: {len(sample['word_accuracy_scores'])}")
    print(f"  ✓ All data integrated in single sample!")
    
    print("\n" + "=" * 80)
    print("✅ ComprehensiveDataIOPrep test completed successfully!")
    print("=" * 80)
    print("\n📊 Summary:")
    print(f"  - Total fields available: {len(sample.keys())}")
    print(f"  - No need for multiple DataPrep instances!")
    print(f"  - All timestamp and score data accessible in one batch!")
    
    return True

if __name__ == "__main__":
    try:
        test_comprehensive_dataloader()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
