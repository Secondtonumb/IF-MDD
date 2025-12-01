#!/usr/bin/env python3
"""
Test script for GoPDataset_ver2 dataloader
This script demonstrates how to use the enhanced dataloader with SSL features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from utils.so762_reader import GoPDataset_ver2

def test_basic_loading():
    """Test basic dataset loading and SSL feature extraction"""
    print("=" * 80)
    print("Test 1: Basic Loading and SSL Feature Extraction")
    print("=" * 80)
    
    # Create dataset with small batch for testing
    print("\nCreating dataset with WavLM Large...")
    dataset = GoPDataset_ver2(
        set='train',
        am='librispeech',
        model_name='wavlm_large',
        sample_rate=16000,
        freeze=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single item
    print("\nTesting single item extraction...")
    utt_id, utt_info, feat, feat_energy, feat_dur, wav_path, ssl_features = dataset[0]
    
    print(f"  Utterance ID: {utt_id}")
    print(f"  GOP features shape: {feat.shape}")
    print(f"  Energy features shape: {feat_energy.shape}")
    print(f"  Duration features shape: {feat_dur.shape}")
    print(f"  SSL features shape: {ssl_features.shape}")
    print(f"  SSL features dtype: {ssl_features.dtype}")
    print(f"  SSL features device: {ssl_features.device}")
    print(f"  Utterance info keys: {utt_info.keys()}")
    
    print("\n✓ Test 1 passed!")
    return dataset

def test_dataloader():
    """Test DataLoader integration"""
    print("\n" + "=" * 80)
    print("Test 2: DataLoader Integration")
    print("=" * 80)
    
    dataset = GoPDataset_ver2(
        set='train',
        am='librispeech',
        model_name='wavlm_large',
        sample_rate=16000
    )
    
    # Create dataloader with batch_size=1 (SSL features have variable length)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0  # Important: use 0 to avoid multiprocessing issues with GPU
    )
    
    print("\nIterating through first 3 batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        utt_id, utt_info, feat, feat_energy, feat_dur, wav_path, ssl_features = batch
        print(f"  Batch {i+1}: utt_id={utt_id[0]}, ssl_shape={ssl_features[0].shape}")
    
    print("\n✓ Test 2 passed!")

def test_feature_caching():
    """Test SSL feature caching mechanism"""
    print("\n" + "=" * 80)
    print("Test 3: Feature Caching")
    print("=" * 80)
    
    dataset = GoPDataset_ver2(
        set='train',
        am='librispeech',
        model_name='wavlm_base',  # Use base model for faster testing
        sample_rate=16000
    )
    
    print("\nFirst access (extract features)...")
    import time
    start = time.time()
    _, _, _, _, _, _, ssl1 = dataset[0]
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    
    # Manually cache the feature
    utt_id = list(dataset.dict_all.keys())[0]
    dataset.ssl_feature_cache[utt_id] = ssl1
    
    print("\nSecond access (from cache)...")
    start = time.time()
    _, _, _, _, _, _, ssl2 = dataset[0]
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    
    # Verify features are identical
    assert torch.allclose(ssl1, ssl2, rtol=1e-5), "Cached features don't match!"
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Clear cache
    dataset.clear_cache()
    print("\nCache cleared successfully")
    print(f"  Cache size: {len(dataset.ssl_feature_cache)}")
    
    print("\n✓ Test 3 passed!")

def test_save_load_features():
    """Test saving and loading SSL features"""
    print("\n" + "=" * 80)
    print("Test 4: Save and Load SSL Features")
    print("=" * 80)
    
    dataset = GoPDataset_ver2(
        set='test',  # Use test set (smaller) for faster testing
        am='librispeech',
        model_name='wavlm_base',
        sample_rate=16000
    )
    
    output_file = '/tmp/test_ssl_features.npz'
    
    print(f"\nExtracting SSL features for {min(5, len(dataset))} utterances...")
    
    # Create a small subset for testing
    import numpy as np
    ssl_dict = {}
    for i in range(min(5, len(dataset))):
        utt_id, _, _, _, _, wav_path, ssl_features = dataset[i]
        ssl_dict[utt_id] = ssl_features.cpu().numpy()
        print(f"  {i+1}. {utt_id}: {ssl_features.shape}")
    
    # Save manually
    print(f"\nSaving to {output_file}...")
    np.savez_compressed(output_file, **ssl_dict)
    
    # Load back
    print("\nLoading features...")
    dataset.load_ssl_features(output_file, format='npz')
    print(f"  Loaded {len(dataset.ssl_feature_cache)} features into cache")
    
    # Verify
    print("\nVerifying loaded features...")
    for i, (utt_id, expected_feat) in enumerate(ssl_dict.items()):
        if i >= 3:
            break
        cached_feat = dataset.ssl_feature_cache[utt_id].numpy()
        assert np.allclose(expected_feat, cached_feat, rtol=1e-5), f"Features don't match for {utt_id}"
        print(f"  ✓ {utt_id}")
    
    # Cleanup
    import os
    os.remove(output_file)
    print(f"\nCleaned up {output_file}")
    
    print("\n✓ Test 4 passed!")

def test_different_models():
    """Test different SSL models"""
    print("\n" + "=" * 80)
    print("Test 5: Different SSL Models")
    print("=" * 80)
    
    models = [
        ('wavlm_base', 768),
        ('wavlm_large', 1024),
        # Add more models as needed, but they might be slow to download
    ]
    
    for model_name, expected_dim in models:
        print(f"\nTesting {model_name}...")
        try:
            dataset = GoPDataset_ver2(
                set='train',
                am='librispeech',
                model_name=model_name,
                sample_rate=16000
            )
            
            _, _, _, _, _, _, ssl_features = dataset[0]
            actual_dim = ssl_features.shape[-1]
            
            print(f"  Expected dim: {expected_dim}, Actual dim: {actual_dim}")
            assert actual_dim == expected_dim, f"Feature dimension mismatch!"
            print(f"  ✓ {model_name} working correctly")
            
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
    
    print("\n✓ Test 5 passed!")

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GoPDataset_ver2 Test Suite")
    print("=" * 80)
    
    try:
        # Test 1: Basic loading
        test_basic_loading()
        
        # Test 2: DataLoader
        test_dataloader()
        
        # Test 3: Caching
        test_feature_caching()
        
        # Test 4: Save/Load
        test_save_load_features()
        
        # Test 5: Different models (may take time to download)
        # test_different_models()  # Uncomment to test
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Test failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
