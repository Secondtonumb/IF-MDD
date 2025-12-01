#!/usr/bin/env python3
"""
Test script to verify the label priors logging functionality.
Simulates the prior update logic without running full training.
"""

import torch
import os

def test_prior_logging():
    """Test the enhanced prior logging logic."""
    
    # Simulate priors (44 tokens: blank + 41 phonemes + bos/eos)
    num_tokens = 44
    
    # Simulate first epoch: blank token dominates (peaky behavior)
    print("\n" + "="*80)
    print("EPOCH 1 (Initial - Peaky Behavior)")
    print("="*80)
    
    # Create peaky distribution (blank ~0.6, others small)
    log_priors_epoch1 = torch.zeros(1, num_tokens)
    log_priors_epoch1[0, 0] = torch.log(torch.tensor(0.6))  # blank
    remaining_mass = 0.4 / (num_tokens - 1)
    for i in range(1, num_tokens):
        log_priors_epoch1[0, i] = torch.log(torch.tensor(remaining_mass))
    
    # Apply threshold
    prior_threshold = -12.0
    log_priors_epoch1 = torch.where(
        log_priors_epoch1 < prior_threshold,
        torch.tensor(prior_threshold),
        log_priors_epoch1
    )
    
    print_priors(log_priors_epoch1, None, 1)
    
    # Simulate second epoch: blank reduces (desired behavior)
    print("\n" + "="*80)
    print("EPOCH 2 (After Training - Reduced Peakiness)")
    print("="*80)
    
    # Create less peaky distribution (blank ~0.4, others larger)
    log_priors_epoch2 = torch.zeros(1, num_tokens)
    log_priors_epoch2[0, 0] = torch.log(torch.tensor(0.4))  # blank reduced
    remaining_mass = 0.6 / (num_tokens - 1)
    for i in range(1, num_tokens):
        # Add some variation to make it more realistic
        variation = torch.randn(1).item() * 0.002
        log_priors_epoch2[0, i] = torch.log(torch.tensor(remaining_mass + variation))
    
    # Apply threshold
    log_priors_epoch2 = torch.where(
        log_priors_epoch2 < prior_threshold,
        torch.tensor(prior_threshold),
        log_priors_epoch2
    )
    
    print_priors(log_priors_epoch2, log_priors_epoch1, 2)

def print_priors(new_log_priors, old_log_priors, epoch):
    """Print priors in the format used in training."""
    
    print(f"Total frames processed: 100000 (simulated)")
    print(f"Prior scaling factor (α): 0.6")
    
    # Convert to probability space
    new_priors_prob = new_log_priors.exp()
    
    # Print formatted priors (first 10 tokens for brevity)
    print(f"\nNew priors (probability) - First 10 tokens:")
    priors_list = new_priors_prob[0][:10].tolist()
    print("  " + ", ".join([f"{p:.4f}" for p in priors_list]))
    
    # Print formatted log-priors
    print(f"\nNew log-priors - First 10 tokens:")
    log_priors_list = new_log_priors[0][:10].tolist()
    print("  " + ", ".join([f"{lp:.2f}" for lp in log_priors_list]))
    
    # If we have previous priors, show the change
    if old_log_priors is not None:
        old_priors_prob = old_log_priors.exp()
        diff_percent = ((new_priors_prob - old_priors_prob) / old_priors_prob * 100)[0].tolist()
        print(f"\nChange from previous epoch (%) - First 10 tokens:")
        print("  " + ", ".join([f"{d:+.2f}" for d in diff_percent[:10]]))
        
        # Highlight the most changed tokens
        abs_diff = [abs(d) for d in diff_percent]
        top_changed_indices = sorted(range(len(abs_diff)), key=lambda i: abs_diff[i], reverse=True)[:5]
        print(f"\nTop 5 most changed tokens:")
        
        # Load token names (if available)
        token_names = load_token_names()
        for idx in top_changed_indices:
            token_name = token_names.get(idx, f"token_{idx}")
            print(f"  {token_name:>10s} (idx={idx:2d}): {old_priors_prob[0][idx].item():.4f} → {new_priors_prob[0][idx].item():.4f} ({diff_percent[idx]:+.2f}%)")
    
    # Highlight blank token (assume idx=0)
    blank_idx = 0
    blank_prior = new_priors_prob[0][blank_idx].item()
    print(f"\n🎯 Blank token (idx={blank_idx}) prior: {blank_prior:.4f} ({blank_prior*100:.2f}%)")
    
    # Check clipping
    prior_threshold = -12.0
    num_clipped = (new_log_priors == prior_threshold).sum().item()
    if num_clipped > 0:
        print(f"⚠️  Clipped {num_clipped} priors to threshold {prior_threshold}")
    
    print("="*80 + "\n")

def load_token_names():
    """Load token names from label encoder (if available)."""
    token_names = {0: "<blank>"}
    
    # Try to load from actual label encoder file
    label_encoder_path = "exp_l2arctic/wavlm_large_None_PhnMonoSSL_wavlm_ctc/save/label_encoder.txt"
    if os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                                # Format: 'token' => index
                                if '=>' in line:
                                    name, idx = line.split('=>')
                                    name = name.strip().strip("'\"")
                                    idx = idx.strip()
                                    token_names[int(idx)] = name
                        except:
                            pass
            print(f"✅ Loaded {len(token_names)} token names from {label_encoder_path}")
        except Exception as e:
            print(f"⚠️  Could not load token names: {e}")
    else:
        # Use dummy names for testing
        phonemes = ["sh", "ae", "l", "ay", "k", "r", "iy", "y", "uw", "sil", 
                   "aa", "eh", "d", "w", "ah", "z", "ch", "ey", "n", "jh",
                   "aw", "dh", "s", "hh", "f", "th", "ih", "m", "ao", "ow",
                   "v", "er", "g", "t", "uh", "zh", "ng", "err", "p", "b", "oy"]
        for i, phn in enumerate(phonemes, start=1):
            token_names[i] = phn
        token_names[42] = "<bos>"
        token_names[43] = "<eos>"
        print(f"ℹ️  Using dummy token names (label encoder not found)")
    
    return token_names

if __name__ == "__main__":
    print("\n🧪 Testing Label Priors Logging Functionality")
    print("="*80)
    test_prior_logging()
    print("\n✅ Test completed successfully!")
