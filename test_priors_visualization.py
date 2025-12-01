#!/usr/bin/env python3
"""
Test the enhanced priors logging with dictionary format and visualization.
"""

import torch
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def create_test_priors():
    """Create test priors data."""
    # Simulate 44 tokens
    priors = torch.tensor([
        0.7082, 0.0066, 0.0117, 0.0052, 0.0067, 0.0091, 0.0130, 0.0028, 0.0041, 0.0257,
        0.0036, 0.0067, 0.0123, 0.0077, 0.0208, 0.0060, 0.0024, 0.0049, 0.0169, 0.0022,
        0.0022, 0.0038, 0.0148, 0.0088, 0.0072, 0.0018, 0.0125, 0.0108, 0.0061, 0.0042,
        0.0047, 0.0043, 0.0025, 0.0133, 0.0023, 0.0011, 0.0040, 0.0015, 0.0047, 0.0053,
        0.0014, 0.0034, 0.0010, 0.0019
    ])
    log_priors = torch.log(priors)
    return priors, log_priors

def load_token_names():
    """Load token names from label encoder."""
    token_names = {}
    label_encoder_path = "exp_l2arctic/wavlm_large_None_PhnMonoSSL_wavlm_ctc/save/label_encoder.txt"
    
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=>' in line:
                    try:
                        name, idx = line.split('=>')
                        name = name.strip().strip("'\"")
                        idx = int(idx.strip())
                        token_names[idx] = name
                    except:
                        pass
    
    # Add blank if not present
    if 0 not in token_names:
        token_names[0] = '<blank>'
    
    print(f"✅ Loaded {len(token_names)} token names")
    return token_names

def test_dictionary_format(priors, log_priors, token_names):
    """Test dictionary format output."""
    print("\n" + "="*80)
    print("📋 Testing Dictionary Format")
    print("="*80)
    
    # Create dictionaries
    priors_dict = {}
    log_priors_dict = {}
    
    for idx, (prior, log_prior) in enumerate(zip(priors.tolist(), log_priors.tolist())):
        token_name = token_names.get(idx, f"token_{idx}")
        priors_dict[token_name] = prior
        log_priors_dict[token_name] = log_prior
    
    # Sort by value
    sorted_priors = sorted(priors_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Top 15 Tokens by Prior:")
    for token_name, prior in sorted_priors[:15]:
        log_prior = log_priors_dict[token_name]
        print(f"  {token_name:>10s}: {prior:.4f} (log: {log_prior:.2f})")
    
    return priors_dict, log_priors_dict

def test_json_export(priors_dict, log_priors_dict, output_dir="test_output"):
    """Test JSON export."""
    print("\n" + "="*80)
    print("💾 Testing JSON Export")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_data = {
        "epoch": 1,
        "num_samples": 247820,
        "prior_scaling_factor": 0.6,
        "priors": priors_dict,
        "log_priors": log_priors_dict
    }
    
    json_path = os.path.join(output_dir, "priors_test.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Saved JSON to: {json_path}")
    
    # Verify by loading
    with open(json_path, 'r') as f:
        loaded_data = json.load(f)
    
    print(f"✅ Verified JSON contains {len(loaded_data['priors'])} priors")
    return json_path

def test_visualization(priors_dict, output_dir="test_output"):
    """Test visualization."""
    print("\n" + "="*80)
    print("📊 Testing Visualization")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Sort tokens by prior value
    sorted_items = sorted(priors_dict.items(), key=lambda x: x[1], reverse=True)
    tokens_sorted = [item[0] for item in sorted_items]
    priors_sorted = [item[1] for item in sorted_items]
    
    # Plot 1: Bar chart of all priors (sorted)
    colors = ['red' if '<blank>' in name or name == 'blank' else 'steelblue' 
             for name in tokens_sorted]
    bars = ax1.bar(range(len(tokens_sorted)), priors_sorted, color=colors, alpha=0.8)
    ax1.set_xlabel('Token Index (sorted by prior)', fontsize=12)
    ax1.set_ylabel('Prior (Probability)', fontsize=12)
    ax1.set_title('Label Priors Distribution - Test', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add reference lines
    ax1.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Target ~0.4')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.legend()
    
    # Add value labels for top 10
    for i in range(min(10, len(priors_sorted))):
        ax1.text(i, priors_sorted[i], f'{priors_sorted[i]:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Top 20 tokens with names
    top_n = min(20, len(tokens_sorted))
    tokens_top = tokens_sorted[:top_n]
    priors_top = priors_sorted[:top_n]
    colors_top = ['red' if '<blank>' in name or name == 'blank' else 'steelblue' 
                 for name in tokens_top]
    
    bars2 = ax2.barh(range(top_n), priors_top, color=colors_top, alpha=0.8)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(tokens_top, fontsize=10)
    ax2.set_xlabel('Prior (Probability)', fontsize=12)
    ax2.set_ylabel('Token', fontsize=12)
    ax2.set_title(f'Top {top_n} Tokens by Prior', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (token, prior) in enumerate(zip(tokens_top, priors_top)):
        ax2.text(prior, i, f' {prior:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "priors_distribution_test.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved plot to: {plot_path}")
    return plot_path

def main():
    print("🧪 Testing Enhanced Priors Logging\n")
    
    # Create test data
    priors, log_priors = create_test_priors()
    print("✅ Created test priors")
    
    # Load token names
    token_names = load_token_names()
    
    # Test dictionary format
    priors_dict, log_priors_dict = test_dictionary_format(priors, log_priors, token_names)
    
    # Test JSON export
    json_path = test_json_export(priors_dict, log_priors_dict)
    
    # Test visualization
    plot_path = test_visualization(priors_dict)
    
    print("\n" + "="*80)
    print("✅ All Tests Passed!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 Plot: {plot_path}")
    print(f"\nYou can view the plot with:")
    print(f"  eog {plot_path}")
    print(f"\nYou can view the JSON with:")
    print(f"  cat {json_path} | jq '.priors | to_entries | sort_by(-.value) | .[0:10]'")

if __name__ == "__main__":
    main()
