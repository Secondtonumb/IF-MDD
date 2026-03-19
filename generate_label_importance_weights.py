"""
Generate label importance weights based on label frequency and confusion rate.

This script analyzes the label encoder and confusion matrix to assign higher weights
to phonemes that:
1. Appear less frequently (lower occurrence)
2. Are more likely to be confused (higher error rate)

The weight formula combines inverse frequency and error rate to boost rare and difficult phonemes.
"""

import json
import numpy as np
from pathlib import Path


def load_label_encoder(label_enc_path):
    """
    Load label encoder from text file.
    只解析 ================ 分隔线之上的 label 映射。
    
    Returns:
        dict: {label: index}
    """
    label2idx = {}
    with open(label_enc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 遇到分隔线就停止解析
            if line == '================':
                break
            # 解析 'label' => index 格式
            if '=>' in line:
                parts = line.split(' => ')
                if len(parts) == 2:
                    label = parts[0].strip("'")
                    try:
                        idx = int(parts[1])
                        label2idx[label] = idx
                    except ValueError:
                        # 跳过非数字行（不应该出现在分隔线之上）
                        continue
    return label2idx


def load_confusion_matrix(confusion_path):
    """
    Load confusion matrix from JSON file.
    
    Returns:
        dict: confusion matrix data
    """
    with open(confusion_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['confusion_matrix']


def calculate_label_weights(label2idx, confusion_matrix, 
                            freq_weight=0.5, error_weight=0.5,
                            min_weight=1.0, max_weight=3.0,
                            inverse_logic=True):
    """
    Calculate importance weights for each label.
    
    Weight formula (when inverse_logic=True):
        - 容易错的音素 → 较小的权重
        - 不容易错的音素 → 较大的权重
        raw_weight = freq_weight * sqrt(frequency) + error_weight * (1 - error_rate)
    
    Weight formula (when inverse_logic=False):
        - 容易错的音素 → 较大的权重
        - 不容易错的音素 → 较小的权重
        raw_weight = freq_weight * (1 / sqrt(frequency)) + error_weight * error_rate
    
    Then normalize to [min_weight, max_weight] range.
    
    Args:
        label2idx: Dictionary mapping labels to indices
        confusion_matrix: Confusion matrix data
        freq_weight: Weight for frequency component (default: 0.5)
        error_weight: Weight for error rate component (default: 0.5)
        min_weight: Minimum weight value (default: 1.0)
        max_weight: Maximum weight value (default: 3.0)
        inverse_logic: If True, easy labels get higher weights; if False, hard labels get higher weights
        
    Returns:
        dict: {label: weight}
        dict: {label: stats} for analysis
    """
    # Extract phoneme labels (exclude special tokens)
    special_tokens = {'<bos>', '<eos>', '<blank>', 'err', 'sil'}
    phoneme_labels = [label for label in label2idx.keys() 
                     if label not in special_tokens]
    
    # Calculate statistics
    stats = {}
    raw_weights = {}
    
    for label in phoneme_labels:
        if label in confusion_matrix:
            conf_data = confusion_matrix[label]
            total_count = conf_data.get('total_ref_count', 1)
            
            # Calculate total error rate
            total_errors = sum(item['count'] for item in conf_data.get('confusions', []))
            error_rate = total_errors / total_count if total_count > 0 else 0
            
            if inverse_logic:
                # 反转逻辑：容易错的音素权重更小
                # Frequency component: frequent phonemes get higher weight
                freq_component = np.sqrt(total_count) if total_count > 0 else 1.0
                
                # Error component: low error rate gets higher weight
                error_component = 1.0 - error_rate
                
                # Combined raw weight (higher = easier/more frequent)
                raw_weight = freq_weight * freq_component + error_weight * error_component
            else:
                # 原始逻辑：容易错的音素权重更大
                # Inverse frequency component (rare phonemes get higher weight)
                freq_component = 1.0 / np.sqrt(total_count) if total_count > 0 else 1.0
                
                # Error rate component (confused phonemes get higher weight)
                error_component = error_rate
                
                # Combined raw weight (higher = harder/rarer)
                raw_weight = freq_weight * freq_component + error_weight * error_component
            
            stats[label] = {
                'total_count': total_count,
                'error_rate': error_rate,
                'freq_component': freq_component,
                'error_component': error_component,
                'raw_weight': raw_weight
            }
            raw_weights[label] = raw_weight
        else:
            # If not in confusion matrix, give default
            stats[label] = {
                'total_count': 0,
                'error_rate': 0,
                'freq_component': 1.0,
                'error_component': 0,
                'raw_weight': 1.0
            }
            raw_weights[label] = 1.0
    
    # Normalize weights to [min_weight, max_weight]
    raw_weight_values = list(raw_weights.values())
    min_raw = min(raw_weight_values)
    max_raw = max(raw_weight_values)
    
    normalized_weights = {}
    for label, raw_w in raw_weights.items():
        if max_raw > min_raw:
            # Linear normalization: map [min_raw, max_raw] -> [min_weight, max_weight]
            normalized = min_weight + (raw_w - min_raw) / (max_raw - min_raw) * (max_weight - min_weight)
        else:
            normalized = (min_weight + max_weight) / 2
        normalized_weights[label] = round(normalized, 4)
        stats[label]['normalized_weight'] = normalized
    
    # Add weights for special tokens (typically middle value)
    middle_weight = (min_weight + max_weight) / 2
    for token in special_tokens:
        if token in label2idx:
            normalized_weights[token] = round(middle_weight, 4)
            stats[token] = {
                'total_count': 'N/A',
                'error_rate': 0,
                'normalized_weight': middle_weight
            }
    
    return normalized_weights, stats


def save_weights(weights, stats, output_dir, label_enc_path, label2idx):
    """
    Save weights and statistics to files.
    
    Args:
        weights: Dictionary of label weights
        stats: Dictionary of label statistics
        output_dir: Output directory path
        label_enc_path: Path to original label encoder (for copying)
        label2idx: Dictionary mapping labels to indices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save weights dictionary (label as key)
    weights_file = output_dir / 'label_importance_weights.json'
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved weights to: {weights_file}")
    
    # 2. Save weights dictionary (label id as key)
    weights_by_id = {}
    for label, weight in weights.items():
        if label in label2idx:
            label_id = label2idx[label]
            weights_by_id[label_id] = weight
    
    weights_id_file = output_dir / 'label_importance_weights_by_id.json'
    with open(weights_id_file, 'w', encoding='utf-8') as f:
        json.dump(weights_by_id, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"✓ Saved weights (by ID) to: {weights_id_file}")
    
    # 3. Save detailed statistics
    stats_file = output_dir / 'label_weight_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved statistics to: {stats_file}")
    
    # 4. Copy label encoder to the same directory (if not already there)
    import shutil
    label_enc_copy = output_dir / 'label_encoder.txt'
    if Path(label_enc_path).resolve() != label_enc_copy.resolve():
        shutil.copy(label_enc_path, label_enc_copy)
        print(f"✓ Copied label encoder to: {label_enc_copy}")
    else:
        print(f"✓ Label encoder already in target directory: {label_enc_copy}")
    
    # 5. Generate analysis report
    report_file = output_dir / 'weight_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LABEL IMPORTANCE WEIGHTS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by weight (descending)
        sorted_items = sorted(stats.items(), 
                            key=lambda x: x[1].get('normalized_weight', 0), 
                            reverse=True)
        
        f.write("Top 10 Highest Weighted Labels (need more attention):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Label':<10} {'Weight':<10} {'Freq':<12} {'Error Rate':<12} {'Raw Weight':<12}\n")
        f.write("-" * 80 + "\n")
        
        for label, stat in sorted_items[:10]:
            if isinstance(stat.get('total_count'), int):
                f.write(f"{label:<10} {stat['normalized_weight']:<10.4f} "
                       f"{stat['total_count']:<12} {stat['error_rate']:<12.4f} "
                       f"{stat['raw_weight']:<12.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Bottom 10 Lowest Weighted Labels (easier/more frequent):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Label':<10} {'Weight':<10} {'Freq':<12} {'Error Rate':<12} {'Raw Weight':<12}\n")
        f.write("-" * 80 + "\n")
        
        for label, stat in sorted_items[-10:]:
            if isinstance(stat.get('total_count'), int):
                f.write(f"{label:<10} {stat['normalized_weight']:<10.4f} "
                       f"{stat['total_count']:<12} {stat['error_rate']:<12.4f} "
                       f"{stat['raw_weight']:<12.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Total labels: {len(weights)}\n")
        f.write(f"  Weight range: [{min(weights.values()):.4f}, {max(weights.values()):.4f}]\n")
        f.write(f"  Mean weight: {np.mean(list(weights.values())):.4f}\n")
        f.write(f"  Median weight: {np.median(list(weights.values())):.4f}\n")
    
    print(f"✓ Saved analysis report to: {report_file}")
    print(f"\n{'='*80}")
    print("All files saved successfully!")
    print(f"{'='*80}")


def main():
    # Configuration
    label_enc_path = "/home/m64000/work/IF-MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_ottc_confEnc_kernal/save/label_encoder.txt"
    confusion_path = "/home/m64000/work/IF-MDD/utils/l2_arctic_conf.json"
    output_dir = "/home/m64000/work/IF-MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_ottc_confEnc_kernal/save"
    
    # Parameters for weight calculation
    freq_weight = 0.5      # Weight for frequency component
    error_weight = 0.5     # Weight for error rate component
    min_weight = 1.0       # Minimum weight value
    max_weight = 3.0       # Maximum weight value
    inverse_logic = True   # True: 容易错的权重更小; False: 容易错的权重更大
    
    print("=" * 80)
    print("GENERATING LABEL IMPORTANCE WEIGHTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Label encoder: {label_enc_path}")
    print(f"  Confusion matrix: {confusion_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Frequency weight: {freq_weight}")
    print(f"  Error weight: {error_weight}")
    print(f"  Weight range: [{min_weight}, {max_weight}]")
    print(f"  Inverse logic: {inverse_logic} (容易错的权重更{'小' if inverse_logic else '大'})")
    print()
    
    # Load data
    print("Loading data...")
    label2idx = load_label_encoder(label_enc_path)
    print(f"✓ Loaded {len(label2idx)} labels from label encoder")
    
    confusion_matrix = load_confusion_matrix(confusion_path)
    print(f"✓ Loaded confusion matrix with {len(confusion_matrix)} phonemes")
    
    # Calculate weights
    print("\nCalculating importance weights...")
    weights, stats = calculate_label_weights(
        label2idx, confusion_matrix,
        freq_weight=freq_weight,
        error_weight=error_weight,
        min_weight=min_weight,
        max_weight=max_weight,
        inverse_logic=inverse_logic,
    )
    print(f"✓ Calculated weights for {len(weights)} labels")
    
    # Save results
    print("\nSaving results...")
    save_weights(weights, stats, output_dir, label_enc_path, label2idx)
    
    # Display top weighted labels
    print(f"\n{'='*80}")
    print("TOP 10 LABELS WITH HIGHEST WEIGHTS (need more attention):")
    print(f"{'='*80}")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for i, (label, weight) in enumerate(sorted_weights[:10], 1):
        if label in stats and isinstance(stats[label].get('total_count'), int):
            stat = stats[label]
            print(f"{i:2d}. {label:<6} weight={weight:.4f}  "
                  f"(count={stat['total_count']:5d}, error_rate={stat['error_rate']:.4f})")


if __name__ == "__main__":
    main()
