#!/usr/bin/env python3
"""
Seq Decoder Hallucination Analysis & Verification Script
用于分析和验证 seq decoder 的多余生成问题
"""

import re
from pathlib import Path
from collections import Counter

def parse_seq_decoder_output(file_path):
    """
    解析 mpd_PER_seq_seq.txt 文件，分析 seq decoder 的输出质量
    """
    results = {
        'total_samples': 0,
        'eps_insertion_ratio': [],
        'sequence_length_ratio': [],
        'hallucination_samples': [],
        'normal_samples': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按样本分割
    samples = content.split('=' * 80)
    
    for sample in samples[1:]:  # 跳过第一个元数据块
        lines = sample.strip().split('\n')
        if len(lines) < 5:
            continue
        
        # 查找 "Model Prediction" 部分
        try:
            # 找到 Canonical 行
            canonical_line = None
            hypothesis_line = None
            alignment_line = None
            
            for i, line in enumerate(lines):
                if 'Model Prediction: Canonical vs Hypothesis:' in line:
                    if i + 1 < len(lines):
                        canonical_line = lines[i + 1]
                    if i + 2 < len(lines):
                        hypothesis_line = lines[i + 2]
                    if i + 3 < len(lines):
                        alignment_line = lines[i + 3]
                    break
            
            if not canonical_line or not hypothesis_line:
                continue
            
            results['total_samples'] += 1
            
            # 解析 phoneme 序列
            cano_phns = [p.strip() for p in canonical_line.split(';') if p.strip()]
            hyp_phns = [p.strip() for p in hypothesis_line.split(';') if p.strip()]
            
            # 计算指标
            cano_len = len(cano_phns)
            hyp_len = len(hyp_phns)
            eps_count = hyp_phns.count('<eps>')
            
            # EPS 插入比例
            if hyp_len > 0:
                eps_ratio = eps_count / hyp_len
                results['eps_insertion_ratio'].append(eps_ratio)
            
            # 序列长度比例
            if cano_len > 0:
                len_ratio = hyp_len / cano_len
                results['sequence_length_ratio'].append(len_ratio)
            
            # 判断是否为幻觉样本（>30% 是 <eps> 或长度膨胀 > 50%）
            is_hallucination = eps_ratio > 0.3 or len_ratio > 1.5
            
            sample_info = {
                'canonical_len': cano_len,
                'hypothesis_len': hyp_len,
                'eps_count': eps_count,
                'eps_ratio': eps_ratio,
                'length_ratio': len_ratio,
                'is_hallucination': is_hallucination
            }
            
            if is_hallucination:
                results['hallucination_samples'].append(sample_info)
            else:
                results['normal_samples'].append(sample_info)
                
        except Exception as e:
            print(f"Error parsing sample: {e}")
            continue
    
    return results

def print_analysis_report(results):
    """生成分析报告"""
    
    print("\n" + "=" * 80)
    print("SEQ DECODER HALLUCINATION ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total samples analyzed: {results['total_samples']}")
    print(f"   Hallucination samples: {len(results['hallucination_samples'])}")
    print(f"   Normal samples: {len(results['normal_samples'])}")
    
    if results['total_samples'] > 0:
        hallucination_rate = len(results['hallucination_samples']) / results['total_samples'] * 100
        print(f"   Hallucination rate: {hallucination_rate:.1f}%")
    
    print(f"\n📈 EPS Insertion Ratio:")
    if results['eps_insertion_ratio']:
        avg_eps = sum(results['eps_insertion_ratio']) / len(results['eps_insertion_ratio'])
        max_eps = max(results['eps_insertion_ratio'])
        print(f"   Average: {avg_eps:.1%}")
        print(f"   Max: {max_eps:.1%}")
        print(f"   ⚠️  Threshold: 30% (samples > 30% are hallucinating)")
    
    print(f"\n📏 Sequence Length Ratio (hypothesis / canonical):")
    if results['sequence_length_ratio']:
        avg_ratio = sum(results['sequence_length_ratio']) / len(results['sequence_length_ratio'])
        max_ratio = max(results['sequence_length_ratio'])
        print(f"   Average: {avg_ratio:.2f}x")
        print(f"   Max: {max_ratio:.2f}x")
        print(f"   ⚠️  Threshold: 1.5x (sequences > 1.5x are bloated)")
    
    print(f"\n🔴 Top 5 Worst Hallucination Samples:")
    sorted_hallu = sorted(results['hallucination_samples'], 
                         key=lambda x: x['eps_ratio'] + x['length_ratio']*0.5, 
                         reverse=True)
    for i, sample in enumerate(sorted_hallu[:5], 1):
        print(f"   {i}. EPS: {sample['eps_ratio']:.1%}, Len ratio: {sample['length_ratio']:.2f}x "
              f"({sample['canonical_len']} → {sample['hypothesis_len']})")
    
    print(f"\n✅ Top 5 Best Normal Samples:")
    sorted_normal = sorted(results['normal_samples'], 
                          key=lambda x: x['eps_ratio'] + x['length_ratio']*0.5)
    for i, sample in enumerate(sorted_normal[:5], 1):
        print(f"   {i}. EPS: {sample['eps_ratio']:.1%}, Len ratio: {sample['length_ratio']:.2f}x "
              f"({sample['canonical_len']} → {sample['hypothesis_len']})")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if results['total_samples'] > 0:
        hallucination_rate = len(results['hallucination_samples']) / results['total_samples']
        avg_eps = sum(results['eps_insertion_ratio']) / len(results['eps_insertion_ratio']) if results['eps_insertion_ratio'] else 0
        avg_len_ratio = sum(results['sequence_length_ratio']) / len(results['sequence_length_ratio']) if results['sequence_length_ratio'] else 0
        
        print(f"\n1️⃣  Hallucination Rate: {hallucination_rate:.1%}")
        if hallucination_rate > 0.3:
            print("   ⚠️  CRITICAL: >30% hallucination rate detected!")
            print("   ✅ Fix: Apply using_eos_threshold=True and reduce max_decode_ratio")
        elif hallucination_rate > 0.1:
            print("   ⚠️  WARNING: 10-30% hallucination rate")
            print("   ✅ Fix: Fine-tune max_decode_ratio and beam_size")
        else:
            print("   ✅ GOOD: <10% hallucination rate")
        
        print(f"\n2️⃣  Average EPS Insertion: {avg_eps:.1%}")
        if avg_eps > 0.2:
            print("   ⚠️  CRITICAL: Model fills outputs with <eps> tokens")
            print("   ✅ Fix: Enable using_eos_threshold in beam search config")
        
        print(f"\n3️⃣  Average Length Ratio: {avg_len_ratio:.2f}x")
        if avg_len_ratio > 1.3:
            print("   ⚠️  CRITICAL: Output sequences too long")
            print("   ✅ Fix: Reduce max_decode_ratio from 1.0 to 0.6-0.7")
        elif avg_len_ratio > 1.1:
            print("   ⚠️  WARNING: Slight length bloat")
            print("   ✅ Fix: Fine-tune max_decode_ratio")
    
    print("\n")

if __name__ == "__main__":
    import sys
    
    # 默认分析路径
    file_path = Path("/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_Trans_IFMDD_ConPCO_ver2_Trans_IFMDD_ConPCO_ver2_l2norm_unfrz/mpd_PER_seq_seq.txt")
    
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    print(f"📂 Analyzing: {file_path}")
    results = parse_seq_decoder_output(file_path)
    print_analysis_report(results)
