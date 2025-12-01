#!/usr/bin/env python3
"""
分析测试集解码结果

使用方法:
    python analyze_test_decoding.py --test_dir <path_to_test_decoding_folder>
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def load_results(test_dir):
    """加载所有测试解码结果"""
    results = defaultdict(list)
    test_path = Path(test_dir)
    
    for speaker_dir in test_path.glob("*/"):
        speaker = speaker_dir.name
        # Match files with format: decode_{SPEAKER}_{FILE_ID}.json
        for json_file in speaker_dir.glob("decode_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    data['file'] = str(json_file)
                    # Extract file_id from filename (e.g., decode_TLV_arctic_a0001.json -> arctic_a0001)
                    filename = json_file.stem  # Remove .json
                    parts = filename.split('_', 2)  # Split on first 2 underscores: decode, SPEAKER, REST
                    if len(parts) >= 3:
                        data['file_id'] = parts[2]  # FILE_ID part
                    results[speaker].append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to load {json_file}: {e}")
    
    return results


def compute_statistics(results_by_speaker):
    """计算统计信息"""
    stats = {}
    all_per = []
    all_mpd = []
    
    for speaker, results in results_by_speaker.items():
        if not results:
            continue
        
        per_scores = [r.get('per_score', 0) for r in results]
        mpd_scores = [r.get('mpd_score', 0) for r in results]
        
        all_per.extend(per_scores)
        all_mpd.extend(mpd_scores)
        
        stats[speaker] = {
            'num_samples': len(results),
            'per_mean': statistics.mean(per_scores) if per_scores else 0,
            'per_std': statistics.stdev(per_scores) if len(per_scores) > 1 else 0,
            'mpd_mean': statistics.mean(mpd_scores) if mpd_scores else 0,
        }
    
    stats['_global'] = {
        'num_speakers': len(results_by_speaker),
        'total_samples': len(all_per),
        'per_mean': statistics.mean(all_per) if all_per else 0,
        'per_std': statistics.stdev(all_per) if len(all_per) > 1 else 0,
        'mpd_mean': statistics.mean(all_mpd) if all_mpd else 0,
    }
    
    return stats


def print_summary(stats):
    """打印统计摘要"""
    global_stats = stats.pop('_global', {})
    
    print("\n" + "="*80)
    print("📊 TEST DECODING ANALYSIS SUMMARY")
    print("="*80)
    
    if global_stats:
        print(f"\n🌍 GLOBAL STATISTICS:")
        print(f"  Speakers: {global_stats.get('num_speakers', 0)}")
        print(f"  Total samples: {global_stats.get('total_samples', 0)}")
        print(f"  PER Mean: {global_stats.get('per_mean', 0):.4f}")
        print(f"  MPD F1 Mean: {global_stats.get('mpd_mean', 0):.4f}")
    
    print(f"\n🎤 PER-SPEAKER:")
    print(f"{'Speaker':<12} {'Samples':<10} {'PER Mean':<12} {'MPD Mean':<12}")
    print("-" * 50)
    
    for speaker in sorted(stats.keys()):
        s = stats[speaker]
        print(f"{speaker:<12} {s['num_samples']:<10} {s['per_mean']:<12.4f} {s['mpd_mean']:<12.4f}")
    
    print("=" * 80)


def find_worst_samples(results_by_speaker, top_n=5):
    """找出错误率最高的样本"""
    all_results = []
    
    for speaker, results in results_by_speaker.items():
        for result in results:
            result['speaker'] = speaker
            all_results.append(result)
    
    all_results.sort(key=lambda x: x.get('per_score', 0), reverse=True)
    
    print("\n" + "="*80)
    print(f"❌ TOP {top_n} WORST SAMPLES")
    print("="*80)
    
    for i, r in enumerate(all_results[:top_n]):
        file_id = r.get('file_id', Path(r['sample_id']).stem)
        print(f"\n{i+1}. {r['speaker']}_{file_id}: PER={r.get('per_score', 0):.4f}")
        print(f"   {r.get('wrd', 'N/A')}")


def find_best_samples(results_by_speaker, top_n=5):
    """找出表现最好的样本"""
    all_results = []
    
    for speaker, results in results_by_speaker.items():
        for result in results:
            result['speaker'] = speaker
            all_results.append(result)
    
    all_results.sort(key=lambda x: x.get('per_score', 0))
    
    print("\n" + "="*80)
    print(f"✅ TOP {top_n} BEST SAMPLES")
    print("="*80)
    
    for i, r in enumerate(all_results[:top_n]):
        file_id = r.get('file_id', Path(r['sample_id']).stem)
        print(f"\n{i+1}. {r['speaker']}_{file_id}: PER={r.get('per_score', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="分析测试集解码结果")
    parser.add_argument("--test_dir", type=str, required=True, help="测试解码输出目录")
    parser.add_argument("--top_n", type=int, default=5, help="显示前N个样本")
    args = parser.parse_args()
    
    print(f"📂 Loading from {args.test_dir}...")
    results = load_results(args.test_dir)
    
    if not results:
        print("❌ No results found")
        sys.exit(1)
    
    print(f"✅ Loaded {sum(len(r) for r in results.values())} samples")
    
    stats = compute_statistics(results)
    print_summary(stats)
    find_worst_samples(results, top_n=args.top_n)
    find_best_samples(results, top_n=args.top_n)


if __name__ == "__main__":
    main()
