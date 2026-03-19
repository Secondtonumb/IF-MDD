"""
快速预览提取的 perceived 和 canonical 数据
"""

import json
from pathlib import Path
from collections import Counter

data_file = '/home/m64000/work/dataset/data_iqra/iqra_train.json'

print("📊 数据分析\n")

print(f"📖 读取数据: {data_file}")
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✓ 加载 {len(data)} 条记录\n")

# 提取 canonical 和 perceived
canonical_sequences = []
perceived_sequences = []
mismatches = 0
same_matches = 0

for wav_path, record in list(data.items())[:1000]:  # 先看前 1000 条
    canonical = record.get('canonical_aligned', '').strip()
    perceived = record.get('perceived_aligned', '').strip()
    
    if canonical and perceived:
        canonical_sequences.append(canonical)
        perceived_sequences.append(perceived)
        
        if canonical == perceived:
            same_matches += 1
        else:
            mismatches += 1

print(f"✓ 提取样本统计 (前1000条):")
print(f"  - 有效样本: {len(canonical_sequences)}")
print(f"  - Canonical == Perceived: {same_matches}")
print(f"  - Canonical != Perceived: {mismatches}")
print(f"  - 不匹配率: {mismatches/len(canonical_sequences)*100:.1f}%\n")

# 统计序列长度
canonical_lengths = [len(s.split()) for s in canonical_sequences]
perceived_lengths = [len(s.split()) for s in perceived_sequences]

print(f"📏 序列长度统计:")
print(f"  - Canonical: min={min(canonical_lengths)}, max={max(canonical_lengths)}, avg={sum(canonical_lengths)/len(canonical_lengths):.1f}")
print(f"  - Perceived: min={min(perceived_lengths)}, max={max(perceived_lengths)}, avg={sum(perceived_lengths)/len(perceived_lengths):.1f}\n")

# 显示样本
print(f"📝 样本 (前10条):\n")
for i in range(min(10, len(canonical_sequences))):
    print(f"{i+1}. Canonical: {canonical_sequences[i]}")
    print(f"   Perceived: {perceived_sequences[i]}")
    if canonical_sequences[i] != perceived_sequences[i]:
        print(f"   ⚠️ 不匹配！")
    print()

print(f"=" * 60)
print(f"✓ 初步分析完成你可以现在运行完整提取和训练：")
print(f"")
print(f"  python extract_and_train_dual_lm.py \\")
print(f"      --data /home/m64000/work/dataset/data_iqra/iqra_train.json \\")
print(f"      --order 3 \\")
print(f"      --output-dir ./lm_models")
print(f"")
print(f"或者包含多个阶数：")
print(f"")
print(f"  python extract_and_train_dual_lm.py \\")
print(f"      --data /home/m64000/work/dataset/data_iqra/iqra_train.json \\")
print(f"      --order 2 3 4 \\")
print(f"      --output-dir ./lm_models")
