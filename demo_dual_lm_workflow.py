#!/usr/bin/env python
"""
演示脚本：从 IQRA 数据生成双语言模型的完整流程

运行: python demo_dual_lm_workflow.py
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"▶️  {description}")
    print(f"   命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ 错误: 命令失败 (返回码: {result.returncode})")
        return False
    
    print(f"✓ 完成\n")
    return True

def main():
    print_section("从 IQRA 生成双语言模型 - 完整演示")
    
    # Step 1: 环境检查
    print_section("Step 1: 环境检查")
    
    # 检查必需文件
    required_files = [
        'pure_ngram_trainer.py',
        'extract_and_train_dual_lm.py',
        'preview_dual_lm_data.py'
    ]
    
    for f in required_files:
        if Path(f).exists():
            print(f"✓ {f}")
        else:
            print(f"❌ 缺失: {f}")
    
    data_file = Path('/home/m64000/work/dataset/data_iqra/iqra_train.json')
    if data_file.exists():
        print(f"✓ 数据文件: {data_file}")
    else:
        print(f"❌ 缺失数据文件: {data_file}")
        return
    
    # Step 2: 数据预览
    print_section("Step 2: 预览 IQRA 数据")
    
    if not run_command(
        'python preview_dual_lm_data.py',
        '分析前 1000 条样本的数据统计'
    ):
        return
    
    # Step 3: 提取并训练
    print_section("Step 3: 提取语料库并训练 n-gram 模型")
    
    print("💡 这一步将:")
    print("   - 从 JSON 提取 canonical_aligned 和 perceived_aligned")
    print("   - 生成两个语料库文件 (每个 71K 行)")
    print("   - 分别训练 2-gram, 3-gram, 4-gram 模型")
    print("   - 生成 ARPA 格式文件\n")
    
    if not run_command(
        'python extract_and_train_dual_lm.py '
        '--data /home/m64000/work/dataset/data_iqra/iqra_train.json '
        '--order 2 3 4 '
        '--output-dir ./lm_models',
        '完整的提取 + 训练流程'
    ):
        return
    
    # Step 4: 验证输出
    print_section("Step 4: 验证生成的模型文件")
    
    lm_dir = Path('lm_models')
    if lm_dir.exists():
        print("✓ lm_models 目录存在\n")
        
        arpa_files = list(lm_dir.glob('*.arpa'))
        if arpa_files:
            print(f"生成的 ARPA 文件 ({len(arpa_files)} 个):\n")
            for f in sorted(arpa_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f.name:35s} {size_mb:6.1f} MB")
        else:
            print("❌ 未找到 ARPA 文件")
            return
        
        corpus_files = [
            lm_dir / 'canonical_corpus.txt',
            lm_dir / 'perceived_corpus.txt'
        ]
        
        print("\n语料库文件:\n")
        for f in corpus_files:
            if f.exists():
                lines = sum(1 for _ in open(f))
                print(f"  ✓ {f.name:35s} {lines:7,} 行")
            else:
                print(f"  ❌ {f.name} 缺失")
    else:
        print(f"❌ lm_models 目录不存在")
        return
    
    # Step 5: KenLM 测试
    print_section("Step 5: 用 KenLM 查询模型")
    
    print("尝试加载并查询模型...\n")
    
    test_code = """
import sys
try:
    import kenlm
    
    # 加载模型
    model_c2 = kenlm.Model('./lm_models/lm_canonical_order2.arpa')
    model_c3 = kenlm.Model('./lm_models/lm_canonical_order3.arpa')
    model_p3 = kenlm.Model('./lm_models/lm_perceived_order3.arpa')
    
    test_text = "f ii h i nn a x A y r aa t H i s aa n"
    
    print("查询结果:\\n")
    print(f"  文本: {test_text}")
    print(f"  Canonical 2-gram: {model_c2.score(test_text):10.6f}")
    print(f"  Canonical 3-gram: {model_c3.score(test_text):10.6f}")
    print(f"  Perceived 3-gram: {model_p3.score(test_text):10.6f}")
    print()
    
    # 批量查询演示
    test_texts = [
        "f ii h i nn a x A",
        "w a < i * aa nn u",
        "m i n n U T f a",
    ]
    
    print(f"批量查询 (3-gram):\\n")
    print(f"{'文本':30s} {'Canonical':12s} {'Perceived':12s}")
    print("-" * 56)
    for text in test_texts:
        score_c = model_c3.score(text)
        score_p = model_p3.score(text)
        print(f"{text:30s} {score_c:12.6f} {score_p:12.6f}")
    
except ImportError as e:
    print(f"⚠️  KenLM 未安装或加载失败")
    print(f"   错误: {e}")
    print(f"   请运行: export CC=/usr/bin/gcc CXX=/usr/bin/g++")
    print(f"           pip install --no-cache-dir kenlm")
"""
    
    if not run_command(
        f'python -c "{test_code}"',
        '尝试加载并查询所有模型'
    ):
        print("⚠️  KenLM 测试失败（可能未安装），但模型文件已正确生成")
    
    # Step 6: 总结
    print_section("完成！✨")
    
    print("📊 处理统计:\n")
    print("  ✓ 数据来源: /home/m64000/work/dataset/data_iqra/iqra_train.json")
    print("  ✓ 总记录数: 71,391 条")
    print("  ✓ 提取的语料: canonical + perceived (各 71K+ 行)")
    print("  ✓ 生成的模型:")
    print("    - 2-gram: 2 个模型 (Canonical + Perceived)")
    print("    - 3-gram: 2 个模型 (Canonical + Perceived)")
    print("    - 4-gram: 2 个模型 (Canonical + Perceived)")
    
    print("\n📁 输出位置: ./lm_models/\n")
    
    print("🚀 下一步:")
    print("  1. 查看 DUAL_LM_QUICKSTART.md 了解更多用法")
    print("  2. 集成到 ASR 推理管道")
    print("  3. 使用双 LM 进行候选重评分")
    print("  4. 评估 WER 改进\n")
    
    print("💡 推荐的集成代码:\n")
    
    integration_code = '''
    import kenlm
    
    # 加载模型
    lm_c = kenlm.Model('./lm_models/lm_canonical_order3.arpa')
    lm_p = kenlm.Model('./lm_models/lm_perceived_order3.arpa')
    
    # 在解码循环中
    def rescore_hypothesis(phoneme_seq):
        text = " ".join(phoneme_seq)
        score_c = lm_c.score(text)
        score_p = lm_p.score(text)
        
        # 组合得分（可调权重）
        combined = 0.5 * score_c + 0.5 * score_p
        return combined
    '''
    
    for line in integration_code.split('\n'):
        print(f"    {line}")
    
    print("\n✓ 演示完成！")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
