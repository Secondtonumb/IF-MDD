#!/usr/bin/env python
"""
从 IQRA TTS 数据提取语料库并训练 n-gram 模型

用法:
  python prepare_tts_corpus_and_train.py \
      --data /home/m64000/work/dataset/data_iqra_tts/tts_train/iqra_tts_all_aligned.json \
      --order 3 \
      --output-dir ./lm_models_tts
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import subprocess
import sys

class TTSCorpusExtractor:
    """从 TTS JSON 数据提取语料库"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.canonical_corpus_file = self.output_dir / "canonical_corpus.txt"
        self.perceived_corpus_file = self.output_dir / "perceived_corpus.txt"
    
    def extract_corpus(self, data_file: str):
        """
        从 JSON 提取语料库
        
        Args:
            data_file: JSON 数据文件路径
        """
        print(f"📖 读取数据文件: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 加载 {len(data)} 条记录\n")
        
        canonical_file = open(self.canonical_corpus_file, 'w', encoding='utf-8')
        perceived_file = open(self.perceived_corpus_file, 'w', encoding='utf-8')
        
        total_samples = 0
        skipped = 0
        
        try:
            for wav_path, record in data.items():
                total_samples += 1
                
                # 提取对齐的音素序列
                canonical = record.get('canonical_aligned', '').strip()
                perceived = record.get('perceived_aligned', '').strip()
                
                if not canonical or not perceived:
                    skipped += 1
                    continue
                
                # 清理多余的空格
                canonical = ' '.join(canonical.split())
                perceived = ' '.join(perceived.split())
                
                # 写入语料库
                canonical_file.write(canonical + '\n')
                perceived_file.write(perceived + '\n')
                
                # 进度显示
                if total_samples % 10000 == 0:
                    print(f"   已处理: {total_samples:,} 条记录...")
        
        finally:
            canonical_file.close()
            perceived_file.close()
        
        print(f"\n✓ 语料库提取完成！")
        print(f"   总样本数: {total_samples:,}")
        print(f"   跳过数: {skipped:,}")
        print(f"   实际样本: {total_samples - skipped:,}")
        print(f"   - Canonical: {self.canonical_corpus_file}")
        print(f"   - Perceived: {self.perceived_corpus_file}")
        
        # 统计行数
        canonical_lines = sum(1 for _ in open(self.canonical_corpus_file))
        perceived_lines = sum(1 for _ in open(self.perceived_corpus_file))
        print(f"\n   Canonical 行数: {canonical_lines:,}")
        print(f"   Perceived 行数: {perceived_lines:,}")
        
        return total_samples - skipped
    
    def train_with_lmplz(self, orders: list = None):
        """
        用 lmplz 训练模型
        
        Args:
            orders: n-gram 阶数列表
        """
        if orders is None:
            orders = [3]
        
        # 检查 lmplz 是否存在
        lmplz_path = Path('/home/m64000/work/IF-MDD/kenlm/install/bin/lmplz')
        if not lmplz_path.exists():
            print(f"❌ 错误: lmplz 不存在: {lmplz_path}")
            print("   请先安装 KenLM 工具")
            return False
        
        print(f"\n" + "=" * 60)
        print("用 lmplz 训练 n-gram 模型")
        print("=" * 60)
        
        # 临时目录
        tmp_dir = Path('/home/m64000/work/.tmp')
        tmp_dir.mkdir(exist_ok=True)
        
        for model_type in ['canonical', 'perceived']:
            corpus_file = {
                'canonical': self.canonical_corpus_file,
                'perceived': self.perceived_corpus_file
            }[model_type]
            
            print(f"\n🚀 训练 {model_type} 模型...")
            
            for order in orders:
                output_file = self.output_dir / f"lm_{model_type}_order{order}.arpa"
                
                print(f"\n   {model_type} {order}-gram → {output_file.name}")
                
                # 构建命令
                cmd = f"""
                cat {corpus_file} | \
                {lmplz_path} -o {order} \
                    -T {tmp_dir} \
                    -S 80% \
                    --discount_fallback \
                    > {output_file}
                """
                
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode == 0:
                        file_size_mb = output_file.stat().st_size / (1024 * 1024)
                        print(f"   ✓ 完成 ({file_size_mb:.1f} MB)")
                    else:
                        print(f"   ❌ 失败")
                        if result.stderr:
                            print(f"   错误: {result.stderr[:200]}")
                        return False
                
                except subprocess.TimeoutExpired:
                    print(f"   ❌ 超时")
                    return False
                except Exception as e:
                    print(f"   ❌ 错误: {e}")
                    return False
        
        return True
    
    def verify_models(self):
        """验证生成的模型"""
        try:
            import kenlm
        except ImportError:
            print("\n⚠️ KenLM 未安装，跳过验证")
            return
        
        print(f"\n" + "=" * 60)
        print("验证生成的模型")
        print("=" * 60)
        
        test_text = "<sil> < a l h a n d"
        
        arpa_files = list(self.output_dir.glob('*.arpa'))
        
        if not arpa_files:
            print("❌ 未找到 ARPA 文件")
            return
        
        print(f"\n{len(arpa_files)} 个模型:\n")
        
        for arpa_file in sorted(arpa_files):
            try:
                model = kenlm.Model(str(arpa_file))
                score = model.score(test_text)
                print(f"  ✓ {arpa_file.name:40s} {score:10.6f}")
            except Exception as e:
                print(f"  ❌ {arpa_file.name:40s} {str(e)[:50]}")


def main():
    parser = argparse.ArgumentParser(
        description="从 IQRA TTS 数据提取语料库并训练 n-gram 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 3-gram 模型
  python prepare_tts_corpus_and_train.py \\
      --data /home/m64000/work/dataset/data_iqra_tts/tts_train/iqra_tts_all_aligned.json \\
      --order 3 \\
      --output-dir ./lm_models_tts
  
  # 训练多个阶数
  python prepare_tts_corpus_and_train.py \\
      --data /home/m64000/work/dataset/data_iqra_tts/tts_train/iqra_tts_all_aligned.json \\
      --order 2 3 4 \\
      --output-dir ./lm_models_tts
        """
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='TTS JSON 数据文件路径'
    )
    parser.add_argument(
        '--order',
        type=int,
        nargs='+',
        default=[3],
        help='n-gram 阶数（可指定多个）'
    )
    parser.add_argument(
        '--output-dir',
        default='./lm_models_tts',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.data).exists():
        print(f"❌ 错误：数据文件不存在: {args.data}")
        sys.exit(1)
    
    # 开始处理
    print("\n" + "=" * 60)
    print("TTS 数据 → n-gram 语言模型")
    print("=" * 60 + "\n")
    
    extractor = TTSCorpusExtractor(args.output_dir)
    
    # Step 1: 提取语料库
    sample_count = extractor.extract_corpus(args.data)
    
    if sample_count == 0:
        print("❌ 错误：没有有效的样本")
        sys.exit(1)
    
    # Step 2: 训练模型
    success = extractor.train_with_lmplz(args.order)
    
    if not success:
        print("\n❌ 训练失败")
        sys.exit(1)
    
    # Step 3: 验证模型
    extractor.verify_models()
    
    # 总结
    print("\n" + "=" * 60)
    print("✓ 完成！")
    print("=" * 60)
    print(f"\n生成的文件位置: {args.output_dir}")
    print(f"\n使用方法:")
    print(f"""
  import kenlm
  model = kenlm.Model('{args.output_dir}/lm_canonical_order3.arpa')
  score = model.score('<sil> < a l h a n d')
  print(score)
    """)


if __name__ == '__main__':
    main()
