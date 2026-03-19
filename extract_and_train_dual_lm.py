"""
从 IQRA 数据中提取 perceived_aligned 和 canonical_aligned
分别生成两个 n-gram 语言模型

用法:
  python extract_and_train_dual_lm.py \
      --data /path/to/iqra_train.json \
      --order 3 \
      --output-dir ./lm_models
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# 导入纯 Python n-gram 训练器
from pure_ngram_trainer import PureNGramTrainer


class DualLMTrainer:
    """从 IQRA 训练双语言模型"""
    
    def __init__(self, output_dir: str):
        """
        初始化
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.canonical_corpus_file = self.output_dir / "canonical_corpus.txt"
        self.perceived_corpus_file = self.output_dir / "perceived_corpus.txt"
        
        self.stats = {
            'canonical': defaultdict(int),
            'perceived': defaultdict(int)
        }
    
    def extract_corpus(self, data_file: str):
        """
        从 JSON 文件中提取语料库
        
        Args:
            data_file: IQRA JSON 文件路径
        """
        print(f"📖 读取数据文件: {data_file}")
        
        canonical_file = open(self.canonical_corpus_file, 'w', encoding='utf-8')
        perceived_file = open(self.perceived_corpus_file, 'w', encoding='utf-8')
        
        total_samples = 0
        skipped = 0
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for wav_path, record in data.items():
                total_samples += 1
                
                # 提取 canonical_aligned
                canonical = record.get('canonical_aligned', '').strip()
                perceived = record.get('perceived_aligned', '').strip()
                
                if not canonical or not perceived:
                    skipped += 1
                    continue
                
                # 写入语料库文件
                canonical_file.write(canonical + '\n')
                perceived_file.write(perceived + '\n')
                
                # 统计字段值（用于分析）
                self.stats['canonical'][canonical] += 1
                self.stats['perceived'][perceived] += 1
                
                # 进度显示
                if total_samples % 10000 == 0:
                    print(f"   已处理: {total_samples} 条记录...")
        
        finally:
            canonical_file.close()
            perceived_file.close()
        
        print(f"\n✓ 数据提取完成！")
        print(f"   总样本数: {total_samples}")
        print(f"   跳过数: {skipped}")
        print(f"   实际样本: {total_samples - skipped}")
        print(f"   - Canonical: {self.canonical_corpus_file}")
        print(f"   - Perceived: {self.perceived_corpus_file}")
        
        return total_samples - skipped
    
    def train_models(self, orders: list = None):
        """
        训练两个 n-gram 模型
        
        Args:
            orders: n-gram 阶数列表（默认 [2, 3, 4]）
        """
        if orders is None:
            orders = [3]
        
        print(f"\n" + "=" * 60)
        print("训练 Canonical 语言模型")
        print("=" * 60)
        
        self._train_single_model(
            self.canonical_corpus_file,
            'canonical',
            orders
        )
        
        print(f"\n" + "=" * 60)
        print("训练 Perceived 语言模型")
        print("=" * 60)
        
        self._train_single_model(
            self.perceived_corpus_file,
            'perceived',
            orders
        )
    
    def _train_single_model(self, corpus_file: Path, model_name: str, orders: list):
        """训练单个模型"""
        for order in orders:
            print(f"\n🚀 训练 {model_name} {order}-gram 模型...")
            
            trainer = PureNGramTrainer(order=order, smooth_method='kneser-ney')
            trainer.train(str(corpus_file))
            
            # 保存 ARPA 格式
            output_arpa = self.output_dir / f"lm_{model_name}_order{order}.arpa"
            trainer.save_arpa(str(output_arpa))
            
            # 保存 JSON 格式（可选）
            output_json = self.output_dir / f"lm_{model_name}_order{order}.json"
            trainer.save_json(str(output_json))
            
            # 统计信息
            print(f"\n   模型统计:")
            for n in range(1, order + 1):
                ngram_count = len(trainer.ngrams[n])
                total_count = sum(trainer.ngrams[n].values())
                print(f"   - {n}-gram: {ngram_count} 种类, {total_count} 总数")


def main():
    parser = argparse.ArgumentParser(
        description="从 IQRA 数据提取两个 n-gram 语言模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 3-gram 模型
  python extract_and_train_dual_lm.py \\
      --data /home/m64000/work/dataset/data_iqra/iqra_train.json \\
      --order 3 \\
      --output-dir ./lm_models
  
  # 训练多个模型（2-gram, 3-gram, 4-gram）
  python extract_and_train_dual_lm.py \\
      --data /home/m64000/work/dataset/data_iqra/iqra_train.json \\
      --order 2 3 4 \\
      --output-dir ./lm_models
        """
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='IQRA JSON 数据文件路径'
    )
    parser.add_argument(
        '--order',
        type=int,
        nargs='+',
        default=[3],
        help='n-gram 阶数（可指定多个，例如 2 3 4）'
    )
    parser.add_argument(
        '--output-dir',
        default='./lm_models',
        help='输出目录（默认 ./lm_models）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.data).exists():
        print(f"❌ 错误：数据文件不存在: {args.data}")
        sys.exit(1)
    
    # 开始处理
    print("\n" + "=" * 60)
    print("IQRA 数据 → 双 n-gram 语言模型")
    print("=" * 60)
    
    trainer = DualLMTrainer(args.output_dir)
    
    # Step 1: 提取语料库
    sample_count = trainer.extract_corpus(args.data)
    
    if sample_count == 0:
        print("❌ 错误：没有有效的样本")
        sys.exit(1)
    
    # Step 2: 训练模型
    trainer.train_models(args.order)
    
    # 总结
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    output_dir = Path(args.output_dir)
    arpa_files = list(output_dir.glob("*.arpa"))
    for arpa_file in sorted(arpa_files):
        size_mb = arpa_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ {arpa_file.name} ({size_mb:.2f} MB)")
    
    print(f"\n💡 下一步:")
    print(f"  1. 检查生成的 .arpa 文件")
    print(f"  2. 用 KenLM 进行快速查询:")
    print(f"")
    print(f"     import kenlm")
    print(f"     model_canonical = kenlm.Model('{output_dir}/lm_canonical_order3.arpa')")
    print(f"     model_perceived = kenlm.Model('{output_dir}/lm_perceived_order3.arpa')")
    print(f"")
    print(f"     score_c = model_canonical.score('f ii h i nn a')")
    print(f"     score_p = model_perceived.score('f ii h i nn a')")
    print(f"")
    print(f"  3. 集成到推理管道中进行二阶段解码")


if __name__ == '__main__':
    main()
