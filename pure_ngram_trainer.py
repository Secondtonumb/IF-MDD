"""
Pure Python N-gram Language Model Trainer - Generates ARPA format
No external dependencies except numpy (可选)

纯 Python n-gram 语言模型训练工具，生成标准 ARPA 格式
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Set
import math
import argparse
from dataclasses import dataclass


@dataclass
class NGramStats:
    """N-gram 统计信息"""
    count: int
    backoff_weight: float = 0.0


class PureNGramTrainer:
    """
    纯 Python n-gram 语言模型训练器
    支持 Kneser-Ney 平滑和简单回退
    """
    
    def __init__(self, order: int = 3, smooth_method: str = "kneser-ney"):
        """
        初始化训练器
        
        Args:
            order: n-gram 阶数（2-5）
            smooth_method: "kneser-ney" 或 "backoff"
        """
        self.order = order
        self.smooth_method = smooth_method
        self.ngrams = defaultdict(lambda: defaultdict(int))  # {n: {tuple: count}}
        self.backoff_weights = defaultdict(float)  # {tuple: weight}
        self.vocab = set()
        self.unk_token = "<unk>"
        self.start_token = "<bos>"
        self.end_token = "<eos>"
        
    def load_corpus(self, corpus_file: str, limit_vocab: int = None):
        """准备语料库"""
        self.vocab = {self.unk_token, self.start_token, self.end_token}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    self.vocab.update(tokens)
        
        # 限制词汇表大小（可选）
        if limit_vocab and len(self.vocab) > limit_vocab:
            # 保留最常见的词汇
            self.vocab = {self.unk_token, self.start_token, self.end_token}
            # 这里可以添加频率计数来筛选
    
    def _tokenize_line(self, line: str) -> List[str]:
        """分词"""
        tokens = line.strip().split()
        return [self.start_token] * (self.order - 1) + tokens + [self.end_token]
    
    def _extract_ngrams(self, tokens: List[str]):
        """从句子中提取所有 n-gram"""
        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                self.ngrams[n][ngram] += 1
    
    def train(self, corpus_file: str):
        """训练模型"""
        print(f"📖 读取语料库: {corpus_file}")
        
        # 第一遍：收集词汇表
        self.load_corpus(corpus_file)
        print(f"   词汇表大小: {len(self.vocab)}")
        
        # 第二遍：抽取 n-gram 并计数
        line_count = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                tokens = self._tokenize_line(line)
                self._extract_ngrams(tokens)
                line_count += 1
        
        print(f"   处理行数: {line_count}")
        
        # 统计 n-gram 数量
        for n in range(1, self.order + 1):
            count = len(self.ngrams[n])
            total = sum(self.ngrams[n].values())
            print(f"   {n}-gram: {count} 种类, {total} 总数")
        
        # 计算回退权重
        if self.smooth_method == "kneser-ney":
            self._compute_kneser_ney_weights()
        else:
            self._compute_simple_backoff_weights()
    
    def _compute_simple_backoff_weights(self):
        """计算简单回退权重"""
        # 简单实现：对每个 n-gram 计算回退权重
        for n in range(1, self.order):
            for ngram, count in self.ngrams[n].items():
                # 计算该 n-gram 出现后被折扣的概率
                prefix = ngram[:-1]
                suffix = ngram[-1]
                
                # 计算折扣率
                if len(self.ngrams[n+1]) > 0:
                    # 简单折扣：0.75
                    discount = 0.75
                    
                    # 回退权重 = -log10(P(suffix | prefix in n+1-gram))
                    sum_after_prefix = sum(
                        c for ng, c in self.ngrams[n+1].items() 
                        if ng[:-1] == prefix
                    )
                    
                    if sum_after_prefix > 0:
                        weight = math.log10(sum_after_prefix / sum(self.ngrams[n].values()))
                        self.backoff_weights[ngram] = weight
    
    def _compute_kneser_ney_weights(self):
        """计算 Kneser-Ney 平滑权重（简化版）"""
        # 这是 Kneser-Ney 的简化实现
        # 完整版本较为复杂，这里提供基础版本
        
        for n in range(1, self.order):
            for ngram in self.ngrams[n]:
                # 基础 Kneser-Ney 权重计算
                prefix = ngram[:-1]
                count = self.ngrams[n][ngram]
                
                # 简化：使用折扣因子
                discount = 0.75
                
                # 计算权重
                weight = math.log10(max(count - discount, 0.1))
                self.backoff_weights[ngram] = weight
    
    def save_arpa(self, output_file: str):
        """保存为 ARPA 格式"""
        print(f"\n📝 生成 ARPA 文件: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入 ARPA 头
            f.write("\\data\\\n")
            for n in range(1, self.order + 1):
                count = len(self.ngrams[n])
                f.write(f"ngram {n}={count}\n")
            f.write("\n")
            
            # 写入每阶 n-gram
            for n in range(1, self.order + 1):
                f.write(f"\\{n}-grams:\n")
                
                for ngram, count in sorted(
                    self.ngrams[n].items(),
                    key=lambda x: -x[1]  # 按频率降序
                ):
                    # 计算概率
                    prefix = ngram[:-1]
                    if prefix in self.ngrams[n-1] if n > 1 else False:
                        prefix_count = self.ngrams[n-1][prefix]
                    else:
                        prefix_count = sum(self.ngrams[n-1].values())
                    
                    if prefix_count > 0:
                        prob = math.log10(count / prefix_count)
                    else:
                        prob = -99  # 未见过
                    
                    # ARPA 格式: prob [backoff_weight]
                    ngram_str = " ".join(ngram)
                    
                    if n < self.order:
                        # 有回退权重
                        backoff = self.backoff_weights.get(ngram, 0.0)
                        f.write(f"{prob:.6f}\t{ngram_str}\t{backoff:.6f}\n")
                    else:
                        # 最后一阶，无回退权重
                        f.write(f"{prob:.6f}\t{ngram_str}\n")
                
                f.write("\n")
            
            # 写入结束标记
            f.write("\\end\\\n")
        
        file_size = Path(output_file).stat().st_size
        print(f"   文件大小: {file_size / 1024:.1f} KB")
        print(f"✓ ARPA 文件已生成！")
    
    def save_json(self, output_file: str):
        """保存为 JSON 格式（便于 Python 使用）"""
        data = {
            "order": self.order,
            "vocab_size": len(self.vocab),
            "ngrams": {}
        }
        
        for n in range(1, self.order + 1):
            data["ngrams"][str(n)] = {
                " ".join(k): v 
                for k, v in self.ngrams[n].items()
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON 文件已生成: {output_file}")
    
    def query(self, text: str) -> float:
        """查询文本得分（对数概率）"""
        tokens = self._tokenize_line(text)
        prob = 0.0
        
        for i in range(len(tokens) - 1):
            ngram = tuple(tokens[max(0, i - self.order + 2):i+2])
            
            if len(ngram) == self.order:
                if ngram in self.ngrams[self.order]:
                    prefix = ngram[:-1]
                    prefix_count = sum(
                        c for ng, c in self.ngrams[self.order-1].items()
                        if ng[:-1] == prefix
                    )
                    
                    if prefix_count > 0:
                        prob += math.log10(self.ngrams[self.order][ngram] / prefix_count)
                    else:
                        prob -= 5.0  # 未见过的前缀
        
        return prob
    
    def compute_perplexity(self, test_file: str, smooth: bool = True) -> float:
        """计算测试集困惑度"""
        total_log_prob = 0.0
        token_count = 0
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                tokens = self._tokenize_line(line)
                prob = self.query(line)
                
                total_log_prob += prob
                token_count += len(tokens) - 1
        
        if token_count > 0:
            perplexity = 10 ** (-total_log_prob / token_count)
            return perplexity
        return float('inf')


def main():
    parser = argparse.ArgumentParser(
        description="纯 Python n-gram 语言模型训练器 (生成 ARPA 格式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 3-gram 模型
  python pure_ngram_trainer.py train \\
      --input corpus.txt \\
      --output model.arpa \\
      --order 3
  
  # 生成 JSON 格式（可选）
  python pure_ngram_trainer.py train \\
      --input corpus.txt \\
      --output model \\
      --order 3 \\
      --format both
  
  # 查询文本得分
  python pure_ngram_trainer.py query \\
      --model model.arpa \\
      --text "< i y aa k"
  
  # 评估困惑度
  python pure_ngram_trainer.py evaluate \\
      --model model.arpa \\
      --test test.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='训练语言模型')
    train_parser.add_argument('--input', required=True, help='输入语料库文件')
    train_parser.add_argument('--output', required=True, help='输出文件名（不含后缀）')
    train_parser.add_argument('--order', type=int, default=3, help='n-gram 阶数（默认 3）')
    train_parser.add_argument('--smooth', choices=['kneser-ney', 'backoff'], 
                            default='kneser-ney', help='平滑方法')
    train_parser.add_argument('--format', choices=['arpa', 'json', 'both'], 
                            default='arpa', help='输出格式')
    
    # query 子命令
    query_parser = subparsers.add_parser('query', help='查询文本得分')
    query_parser.add_argument('--model', required=True, help='ARPA 模型文件')
    query_parser.add_argument('--text', required=True, help='查询文本')
    
    # evaluate 子命令
    eval_parser = subparsers.add_parser('evaluate', help='评估困惑度')
    eval_parser.add_argument('--model', required=True, help='ARPA 模型文件')
    eval_parser.add_argument('--test', required=True, help='测试集文件')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        trainer = PureNGramTrainer(order=args.order, smooth_method=args.smooth)
        print(f"🚀 开始训练 {args.order}-gram 模型")
        print(f"   平滑方法: {args.smooth}\n")
        
        trainer.train(args.input)
        
        if args.format in ['arpa', 'both']:
            trainer.save_arpa(f"{args.output}.arpa")
        
        if args.format in ['json', 'both']:
            trainer.save_json(f"{args.output}.json")
    
    elif args.command == 'query':
        print("📝 查询功能需要从 ARPA 文件加载模型（演示用）")
        print("   请使用 kenlm 或自己实现加载逻辑")
        # 这里可以添加 ARPA 文件加载和查询
    
    elif args.command == 'evaluate':
        print("📊 评估功能需要从 ARPA 文件加载模型（演示用）")
        # 同上
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


# ==================== 使用示例 ====================
"""
# 1. 快速使用（Python 脚本）

from pure_ngram_trainer import PureNGramTrainer

# 创建训练器
trainer = PureNGramTrainer(order=3, smooth_method='kneser-ney')

# 训练
trainer.train('corpus.txt')

# 保存 ARPA
trainer.save_arpa('model.arpa')

# 保存 JSON（可选）
trainer.save_json('model.json')

# 查询得分
score = trainer.query('< i y aa k')
print(f"Score: {score}")

# 困惑度
ppl = trainer.compute_perplexity('test.txt')
print(f"Perplexity: {ppl}")


# 2. 命令行使用

# 训练
python pure_ngram_trainer.py train \\
    --input corpus.txt \\
    --output model \\
    --order 3 \\
    --format both

# 输出文件:
#   model.arpa  - 标准 ARPA 格式（可用于 KenLM）
#   model.json  - JSON 格式（可用于 Python）


# 3. 与 KenLM 的配合

# 用纯 Python 生成 ARPA
python pure_ngram_trainer.py train --input corpus.txt --output model --order 3

# 用 KenLM 快速加载和查询（10 倍速）
import kenlm
model = kenlm.Model('model.arpa')
score = model.score('< i y aa k')
"""
