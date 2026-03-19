#!/usr/bin/env python3
"""
KenLM 语言模型训练和使用工具

功能：
1. 从文本语料库训练 n-gram LM
2. 编译为二进制格式
3. 查询和评估模型

使用方法：
    python kenlm_train_tool.py train --input corpus.txt --order 3 --output lm_dir
    python kenlm_train_tool.py query --model lm_path.arpa --text "< i y aa"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_srilm_installed() -> bool:
    """检查 SRILM 是否已安装"""
    result = subprocess.run(['which', 'ngram-count'], capture_output=True)
    return result.returncode == 0


def check_kenlm_installed() -> bool:
    """检查 KenLM 是否已安装"""
    try:
        import kenlm
        return True
    except ImportError:
        return False


def train_lm_with_srilm(corpus_file: str, output_dir: str, order: int = 3, 
                        discount: str = "kndiscount", interpolate: bool = True) -> str:
    """
    使用 SRILM 训练语言模型
    
    Args:
        corpus_file: 输入语料库文件
        output_dir: 输出目录
        order: n-gram 阶数
        discount: 折扣方法 (kndiscount, witten-bell 等)
        interpolate: 是否使用插值
    
    Returns:
        arpa_file: 生成的 ARPA 文件路径
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    arpa_file = output_path / f"lm_order{order}.arpa"
    
    logger.info(f"Training {order}-gram language model with SRILM...")
    logger.info(f"Input corpus: {corpus_file}")
    logger.info(f"Output: {arpa_file}")
    
    # 构建命令
    cmd = [
        'ngram-count',
        '-order', str(order),
        '-text', corpus_file,
        '-lm', str(arpa_file),
    ]
    
    if discount:
        cmd.extend(['-' + discount])
    
    if interpolate:
        cmd.append('-interpolate')
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ ARPA model saved to {arpa_file}")
        return str(arpa_file)
    except subprocess.CalledProcessError as e:
        logger.error(f"SRILM training failed: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("ngram-count not found. Is SRILM installed?")
        logger.error("Install SRILM from: http://www.speech.sri.com/projects/srilm/")
        sys.exit(1)


def compile_to_binary(arpa_file: str, output_file: Optional[str] = None) -> str:
    """
    将 ARPA 模型编译为二进制格式
    
    Args:
        arpa_file: ARPA 文件路径
        output_file: 输出二进制文件路径
    
    Returns:
        binary_file: 二进制文件路径
    """
    
    if not Path(arpa_file).exists():
        logger.error(f"ARPA file not found: {arpa_file}")
        sys.exit(1)
    
    if output_file is None:
        output_file = arpa_file.replace('.arpa', '.binary')
    
    logger.info(f"Compiling to binary format...")
    logger.info(f"Input: {arpa_file}")
    logger.info(f"Output: {output_file}")
    
    cmd = ['python', '-m', 'kenlm.model_builder', arpa_file, output_file]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ Binary model saved to {output_file}")
        
        # 检查文件大小
        arpa_size = Path(arpa_file).stat().st_size / (1024*1024)  # MB
        binary_size = Path(output_file).stat().st_size / (1024*1024)
        logger.info(f"  - ARPA size: {arpa_size:.1f} MB")
        logger.info(f"  - Binary size: {binary_size:.1f} MB")
        logger.info(f"  - Compression ratio: {arpa_size/binary_size:.1f}x")
        
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("KenLM model builder not found")
        sys.exit(1)


def query_lm(model_file: str, text: str) -> float:
    """
    查询语言模型得分
    
    Args:
        model_file: 模型文件路径 (.arpa 或 .binary)
        text: 查询文本
    
    Returns:
        score: 模型得分
    """
    
    try:
        import kenlm
    except ImportError:
        logger.error("KenLM not installed. Run: pip install kenlm")
        sys.exit(1)
    
    if not Path(model_file).exists():
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)
    
    logger.info(f"Loading model: {model_file}")
    model = kenlm.Model(model_file)
    
    logger.info(f"Model order: {model.order}")
    logger.info(f"Query text: {text}")
    
    score = model.score(text)
    logger.info(f"Score: {score:.6f}")
    
    return score


def evaluate_lm(model_file: str, test_file: str) -> float:
    """
    评估模型困惑度（Perplexity）
    
    Args:
        model_file: 模型文件路径
        test_file: 测试集文件
    
    Returns:
        perplexity: 困惑度分数
    """
    
    logger.info(f"Evaluating model perplexity...")
    logger.info(f"Model: {model_file}")
    logger.info(f"Test file: {test_file}")
    
    # 使用 SRILM 的 ngram 工具
    cmd = ['ngram', '-lm', model_file, '-ppl', test_file]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        
        # 解析输出获得困惑度
        logger.info(output)
        
        # 提取困惑度值（最后一行包含 "ppl= X"）
        for line in output.split('\n'):
            if 'ppl=' in line:
                ppl_str = line.split('ppl=')[1].strip().split()[0]
                ppl = float(ppl_str)
                logger.info(f"✓ Perplexity: {ppl:.4f}")
                return ppl
        
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("ngram tool not found. Is SRILM installed?")
        return None


def main():
    parser = argparse.ArgumentParser(description='KenLM Training and Query Tool')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 训练子命令
    train_parser = subparsers.add_parser('train', help='Train language model')
    train_parser.add_argument('--input', required=True, help='Input corpus file')
    train_parser.add_argument('--output', required=True, help='Output directory')
    train_parser.add_argument('--order', type=int, default=3, help='N-gram order (default: 3)')
    train_parser.add_argument('--discount', default='kndiscount', help='Discount method')
    train_parser.add_argument('--no-binary', action='store_true', help='Do not compile to binary')
    
    # 查询子命令
    query_parser = subparsers.add_parser('query', help='Query language model')
    query_parser.add_argument('--model', required=True, help='Model file path')
    query_parser.add_argument('--text', required=True, help='Text to query')
    
    # 评估子命令
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', required=True, help='Model file path')
    eval_parser.add_argument('--test', required=True, help='Test file path')
    
    # 检查子命令
    check_parser = subparsers.add_parser('check', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        logger.info("Checking dependencies...")
        srilm_ok = check_srilm_installed()
        kenlm_ok = check_kenlm_installed()
        
        logger.info(f"SRILM: {'✓ installed' if srilm_ok else '✗ not installed'}")
        logger.info(f"KenLM: {'✓ installed' if kenlm_ok else '✗ not installed'}")
        
        if not kenlm_ok:
            logger.warning("Install KenLM: pip install kenlm")
        if not srilm_ok:
            logger.warning("Install SRILM from: http://www.speech.sri.com/projects/srilm/")
    
    elif args.command == 'train':
        if not check_srilm_installed():
            logger.error("SRILM not found. Please install it first.")
            logger.error("Visit: http://www.speech.sri.com/projects/srilm/")
            sys.exit(1)
        
        # 训练模型
        arpa_file = train_lm_with_srilm(
            args.input, 
            args.output, 
            order=args.order,
            discount=args.discount
        )
        
        # 编译为二进制（可选）
        if not args.no_binary:
            if check_kenlm_installed():
                binary_file = compile_to_binary(arpa_file)
                logger.info(f"Training complete!")
                logger.info(f"ARPA model: {arpa_file}")
                logger.info(f"Binary model: {binary_file}")
            else:
                logger.warning("KenLM not installed. Skipping binary compilation.")
                logger.warning("Install with: pip install kenlm")
    
    elif args.command == 'query':
        if not check_kenlm_installed():
            logger.error("KenLM not installed. Please install it first.")
            logger.error("Run: pip install kenlm")
            sys.exit(1)
        
        query_lm(args.model, args.text)
    
    elif args.command == 'evaluate':
        evaluate_lm(args.model, args.test)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
