#!/usr/bin/env python
"""
为语料库添加 <bos> 和 <eos> 标记，然后重新训练模型
"""

import subprocess
from pathlib import Path

def add_bos_eos_to_corpus(corpus_file, output_file):
    """为语料库添加 BOS/EOS 标记"""
    print(f"📝 处理: {corpus_file.name}")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if line:
                # 添加 <bos> 和 <eos>
                new_line = f"<bos> {line} <eos>\n"
                f.write(new_line)
    
    print(f"✓ 已生成: {output_file} ({len(lines):,} 行)")

def train_with_lmplz(corpus_file, output_arpa, order=3, tmp_dir='/home/m64000/work/.tmp'):
    """用 lmplz 训练模型"""
    
    lmplz_path = Path('/home/m64000/work/IF-MDD/kenlm/install/bin/lmplz')
    
    print(f"\n🚀 训练 {order}-gram 模型: {output_arpa.name}")
    
    cmd = f"""
    cat {corpus_file} | \
    {lmplz_path} -o {order} \
        -T {tmp_dir} \
        -S 80% \
        --discount_fallback \
        > {output_arpa}
    """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        size_mb = output_arpa.stat().st_size / (1024 * 1024)
        print(f"✓ 完成 ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ 失败: {result.stderr[:200]}")
        return False

def compile_to_binary(arpa_file):
    """编译为二进制"""
    
    build_binary = Path('/home/m64000/work/IF-MDD/kenlm/install/bin/build_binary')
    bin_file = arpa_file.with_suffix('.bin')
    
    print(f"\n⚡ 编译二进制: {arpa_file.name} → {bin_file.name}")
    
    cmd = f"{build_binary} {arpa_file} {bin_file}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        arpa_size = arpa_file.stat().st_size / (1024 * 1024)
        bin_size = bin_file.stat().st_size / (1024 * 1024)
        print(f"✓ 完成")
        print(f"  ARPA: {arpa_size:.1f} MB → Binary: {bin_size:.1f} MB (压缩率 {bin_size/arpa_size*100:.1f}%)")
        return True
    else:
        print(f"❌ 失败（可能需要更多磁盘空间）")
        return False

def main():
    output_dir = Path('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("为语料库添加 <bos> 和 <eos>，然后重新训练")
    print("=" * 70)
    
    # 处理两个语料库
    corpus_files = {
        'canonical': Path('/home/m64000/work/IF-MDD/lm_models_tts/canonical_corpus.txt'),
        'perceived': Path('/home/m64000/work/IF-MDD/lm_models_tts/perceived_corpus.txt'),
    }
    
    # Step 1: 添加 BOS/EOS
    print("\n" + "=" * 70)
    print("Step 1: 添加 <bos> 和 <eos> 到语料库")
    print("=" * 70 + "\n")
    
    processed_files = {}
    
    for corpus_type, corpus_file in corpus_files.items():
        if not corpus_file.exists():
            print(f"⚠️ 文件不存在: {corpus_file}")
            continue
        
        output_corpus = output_dir / f"{corpus_type}_corpus_with_bos_eos.txt"
        add_bos_eos_to_corpus(corpus_file, output_corpus)
        processed_files[corpus_type] = output_corpus
    
    # Step 2: 训练模型
    print("\n" + "=" * 70)
    print("Step 2: 用 lmplz 训练 n-gram 模型")
    print("=" * 70)
    
    orders = [3, 4]
    arpa_files = []
    
    for corpus_type, corpus_file in processed_files.items():
        for order in orders:
            output_arpa = output_dir / f"lm_{corpus_type}_order{order}_with_bos_eos.arpa"
            
            if train_with_lmplz(corpus_file, output_arpa, order):
                arpa_files.append(output_arpa)
    
    # Step 3: 编译二进制
    print("\n" + "=" * 70)
    print("Step 3: 编译为二进制格式（可选，加速 10 倍）")
    print("=" * 70)
    
    for arpa_file in arpa_files:
        try:
            compile_to_binary(arpa_file)
        except Exception as e:
            print(f"⚠️ 编译失败（磁盘空间不足？）: {e}")
    
    # Step 4: 验证
    print("\n" + "=" * 70)
    print("Step 4: 验证模型")
    print("=" * 70 + "\n")
    
    try:
        import kenlm
        
        test_text = "<bos> <sil> < a l h a n d a l i l l a E i r a b b i l E aa l a m ii n <eos>"
        
        for arpa_file in sorted(arpa_files):
            try:
                model = kenlm.Model(str(arpa_file))
                score = model.score(test_text)
                print(f"✓ {arpa_file.name:50s} {score:10.6f}")
            except Exception as e:
                print(f"❌ {arpa_file.name:50s} {str(e)[:50]}")
        
        # 也尝试加载二进制版本
        print("\n二进制版本:")
        for bin_file in sorted(output_dir.glob('*.bin')):
            try:
                model = kenlm.Model(str(bin_file))
                score = model.score(test_text)
                print(f"✓ {bin_file.name:50s} {score:10.6f}")
            except Exception as e:
                print(f"❌ {bin_file.name:50s} {str(e)[:50]}")
    
    except ImportError:
        print("⚠️ KenLM 未安装，跳过验证")
    
    # 总结
    print("\n" + "=" * 70)
    print("✓ 完成！")
    print("=" * 70)
    print(f"\n生成的文件位置: {output_dir}")
    print(f"\n使用方法:")
    print(f"""
  import kenlm
  
  # 使用 ARPA 文件
  model = kenlm.Model('{output_dir}/lm_canonical_order3_with_bos_eos.arpa')
  
  # 或使用二进制文件（更快）
  model = kenlm.Model('{output_dir}/lm_canonical_order3_with_bos_eos.bin')
  
  # 查询时也要添加 <bos> 和 <eos>
  score = model.score('<bos> <sil> < a l h a n d <eos>')
  print(score)
    """)

if __name__ == '__main__':
    main()
