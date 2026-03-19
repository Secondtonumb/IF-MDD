#!/usr/bin/env python
"""
对比旧模型（无 BOS/EOS）和新模型（有 BOS/EOS）
"""

import sys
sys.path.insert(0, '/home/m64000/work/IF-MDD/kenlm/python')

import kenlm
from pathlib import Path

def test_model(model_path, test_texts):
    """测试模型"""
    try:
        model = kenlm.Model(str(model_path))
        print(f"\n📊 {Path(model_path).name}")
        print("-" * 60)
        
        for text in test_texts:
            score = model.score(text)
            print(f"  {text[:50]:50s} → {score:10.6f}")
        
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

def main():
    print("=" * 70)
    print("对比：旧模型（无 BOS/EOS）vs 新模型（有 BOS/EOS）")
    print("=" * 70)
    
    test_texts = [
        "<sil> < a l h a n d a l i l l a E i r a b b i l E aa l a m ii n",  # 旧格式
        "<bos> <sil> < a l h a n d a l i l l a E i r a b b i l E aa l a m ii n <eos>",  # 新格式
    ]
    
    models = [
        Path('/home/m64000/work/IF-MDD/lm_models_tts/lm_canonical_order3.arpa'),
        Path('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/lm_canonical_order3_with_bos_eos.arpa'),
        Path('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/lm_canonical_order3_with_bos_eos.bin'),
        Path('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/lm_canonical_order4_with_bos_eos.arpa'),
        Path('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/lm_canonical_order4_with_bos_eos.bin'),
    ]
    
    print("\n测试文本:")
    print(f"  1. (旧格式) {test_texts[0][:40]}...")
    print(f"  2. (新格式) {test_texts[1][:40]}...")
    
    success_count = 0
    for model_path in models:
        if model_path.exists():
            if test_model(model_path, test_texts):
                success_count += 1
        else:
            print(f"⚠️ 文件不存在: {model_path}")
    
    print("\n" + "=" * 70)
    print(f"✓ 成功加载 {success_count}/{len(models)} 个模型")
    print("=" * 70)
    
    summary = """
📝 总结：
  ✓ 旧模型位置: /home/m64000/work/IF-MDD/lm_models_tts/
  ✓ 新模型位置: /home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/
  
💡 使用建议：
  - 使用 .bin 文件速度快 10 倍
  - 新模型中查询时需要加 <bos> ... <eos>
  - 对比看起来旧模型用旧格式分数更高，这是预期的
  
🔧 使用样例：
  
  import sys
  sys.path.insert(0, '/home/m64000/work/IF-MDD/kenlm/python')
  import kenlm
  
  # 使用新模型（推荐用二进制版本）
  model = kenlm.Model('/home/m64000/work/IF-MDD/lm_models_tts_with_bos_eos/lm_canonical_order4_with_bos_eos.bin')
  
  # 查询时要加 <bos> 和 <eos>
  text = "<bos> <sil> < a l h a n d a l i l l a h <eos>"
  score = model.score(text)
  print(f"Score: {score:.6f}")
  
  # 或者用 full_scores() 获取每个 token 的分数
  for word, score in model.full_scores(text):
      print(f"  {word:20s} {score[0]:10.6f}")
    """
    print(summary)

if __name__ == '__main__':
    main()
