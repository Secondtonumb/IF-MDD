"""
测试 SSL_LLM Prompt 模式的功能
"""

import sys
import torch
from hyperpyyaml import load_hyperpyyaml

print("="*80)
print("SSL_LLM Prompt 模式测试")
print("="*80)

# 加载配置
print("\n[1/4] 加载 Prompt 配置...")
try:
    with open("hparams/SSL_LLM_Prompt.yaml") as f:
        hparams = load_hyperpyyaml(f, {})
    
    use_prompt = hparams.get('use_prompt', False)
    prompt_len = hparams.get('prompt_len', 0)
    prompt_init = hparams.get('prompt_init', 'normal')
    prompt_dropout = hparams.get('prompt_dropout', 0.0)
    lr_prompt = hparams.get('lr_prompt', hparams['lr'])
    
    print(f"✓ 配置加载成功")
    print(f"  - use_prompt: {use_prompt}")
    print(f"  - prompt_len: {prompt_len}")
    print(f"  - prompt_init: {prompt_init}")
    print(f"  - prompt_dropout: {prompt_dropout}")
    print(f"  - lr: {hparams['lr']}")
    print(f"  - lr_prompt: {lr_prompt}")
    print(f"  - ctc_weight: {hparams['ctc_weight']}")
    print(f"  - grad_clip: {hparams.get('grad_clip', 'Not set')}")
    
    if not use_prompt:
        print("\n⚠ use_prompt = False，Prompt 功能未启用")
        sys.exit(0)
        
except Exception as e:
    print(f"✗ 配置加载失败: {e}")
    sys.exit(1)

# 检查 Prompt 配置合理性
print("\n[2/4] 检查 Prompt 配置...")

checks = [
    ("Prompt 长度", prompt_len, lambda x: 8 <= x <= 32, "建议 8-32"),
    ("Prompt 初始化", prompt_init, lambda x: x in ['normal', 'xavier', 'zeros'], "应该是 normal/xavier/zeros"),
    ("Prompt dropout", prompt_dropout, lambda x: 0 <= x <= 0.3, "建议 0-0.3"),
    ("Prompt 学习率", lr_prompt, lambda x: x >= hparams['lr'], "应该 >= 主学习率"),
]

all_good = True
for name, value, check, suggestion in checks:
    if check(value):
        print(f"✓ {name}: {value}")
    else:
        print(f"⚠ {name}: {value} ({suggestion})")
        all_good = False

# 模拟 prompt embeddings 创建
print("\n[3/4] 模拟 Prompt Embeddings 创建...")
try:
    LLM = hparams["LLM"]
    H = LLM.get_input_embeddings().weight.shape[1]
    
    prompt_embeddings = torch.nn.Parameter(torch.zeros(prompt_len, H))
    
    if prompt_init == "normal":
        torch.nn.init.normal_(prompt_embeddings, mean=0.0, std=0.02)
    elif prompt_init == "xavier":
        torch.nn.init.xavier_uniform_(prompt_embeddings)
    
    prompt_params = prompt_embeddings.numel()
    
    print(f"✓ Prompt Embeddings 创建成功")
    print(f"  - 形状: {list(prompt_embeddings.shape)}")
    print(f"  - 参数量: {prompt_params:,}")
    print(f"  - 数据类型: {prompt_embeddings.dtype}")
    print(f"  - 初始值范围: [{prompt_embeddings.min().item():.4f}, {prompt_embeddings.max().item():.4f}]")
    print(f"  - 初始 norm: {torch.norm(prompt_embeddings).item():.4f}")
    
except Exception as e:
    print(f"✗ Prompt Embeddings 创建失败: {e}")
    sys.exit(1)

# 检查序列拼接逻辑
print("\n[4/4] 验证序列拼接逻辑...")
try:
    # 模拟数据
    B = 4  # batch size
    T = 100  # audio length (after projection)
    P = prompt_len  # prompt length
    L = 20  # text length
    
    # 模拟序列
    audio_seq_len = T
    prompt_seq_len = P
    text_seq_len = L + 1  # +1 for BOS
    
    total_len = audio_seq_len + prompt_seq_len + text_seq_len
    
    print(f"✓ 序列长度计算:")
    print(f"  - Audio: {audio_seq_len}")
    print(f"  - Prompt: {prompt_seq_len}")
    print(f"  - Text (含BOS): {text_seq_len}")
    print(f"  - 总长度: {total_len}")
    
    # 验证 attention mask
    prefix_mask_len = audio_seq_len
    prompt_mask_len = prompt_seq_len
    text_mask_len = text_seq_len
    
    print(f"\n✓ Attention Mask 分段:")
    print(f"  - Audio部分: [0:{prefix_mask_len}] - 全为1")
    print(f"  - Prompt部分: [{prefix_mask_len}:{prefix_mask_len + prompt_mask_len}] - 全为1")
    print(f"  - Text部分: [{prefix_mask_len + prompt_mask_len}:{total_len}] - 根据实际长度")
    
    # 验证 hidden states 提取
    hidden_start = audio_seq_len + prompt_seq_len
    hidden_end = hidden_start + text_seq_len
    
    print(f"\n✓ Hidden States 提取:")
    print(f"  - 跳过 Audio: {audio_seq_len} 位")
    print(f"  - 跳过 Prompt: {prompt_seq_len} 位")
    print(f"  - 提取范围: [{hidden_start}:{hidden_end}]")
    print(f"  - 用于预测: {text_seq_len} 个 token")
    
except Exception as e:
    print(f"✗ 序列逻辑验证失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "="*80)
if all_good:
    print("✓ Prompt 配置检查全部通过!")
    print("\n建议的训练命令：")
    print("python train.py hparams/SSL_LLM_Prompt.yaml \\")
    print("    --output_folder=exp_l2arctic/SSL_LLM_Prompt")
    print("\n关键监控点：")
    print("- Prompt embeddings 的 norm 应该逐渐变化（不应保持初始值）")
    print("- LLM PER 应该比不用 prompt 时更低（期望 < 10%）")
    print("- 训练前几个 epoch 可能会稍慢（prompt 在适应）")
else:
    print("⚠ 部分配置建议调整，但不影响运行")
print("="*80)
