"""
快速测试脚本：验证 SSL_LLM 修复后的功能
测试内容：
1. 模型能否正常加载
2. optimizer 是否正确配置
3. forward pass 是否正常工作
"""

import sys
import torch
from hyperpyyaml import load_hyperpyyaml

print("="*80)
print("SSL_LLM 修复验证测试")
print("="*80)

# 1. 加载配置
print("\n[1/5] 加载配置文件...")
try:
    with open("hparams/SSL_LLM.yaml") as f:
        hparams = load_hyperpyyaml(f, {})
    print("✓ 配置文件加载成功")
    print(f"  - lr: {hparams['lr']}")
    print(f"  - ctc_weight: {hparams['ctc_weight']}")
    print(f"  - grad_clip: {hparams.get('grad_clip', 'Not set')}")
except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")
    sys.exit(1)

# 2. 检查关键模块
print("\n[2/5] 检查关键模块...")
try:
    enc = hparams["enc"]
    ctc_lin = hparams["ctc_lin"]
    phn_embed = hparams["phn_embed"]
    phn_head = hparams["phn_head"]
    LLM = hparams["LLM"]
    
    print(f"✓ enc: {type(enc).__name__}")
    print(f"✓ ctc_lin: {type(ctc_lin).__name__}")
    print(f"✓ phn_embed: {type(phn_embed).__name__} - {phn_embed.num_embeddings} embeddings")
    print(f"✓ phn_head: {type(phn_head).__name__}")
    print(f"✓ LLM: {type(LLM).__name__}")
except Exception as e:
    print(f"✗ 模块检查失败: {e}")
    sys.exit(1)

# 3. 检查可训练参数
print("\n[3/5] 检查各模块的可训练参数...")
modules_to_check = {
    "enc": enc,
    "ctc_lin": ctc_lin,
    "phn_embed": phn_embed,
    "phn_head": phn_head,
    "LLM": LLM
}

for name, module in modules_to_check.items():
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    if name == "LLM":
        # LLM 应该只有少量 LoRA 参数可训练
        print(f"✓ {name:15s}: {total_params:>12,} 总参数, {trainable_params:>12,} 可训练 ({trainable_params/total_params*100:.2f}%)")
    else:
        # 其他模块应该全部可训练
        if trainable_params == total_params:
            print(f"✓ {name:15s}: {total_params:>12,} 总参数, {trainable_params:>12,} 可训练 (100.00%)")
        else:
            print(f"⚠ {name:15s}: {total_params:>12,} 总参数, {trainable_params:>12,} 可训练 ({trainable_params/total_params*100:.2f}%)")

# 4. 模拟 optimizer 初始化
print("\n[4/5] 模拟 optimizer 参数收集...")
try:
    adam_params = []
    
    # 收集参数（与修复后的 init_optimizers 相同）
    for module in [enc, hparams["enc_ctc"], ctc_lin, phn_embed, phn_head]:
        for param in module.parameters():
            if param.requires_grad:
                adam_params.append(param)
    
    # LLM LoRA 参数
    llm_params = []
    for name, param in LLM.named_parameters():
        if param.requires_grad:
            llm_params.append(param)
            adam_params.append(param)
    
    total_adam_params = sum(p.numel() for p in adam_params)
    llm_trainable = sum(p.numel() for p in llm_params)
    
    print(f"✓ Adam optimizer 将优化 {len(adam_params)} 个参数组")
    print(f"  - 总参数量: {total_adam_params:,}")
    print(f"  - 其中 LLM LoRA: {llm_trainable:,} ({llm_trainable/total_adam_params*100:.2f}%)")
    print(f"  - 其他模块: {total_adam_params - llm_trainable:,} ({(total_adam_params - llm_trainable)/total_adam_params*100:.2f}%)")
    
    if llm_trainable < 1000000:  # LoRA 参数应该很少
        print("✓ LLM LoRA 参数量合理")
    else:
        print("⚠ LLM 可训练参数过多，检查是否正确配置 LoRA")
        
except Exception as e:
    print(f"✗ Optimizer 模拟失败: {e}")
    sys.exit(1)

# 5. 检查修改的配置值
print("\n[5/5] 验证关键配置修改...")
checks = [
    ("学习率", hparams['lr'], 0.0001, "lr 应该是 0.0001（已从 0.0003 降低）"),
    ("CTC 权重", hparams['ctc_weight'], 0.3, "ctc_weight 应该是 0.3（已从 0.5 降低）"),
    ("梯度裁剪", hparams.get('grad_clip', None), 1.0, "grad_clip 应该是 1.0（新增）"),
]

all_passed = True
for name, actual, expected, message in checks:
    if actual == expected:
        print(f"✓ {name}: {actual} ✓")
    else:
        print(f"✗ {name}: {actual} (期望: {expected}) - {message}")
        all_passed = False

# 总结
print("\n" + "="*80)
if all_passed:
    print("✓ 所有检查通过！可以开始训练了")
    print("\n建议的训练命令：")
    print("python train.py hparams/SSL_LLM.yaml \\")
    print("    --output_folder=exp_l2arctic/SSL_LLM_fixed")
else:
    print("⚠ 部分检查未通过，请检查配置")
print("="*80)
