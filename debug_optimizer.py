"""
调试 SSL_LLM 的优化器配置
检查哪些参数被优化，哪些被冻结
"""

import torch
from hyperpyyaml import load_hyperpyyaml
import sys

# 加载配置
hparams_file = "hparams/SSL_LLM.yaml"
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, {})

print("=" * 80)
print("检查 model 中的模块和参数")
print("=" * 80)

# 检查 model 包含的模块
model = hparams["model"]
print(f"\nmodel 类型: {type(model)}")
print(f"model 包含 {len(model)} 个模块:\n")

total_params = 0
trainable_params = 0
frozen_params = 0

for idx, module in enumerate(model):
    module_name = module.__class__.__name__
    module_params = sum(p.numel() for p in module.parameters())
    module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    module_frozen = module_params - module_trainable
    
    total_params += module_params
    trainable_params += module_trainable
    frozen_params += module_frozen
    
    print(f"{idx}. {module_name:30s} | 总参数: {module_params:>12,} | 可训练: {module_trainable:>12,} | 冻结: {module_frozen:>12,}")

print("\n" + "=" * 80)
print(f"{'总计':30s} | 总参数: {total_params:>12,} | 可训练: {trainable_params:>12,} | 冻结: {frozen_params:>12,}")
print("=" * 80)

# 检查关键模块
print("\n" + "=" * 80)
print("检查关键模块的参数")
print("=" * 80)

# phn_head 参数
phn_head = hparams["phn_head"]
print(f"\nphn_head 参数:")
for name, param in phn_head.named_parameters():
    print(f"  {name:40s} | shape: {str(param.shape):25s} | requires_grad: {param.requires_grad}")

# phn_embed 参数
phn_embed = hparams["phn_embed"]
print(f"\nphn_embed 参数:")
for name, param in phn_embed.named_parameters():
    print(f"  {name:40s} | shape: {str(param.shape):25s} | requires_grad: {param.requires_grad}")

# LLM 参数（只显示前10个和 LoRA 相关的）
LLM = hparams["LLM"]
print(f"\nLLM 参数 (前10个 + LoRA相关):")
count = 0
for name, param in LLM.named_parameters():
    if count < 10 or "lora" in name.lower():
        print(f"  {name:60s} | shape: {str(param.shape):25s} | requires_grad: {param.requires_grad}")
    count += 1

print(f"\n  ... (共 {count} 个参数)")

# 检查 optimizer 会收集哪些参数
print("\n" + "=" * 80)
print("检查 adam_optimizer 会优化哪些参数")
print("=" * 80)

# 模拟创建 optimizer
adam_opt_config = hparams["adam_opt_class"]
print(f"\nOptimizer 配置:")
print(f"  类型: {adam_opt_config.keywords.get('cls', 'Unknown')}")
print(f"  学习率: {adam_opt_config.keywords.get('lr', 'Unknown')}")

# 统计会被优化的参数
opt_params = [p for p in model.parameters() if p.requires_grad]
opt_param_count = sum(p.numel() for p in opt_params)

print(f"\nadam_optimizer 将优化:")
print(f"  参数组数: {len(opt_params)}")
print(f"  总参数量: {opt_param_count:,}")
print(f"  占总参数比例: {opt_param_count / total_params * 100:.2f}%")

# 按模块统计
print("\n按模块统计会被优化的参数:")
for idx, module in enumerate(model):
    module_name = module.__class__.__name__
    module_opt_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if module_opt_params > 0:
        print(f"  {module_name:30s}: {module_opt_params:>15,} 个参数")
