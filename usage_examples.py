#!/usr/bin/env python3
"""
简单的使用示例：如何在训练中加载预训练模型的特定部分

这个脚本展示了三种使用方式：
1. 通过配置文件自动加载
2. 手动在代码中加载
3. 在训练过程中动态加载
"""

import os
import torch
from hyperpyyaml import load_hyperpyyaml
from models.Transformer import TransformerMDD

def example_1_config_based_loading():
    """示例1: 通过配置文件自动加载"""
    print("🔧 示例1: 配置文件方式加载预训练组件")
    
    # 修改配置文件中的设置
    hparams_file = "hparams/transformer.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    # 设置预训练组件加载
    hparams["load_pretrained_components"] = True
    hparams["pretrained_model_path"] = "/path/to/your/checkpoint/save/"
    hparams["components_to_load"] = ["ssl", "encoder"]  # 加载SSL和encoder
    hparams["freeze_loaded_components"] = True  # 冻结加载的组件
    
    # 创建模型时会自动加载
    model = TransformerMDD(
        modules=hparams["modules"],
        opt_class=hparams["adam_opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    
    # 查看参数状态
    model.print_parameter_status()
    return model

def example_2_manual_loading():
    """示例2: 手动在代码中加载"""
    print("🔧 示例2: 手动加载预训练组件")
    
    # 正常创建模型
    hparams_file = "hparams/transformer.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    model = TransformerMDD(
        modules=hparams["modules"],
        opt_class=hparams["adam_opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    
    # 手动加载特定组件
    checkpoint_path = "/path/to/your/checkpoint/save/"
    if os.path.exists(checkpoint_path):
        # 方法1: 使用简化接口
        model.load_from_checkpoint_manual(
            checkpoint_path=checkpoint_path,
            ssl_only=False,  # 不只加载SSL
            encoder_only=False,  # 不只加载encoder  
            freeze_ssl=True,  # 冻结SSL
            freeze_encoder=True  # 冻结encoder
        )
        
        # 方法2: 使用详细接口
        # model.load_pretrained_components(
        #     checkpoint_path=checkpoint_path,
        #     components_to_load=["ssl", "encoder"],
        #     freeze_loaded=True
        # )
    
    # 查看参数状态
    model.print_parameter_status()
    return model

def example_3_dynamic_loading():
    """示例3: 在训练过程中动态加载"""
    print("🔧 示例3: 训练过程中动态加载")
    
    # 创建模型
    hparams_file = "hparams/transformer.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    model = TransformerMDD(
        modules=hparams["modules"],
        opt_class=hparams["adam_opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    
    print("📊 训练前的参数状态:")
    model.print_parameter_status()
    
    # 假设训练了一段时间后...
    print("\n⏰ 假设训练了20个epoch后，现在要加载预训练的SSL模型...")
    
    checkpoint_path = "/path/to/your/checkpoint/save/"
    if os.path.exists(checkpoint_path):
        # 只加载SSL模型，冻结它
        model.load_pretrained_components(
            checkpoint_path=checkpoint_path,
            components_to_load=["ssl"],
            freeze_loaded=True
        )
        
        print("\n📊 加载SSL后的参数状态:")
        model.print_parameter_status()
        
        # 继续训练时，只有encoder和decoder会更新
        print("\n🎯 现在可以继续训练，SSL模型被冻结，只训练encoder和decoder")
    
    return model

def practical_usage_examples():
    """实际使用场景示例"""
    print("\n🎯 实际使用场景:")
    
    scenarios = [
        {
            "name": "场景1: 迁移学习",
            "description": "从一个数据集训练的模型迁移到新数据集",
            "components": ["ssl", "encoder"],
            "freeze": True,
            "note": "冻结特征提取部分，只训练分类头"
        },
        {
            "name": "场景2: 消融实验",
            "description": "测试不同encoder架构的效果",
            "components": ["ssl"],
            "freeze": True,
            "note": "固定SSL特征，比较不同encoder"
        },
        {
            "name": "场景3: 计算受限训练",
            "description": "在有限计算资源下训练",
            "components": ["ssl", "encoder"],
            "freeze": True,
            "note": "冻结大部分参数，只训练轻量级组件"
        },
        {
            "name": "场景4: 领域适应",
            "description": "将模型适应到新的语音领域",
            "components": ["ssl"],
            "freeze": False,
            "note": "微调所有组件，但从预训练SSL开始"
        },
        {
            "name": "场景5: 分阶段训练",
            "description": "先训练encoder，再训练decoder",
            "components": ["encoder", "ctc_head"],
            "freeze": True,
            "note": "固定已训练的encoder，专注训练decoder"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}: {scenario['description']}")
        print(f"   加载组件: {scenario['components']}")
        print(f"   是否冻结: {scenario['freeze']}")
        print(f"   说明: {scenario['note']}")

if __name__ == "__main__":
    print("🚀 预训练组件加载功能使用示例\n")
    
    # 显示实际使用场景
    practical_usage_examples()
    
    print("\n" + "="*60)
    print("📝 使用方法:")
    print("1. 修改transformer.yaml中的load_pretrained_components设置")
    print("2. 或者在代码中手动调用load_pretrained_components方法")
    print("3. 使用print_parameter_status()查看参数状态")
    print("4. 只有未冻结的参数会在训练中更新")
    
    # 如果要运行示例，取消下面的注释
    # print("\n🔧 运行示例1:")
    # example_1_config_based_loading()
    
    # print("\n🔧 运行示例2:")  
    # example_2_manual_loading()
    
    # print("\n🔧 运行示例3:")
    # example_3_dynamic_loading()
