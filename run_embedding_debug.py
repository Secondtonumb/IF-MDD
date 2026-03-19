#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速调试脚本：在单个样本上运行完整的 embedding 诊断
使用方法: python run_embedding_debug.py
"""

import torch
import sys
import os
sys.path.append("/work/gm64/m64000/IF-MDD")

from debug_input_embedding import InputEmbeddingDebugger
import speechbrain as sb


def phn_list_to_seq(batch):
    """
    Args:
        batch:[["sil", "aa", "x"], ["sil", "xa", "th"]]
    return
        batch ["sil aa x", "sil xa th"]
    """
    result = []
    for phn_list in batch:
        result.append(" ".join(x for x in phn_list))
    return result

def run_debug_on_single_sample(
    checkpoint_path,
    hparams_file,
    test_data_csv,
    sample_index=0
):
    """
    在单个样本上运行完整的 embedding 诊断
    
    Args:
        checkpoint_path: 模型checkpoint路径
        hparams_file: hparams YAML文件
        test_data_csv: 测试数据CSV
        sample_index: 要调试的样本索引（默认0）
    """
    from hyperpyyaml import load_hyperpyyaml
    from models.SSL_LLM_MultiTarget_ver1 import SSL_LLM_MultiTarget_ver1
    
    print(f"\n{'#'*100}")
    print(f"# 加载模型和数据")
    print(f"{'#'*100}\n")
    
    # 加载 hparams
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    # 加载 model
    brain = SSL_LLM_MultiTarget_ver1(
        modules=hparams["modules"],
        opt_class=hparams["adam_opt_class"],
        hparams=hparams,
        run_opts={"device": "cuda"}
    )
    
    # 加载 checkpoint
    print(f"[INFO] 加载 checkpoint: {checkpoint_path}")
    brain.checkpointer.recover_if_possible()
    brain.modules.eval()
    
    # 确保初始化
    brain._ensure_initialized()
    
    # 创建调试器
    debugger = InputEmbeddingDebugger(
        brain=brain,
        tokenizer=hparams["LLM_tokenizer"],
        output_dir="./debug_embedding_outputs"
    )
    
    # 加载测试数据
    from speechbrain.dataio.dataset import DynamicItemDataset
    test_data = DynamicItemDataset.from_csv(csv_path=test_data_csv)
    
    # 获取指定样本
    sample_items = list(test_data.data.items())
    if sample_index >= len(sample_items):
        print(f"[ERROR] 样本索引 {sample_index} 超出范围 (max: {len(sample_items)-1})")
        return
    
    sample_id, sample = sample_items[sample_index]
    
    print(f"\n{'#'*100}")
    print(f"# 调试样本: {sample_id} (索引: {sample_index})")
    print(f"{'#'*100}\n")
    
    # 构造 batch（单样本）
    batch = hparams["valid_dataloader"].collate_fn([sample])
    batch = batch.to("cuda")
    
    # =========================================================================
    # Step 1-3: 基础诊断
    # =========================================================================
    debugger.full_diagnosis(batch, sb.Stage.TEST)
    
    # =========================================================================
    # Step 4: Compact IDs 构建
    # =========================================================================
    with torch.no_grad():
        tok = hparams["LLM_tokenizer"]
        device = brain.device
        
        # 获取训练目标
        if getattr(brain.hparams, "training_target") == "target":
            phn_list_training_target = batch.phn_list_target
        else:
            phn_list_training_target = batch.phn_list_target
        
        phn_tgt_seq = phn_list_to_seq(phn_list_training_target)
        phn_tgt_tokens = tok(phn_tgt_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        phn_can_seq = phn_list_to_seq(batch.phn_list_canonical)
        phn_can_tokens = tok(phn_can_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        wrd_tokens = tok(batch.wrd, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        newline_tokens = tok("\n", return_tensors="pt", add_special_tokens=False).to(device)
        SEP_TGT_ID = newline_tokens["input_ids"][0, 0].item()
        EOS_ID = tok.eos_token_id
        
        debugger.check_compact_ids_construction(
            wrd_ids=wrd_tokens["input_ids"],
            wrd_mask=wrd_tokens["attention_mask"],
            phn_can_ids=phn_can_tokens["input_ids"],
            phn_can_mask=phn_can_tokens["attention_mask"],
            phn_tgt_ids=phn_tgt_tokens["input_ids"],
            phn_tgt_mask=phn_tgt_tokens["attention_mask"],
            SEP_TGT_ID=SEP_TGT_ID,
            EOS_ID=EOS_ID
        )
    
    # =========================================================================
    # Step 5-6: 完整 forward + 生成
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"[STEP] 运行完整 forward pass (TEST stage)")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        predictions = brain.compute_forward(batch, sb.Stage.TEST)
        
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, wav_lens = predictions
            
            if isinstance(ce_targets, dict):
                # 检查生成输出
                if "generated_ids" in ce_targets:
                    debugger.check_generated_output(ce_targets["generated_ids"])
    
    print(f"\n{'#'*100}")
    print(f"# 调试完成！结果保存在: ./debug_embedding_outputs/")
    print(f"# 查看详细信息: ls -lh debug_embedding_outputs/")
    print(f"{'#'*100}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Input Embedding 调试工具")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--hparams", type=str, required=True,
                        help="hparams YAML 文件路径")
    parser.add_argument("--data", type=str, required=True,
                        help="测试数据 CSV 文件路径")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="要调试的样本索引（默认: 0）")
    
    args = parser.parse_args()
    
    # 示例调用
    run_debug_on_single_sample(
        checkpoint_path=args.checkpoint,
        hparams_file=args.hparams,
        test_data_csv=args.data,
        sample_index=args.sample_idx
    )
