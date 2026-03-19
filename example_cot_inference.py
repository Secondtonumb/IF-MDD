#!/usr/bin/env python3
"""
Chain of Thought (CoT) 推理示例脚本

这个脚本展示如何使用新添加的 CoT 推理功能进行发音错误检测。

使用方法：
    python example_cot_inference.py \
        --checkpoint /path/to/checkpoint \
        --hparams_file /path/to/hparams.yaml \
        --test_csv /path/to/test.csv \
        --output_file results/cot_output.jsonl
"""

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.SSL_LLM_CoT import SSL_LLM_COT

def main():
    """主函数：运行 CoT 推理"""
    
    # ===== 配置参数 =====
    # 这些参数需要根据你的实际情况修改
    hparams_file = "hparams/SSL_LLM_CoT.yaml"  # 你的 hparams 文件路径
    checkpoint_path = "results/SSL_LLM_CoT/1234/save/CKPT+2024-01-01+00-00-00+00"  # checkpoint 路径
    test_csv = "data/test.csv"  # 测试集 CSV 文件
    output_file = "results/cot_inference_results.jsonl"  # 输出文件路径
    
    # CoT 推理参数
    max_new_tokens = 300  # CoT 需要更多的 tokens 来生成完整的推理过程
    do_sample = True      # 使用采样而非贪婪解码
    temperature = 0.7     # 采样温度（0.7 比较保守，可以调整）
    top_k = 50
    top_p = 0.9
    
    # ===== 加载配置 =====
    print("=" * 80)
    print("Loading hyperparameters...")
    print("=" * 80)
    
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    # ===== 准备数据集 =====
    print("\nPreparing test dataset...")
    from speechbrain.dataio.dataset import DynamicItemDataset
    from speechbrain.dataio.dataloader import make_dataloader
    
    # 这里假设你的 CSV 文件至少包含 ID 和音频路径
    # 根据你的实际数据格式调整
    test_data = DynamicItemDataset.from_csv(
        csv_path=test_csv,
        replacements={"data_root": hparams.get("data_folder", ".")}
    )
    
    # 添加音频加载管道
    test_data.add_dynamic_item(sb.dataio.dataset.audio_pipeline)
    test_data.set_output_keys(["id", "sig"])
    
    # ===== 初始化模型 =====
    print("\nInitializing SSL_LLM_COT model...")
    
    brain = SSL_LLM_COT(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams.get("checkpointer", None),
    )
    
    # 加载 checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        brain.checkpointer.recover_if_possible()
    
    # ===== 运行 CoT 推理 =====
    print("\n" + "=" * 80)
    print("Running Chain-of-Thought Inference")
    print("=" * 80)
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Do sample: {do_sample}")
    print(f"Output file: {output_file}")
    print("=" * 80 + "\n")
    
    results = brain.inference(
        test_set=test_data,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        output_file=output_file,
        use_cot=True,  # 🔥 关键：启用 CoT 模式
    )
    
    # ===== 输出示例结果 =====
    print("\n" + "=" * 80)
    print("Sample Results (first 3 utterances):")
    print("=" * 80)
    
    for i, batch_result in enumerate(results[:1]):  # 只显示第一个 batch
        for j in range(min(3, len(batch_result["ids"]))):
            print(f"\n--- Utterance {i*len(batch_result['ids'])+j+1} ---")
            print(f"ID: {batch_result['ids'][j]}")
            print(f"\nRaw CoT Output:\n{batch_result['generated_text'][j]}")
            
            if "parsed_results" in batch_result:
                parsed = batch_result["parsed_results"][j]
                print(f"\n📊 Parsed Structured Output:")
                print(f"  INTENT: {parsed.get('intent', 'N/A')}")
                print(f"  ERROR_REGION: {parsed.get('error_region', 'N/A')}")
                print(f"  PHONEME_REAL: {parsed.get('phoneme_real', 'N/A')}")
                print(f"  DECISION: {parsed.get('decision', 'N/A')}")
            
            if batch_result.get("ctc_predictions"):
                print(f"\n🔤 CTC Prediction: {batch_result['ctc_predictions'][j]}")
            
            print("-" * 80)
    
    print(f"\n✅ All results saved to: {output_file}")
    print(f"📁 LLM predictions saved to: {output_file.replace('.jsonl', '_LLM.csv')}")
    print(f"📁 Full details saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)


def compare_standard_vs_cot():
    """
    可选：对比标准推理和 CoT 推理的结果
    """
    print("\n" + "=" * 80)
    print("Comparing Standard vs CoT Inference")
    print("=" * 80)
    
    # 配置参数（同上）
    # ... [省略配置代码，与 main() 相同]
    
    # 1. 运行标准推理
    print("\n[1/2] Running standard inference...")
    # results_standard = brain.inference(
    #     test_set=test_data,
    #     use_cot=False,  # 标准模式
    #     output_file="results/standard_inference.jsonl"
    # )
    
    # 2. 运行 CoT 推理
    print("\n[2/2] Running CoT inference...")
    # results_cot = brain.inference(
    #     test_set=test_data,
    #     use_cot=True,  # CoT 模式
    #     max_new_tokens=300,
    #     output_file="results/cot_inference.jsonl"
    # )
    
    # 3. 对比分析
    print("\n📊 Comparison Summary:")
    print("  Standard: Direct phoneme prediction")
    print("  CoT: Structured reasoning with INTENT/ERROR_REGION/PHONEME_REAL/DECISION")
    print("\nPlease analyze the output files for detailed comparison.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain of Thought Inference")
    parser.add_argument("--mode", choices=["inference", "compare"], default="inference",
                        help="Mode: 'inference' (run CoT) or 'compare' (standard vs CoT)")
    args = parser.parse_args()
    
    if args.mode == "inference":
        main()
    elif args.mode == "compare":
        compare_standard_vs_cot()
