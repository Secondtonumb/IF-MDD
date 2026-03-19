#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 CoT 监控功能
Demo CoT Monitoring Features
"""

import re


class CoTMonitor:
    """CoT 训练和推理监控工具"""
    
    def __init__(self, log_interval=50):
        self.step_count = 0
        self.log_interval = log_interval
    
    def extract_thinking(self, text):
        """提取 <think> 部分的内容"""
        match = re.search(r"<think>\s*(.+?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def extract_final_output(self, text):
        """提取 </think> 之后的最终输出"""
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[1].strip()
        return text.strip()
    
    def parse_thinking_steps(self, thinking_content):
        """解析 thinking 中的各个步骤"""
        result = {}
        
        # 提取 INTENT_WORD
        intent_match = re.search(r"INTENT[_\s]*WORD:\s*(.+?)(?=\n|$)", thinking_content, re.IGNORECASE)
        if intent_match:
            result["intent_word"] = intent_match.group(1).strip()
        
        # 提取 CANONICAL_PHONEME
        canonical_match = re.search(r"CANONICAL[_\s]*PHONEME:\s*(.+?)(?=\n|$)", thinking_content, re.IGNORECASE)
        if canonical_match:
            result["canonical_phoneme"] = canonical_match.group(1).strip()
        
        # 提取 ERROR_ANALYSIS
        error_match = re.search(r"ERROR[_\s]*ANALYSIS:\s*(.+?)(?=\n|$)", thinking_content, re.IGNORECASE)
        if error_match:
            result["error_analysis"] = error_match.group(1).strip()
        
        return result
    
    def analyze_errors(self, error_analysis):
        """分析错误标签统计"""
        if not error_analysis:
            return {}
        
        labels = error_analysis.split()
        stats = {
            "correct": labels.count("C"),
            "substitution": labels.count("S"),
            "deletion": labels.count("D"),
            "insertion": labels.count("I"),
            "total": len(labels)
        }
        
        error_count = stats["substitution"] + stats["deletion"] + stats["insertion"]
        stats["error_rate"] = error_count / stats["total"] if stats["total"] > 0 else 0
        
        return stats
    
    def monitor_training_step(self, sample_id, cot_target, force_log=False):
        """监控训练步骤"""
        self.step_count += 1
        
        if not force_log and self.step_count % self.log_interval != 0:
            return  # 不在日志间隔内，跳过
        
        print("=" * 80)
        print(f"[CoT Training Monitor] Step {self.step_count}")
        print("-" * 80)
        print(f"Sample ID: {sample_id}")
        print()
        
        # 提取 thinking 部分
        thinking = self.extract_thinking(cot_target)
        if thinking:
            print("🧠 Reasoning Process (<think>):")
            print(thinking)
            print()
            
            # 解析各个步骤
            steps = self.parse_thinking_steps(thinking)
            if steps:
                print("📊 Parsed Steps:")
                for key, value in steps.items():
                    print(f"  {key}: {value}")
                print()
                
                # 分析错误
                if "error_analysis" in steps:
                    error_stats = self.analyze_errors(steps["error_analysis"])
                    print("📈 Error Statistics:")
                    print(f"  Correct: {error_stats.get('correct', 0)}")
                    print(f"  Substitution: {error_stats.get('substitution', 0)}")
                    print(f"  Deletion: {error_stats.get('deletion', 0)}")
                    print(f"  Insertion: {error_stats.get('insertion', 0)}")
                    print(f"  Error Rate: {error_stats.get('error_rate', 0):.1%}")
                    print()
        
        # 提取最终输出
        final_output = self.extract_final_output(cot_target)
        print(f"📝 Expected Output: {final_output}")
        print("=" * 80)
        print()
    
    def monitor_inference(self, sample_id, generated_text):
        """监控推理输出"""
        print("=" * 80)
        print(f"[CoT Inference Monitor] ID: {sample_id}")
        print("-" * 80)
        
        # 显示完整输出
        print("Full Generated Output:")
        print(generated_text)
        print("-" * 80)
        
        # 提取并分析 thinking
        thinking = self.extract_thinking(generated_text)
        if thinking:
            print("🧠 Reasoning Process (<think>):")
            print(thinking)
            print("-" * 80)
            
            # 解析步骤
            steps = self.parse_thinking_steps(thinking)
            if steps:
                print("📊 Parsed Steps:")
                for key, value in steps.items():
                    print(f"  {key}: {value}")
                
                # 分析错误
                if "error_analysis" in steps:
                    error_stats = self.analyze_errors(steps["error_analysis"])
                    print()
                    print("📈 Error Statistics:")
                    print(f"  Correct: {error_stats.get('correct', 0)}")
                    print(f"  Substitution: {error_stats.get('substitution', 0)}")
                    print(f"  Deletion: {error_stats.get('deletion', 0)}")
                    print(f"  Insertion: {error_stats.get('insertion', 0)}")
                    print(f"  Error Rate: {error_stats.get('error_rate', 0):.1%}")
            
            print("-" * 80)
        else:
            print("⚠️ Warning: No <think> tags found in output")
            print("-" * 80)
        
        # 提取最终输出
        final_output = self.extract_final_output(generated_text)
        print(f"📝 Final Output: {final_output}")
        print("=" * 80)
        print()


def demo_training_monitoring():
    """演示训练时的监控"""
    print("\n" + "=" * 80)
    print("Demo 1: Training Monitoring")
    print("=" * 80 + "\n")
    
    monitor = CoTMonitor(log_interval=1)  # 每步都记录（演示用）
    
    # 模拟训练样本
    samples = [
        {
            "id": "sample_001.wav",
            "cot_target": """<think>
Step 1: Identify the intended word
INTENT_WORD: كِتَابٌ

Step 2: Get canonical phoneme sequence
CANONICAL_PHONEME: k i t aa b u n

Step 3: Analyze each phoneme for errors
ERROR_ANALYSIS: C C C C C C C
</think>

k i t aa b u n"""
        },
        {
            "id": "sample_002.wav",
            "cot_target": """<think>
Step 1: Identify the intended word
INTENT_WORD: بَيْتٌ

Step 2: Get canonical phoneme sequence
CANONICAL_PHONEME: b a y t u n

Step 3: Analyze each phoneme for errors
ERROR_ANALYSIS: C C C S C C
</think>

b a y d u n"""
        }
    ]
    
    for sample in samples:
        monitor.monitor_training_step(sample["id"], sample["cot_target"], force_log=True)


def demo_inference_monitoring():
    """演示推理时的监控"""
    print("\n" + "=" * 80)
    print("Demo 2: Inference Monitoring")
    print("=" * 80 + "\n")
    
    monitor = CoTMonitor()
    
    # 模拟推理输出
    generated_text = """<think>
Step 1: Identify the intended word
INTENT_WORD: مَدْرَسَةٌ

Step 2: Get canonical phoneme sequence
CANONICAL_PHONEME: m a d r a s a t u n

Step 3: Analyze each phoneme for errors
ERROR_ANALYSIS: C C S C C S C C C C
</think>

m a t r a z a t u n"""
    
    monitor.monitor_inference("test_003.wav", generated_text)


def demo_error_analysis():
    """演示错误分析功能"""
    print("\n" + "=" * 80)
    print("Demo 3: Error Analysis")
    print("=" * 80 + "\n")
    
    monitor = CoTMonitor()
    
    test_cases = [
        ("All Correct", "C C C C C C C"),
        ("Single Substitution", "C C C S C C"),
        ("Multiple Errors", "C C S C C S C C C C"),
        ("Mixed Errors", "C S D I C C")
    ]
    
    for name, error_labels in test_cases:
        stats = monitor.analyze_errors(error_labels)
        print(f"Case: {name}")
        print(f"  Labels: {error_labels}")
        print(f"  Stats: {stats}")
        print()


if __name__ == "__main__":
    print("\n" + "🔍 CoT Monitoring Demo" + "\n")
    
    demo_training_monitoring()
    demo_inference_monitoring()
    demo_error_analysis()
    
    print("\n" + "=" * 80)
    print("💡 Usage in Your Model:")
    print("=" * 80)
    print("""
1. Training Monitoring:
   - Automatically logs every 50 steps (configurable)
   - Shows thinking process and expected output
   - Provides error statistics
   
2. Inference Monitoring:
   - Shows full generated output
   - Extracts and displays <think> content
   - Parses structured fields
   
3. Configuration in hparams.yaml:
   use_cot_training: True
   cot_training_prob: 0.3
   cot_monitor_interval: 50  # Log every 50 steps
   
4. Inference with monitoring:
   brain.inference(test_set, use_cot=True)
""")
    print("=" * 80 + "\n")
