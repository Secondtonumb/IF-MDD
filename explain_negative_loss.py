#!/usr/bin/env python3
"""
演示为什么 CTC + Label Priors 会产生负数 Loss

这个脚本通过具体的数值例子，展示：
1. 传统 CTC Loss 的计算（总是非负）
2. 加入 Label Priors 后的调整
3. 为什么调整后的 Loss 可能是负数
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

def demonstrate_negative_loss():
    """
    演示 Label Priors 如何导致负数 loss
    """
    
    print("=" * 80)
    print("🔍 演示：为什么 CTC + Label Priors 会产生负数 Loss")
    print("=" * 80)
    
    # ========== 场景设置 ==========
    print("\n【场景设置】")
    print("-" * 80)
    
    # 假设一个简单的例子：识别单词 "ah"
    # CTC 路径: blank-a-a-h-h-blank
    
    num_frames = 6
    blank_idx = 0
    a_idx = 1
    h_idx = 2
    
    print(f"任务：识别音素序列 'ah'")
    print(f"CTC 路径：blank-a-a-h-h-blank (共 {num_frames} 帧)")
    print(f"Token 映射：blank=0, 'a'=1, 'h'=2")
    
    # ========== 情况 1：传统 CTC（无 priors）==========
    print("\n" + "=" * 80)
    print("📊 情况 1：传统 CTC Loss（无 Label Priors）")
    print("=" * 80)
    
    # 模型输出的概率（未调整）
    # 注意：blank 的概率很高（这是 CTC 的常见问题）
    probs_original = torch.tensor([
        [0.7, 0.2, 0.1],   # Frame 0: 倾向于 blank
        [0.3, 0.6, 0.1],   # Frame 1: 倾向于 a
        [0.3, 0.6, 0.1],   # Frame 2: 倾向于 a
        [0.4, 0.1, 0.5],   # Frame 3: 倾向于 h
        [0.4, 0.1, 0.5],   # Frame 4: 倾向于 h
        [0.7, 0.2, 0.1],   # Frame 5: 倾向于 blank
    ])
    
    print("\n模型输出概率（原始）：")
    print("  Frame | P(blank) | P(a)  | P(h)")
    print("  ------|----------|-------|------")
    for i, prob in enumerate(probs_original):
        print(f"    {i}   |  {prob[0]:.3f}   | {prob[1]:.3f} | {prob[2]:.3f}")
    
    # 计算路径概率（简化：假设只有一条最优路径）
    path_prob_original = probs_original[0, 0] * probs_original[1, 1] * \
                        probs_original[2, 1] * probs_original[3, 2] * \
                        probs_original[4, 2] * probs_original[5, 0]
    
    log_prob_original = torch.log(path_prob_original)
    loss_original = -log_prob_original
    
    print(f"\n路径概率：P(path) = {path_prob_original:.6e}")
    print(f"Log 概率：log P(path) = {log_prob_original:.4f}")
    print(f"CTC Loss：-log P(path) = {loss_original:.4f}")
    print(f"\n✅ 传统 CTC Loss 总是 ≥ 0（这里是 {loss_original:.4f}）")
    
    # ========== 情况 2：使用 Label Priors ==========
    print("\n" + "=" * 80)
    print("📊 情况 2：CTC + Label Priors")
    print("=" * 80)
    
    # 从训练数据统计的 priors
    # blank 的 prior 很高（70%），因为 CTC 路径中大量是 blank
    priors = torch.tensor([0.7, 0.2, 0.1])  # [blank, a, h]
    alpha = 0.6  # prior_scaling_factor
    
    print(f"\nLabel Priors（从训练数据统计）：")
    print(f"  P(blank) = {priors[0]:.3f} (70% 的帧是 blank)")
    print(f"  P(a)     = {priors[1]:.3f}")
    print(f"  P(h)     = {priors[2]:.3f}")
    print(f"\nPrior Scaling Factor (α) = {alpha}")
    
    # 计算调整因子
    adjustment_factors = priors ** alpha
    
    print(f"\n调整因子：prior^α = prior^{alpha}")
    print(f"  blank: {priors[0]:.3f}^{alpha} = {adjustment_factors[0]:.4f}")
    print(f"  a:     {priors[1]:.3f}^{alpha} = {adjustment_factors[1]:.4f}")
    print(f"  h:     {priors[2]:.3f}^{alpha} = {adjustment_factors[2]:.4f}")
    
    # 应用 Label Priors 调整
    # log P(c|x) - α * log prior(c) = log[P(c|x) / prior(c)^α]
    log_probs_original = torch.log(probs_original)
    log_priors = torch.log(priors)
    log_probs_adjusted = log_probs_original - alpha * log_priors
    
    # 转换回概率空间（注意：这些"概率"不再满足归一化）
    probs_adjusted = torch.exp(log_probs_adjusted)
    
    print("\n调整后的概率（log space 减法 → 概率空间除法）：")
    print("  Frame | P(blank) | P(a)  | P(h)  | Sum")
    print("  ------|----------|-------|-------|------")
    for i, prob in enumerate(probs_adjusted):
        print(f"    {i}   |  {prob[0]:.3f}   | {prob[1]:.3f} | {prob[2]:.3f} | {prob.sum():.3f}")
    
    print("\n⚠️  注意：调整后每行的和 ≠ 1（不再是归一化的概率分布）")
    
    # 计算调整后的路径概率
    path_prob_adjusted = probs_adjusted[0, 0] * probs_adjusted[1, 1] * \
                        probs_adjusted[2, 1] * probs_adjusted[3, 2] * \
                        probs_adjusted[4, 2] * probs_adjusted[5, 0]
    
    log_prob_adjusted = torch.log(path_prob_adjusted)
    loss_adjusted = -log_prob_adjusted
    
    print(f"\n调整后的路径概率：P'(path) = {path_prob_adjusted:.6e}")
    print(f"Log 概率：log P'(path) = {log_prob_adjusted:.4f}")
    print(f"调整后的 Loss：-log P'(path) = {loss_adjusted:.4f}")
    
    # ========== 对比与解释 ==========
    print("\n" + "=" * 80)
    print("🎯 关键发现")
    print("=" * 80)
    
    print(f"\n原始路径概率：{path_prob_original:.6e}")
    print(f"调整后概率：  {path_prob_adjusted:.6e}")
    print(f"比值：        {path_prob_adjusted / path_prob_original:.2f}x")
    
    if path_prob_adjusted > 1.0:
        print(f"\n⚠️  调整后的概率 > 1！({path_prob_adjusted:.6e})")
        print(f"    这是因为我们'压低'了高频 token (blank) 的影响")
    
    print(f"\n原始 CTC Loss：   {loss_original:.4f}")
    print(f"调整后的 Loss：   {loss_adjusted:.4f}")
    print(f"变化：           {loss_adjusted - loss_original:+.4f}")
    
    if loss_adjusted < 0:
        print(f"\n✅ Loss 变成了负数！({loss_adjusted:.4f})")
        print(f"   这是正常的，因为调整后的概率 > 1")
        print(f"   log(>1) > 0，加负号后变成负数")
    else:
        print(f"\n⚠️  这个例子中 Loss 还是正数 ({loss_adjusted:.4f})")
        print(f"   但在实际训练中（序列更长、调整更明显），Loss 经常变负")
    
    # ========== 可视化 ==========
    print("\n" + "=" * 80)
    print("📈 生成可视化图表...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CTC + Label Priors: 为什么 Loss 会变成负数？', fontsize=16, fontweight='bold')
    
    # 图 1: 原始概率分布
    ax1 = axes[0, 0]
    frames = np.arange(num_frames)
    width = 0.25
    
    ax1.bar(frames - width, probs_original[:, 0], width, label='P(blank)', color='red', alpha=0.7)
    ax1.bar(frames, probs_original[:, 1], width, label="P(a)", color='steelblue', alpha=0.7)
    ax1.bar(frames + width, probs_original[:, 2], width, label="P(h)", color='green', alpha=0.7)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Probability')
    ax1.set_title('(1) 原始模型输出概率')
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # 图 2: 调整后的概率分布
    ax2 = axes[0, 1]
    
    ax2.bar(frames - width, probs_adjusted[:, 0], width, label='P(blank)', color='red', alpha=0.7)
    ax2.bar(frames, probs_adjusted[:, 1], width, label="P(a)", color='steelblue', alpha=0.7)
    ax2.bar(frames + width, probs_adjusted[:, 2], width, label="P(h)", color='green', alpha=0.7)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Adjusted Probability')
    ax2.set_title('(2) 调整后的概率（P / prior^α）')
    ax2.legend()
    max_prob = probs_adjusted.max().item()
    ax2.set_ylim([0, max(1.2, max_prob * 1.1)])
    ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='y=1 (归一化边界)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    if probs_adjusted.max() > 1:
        ax2.text(0.5, 0.95, '⚠️ 某些概率 > 1', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 图 3: Priors 的影响
    ax3 = axes[1, 0]
    
    token_names = ['blank', 'a', 'h']
    x_pos = np.arange(len(token_names))
    
    bars1 = ax3.bar(x_pos - 0.2, priors.numpy(), 0.4, label='Prior', color='orange', alpha=0.7)
    bars2 = ax3.bar(x_pos + 0.2, adjustment_factors.numpy(), 0.4, 
                    label=f'prior^{alpha}', color='purple', alpha=0.7)
    
    ax3.set_xlabel('Token')
    ax3.set_ylabel('Value')
    ax3.set_title(f'(3) Prior 值与调整因子 (α={alpha})')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(token_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 图 4: Loss 对比
    ax4 = axes[1, 1]
    
    loss_values = [loss_original.item(), loss_adjusted.item()]
    loss_labels = ['原始 CTC\nLoss', '调整后\nLoss']
    colors = ['green' if l >= 0 else 'red' for l in loss_values]
    
    bars = ax4.bar(loss_labels, loss_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax4.set_ylabel('Loss Value')
    ax4.set_title('(4) Loss 对比')
    ax4.axhline(y=0, color='black', linewidth=2, linestyle='-')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, loss_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', 
                va='bottom' if val >= 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    # 添加解释文本
    y_min, y_max = ax4.get_ylim()
    if loss_adjusted < 0:
        ax4.text(0.5, 0.05, '✅ 负数 Loss 是正常的！\n这是 Label Priors 的预期效果', 
                transform=ax4.transAxes, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    output_path = 'negative_loss_explanation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存可视化图表到: {output_path}")
    
    # ========== 更极端的例子 ==========
    print("\n" + "=" * 80)
    print("🔬 更极端的例子：长序列 + 高 α 值")
    print("=" * 80)
    
    # 假设一个 100 帧的序列，70% 是 blank
    num_frames_long = 100
    num_blanks = 70
    num_phonemes = 30
    
    # 原始路径概率（假设每帧概率是 0.6）
    frame_prob = 0.6
    path_prob_long_original = frame_prob ** num_frames_long
    
    # 调整因子
    blank_prior = 0.7
    phoneme_prior = 0.01
    alpha_high = 0.8
    
    blank_adj = (blank_prior ** alpha_high) ** num_blanks
    phoneme_adj = (phoneme_prior ** alpha_high) ** num_phonemes
    
    path_prob_long_adjusted = path_prob_long_original / (blank_adj * phoneme_adj)
    
    loss_long_original = -np.log(path_prob_long_original)
    loss_long_adjusted = -np.log(path_prob_long_adjusted)
    
    print(f"\n序列长度：{num_frames_long} 帧")
    print(f"  - {num_blanks} 帧 blank (prior={blank_prior})")
    print(f"  - {num_phonemes} 帧 phonemes (prior={phoneme_prior})")
    print(f"Prior Scaling Factor：α = {alpha_high}")
    
    print(f"\n原始路径概率：{path_prob_long_original:.6e}")
    print(f"调整后概率：  {path_prob_long_adjusted:.6e}")
    
    print(f"\n原始 Loss：   {loss_long_original:.2f}")
    print(f"调整后 Loss： {loss_long_adjusted:.2f}")
    
    if loss_long_adjusted < 0:
        print(f"\n🎯 在这个例子中，Loss 变成了 {loss_long_adjusted:.2f}（很大的负数）")
        print(f"   这在实际训练中很常见！")
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("💡 总结：负数 Loss 的本质")
    print("=" * 80)
    
    print("""
1. **传统 CTC Loss**：
   - Loss = -log P(y|x)
   - 因为 P(y|x) ≤ 1，所以 log P(y|x) ≤ 0
   - 因此 Loss ≥ 0（总是非负）

2. **CTC + Label Priors**：
   - 调整：P'(c|x) = P(c|x) / prior(c)^α
   - 当 prior(c) < 1 且 α > 0 时，prior(c)^α < 1
   - 除以一个 < 1 的数，结果变大
   - 累积效果可能导致 P'(y|x) > 1

3. **为什么 P' > 1 是合理的**：
   - 这不是真实的概率，而是"调整后的分数"
   - 目的是重新平衡 token 的相对重要性
   - 压低高频 token (blank) 的影响

4. **负数 Loss 的意义**：
   - P'(y|x) > 1 ⇒ log P'(y|x) > 0
   - Loss = -log P'(y|x) < 0
   - **这是正常的，不是 bug！**

5. **关注什么**：
   - ❌ 不要关注 Loss 是正是负
   - ✅ 关注 Loss 是否持续下降
   - ✅ 关注 Validation 性能是否提升
   - ✅ 关注 Blank Prior 是否从 0.7 降到 0.3-0.4
   - ✅ 关注对齐质量是否改善
    """)
    
    print("=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print(f"\n📊 可视化图表已保存到: {output_path}")
    print("📚 详细文档请参考: docs/why_negative_loss.md")
    print("\n")

if __name__ == '__main__':
    demonstrate_negative_loss()
