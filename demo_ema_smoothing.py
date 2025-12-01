#!/usr/bin/env python3
"""
演示 EMA (指数移动平均) 如何稳定 Blank Prior 更新

对比两种策略：
1. 无平滑：每个 epoch 直接使用当前 priors（剧烈波动）
2. EMA 平滑：使用指数移动平均（稳定变化）
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def simulate_training_with_fluctuation():
    """模拟训练中的 blank prior 波动"""
    
    print("=" * 80)
    print("🔬 模拟：Blank Prior 波动问题与 EMA 解决方案")
    print("=" * 80)
    
    # 模拟 20 个 epoch 的 blank prior（当前 epoch 的值）
    # 这些值模拟实际训练中的剧烈波动
    np.random.seed(42)
    
    # 基础趋势：从 0.7 逐渐降到 0.35
    epochs = np.arange(1, 21)
    base_trend = 0.7 - 0.35 * (epochs - 1) / 19
    
    # 添加随机波动（模拟不稳定训练）
    noise_magnitude = 0.15  # 15% 的随机波动
    random_noise = np.random.randn(20) * noise_magnitude
    
    # 添加几个异常值（模拟训练崩溃）
    current_blank_priors = base_trend + random_noise
    current_blank_priors[3] = 0.02   # Epoch 4: 崩溃到 2%
    current_blank_priors[7] = 0.85   # Epoch 8: 突然跳到 85%
    current_blank_priors[12] = 0.05  # Epoch 13: 又崩溃到 5%
    
    # 确保在 [0.01, 0.95] 范围内
    current_blank_priors = np.clip(current_blank_priors, 0.01, 0.95)
    
    print(f"\n📊 模拟的当前 Epoch Blank Priors (有剧烈波动):")
    for i, prior in enumerate(current_blank_priors[:10], 1):
        print(f"  Epoch {i:2d}: {prior:.4f} ({prior*100:5.1f}%)")
    print("  ...")
    
    # ========== 策略 1：无平滑（直接使用） ==========
    no_smoothing_priors = current_blank_priors.copy()
    
    # 计算波动幅度
    changes_no_smoothing = np.abs(np.diff(no_smoothing_priors))
    max_change_no_smoothing = np.max(changes_no_smoothing)
    avg_change_no_smoothing = np.mean(changes_no_smoothing)
    
    print(f"\n❌ 策略 1：无平滑（直接使用当前 epoch priors）")
    print(f"   最大变化: {max_change_no_smoothing:.4f} ({max_change_no_smoothing*100:.1f}%)")
    print(f"   平均变化: {avg_change_no_smoothing:.4f} ({avg_change_no_smoothing*100:.1f}%)")
    print(f"   异常 epoch:")
    for i, (curr, next_val) in enumerate(zip(no_smoothing_priors[:-1], no_smoothing_priors[1:]), 1):
        change = abs(next_val - curr)
        if change > 0.2:  # 变化超过 20%
            print(f"     Epoch {i} → {i+1}: {curr:.4f} → {next_val:.4f} (变化 {change:.4f} = {change/curr*100:+.1f}%)")
    
    # ========== 策略 2：EMA 平滑 ==========
    def apply_ema(values, momentum=0.9):
        """应用指数移动平均"""
        ema_values = np.zeros_like(values)
        ema_values[0] = values[0]  # 初始化
        
        for i in range(1, len(values)):
            ema_values[i] = momentum * ema_values[i-1] + (1 - momentum) * values[i]
        
        return ema_values
    
    # 测试不同的 momentum 值
    momentums = [0.7, 0.8, 0.9, 0.95]
    ema_results = {}
    
    print(f"\n✅ 策略 2：EMA 平滑（不同 momentum 值）")
    
    for momentum in momentums:
        ema_priors = apply_ema(current_blank_priors, momentum)
        changes_ema = np.abs(np.diff(ema_priors))
        max_change_ema = np.max(changes_ema)
        avg_change_ema = np.mean(changes_ema)
        
        ema_results[momentum] = {
            'priors': ema_priors,
            'max_change': max_change_ema,
            'avg_change': avg_change_ema
        }
        
        print(f"\n   Momentum = {momentum}:")
        print(f"     最大变化: {max_change_ema:.4f} ({max_change_ema*100:.1f}%) "
              f"[降低 {(1 - max_change_ema/max_change_no_smoothing)*100:.1f}%]")
        print(f"     平均变化: {avg_change_ema:.4f} ({avg_change_ema*100:.1f}%) "
              f"[降低 {(1 - avg_change_ema/avg_change_no_smoothing)*100:.1f}%]")
    
    # ========== 可视化对比 ==========
    print(f"\n📈 生成对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Blank Prior 波动问题：EMA 解决方案', fontsize=16, fontweight='bold')
    
    # 图 1: 原始波动 vs EMA 平滑
    ax1 = axes[0, 0]
    ax1.plot(epochs, current_blank_priors, 'o--', label='Current Epoch (无平滑)', 
             color='red', linewidth=2, markersize=8, alpha=0.6)
    ax1.plot(epochs, ema_results[0.9]['priors'], 's-', label='EMA (momentum=0.9)', 
             color='green', linewidth=2.5, markersize=6)
    
    # 标注异常点
    anomalies = [3, 7, 12]  # 索引（从 0 开始）
    for idx in anomalies:
        ax1.annotate(f'异常!', 
                    xy=(epochs[idx], current_blank_priors[idx]),
                    xytext=(epochs[idx], current_blank_priors[idx] + 0.15),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
    
    ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='目标区间 [0.3, 0.4]')
    ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
    ax1.fill_between(epochs, 0.3, 0.4, alpha=0.1, color='green')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Blank Prior')
    ax1.set_title('(1) 原始波动 vs EMA 平滑')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 图 2: 不同 momentum 的效果
    ax2 = axes[0, 1]
    ax2.plot(epochs, current_blank_priors, 'o:', label='Current (无平滑)', 
             color='gray', linewidth=1, markersize=4, alpha=0.5)
    
    colors = ['blue', 'green', 'purple', 'brown']
    for momentum, color in zip(momentums, colors):
        ax2.plot(epochs, ema_results[momentum]['priors'], 
                marker='s', label=f'EMA (m={momentum})', 
                color=color, linewidth=2, markersize=5, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Blank Prior')
    ax2.set_title('(2) 不同 Momentum 值的效果')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 图 3: 逐 epoch 变化幅度
    ax3 = axes[1, 0]
    
    changes_no_smooth = np.abs(np.diff(no_smoothing_priors))
    ax3.plot(epochs[1:], changes_no_smooth, 'o-', label='无平滑', 
             color='red', linewidth=2, markersize=8, alpha=0.7)
    
    for momentum, color in zip([0.8, 0.9, 0.95], ['blue', 'green', 'purple']):
        changes = np.abs(np.diff(ema_results[momentum]['priors']))
        ax3.plot(epochs[1:], changes, marker='s', label=f'EMA (m={momentum})', 
                color=color, linewidth=2, markersize=5, alpha=0.8)
    
    ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='安全阈值 (5%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('|Prior(t) - Prior(t-1)|')
    ax3.set_title('(3) 逐 Epoch 变化幅度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 图 4: 统计对比
    ax4 = axes[1, 1]
    
    methods = ['无平滑'] + [f'EMA\n(m={m})' for m in momentums]
    max_changes = [max_change_no_smoothing] + [ema_results[m]['max_change'] for m in momentums]
    avg_changes = [avg_change_no_smoothing] + [ema_results[m]['avg_change'] for m in momentums]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, max_changes, width, label='最大变化', 
                    color='red', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, avg_changes, width, label='平均变化', 
                    color='blue', alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('变化幅度')
    ax4.set_title('(4) 统计对比')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = 'ema_smoothing_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存对比图表到: {output_path}")
    
    # ========== 详细分析 ==========
    print(f"\n" + "=" * 80)
    print(f"📊 详细分析：EMA 的效果")
    print(f"=" * 80)
    
    print(f"\n【波动幅度对比】")
    print(f"{'策略':<20s} | {'最大变化':<12s} | {'平均变化':<12s} | {'改善率':<10s}")
    print(f"-" * 65)
    print(f"{'无平滑 (baseline)':<20s} | {max_change_no_smoothing:>10.4f}   | {avg_change_no_smoothing:>10.4f}   | {'N/A':<10s}")
    
    for momentum in momentums:
        max_change = ema_results[momentum]['max_change']
        avg_change = ema_results[momentum]['avg_change']
        improvement = (1 - max_change / max_change_no_smoothing) * 100
        print(f"{'EMA (m=' + str(momentum) + ')':<20s} | {max_change:>10.4f}   | {avg_change:>10.4f}   | {improvement:>9.1f}%")
    
    print(f"\n【推荐 Momentum 值】")
    print(f"""
- **momentum = 0.9** (推荐 ⭐⭐⭐⭐⭐)
  - 平衡新旧信息（90% 旧 + 10% 新）
  - 最大变化从 {max_change_no_smoothing:.4f} 降到 {ema_results[0.9]['max_change']:.4f}
  - 改善 {(1 - ema_results[0.9]['max_change']/max_change_no_smoothing)*100:.1f}%
  - 适合大多数场景

- **momentum = 0.95** (更保守)
  - 更平滑，变化更缓慢
  - 最大变化: {ema_results[0.95]['max_change']:.4f}
  - 适合训练极度不稳定的情况

- **momentum = 0.8** (更激进)
  - 更快适应新分布
  - 最大变化: {ema_results[0.8]['max_change']:.4f}
  - 适合训练相对稳定，需要快速收敛
    """)
    
    # ========== 实际案例分析 ==========
    print(f"\n" + "=" * 80)
    print(f"🎯 实际案例：异常 Epoch 的处理")
    print(f"=" * 80)
    
    for idx in anomalies:
        epoch_num = idx + 1
        current = current_blank_priors[idx]
        ema_09 = ema_results[0.9]['priors'][idx]
        
        print(f"\nEpoch {epoch_num} (异常值):")
        print(f"  当前 epoch prior: {current:.4f} ({current*100:.1f}%)")
        print(f"  EMA prior (m=0.9): {ema_09:.4f} ({ema_09*100:.1f}%)")
        print(f"  EMA 的保护作用: 避免了 {abs(current - ema_09):.4f} ({abs(current - ema_09)/current*100:.1f}%) 的剧烈波动")
        
        if current < 0.1:
            print(f"  ⚠️  如果直接使用当前值 ({current:.4f})，模型可能崩溃！")
            print(f"  ✅  EMA 稳定在 {ema_09:.4f}，避免了崩溃")
    
    # ========== 总结 ==========
    print(f"\n" + "=" * 80)
    print(f"💡 关键要点")
    print(f"=" * 80)
    
    print(f"""
1. **问题严重性**
   - 无平滑: 最大变化 {max_change_no_smoothing:.4f} ({max_change_no_smoothing/max_change_no_smoothing*100:.1f}%)
   - 可能导致训练崩溃、loss 剧烈波动

2. **EMA 的效果**
   - 使用 momentum=0.9: 最大变化降低到 {ema_results[0.9]['max_change']:.4f} ({ema_results[0.9]['max_change']/max_change_no_smoothing*100:.1f}%)
   - 改善 {(1 - ema_results[0.9]['max_change']/max_change_no_smoothing)*100:.1f}%
   - 完全避免异常值的影响

3. **实施简单**
   ```python
   # 只需 3 行代码
   if ema_priors is None:
       ema_priors = current_priors
   else:
       ema_priors = 0.9 * ema_priors + 0.1 * current_priors
   ```

4. **立即可用**
   - 已集成到 CTCLossWithLabelPriors
   - 只需在配置文件添加: prior_momentum: 0.9
   - 训练时自动应用，无需额外操作
    """)
    
    print(f"=" * 80)
    print(f"✅ 演示完成！")
    print(f"=" * 80)
    print(f"\n📊 对比图表: {output_path}")
    print(f"📚 详细文档: docs/fix_blank_prior_fluctuation.md\n")

if __name__ == '__main__':
    simulate_training_with_fluctuation()
