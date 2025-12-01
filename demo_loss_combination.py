#!/usr/bin/env python3
"""
演示如何正确组合 CTC + Label Priors 与其他辅助 Loss

这个脚本展示：
1. 直接加权组合（推荐）
2. 分别记录各项 loss
3. 梯度分析
4. 监控策略
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def demonstrate_loss_combination():
    """演示多 loss 组合"""
    
    print("=" * 80)
    print("🔗 演示：如何组合 CTC + Label Priors 与其他辅助 Loss")
    print("=" * 80)
    
    # ========== 模拟场景 ==========
    print("\n【场景】多任务学习")
    print("-" * 80)
    print("任务 1: CTC 音素识别（使用 Label Priors）")
    print("任务 2: Seq2Seq 解码")
    print("任务 3: 发音错误检测（二分类）")
    print()
    
    # ========== 模拟训练过程 ==========
    num_epochs = 20
    
    # 模拟不同 epoch 的 loss 值
    # CTC loss: 从小负数变成大负数（置信度提高）
    loss_ctc_history = [-5, -8, -12, -15, -18, -22, -25, -28, -30, -32,
                        -34, -35, -36, -37, -38, -38, -39, -39, -40, -40]
    
    # Seq2Seq loss: 逐渐下降（正数）
    loss_seq_history = [45, 42, 38, 35, 32, 30, 28, 26, 24, 23,
                        22, 21, 20, 19.5, 19, 18.5, 18, 17.8, 17.5, 17.3]
    
    # Mispro loss: 逐渐下降（正数）
    loss_mispro_history = [12, 11, 10, 9, 8.5, 8, 7.5, 7, 6.8, 6.5,
                           6.3, 6.1, 6, 5.9, 5.8, 5.7, 5.6, 5.5, 5.4, 5.3]
    
    # ========== 方案 1：直接加权组合 ==========
    print("\n" + "=" * 80)
    print("方案 1：直接加权组合（推荐）")
    print("=" * 80)
    
    ctc_weight = 0.3
    seq_weight = 0.5
    mispro_weight = 0.2
    
    print(f"\n权重设置：")
    print(f"  CTC weight:    {ctc_weight}")
    print(f"  Seq2Seq weight: {seq_weight}")
    print(f"  Mispro weight:  {mispro_weight}")
    
    total_loss_history = []
    
    print(f"\n训练过程（前 10 个 epoch）：")
    print(f"{'Epoch':>6} | {'CTC Loss':>10} | {'Seq Loss':>10} | {'Mispro':>10} | {'Total':>10} | {'Trend':>8}")
    print("-" * 78)
    
    for epoch in range(10):
        loss_ctc = loss_ctc_history[epoch]
        loss_seq = loss_seq_history[epoch]
        loss_mispro = loss_mispro_history[epoch]
        
        # 直接组合（不管符号）
        total_loss = (
            ctc_weight * loss_ctc +
            seq_weight * loss_seq +
            mispro_weight * loss_mispro
        )
        
        total_loss_history.append(total_loss)
        
        # 判断趋势
        if epoch == 0:
            trend = "-"
        else:
            trend = "↓" if total_loss < total_loss_history[epoch-1] else "↑"
        
        print(f"{epoch+1:6d} | {loss_ctc:10.2f} | {loss_seq:10.2f} | {loss_mispro:10.2f} | {total_loss:10.2f} | {trend:>8s}")
    
    print("\n✅ 观察：")
    print(f"  - CTC Loss 变得越来越负（-5 → -32），这是正常的")
    print(f"  - Total Loss 持续下降（{total_loss_history[0]:.2f} → {total_loss_history[9]:.2f}）")
    print(f"  - 尽管 CTC loss 是负数，整体优化方向正确")
    
    # ========== 方案 2：分析各项贡献 ==========
    print("\n" + "=" * 80)
    print("方案 2：分析各项 Loss 的贡献")
    print("=" * 80)
    
    epoch_analyze = 5  # 分析第 5 个 epoch
    loss_ctc = loss_ctc_history[epoch_analyze]
    loss_seq = loss_seq_history[epoch_analyze]
    loss_mispro = loss_mispro_history[epoch_analyze]
    
    # 计算各项贡献
    ctc_contribution = ctc_weight * loss_ctc
    seq_contribution = seq_weight * loss_seq
    mispro_contribution = mispro_weight * loss_mispro
    total = ctc_contribution + seq_contribution + mispro_contribution
    
    print(f"\nEpoch {epoch_analyze + 1} 的详细分析：")
    print(f"{'Task':>15s} | {'Loss':>10s} | {'Weight':>8s} | {'Contribution':>15s} | {'Abs %':>10s}")
    print("-" * 75)
    
    abs_total = abs(ctc_contribution) + abs(seq_contribution) + abs(mispro_contribution)
    
    print(f"{'CTC':>15s} | {loss_ctc:10.2f} | {ctc_weight:8.2f} | {ctc_contribution:15.2f} | {abs(ctc_contribution)/abs_total*100:9.1f}%")
    print(f"{'Seq2Seq':>15s} | {loss_seq:10.2f} | {seq_weight:8.2f} | {seq_contribution:15.2f} | {abs(seq_contribution)/abs_total*100:9.1f}%")
    print(f"{'Mispro':>15s} | {loss_mispro:10.2f} | {mispro_weight:8.2f} | {mispro_contribution:15.2f} | {abs(mispro_contribution)/abs_total*100:9.1f}%")
    print("-" * 75)
    print(f"{'Total':>15s} | {'':>10s} | {'':>8s} | {total:15.2f} | {100.0:9.1f}%")
    
    print(f"\n✅ 解释：")
    print(f"  - CTC 贡献：{ctc_contribution:.2f}（负数，说明这部分置信度高）")
    print(f"  - Seq2Seq 贡献：{seq_contribution:.2f}（正数，主要优化目标）")
    print(f"  - Mispro 贡献：{mispro_contribution:.2f}（正数，辅助任务）")
    print(f"  - 总和：{total:.2f}（综合优化目标）")
    
    # ========== 方案 3：模拟梯度分析 ==========
    print("\n" + "=" * 80)
    print("方案 3：梯度分析")
    print("=" * 80)
    
    # 创建简单的模型和 loss
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_layer = nn.Linear(10, 10)
            self.ctc_head = nn.Linear(10, 5)
            self.seq_head = nn.Linear(10, 5)
            self.mispro_head = nn.Linear(10, 2)
        
        def forward(self, x):
            shared = self.shared_layer(x)
            ctc_out = self.ctc_head(shared)
            seq_out = self.seq_head(shared)
            mispro_out = self.mispro_head(shared)
            return ctc_out, seq_out, mispro_out
    
    model = SimpleModel()
    
    # 模拟输入
    x = torch.randn(2, 10)
    ctc_out, seq_out, mispro_out = model(x)
    
    # 模拟 loss（CTC 是负数）
    loss_ctc_sim = -torch.sum(ctc_out ** 2) * 0.1  # 负数
    loss_seq_sim = torch.sum((seq_out - 1) ** 2)
    loss_mispro_sim = torch.sum((mispro_out) ** 2)
    
    # 组合
    total_loss_sim = 0.3 * loss_ctc_sim + 0.5 * loss_seq_sim + 0.2 * loss_mispro_sim
    
    print(f"\n模拟的 Loss 值：")
    print(f"  CTC Loss:    {loss_ctc_sim.item():10.4f} (负数)")
    print(f"  Seq Loss:    {loss_seq_sim.item():10.4f}")
    print(f"  Mispro Loss: {loss_mispro_sim.item():10.4f}")
    print(f"  Total Loss:  {total_loss_sim.item():10.4f}")
    
    # 反向传播
    total_loss_sim.backward()
    
    # 查看梯度
    print(f"\n各层梯度范数：")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name:20s}: {grad_norm:10.6f}")
    
    print(f"\n✅ 结论：")
    print(f"  - 即使 CTC loss 是负数，梯度仍然正确计算")
    print(f"  - 各层都有合理的梯度更新")
    print(f"  - 优化器会根据总梯度正确更新参数")
    
    # ========== 可视化 ==========
    print("\n" + "=" * 80)
    print("📈 生成可视化图表...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('多 Loss 组合策略', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, num_epochs + 1))
    
    # 完成所有 epoch 的 total loss 计算
    for epoch in range(10, num_epochs):
        total_loss = (
            ctc_weight * loss_ctc_history[epoch] +
            seq_weight * loss_seq_history[epoch] +
            mispro_weight * loss_mispro_history[epoch]
        )
        total_loss_history.append(total_loss)
    
    # 图 1: 各个 Loss 的原始值
    ax1 = axes[0, 0]
    ax1.plot(epochs, loss_ctc_history, 'o-', label='CTC Loss', color='red', linewidth=2)
    ax1.plot(epochs, loss_seq_history, 's-', label='Seq2Seq Loss', color='blue', linewidth=2)
    ax1.plot(epochs, loss_mispro_history, '^-', label='Mispro Loss', color='green', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('(1) 各个 Loss 的原始值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2: Total Loss 趋势
    ax2 = axes[0, 1]
    ax2.plot(epochs, total_loss_history, 'D-', color='purple', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('(2) 组合后的 Total Loss')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(epochs, total_loss_history, 1)
    p = np.poly1d(z)
    ax2.plot(epochs, p(epochs), "--", color='orange', linewidth=2, alpha=0.7, label='Trend')
    ax2.legend()
    
    # 图 3: 各项贡献（堆叠柱状图）
    ax3 = axes[1, 0]
    
    ctc_contributions = [ctc_weight * l for l in loss_ctc_history]
    seq_contributions = [seq_weight * l for l in loss_seq_history]
    mispro_contributions = [mispro_weight * l for l in loss_mispro_history]
    
    # 只显示前 10 个 epoch
    epochs_subset = epochs[:10]
    
    ax3.bar(epochs_subset, ctc_contributions[:10], label='CTC', color='red', alpha=0.7)
    ax3.bar(epochs_subset, seq_contributions[:10], 
            bottom=ctc_contributions[:10], label='Seq2Seq', color='blue', alpha=0.7)
    ax3.bar(epochs_subset, mispro_contributions[:10], 
            bottom=[c+s for c, s in zip(ctc_contributions[:10], seq_contributions[:10])], 
            label='Mispro', color='green', alpha=0.7)
    
    ax3.axhline(y=0, color='black', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Contribution to Total Loss')
    ax3.set_title('(3) 各项 Loss 的贡献（前 10 epochs）')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图 4: 相对变化率
    ax4 = axes[1, 1]
    
    # 计算相对于初始值的变化率
    ctc_change = [(l - loss_ctc_history[0]) / abs(loss_ctc_history[0]) * 100 for l in loss_ctc_history]
    seq_change = [(l - loss_seq_history[0]) / loss_seq_history[0] * 100 for l in loss_seq_history]
    mispro_change = [(l - loss_mispro_history[0]) / loss_mispro_history[0] * 100 for l in loss_mispro_history]
    total_change = [(l - total_loss_history[0]) / abs(total_loss_history[0]) * 100 for l in total_loss_history]
    
    ax4.plot(epochs, ctc_change, 'o-', label='CTC', color='red', linewidth=2, alpha=0.7)
    ax4.plot(epochs, seq_change, 's-', label='Seq2Seq', color='blue', linewidth=2, alpha=0.7)
    ax4.plot(epochs, mispro_change, '^-', label='Mispro', color='green', linewidth=2, alpha=0.7)
    ax4.plot(epochs, total_change, 'D-', label='Total', color='purple', linewidth=2.5)
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Relative Change (%)')
    ax4.set_title('(4) 相对于初始值的变化率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'loss_combination_strategy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存可视化图表到: {output_path}")
    
    # ========== 权重调整建议 ==========
    print("\n" + "=" * 80)
    print("🎯 权重调整建议")
    print("=" * 80)
    
    print(f"""
根据模拟结果，当前权重设置：
  - CTC weight:    {ctc_weight} (CTC Loss 从 {loss_ctc_history[0]:.1f} 降到 {loss_ctc_history[-1]:.1f})
  - Seq2Seq weight: {seq_weight} (Seq Loss 从 {loss_seq_history[0]:.1f} 降到 {loss_seq_history[-1]:.1f})
  - Mispro weight:  {mispro_weight} (Mispro Loss 从 {loss_mispro_history[0]:.1f} 降到 {loss_mispro_history[-1]:.1f})

如何调整？

1. 如果 Validation WER 不理想：
   → 增加 CTC weight (如 0.3 → 0.5)
   
2. 如果 Seq2Seq 解码质量差：
   → 增加 Seq weight (如 0.5 → 0.6)
   
3. 如果 Mispro 检测不准：
   → 增加 Mispro weight (如 0.2 → 0.3)
   
4. 如果某个 loss 收敛太快（接近 0）：
   → 降低其 weight，让其他任务有更多训练机会

记住：权重总和不一定要等于 1.0！
可以是 0.7 + 0.5 + 0.3 = 1.5（各任务都重要）
也可以是 0.3 + 0.2 + 0.1 = 0.6（降低整体 loss scale）
    """)
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("💡 关键要点总结")
    print("=" * 80)
    
    print("""
1. **直接组合，不怕负数**
   total_loss = w1 * loss1 + w2 * loss2 + w3 * loss3
   即使某个 loss 是负数也没问题！

2. **关注总体趋势，不是绝对值**
   - Total loss 持续下降 ✓
   - Validation 性能提升 ✓
   - 这才是成功的标志

3. **分别监控各项 loss**
   - 记录每个 loss 的值
   - 可视化趋势
   - 方便调试和调权重

4. **不要用 abs()！**
   - abs() 会破坏梯度方向
   - 直接用原始 loss 值

5. **权重调整靠实验**
   - 没有万能的权重
   - 根据 validation 性能调整
   - 可以动态调整（早期 vs 后期）

6. **梯度检查很重要**
   - 定期检查各层梯度范数
   - 确保没有梯度爆炸/消失
   - 确保各任务都有梯度贡献
    """)
    
    print("=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print(f"\n📊 可视化图表已保存到: {output_path}")
    print("📚 详细文档请参考: docs/combining_losses_with_label_priors.md")
    print("\n")

if __name__ == '__main__':
    demonstrate_loss_combination()
