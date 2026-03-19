import matplotlib.pyplot as plt
import numpy as np

# 数据准备
x_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P10']
x = np.arange(len(x_labels))

# 指标数据
Pre = [95.91, 94.74, 95.02, 94.12, 93.69, 83.96]
Rec = [85.00, 88.26, 88.74, 87.73, 87.52, 84.47]
F1  = [90.12, 91.38, 91.78, 90.81, 90.50, 84.21]

PER = [5.17, 5.15, 5.04, 5.31, 5.53, 8.02]
FRR = [0.61, 0.82, 0.78, 0.92, 0.99, 2.72]
FAR = [15.00, 11.74, 11.26, 12.27, 12.48, 15.53]
EDR = [24.59, 24.80, 24.72, 24.74, 24.99, 25.83]

# 设置全局字体和样式（适配学术论文）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 图 1：综合性能表现 (Pre, Rec, F1) ---
ax1.plot(x, Pre, marker='o', linestyle='-', linewidth=2, label='Pre')
ax1.plot(x, Rec, marker='s', linestyle='--', linewidth=2, label='Rec')
ax1.plot(x, F1,  marker='^', linestyle='-', linewidth=2, color='red', label='F1')

ax1.set_title('Performance Metrics (Pre, Rec, F1)')
ax1.set_xlabel('Parameter Configuration')
ax1.set_ylabel('Percentage (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# --- 图 2：错误率表现 (PER, FRR, FAR, EDR) ---
ax2.plot(x, PER, marker='o', linestyle='-', linewidth=2, color='purple', label='PER')
ax2.plot(x, FRR, marker='s', linestyle='--', linewidth=2, label='FRR')
ax2.plot(x, FAR, marker='^', linestyle='-.', linewidth=2, label='FAR')
ax2.plot(x, EDR, marker='d', linestyle=':', linewidth=2, color='gray', label='EDR')

ax2.set_title('Error Rates (PER, FRR, FAR, EDR)')
ax2.set_xlabel('Parameter Configuration')
ax2.set_ylabel('Percentage (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

# 调整布局并保存/显示
plt.tight_layout()
plt.savefig('mdd_performance_trends.png', dpi=300, bbox_inches='tight') # 保存为高清图片
plt.show()