import matplotlib.pyplot as plt
import numpy as np

# 数据
ifr_queue = list(range(0, 14))
w0_bandwidths = [1.06, 0.91, 1.57, 1.39, 1.32, 1.47, 1.21, 1.52, 0.10, 0.10, 0.76, 0.93, 0.55, 0.86] 
w1_bandwidths = [2.12, 1.43, 1.19, 1.08, 1.42, 1.36, 1.00, 1.42, 0.10, 0.11, 0.82, 0.74, 0.92, 0.76]
# 将 w0_bandwidths 和 w1_bandwidths 的值都乘以 8
w0_bandwidths = [x * 8 for x in w0_bandwidths]
w1_bandwidths = [x * 8 for x in w1_bandwidths]

best_split_points = [5, 10, 5, 10, 5, 5, 10, 5, 40, 40, 17, 17, 17, 17]

# 创建图表
fig, ax1 = plt.subplots()

# 绘制 w0 和 w1 的带宽折线图，并添加点标记
ax1.plot(ifr_queue, w0_bandwidths, color='lightblue', marker='x', label=r'm$\rightarrow$w0')  # 浅绿色
ax1.plot(ifr_queue, w1_bandwidths, color='gray', marker='x', label=r'w0$\rightarrow$w1')  # 浅蓝色

# 设置第一个 y 轴
ax1.set_xlabel('IFR Queue Index', fontsize=14)
ax1.set_ylabel('Bandwidth (Mbps)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xticks(ifr_queue)  # 设置横坐标显示 0-7 的整数
ax1.set_ylim(0, 20)  # 设置 y 轴范围为 0-20

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(ifr_queue, best_split_points, color='green', marker='o', label='split point')  # 绿色

# 设置第二个 y 轴
ax2.set_ylabel('Layer Index', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_ylim(0, 42)  # 设置 y 轴范围为 0-42

# 添加图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# 添加标题
plt.title('Pipeline Execution with Dynamic Bandwidths', fontsize=16)

# 显示图表
plt.tight_layout()  # 调整布局以防止标签被裁剪
plt.savefig('pipeline_dynamic_bandwidths_real.png', dpi=300)  # 保存图表为 PNG 文件