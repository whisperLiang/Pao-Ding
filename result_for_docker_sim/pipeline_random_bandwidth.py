import matplotlib.pyplot as plt
import numpy as np

# 数据
ifr_queue = list(range(0, 20))
w0_bandwidths = [2.27, 1.71, 1.64, 1.42, 1.71, 1.63, 1.59, 1.51, 2.22, 1.94, 1.84, 1.90, 0.11, 0.09, 1.14, 0.90, 0.84, 0.82, 0.85, 0.74] 
w1_bandwidths = [2.23, 2.17, 1.76, 1.91, 1.73, 2.23, 1.89, 1.79, 1.72, 2.21, 1.93, 2.05, 0.06, 0.08, 1.14, 1.14, 1.14, 1.07, 1.14, 1.14]
# 将 w0_bandwidths 和 w1_bandwidths 的值都乘以 8
w0_bandwidths = [x * 8 for x in w0_bandwidths]
w1_bandwidths = [x * 8 for x in w1_bandwidths]

best_split_points = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 40, 40, 17, 17, 17, 17, 17, 17]

# 创建图表
fig, ax1 = plt.subplots()

# 绘制 w0 和 w1 的带宽折线图，并添加点标记
ax1.plot(ifr_queue, w0_bandwidths, color='lightblue', marker='x', label=r'm$\rightarrow$w0 bandwidth')  # 浅绿色
ax1.plot(ifr_queue, w1_bandwidths, color='gray', marker='x', label=r'w0$\rightarrow$w1 bandwidth')  # 浅蓝色

# 设置第一个 y 轴
ax1.set_xlabel('IFR Queue Index', fontsize=14)
ax1.set_ylabel('Bandwidth (Mbps)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xticks(ifr_queue)  # 设置横坐标显示 0-19 的整数
ax1.set_ylim(0, 20)  # 设置 y 轴范围为 0-20

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(ifr_queue, best_split_points, color='green', marker='o', label='Best split point')  # 绿色

# 设置第二个 y 轴
ax2.set_ylabel('Layer Index', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_ylim(0, 42)  # 设置 y 轴范围为 0-42

# 添加图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left', fontsize=12)

# 添加标题
plt.title('Pipeline Execution with Random Bandwidths', fontsize=16)

# 显示图表
plt.tight_layout()  # 调整布局以防止标签被裁剪
plt.savefig('pipeline_random_bandwidths.png', dpi=300)  # 保存图表为 PNG 文件