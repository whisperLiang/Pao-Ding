import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# 实验设置的带宽
bandwidths = [1] + list(range(5, 45, 5))

# 首帧延时数据
first_frame_times = [15.05, 11.36, 9.94, 8.31, 8.24, 8.08, 7.30, 7.28, 5.97]

# 平均推理延时数据
avg_inference_times = [11.71, 8.35, 8.31, 5.24, 5.20, 5.14, 3.08, 3.02, 2.77]

# 最佳分割点选择
best_split_points = [40, 17, 17, 10, 10, 10, 5, 5, 5]

# 创建图表
fig, ax1 = plt.subplots()

# 绘制训练时间的条形图
bar_width = 0.35
index = np.arange(len(bandwidths))
bar1 = ax1.bar(index - bar_width/2, first_frame_times, bar_width, label='First frame time', color='lightgreen')

# 绘制推理时间的条形图
bar2 = ax1.bar(index + bar_width/2, avg_inference_times, bar_width, label='Avg inference time', color='lightblue')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制折线图
line = ax2.plot(best_split_points, 'g-', label='Best split point')

# 在每个点上添加数值标注
for i, txt in enumerate(best_split_points):
    ax2.annotate(txt, (i, txt), textcoords="offset points", xytext=(0,3), ha='center')

# 设置第二个y轴的标签
ax2.set_ylabel('DNN Index', fontsize=16)

# 添加标题和标签
plt.title('Pipeline execution with different bandwidths', fontsize=16)
ax1.set_xlabel('Bandwidth (Mbps)', fontsize=16)
ax1.set_ylabel('Time (seconds)', fontsize=16)

# 设置坐标轴刻度的字体大小
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# 添加图例
ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.85))

# 添加刻度标签
plt.xticks(index, bandwidths)

# 显示图表
plt.tight_layout()  # 调整布局以防止标签被裁剪
plt.savefig('pipeline_different_bandwidths.png')
plt.show()