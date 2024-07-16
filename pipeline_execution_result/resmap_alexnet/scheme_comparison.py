import matplotlib.pyplot as plt
import numpy as np

# 准备数据
categories = ['ResMap', 'Pao-Ding']
values1 = [2.37, 0.37]
values2 = [5, 0.9]

# 创建一个图形和一个子图
fig, ax1 = plt.subplots()

# 计算柱状图的宽度和x位置
bar_width = 0.1
x_positions1 = 0.5 * np.arange(len(categories)) - bar_width / 2
x_positions2 = 0.5 * np.arange(len(categories)) + bar_width / 2

# 绘制第一个柱状图，颜色为浅绿色
bars1 = ax1.bar(x_positions1, values1, color='lightgreen', width=bar_width, align='center')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制第二个柱状图，颜色改为浅蓝色
bars2 = ax2.bar(x_positions2, values2, color='lightblue', width=bar_width, align='center')

# 设置标题和标签
ax1.set_xlabel('Schemes', fontsize=16)
ax1.set_ylabel('trainning time(s)', fontsize=16)
ax2.set_ylabel('inference time(ms)', fontsize=16)

# 设置x轴的刻度位置和标签
ax1.set_xticks(0.5 * np.arange(len(categories)))
ax1.set_xticklabels(categories, fontsize=16)

# 增加坐标轴数字的字体大小
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)

# 设置图例
legend1 = ax1.legend(bars1, ['trainning time'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=16)
legend2 = ax2.legend(bars2, ['inference time'], loc='upper right', bbox_to_anchor=(1, 0.85), fontsize=16)

# 显示图形
plt.savefig('scheme comparison.png')