import matplotlib.pyplot as plt

# 数据
workers_num = list(range(1, 9))
average_inference = [9.26, 5.28, 4.48, 3.04, 2.68, 2.64, 2.48, 2.58]

# 绘制条形图
plt.bar(workers_num, average_inference, color='lightgreen')

# 添加标题和标签
plt.title('Average inference time over different workers')
plt.xlabel('Workers number')
plt.ylabel('Average inference time/s')

# 显示图形
plt.savefig('pipeline_different_workers.png')