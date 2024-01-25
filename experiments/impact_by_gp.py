from typing import List

from matplotlib import pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
lg = {'size': 16}


def show_total(my: List[float], lb: float):
    my = np.array(my)
    print(f"最大提升 {np.max((lb-my)/lb)*100}%")

    plt.tick_params(labelsize=13)
    plt.gca().set_xlabel('调度组大小', fontproperties=lg)
    plt.gca().set_ylabel('总耗时(s)', fontproperties=lg)
    plt.plot(list(range(1, len(my)+1)), my, '-', c='tab:blue', label='DAS')
    plt.plot(list(range(1, len(my)+1)), [lb]*len(my), '--', c='tab:orange', label='LBS')
    plt.legend(prop=lg)
    plt.tight_layout()


# 数据来源：
# 设备配置：pi4G(m) → 4MB/s(实际带宽3) → pi2G(w0) → 4MB/s(实际带宽3) → pi2G1(w1)
# 运行配置：road[ifr_num=10, pd_num=0]
if __name__ == '__main__':
    # my: gp_size分别等于1,2,...,10时，总耗时(total)
    ax_my = [15.91, 11.44, 11.21, 11.43, 11.52, 11.69, 11.68, 11.37, 10.93, 11.74]
    ax_lb = 15.99
    vg_my = [251.91, 158.46, 140.16, 150.49, 150.11, 150.54, 150.18, 151.8, 143.6, 149.93]
    vg_lb = 158.42
    show_total(ax_my, ax_lb)
    plt.figure()
    show_total(vg_my, vg_lb)
    plt.show()
