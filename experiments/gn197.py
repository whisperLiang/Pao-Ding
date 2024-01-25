"""查看GoogLeNet第197层预测不准的原因
原因是数据集本身难以预测，输入的nz全部为1，但是输出的nz值很分散
"""
import pickle

import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(f'dataset/gn.road.480x720.400.lfcnz', 'rb') as f:
        lfcnz = pickle.load(f)
    x_total = [sum(lfcnz[196][f]) / len(lfcnz[196][f]) for f in range(1, len(lfcnz[196]))]
    y_total = [sum(lfcnz[197][f]) / len(lfcnz[197][f]) for f in range(1, len(lfcnz[197]))]
    plt.scatter(x_total, y_total)
    plt.title('whole feature map')
    plt.show()
    for c in range(len(lfcnz[197][0])):
        x, y = [], []
        for f in range(len(lfcnz[197])):
            x.append(lfcnz[196][f][c])
            y.append(lfcnz[197][f][c])
        plt.scatter(x, y)
        plt.title(f'channel{c}')
        plt.show()
