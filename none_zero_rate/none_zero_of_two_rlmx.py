"""读取一个LFCNZ文件，显示各层输出差值的稀疏情况
横轴为层号，纵轴为当前层输出和前一帧差值的非零占比
"""
import pickle
from typing import List

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from core.dag_dnn import DagDNN
from model_split import Node
from dnn_models.any_dnn_split import prepare_alexnet, prepare_vgg16, prepare_resnet50, prepare_googlenet
import numpy as np
from torch.nn import ReLU, MaxPool2d
plt.rc('font',family='Times New Roman')
import statsmodels.api as sm # recommended import according to the docs
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# sm = FontProperties(size=16)
lg = FontProperties(size=15)
import matplotlib
matplotlib.rc('pdf', fonttype=42)


def lfcnz2lfnz(lfcnz: List[List[List[float]]]) -> List[List[float]]:
    """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
    return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]

def hist_cdf(data,bins=10,range=None):
      hist,bin_edges=np.histogram(data,bins=bins)
      hist = np.insert(hist,0,0)
      cdf = np.cumsum(hist)/len(data)
      return cdf,bin_edges

def heatmap(r_layers: List[Node], lfnz: List[List[float]], thres: float = .495):
    """点的颜色表示这个稀疏率出现次数，thres为稀疏阈值"""
    nlayer = len(lfnz)  # 层数
    nframe = len(lfnz[0])  # 帧数
    x, y = [], []  # 每个点为一帧，x为该帧数据所在层，y为该帧数据的非零占比
    for l in range(len(r_layers)):
        x.extend([l] * nframe)
        y.extend(lfnz[l])
    sps_cnt = 0
    for yelm in y:
        if yelm < .5:
            sps_cnt += 1
    print(len(x))
    xedges = [-0.5] + [l+0.5 for l in range(nlayer)]
    yedges = [0.] + [1/nlayer*l for l in range(1, nlayer+1)]
    plt.figure(figsize=(6, 5))
    plt.tick_params(labelsize=15)
    plt.hist2d(x, y, bins=(xedges, yedges), cmap='Greens')
    plt.plot([0, nlayer - 1], [thres, thres], linestyle='--')
    print(f'sparse={sps_cnt}/{nframe*nlayer}={round(sps_cnt/(nframe*nlayer)*100, 2)}%')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    plt.gca().set_ylabel('Nonzero Rate', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.text(7, 0.5, r'$threshold=\eta$', ha='center', va='bottom', fontsize=15)
    plt.show()

def generate_data(CNN_NAME, ORIGINAL):
    VIDEO_NAME = 'parking'  # road, campus, parking
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    NFRAME_SHOW = 400  # 展示数据集中的多少帧
    suffix = ('o_' if ORIGINAL else '') + 'lfcnz'
    file_name = f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.{suffix}"
    cnn_loaders = {'AlexNet': prepare_alexnet,
                   'VGG': prepare_vgg16,
                   'GoogLeNet': prepare_googlenet,
                   'ResNet': prepare_resnet50}
    dag_dnn = DagDNN(cnn_loaders[CNN_NAME]())
    r_layers = dag_dnn.layers
    with open(file_name, 'rb') as f:
        lfcnz = pickle.load(f)
    lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in lfcnz]
    lfnz = lfcnz2lfnz(lfcnz)
    nlayer = len(lfnz)  # 层数
    nframe = len(lfnz[0])  # 帧数

    # 分别存储 ReLU 层和其他层的数据
    x_relu, y_relu = [], []
    x_other, y_other = [], []

    for l in range(len(r_layers)):
        if isinstance(r_layers[l].module, ReLU):
            x_relu.extend([l] * nframe)
            y_relu.extend(lfnz[l])
        else:
            x_other.extend([l] * nframe)
            y_other.extend(lfnz[l])

    # 分别计算稀疏率
    sps_cnt_relu = sum(1 for yelm in y_relu if yelm < .5)
    sps_cnt_other = sum(1 for yelm in y_other if yelm < .5)

    xedges = [-0.5] + [l + 0.5 for l in range(nlayer)]
    yedges = [0.] + [1 / nlayer * l for l in range(1, nlayer + 1)]

    return x_relu, y_relu, x_other, y_other, xedges, yedges, nlayer, nframe, sps_cnt_relu, sps_cnt_other

if __name__ == '__main__':
    ORIGINAL = False  # False为差值数据LFCNZ，True为原始数据OLFCNZ

    # 处理 AlexNet 数据
    x_relu_ax, y_relu_ax, x_other_ax, y_other_ax, xedges_ax, yedges_ax, nlayer_ax, nframe_ax, sps_cnt_relu_ax, sps_cnt_other_ax = generate_data('AlexNet', ORIGINAL)

    # 处理 VGG16 数据
    x_relu_vgg, y_relu_vgg, x_other_vgg, y_other_vgg, xedges_vgg, yedges_vgg, nlayer_vgg, nframe_vgg, sps_cnt_relu_vgg, sps_cnt_other_vgg = generate_data('VGG', ORIGINAL)

    # 处理 GoogLeNet 数据
    x_relu_gn, y_relu_gn, x_other_gn, y_other_gn, xedges_gn, yedges_gn, nlayer_gn, nframe_gn, sps_cnt_relu_gn, sps_cnt_other_gn = generate_data('GoogLeNet', ORIGINAL)

    # 处理 ResNet50 数据
    x_relu_res, y_relu_res, x_other_res, y_other_res, xedges_res, yedges_res, nlayer_res, nframe_res, sps_cnt_relu_res, sps_cnt_other_res = generate_data('ResNet', ORIGINAL)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制 AlexNet 图
    plt.subplot(221)
    plt.tick_params(labelsize=13)
    plt.title('AlexNet')
    plt.hist2d(x_relu_ax, y_relu_ax, bins=(xedges_ax, yedges_ax), cmap='Greens', label='ReLU')
    plt.hist2d(x_other_ax, y_other_ax, bins=(xedges_ax, yedges_ax), cmap='Reds', alpha=0.6, label='Other')
    plt.plot([0, nlayer_ax - 1], [0.5, 0.5], linestyle='--')
    sparse_relu = round(sps_cnt_relu_ax / len(x_relu_ax) * 100, 1)
    sparse_other = round(sps_cnt_other_ax / len(x_other_ax) * 100, 1)
    plt.text(6, 0.07, f'ReLU: {sparse_relu}%', ha='center', va='bottom', fontsize=12, color='green')
    plt.text(6, 0.03, f'Other: {sparse_other}%', ha='center', va='bottom', fontsize=12, color='red')
    plt.gca().set_ylabel('Nonzero-rate', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.set_ticks([])  # 隐藏刻度横线和刻度文字
    # cbar.set_label('Number of Frames', fontproperties=lg)

    # 绘制 VGG16 图
    plt.subplot(222)
    plt.tick_params(labelsize=13)
    plt.title('VGG16')
    plt.hist2d(x_relu_vgg, y_relu_vgg, bins=(xedges_vgg, yedges_vgg), cmap='Greens', label='ReLU')
    plt.hist2d(x_other_vgg, y_other_vgg, bins=(xedges_vgg, yedges_vgg), cmap='Reds', alpha=0.6, label='Other')
    plt.plot([0, nlayer_vgg - 1], [0.5, 0.5], linestyle='--')
    sparse_relu = round(sps_cnt_relu_vgg / len(x_relu_vgg) * 100, 1)
    sparse_other = round(sps_cnt_other_vgg / len(x_other_vgg) * 100, 1)
    plt.text(8, 0.09, f'ReLU: {sparse_relu}%', ha='center', va='bottom', fontsize=12, color='green')
    plt.text(8, 0.05, f'Other: {sparse_other}%', ha='center', va='bottom', fontsize=12, color='red')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.gca().tick_params(left=False, labelleft=False)  # 去除左边刻度线和刻度文字

    # 绘制 ResNet50 图
    plt.subplot(223)
    plt.tick_params(labelsize=13)
    plt.title('ResNet50')
    plt.hist2d(x_relu_res, y_relu_res, bins=(xedges_res, yedges_res), cmap='Greens', label='ReLU')
    plt.hist2d(x_other_res, y_other_res, bins=(xedges_res, yedges_res), cmap='Reds', alpha=0.6, label='Other')
    plt.plot([0, nlayer_res - 1], [0.5, 0.5], linestyle='--')
    sparse_relu = round(sps_cnt_relu_res / len(x_relu_res) * 100, 1)
    sparse_other = round(sps_cnt_other_res / len(x_other_res) * 100, 1)
    plt.text(35, 0.09, f'ReLU: {sparse_relu}%', ha='center', va='bottom', fontsize=12, color='green')
    plt.text(35, 0.05, f'Other: {sparse_other}%', ha='center', va='bottom', fontsize=12, color='red')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    plt.gca().set_ylabel('Nonzero-rate', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.set_ticks([])  # 隐藏刻度横线和刻度文字
    # cbar.set_label('Number of Frames', fontproperties=lg)

    # 绘制 GoogLeNet 图
    plt.subplot(224)
    plt.tick_params(labelsize=13)
    plt.title('GoogLeNet')
    plt.hist2d(x_relu_gn, y_relu_gn, bins=(xedges_gn, yedges_gn), cmap='Greens', label='ReLU')
    plt.hist2d(x_other_gn, y_other_gn, bins=(xedges_gn, yedges_gn), cmap='Reds', alpha=0.6, label='Other')
    plt.plot([0, nlayer_gn - 1], [0.5, 0.5], linestyle='--')
    sparse_relu = round(sps_cnt_relu_gn / len(x_relu_gn) * 100, 1)
    sparse_other = round(sps_cnt_other_gn / len(x_other_gn) * 100, 1)
    plt.text(35, 0.09, f'ReLU: {sparse_relu}%', ha='center', va='bottom', fontsize=12, color='green')
    plt.text(35, 0.05, f'Other: {sparse_other}%', ha='center', va='bottom', fontsize=12, color='red')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.gca().tick_params(left=False, labelleft=False)  # 去除左边刻度线和刻度文字

    # 调整布局并保存图像
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.18)
    plt.savefig('none_zero_rate_of_two_rlmx.png')