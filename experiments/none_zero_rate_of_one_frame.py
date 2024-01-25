"""读取一个LFCNZ文件，显示各层输出差值的稀疏情况
横轴为层号，纵轴为当前层输出和前一帧差值的非零占比
"""
import pickle
from typing import List

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from core.dnn_config import RawLayer
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
import numpy as np
plt.rc('font',family='Times New Roman')
import statsmodels.api as sm # recommended import according to the docs
from scipy.signal import savgol_filter
import matplotlib
matplotlib.rc('pdf', fonttype=42)

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# sm = FontProperties(size=16)
lg = FontProperties(size=15)


def lfcnz2lfnz(lfcnz: List[List[List[float]]]) -> List[List[float]]:
    """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
    return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]


def frame_seq(r_layers: List[RawLayer], lfnz: List[List[float]], thres: float = .495):
    """点的颜色表示该帧在视频中所处位置，thres为稀疏阈值"""
    nlayer = len(lfnz)  # 层数
    nframe = len(lfnz[0])  # 帧数
    ax = plt.gca()
    plt.tick_params(labelsize=13)
    ax.set_ylim(-.05, 1.05)
    # 确保xtick都是int
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('CNN层号', fontproperties=lg)
    ax.set_ylabel('非零元素占比', fontproperties=lg)
    for l in range(len(r_layers)):
        plt.scatter([l] * nframe, lfnz[l], s=5,
                    c=[i / nframe for i in range(nframe)],
                    marker='.', cmap='viridis')
        if nlayer >= 60:  # 超过60层，画垂直辅助线
            # 垂直辅助线：从1到最大值
            plt.plot([l, l], [1, max(lfnz[l])], color=(0.8, 0.8, 0.8), linestyle=':')
    # plt.plot([ax.get_xbound()[0], ax.get_xbound()[1]], [thres, thres], linestyle='--')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('播放进度', fontproperties=lg)
    cbar.set_ticks([0, .2, .4, .6, .8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', 'end'])
    plt.tight_layout()


def hist_cdf(data,bins=10,range=None):
      hist,bin_edges=np.histogram(data,bins=bins)
      hist = np.insert(hist,0,0)
      cdf = np.cumsum(hist)/len(data)
      return cdf,bin_edges

def heatmap(r_layers: List[RawLayer], lfnz: List[List[float]], thres: float = .495):
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
    plt.hist2d(x, y, bins=(xedges, yedges), cmap='Reds')
    plt.plot([0, nlayer - 1], [thres, thres], linestyle='--')
    print(f'sparse={sps_cnt}/{nframe*nlayer}={round(sps_cnt/(nframe*nlayer)*100, 2)}%')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    plt.gca().set_ylabel('Nonzero Rate', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.text(7, 0.5, r'$threshold=\eta$', ha='center', va='bottom', fontsize=15)
    plt.show()
    #画CDF图
    plt.figure(figsize=(6, 5))
    nzr, x_ = hist_cdf(y,[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    print(nzr,x_)
    plt.plot(x_, nzr, 'o-', linewidth=3, label='Festive', zorder=10)
    plt.show()

def generate_data(CNN_NAME,ORIGINAL):
    VIDEO_NAME = 'road'  # road, campus, parking
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    NFRAME_SHOW = 400  # 展示数据集中的多少帧
    suffix = ('o_' if ORIGINAL else '') + 'lfcnz'
    file_name = f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.{suffix}"
    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    r_layers = raw_dnn.layers
    with open(file_name, 'rb') as f:
        lfcnz = pickle.load(f)
    lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in lfcnz]
    lfnz = lfcnz2lfnz(lfcnz)
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
    xedges = [-0.5] + [l + 0.5 for l in range(nlayer)]
    yedges = [0.] + [1 / nlayer * l for l in range(1, nlayer + 1)]
    return x,y,xedges,yedges,nlayer,nframe,sps_cnt

if __name__ == '__main__':
    CNN_NAME = 'ax'  #ax, vg16, gn, rs50
    ORIGINAL = True  # False为差值数据LFCNZ，True为原始数据OLFCNZ
    x_ax, y_ax, xedges_ax, yedges_ax, nlayer_ax, nframe_ax, sps_cnt_ax= generate_data('ax', ORIGINAL)
    x_vgg, y_vgg, xedges_vgg, yedges_vgg, nlayer_vgg, nframe_vgg, sps_cnt_vgg = generate_data('vg16', ORIGINAL)
    x_gn, y_gn, xedges_gn, yedges_gn, nlayer_gn, nframe_gn, sps_cnt_gn = generate_data('gn', ORIGINAL)
    x_res, y_res, xedges_res, yedges_res, nlayer_res, nframe_res, sps_cnt_res = generate_data('rs50', ORIGINAL)
    plt.figure(figsize=(6, 5))

    plt.subplot(221)
    plt.tick_params(labelsize=13)
    plt.title('AlexNet')
    plt.hist2d(x_ax, y_ax, bins=(xedges_ax, yedges_ax), cmap='Reds')
    plt.plot([0, nlayer_ax - 1], [0.5, 0.5], linestyle='--')
    print(f'sparse={sps_cnt_ax}/{nframe_ax*nlayer_ax}={round(sps_cnt_ax/(nframe_ax*nlayer_ax)*100, 2)}%')
    #plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    plt.gca().set_ylabel('Nonzero-rate', fontproperties=lg)
    cbar = plt.colorbar(ticks=[])
    #cbar.ax.tick_params(labelsize=13)
    #cbar.set_label('Number of Frames', fontproperties=lg)
    plt.text(2, 0.09, r'42.86%', ha='center', va='bottom', fontsize=10)
    plt.quiver(0.5, 0.5, 0, -1, color='b', scale=3, width=0.02)

    plt.subplot(222)
    plt.tick_params(labelsize=13)
    plt.title('VGG16')
    plt.hist2d(x_vgg, y_vgg, bins=(xedges_vgg, yedges_vgg), cmap='Reds')
    plt.plot([0, nlayer_vgg - 1], [0.5, 0.5], linestyle='--')
    print(f'sparse={sps_cnt_vgg}/{nframe_vgg * nlayer_vgg}={round(sps_cnt_vgg / (nframe_vgg * nlayer_vgg) * 100, 2)}%')
    #plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    #plt.gca().set_ylabel('Nonzero Rate', fontproperties=lg)
    cbar = plt.colorbar()
    #plt.yticks([])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.text(4, 0.09, r'43.75%', ha='center', va='bottom', fontsize=10)
    plt.quiver(2, 0.5, 0, -1, color='b', scale=3, width=0.02)


    plt.subplot(223)
    plt.tick_params(labelsize=13)
    plt.title('ResNet')
    plt.hist2d(x_res, y_res, bins=(xedges_res, yedges_res), cmap='Reds')
    plt.plot([0, nlayer_res - 1], [0.5, 0.5], linestyle='--')
    print(f'sparse={sps_cnt_res}/{nframe_res * nlayer_res}={round(sps_cnt_res / (nframe_res * nlayer_res) * 100, 2)}%')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    plt.gca().set_ylabel('Nonzero-rate', fontproperties=lg)
    cbar = plt.colorbar(ticks=[])
    cbar.ax.tick_params(labelsize=13)
    #cbar.set_label('Number of Frames', fontproperties=lg)
    #plt.text(20, 0.5, r'$threshold=\eta$', ha='center', va='bottom', fontsize=15)
    plt.text(25, 0.09, r'22.5%', ha='center', va='bottom', fontsize=10)
    plt.quiver(20, 0.5, 0, -1, color='b', scale=3, width=0.02)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)

    plt.subplot(224)
    plt.tick_params(labelsize=13)
    plt.title('GoogLeNet')
    plt.hist2d(x_gn, y_gn, bins=(xedges_gn, yedges_gn), cmap='Reds')
    plt.plot([0, nlayer_gn - 1], [0.5, 0.5], linestyle='--')
    print(f'sparse={sps_cnt_gn}/{nframe_gn * nlayer_gn}={round(sps_cnt_gn / (nframe_gn * nlayer_gn) * 100, 2)}%')
    plt.gca().set_xlabel('CNN Layer Index', fontproperties=lg)
    #plt.gca().set_ylabel('NZR', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Number of Frames', fontproperties=lg)
    plt.text(25, 0.09, r'27.09%', ha='center', va='bottom', fontsize=10)
    plt.quiver(20, 0.5, 0, -1, color='b', scale=3, width=0.02)
    plt.show()


    #画CDF图
    plt.figure(figsize=(6, 5))
    nzr_ax, xax = hist_cdf(y_ax,[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    nzr_vgg, xvgg = hist_cdf(y_vgg,[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    nzr_gn, xgn = hist_cdf(y_gn,[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    nzr_res, xres = hist_cdf(y_res,[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    lowess = sm.nonparametric.lowess
    # nzr_ax = savgol_filter(nzr_ax, 5, 3, mode='nearest')
    # nzr_vgg = savgol_filter(nzr_vgg, 5, 3, mode='nearest')
    # nzr_gn = savgol_filter(nzr_gn, 5, 3, mode='nearest')
    # nzr_res = savgol_filter(nzr_res, 5, 3, mode='nearest')
    plt.tick_params(labelsize=18)
    plt.plot(xax, nzr_ax, 'o-', linewidth=2, label='AlexNet', zorder=10)
    plt.plot(xvgg, nzr_vgg, 'v-', linewidth=2, label='VGG16', zorder=10)
    plt.plot(xgn, nzr_gn, 'o--', linewidth=2, label='GoogLeNet', zorder=10)
    plt.plot(xres, nzr_res, '^:', linewidth=2, label='ResNet', zorder=10)
  #  plt.plot([0.5,0.5],[0,1], 'o--',linewidth=2, color='orchid')
    plt.legend(loc='best',fontsize=18)
    plt.xlabel('Nonzero-rate Distribution', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
   # plt.grid(axis="y", linestyle='-.', zorder=0)
    #plt.quiver(0.5, 0.5, -1, 0, color='r', scale=4, width=0.02)
   # plt.text(0.5, 0.1, 'better', fontsize=20, )
    plt.show()