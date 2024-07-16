"""读取一个LFCNZ文件，显示每层的输入差值非零占比和输出差值非零占比的关系
每层是一个单独的图，横轴为输入差值非零占比，纵轴为输出差值非零占比
"""
import pickle
import sys
from typing import List, Type

import torch.nn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from core.dag_dnn import DagDNN
from model_split import Node
from core.predictor import Predictor
from dnn_models.any_dnn_split import prepare_alexnet, prepare_vgg16, prepare_resnet50, prepare_googlenet
from trainer.trainer import Trainer

plt.rc('font',family='Times New Roman')
import matplotlib
matplotlib.rc('pdf', fonttype=42)

#plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

def lfcnz2lfnz(lfcnz: List[List[List[float]]]) -> List[List[float]]:
    """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
    return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]


def draw_predictor(predictor: Predictor, i_fcnz: List[List[float]], o_fnz: List[float], ax: Axes):
    """对于特定的层，使用predictor预测各帧输出数据的总体非零占比，并绘制到ax上
    注意：这里只处理只有一个前驱的层，不处理多前驱的层
    """
    p_fnz = [0. for _ in i_fcnz]
    for f, cnz in enumerate(i_fcnz):
        p_cnz = predictor.predict([i_fcnz[f]])
        p_fnz[f] = sum(p_cnz) / len(p_cnz)
    i_fnz = [sum(cnz)/len(cnz) for cnz in i_fcnz]
    xarr, yarr = list(zip(*sorted(zip(i_fnz, p_fnz))))
    ax.plot(xarr, yarr, 'g-')
    ax.set_xlabel(ax.get_xlabel() + f" err={round(float(np.sum(np.abs(np.array(o_fnz)-np.array(p_fnz)))), 2)}")


def draw_fit3(i_fnz: List[float], o_fnz: List[float], ax: Axes):
    """对于特定的层，使用三次函数对各帧输出数据的总体非零占比进行拟合，并绘制到ax上
    注意：这里只处理只有一个前驱的层，不处理多前驱的层
    """
    X = np.array(i_fnz).reshape(-1, 1)
    y = np.array(o_fnz)
    X_tr = PolynomialFeatures(degree=3).fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_tr, y)
    y_pred = lr.predict(X_tr)
    xarr, yarr = list(zip(*sorted(zip(i_fnz, y_pred))))
    ax.plot(xarr, yarr, 'g-')
    ax.set_xlabel(ax.get_xlabel() + f" err={round(float(np.sum(np.abs(np.array(o_fnz)-y_pred))), 2)}")


def draw_mlp(i_fnz: List[float], o_fnz: List[float], ax: Axes):
    """对于特定的层，使用感知机对各帧输出数据的总体非零占比进行拟合，并绘制到ax上
    注意：这里只处理只有一个前驱的层，不处理多前驱的层
    """
    mlp = MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500)
    X, y = np.array(i_fnz).reshape(-1, 1), np.array(o_fnz)
    mlp.fit(X, y)
    y_pred = mlp.predict(X)
    xarr, yarr = list(zip(*sorted(zip(i_fnz, y_pred))))
    ax.plot(xarr, yarr, 'g-')
    ax.set_xlabel(ax.get_xlabel() + f" err={round(float(np.sum(np.abs(y - y_pred))), 2)}")


def draw_logistic(i_fnz: List[float], o_fnz: List[float], ax: Axes):
    """对于特定的层，使用Logistic函数对各帧输出数据的总体非零占比进行拟合，并绘制到ax上
    注意：这里只处理只有一个前驱的层，不处理多前驱的层
    """
    xarr, yarr = list(zip(*sorted(zip(i_fnz, o_fnz))))
    xarr, yarr = np.array(xarr), np.array(yarr)
    func = lambda x, k, p, r: (k*p*np.exp(r*x))/(k+p*(np.exp(r*x)-1))  # logistic函数
    popt, pcov = curve_fit(func, xarr, yarr, maxfev=50000)
    yarr_pred = func(xarr, *popt)
    ax.plot(xarr, yarr_pred, 'g-', label='Logistic Fit')
    # ax.set_xlabel(ax.get_xlabel() + f" err={round(float(np.sum(np.abs(yarr - yarr_pred))), 2)}")

lg = {'size': 20}

def target_layers_in_out(cnn_name: str, target_type: Type[torch.nn.Module], uni_scale: bool, show_seq: bool,
                  r_layers: List[Node], lfcnz: List[List[List[float]]], fit: str = None):
    """对于特定类型的所有层，显示输入和输出的关联。一个窗口展示3*5=15个层的数据
    target_type为要查看的layer类型
    uni_scale为是否把刻度统一到[0, 1]区间
    show_seq为是否用点的颜色表示帧的顺序
    fit为使用什么拟合，''不拟合，'predictor'使用Trainer的Predictor拟合，'fit3'使用三次函数拟合
    """
    SUB_NROW, SUB_NCOL = 1, 2
    if fit == 'predictor':
        print("training predictor...", file=sys.stderr)
        predictors = Trainer.train_predictors(dag_dnn, [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz])
    else:
        predictors = None
    lfnz = lfcnz2lfnz(lfcnz)
    nframe = len(lfnz[0])
    cnt = 1
    ahead_relu_l = 0
    print(f"plotting {fit}...", file=sys.stderr)
    for l in range(len(r_layers)):
        layer = r_layers[l].module
        if not isinstance(layer, target_type):
            continue
        xlabel = f"{l}-th Relu Layer"
        ax = plt.subplot(SUB_NROW, SUB_NCOL, cnt)
        ax.set_title(xlabel, lg)
        if uni_scale:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect(1)
            ax.tick_params(labelsize=18)
        i_fnz = lfnz[ahead_relu_l][1:]
        o_fnz = lfnz[l][1:]
        ahead_relu_l = l
        if show_seq:
            plt.scatter(i_fnz, o_fnz, s=2,
                        c=[i / nframe for i in range(nframe-1)],
                        marker='.', cmap='viridis')
        else:
            plt.scatter(i_fnz, o_fnz, s=2, c='lightblue')
        if predictors is not None:
            draw_predictor(predictors[l], i_fnz, o_fnz, ax)
        if fit == 'fit3':
            draw_fit3(i_fnz, o_fnz, ax)
        elif fit == 'mlp':
            draw_mlp(i_fnz, o_fnz, ax)
        elif fit == 'lgi':
            draw_logistic(i_fnz, o_fnz, ax)
        elif fit == '':
            pass
        cnt += 1
        # plt.legend(loc='upper left')
        if cnt > SUB_NROW*SUB_NCOL:
            cnt = 1
            plt.figure(figsize=(8 ,4))
            plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95, bottom=0.1, top=0.9)  # 调整子图间距&去除空白区域


# 控制图片大小的方法：手动调整窗口大小，达到和论文图的相同大小即可
if __name__ == '__main__':
    CNN_NAME = 'ResNet'  #AlexNet, VGG, GoogLeNet, ResNet
    VIDEO_NAME = 'parking'  # road, campus, parking
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数

    NFRAME_SHOW = NFRAME_TOTAL  # 展示数据集中的多少帧
    TARGET_TYPE = 'rl'
    UNI_SCALE = True  # 是否统一刻度到[0, 1]区间
    SEQ_FRAME = False  # 是否用点的颜色表示帧的顺序

    # 拟合方法：predictor，fit3，mlp，lgi。''表示不拟合
    FITS = ['lgi']  # 用哪些方式对NFRAME_SHOW进行拟合（训练集也是NFRAME_SHOW）

    cnn_loaders = {'AlexNet': prepare_alexnet,
                   'VGG': prepare_vgg16,
                   'GoogLeNet': prepare_googlenet,
                   'ResNet': prepare_resnet50}
    dag_dnn = DagDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = dag_dnn.layers
    target_types = {'cv': torch.nn.Conv2d, 'rl': torch.nn.ReLU, 'mp': torch.nn.MaxPool2d}
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        g_lfcnz = pickle.load(f)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]

    for g_fit in FITS:
        fig = plt.figure(figsize=(8 ,4)) # figsize=(8 ,4)
        plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95, bottom=0.1, top=0.9)  # 调整子图间距&去除空白区域
        #fig.suptitle(g_fit)
        target_layers_in_out(CNN_NAME, target_types[TARGET_TYPE], UNI_SCALE, SEQ_FRAME,
                             g_r_layers, g_lfcnz, g_fit)
    plt.show()
