"""读取LFCNZ格式的数据，可视化
"""
import operator
import pickle
import sys
from functools import reduce
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm

from core.executor import Node
from core.predictor import Predictor
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from core.raw_dnn import RawDNN
from master.scheduler import Scheduler, SizedNode
from trainer.trainer import Trainer


def chan_err_mtrx(raw_dnn: RawDNN, lfcnz: List[List[List[float]]]):
    # 逐层展示各通道的预测误差
    NFRAME_TRAIN = 200  # 前多少帧的数据用于训练

    nframe_total = len(lfcnz[0])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=.2)  # colorbar范围统一
    for lid in range(1, len(raw_dnn.layers)):  # 从InputModule后面开始
        pedor = raw_dnn.dnn_cfg.mdl2pred[raw_dnn.layers[lid].module.__class__](raw_dnn.layers[lid].module)
        pedor.fit([lfcnz[al.id_][:NFRAME_TRAIN] for al in raw_dnn.layers[lid].ac_layers], lfcnz[lid][:NFRAME_TRAIN])
        fcnz_prd = []
        for f in range(NFRAME_TRAIN, nframe_total):
            fcnz_prd.append(pedor.predict([lfcnz[al.id_][f] for al in raw_dnn.layers[lid].ac_layers]))
        fcnz_prd = np.array(fcnz_prd)
        fcnz_trh = np.array(lfcnz[lid][NFRAME_TRAIN:nframe_total])
        fcnz_err = np.abs(fcnz_prd - fcnz_trh)
        # 可视化
        fig = plt.figure()
        # 窗口最大化，便于观察
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # 绘制图像
        plt.imshow(fcnz_err, norm=norm)
        plt.colorbar()
        fig.suptitle(f"Layer{lid}: " + raw_dnn.layers[lid].module_type())
        plt.show()


def layer_flnz_mtrx(raw_dnn: RawDNN, lfcnz: List[List[List[float]]]):
    # 逐帧展示各层的整体误差
    NFRAME_TRAIN = 200  # 前多少帧的数据用于训练
    NFRAME_PRED = 100  # 对后面的多少帧进行预测
    ENABLE_FRAME = False

    print("training...", file=sys.stderr)
    predictors = Trainer.train_predictors(raw_dnn, [fcnz[:NFRAME_TRAIN] for fcnz in lfcnz])

    print("predicting...", file=sys.stderr)
    dag = Node.raw2dag(raw_dnn.layers)
    fl_err = []
    flnz_trh, flnz_prd = [], []
    for fid in tqdm(range(NFRAME_TRAIN, NFRAME_TRAIN+NFRAME_PRED)):
        # 真实值
        lnz_trh = [sum(lfcnz[l][fid])/len(lfcnz[l][fid]) for l in range(len(lfcnz))]
        flnz_trh.append(lnz_trh)
        cnz = lfcnz[0][fid]
        lcnz_prd = Scheduler.predict_dag(cnz, dag, predictors)
        lnz_prd = [sum(lcnz_prd[l])/len(lcnz_prd[l]) for l in range(len(lcnz_prd))]
        flnz_prd.append(lnz_prd)
        if ENABLE_FRAME:
            plt.plot(lnz_trh, 'r')
            plt.plot(lnz_prd, 'b')
            plt.title(f'frame{fid}')
            plt.show()
        # 计算误差
        fl_err.append(np.abs(np.array(lnz_prd) - np.array(lnz_trh)))
    fl_err = np.array(fl_err)
    # 误差图
    plt.figure()
    plt.imshow(fl_err)
    plt.colorbar()
    plt.title('error')
    plt.show()
    # 真实值
    plt.figure()
    plt.imshow(flnz_trh)
    plt.colorbar()
    plt.title('truth')
    plt.show()
    # 预测值
    plt.figure()
    plt.imshow(flnz_prd)
    plt.colorbar()
    plt.title('predict')
    plt.show()


def layer_size_mtrx(raw_dnn: RawDNN, lfcnz: List[List[List[float]]], frame_size: Tuple[int, int]):
    # 展示各层的原始数据量base，实际的压缩后数据量truth，预测的压缩后数据量predict
    NFRAME_TRAIN = 200  # 前多少帧的数据用于训练
    NFRAME_PRED = 100  # 对后面的多少帧进行预测

    print("training...", file=sys.stderr)
    predictors = Trainer.train_predictors(raw_dnn, [fcnz[:NFRAME_TRAIN] for fcnz in lfcnz])

    print("predicting...", file=sys.stderr)
    s_dag = SizedNode.raw2dag_sized(raw_dnn, frame_size)
    thresholds = []
    for s_node in s_dag:
        _, R, C = s_node.out_size
        thresholds.append((1 - 1/C - 1/(R*C))/2)
    fls_trh, fls_prd = [], []  # Frame Layer Size：元素个数
    for fid in tqdm(range(NFRAME_TRAIN, NFRAME_TRAIN + NFRAME_PRED)):
        # 真实值
        lcnz_trh = [fcnz_[fid] for fcnz_ in lfcnz]
        fls_trh.append(Scheduler.lcnz2lsz(lcnz_trh, s_dag))
        cnz = lfcnz[0][fid]
        lcnz_prd = Scheduler.predict_dag(cnz, s_dag, predictors)
        fls_prd.append(Scheduler.lcnz2lsz(lcnz_prd, s_dag))
    fls_trh, fls_prd = np.array(fls_trh)*4/1024/1024, np.abs(fls_prd)*4/1024/1024  # 单位：MB
    fls_err = np.abs(fls_prd-fls_trh)
    plt.imshow(fls_err)
    plt.colorbar()
    plt.title(f"Size mean_error={round(float(np.mean(fls_err)), 2)} MB")
    plt.show()
    fls_bas = np.array([[reduce(operator.mul, s_node.out_size)*4/1024/1024 for s_node in s_dag]]*NFRAME_PRED)
    title_data = {'base': fls_bas, 'truth': fls_trh, 'predict': fls_prd,
                  'base-truth': fls_bas-fls_trh, 'base-predict': fls_bas-fls_prd}
    for i, (title, data) in enumerate(title_data.items()):
        ax = plt.subplot(1, 5, i+1)
        im = ax.imshow(data)
        plt.colorbar(im)
        # title中的数字：矩阵中全部的数据量
        ax.set_title(f'{title}:{round(float(np.sum(data)), 2)}MB')
    plt.show()


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    MODE = 'sz'
    NFRAME_TOTAL = 400  # 数据集中的帧数

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    g_raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    with open(f'dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz', 'rb') as lfile:
        g_lfcnz = pickle.load(lfile)
    if MODE == 'ch':
        chan_err_mtrx(g_raw_dnn, g_lfcnz)
    elif MODE == 'ly':
        layer_flnz_mtrx(g_raw_dnn, g_lfcnz)
    elif MODE == 'sz':
        layer_size_mtrx(g_raw_dnn, g_lfcnz, tuple(map(int, RESOLUTION.split('x'))))
