"""读取一个LFCNZ文件，显示各层输出数据的数据量曲线：原始数据(不压缩/压缩)、差值压缩数据(实际值/预测值)
一个子图展示一个帧，横轴为层号，纵轴为数据量（单位MB）
"""
import pickle
from typing import List

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from core.predictor import Predictor
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.scheduler import SizedNode, Scheduler
from schedulers.my_scheduler import MyScheduler

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
lg = {'size': 16}
plt.rc('font',family='Times New Roman')
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rc('pdf', fonttype=42)
def predict_show(s_dag: List[SizedNode], predictors: List[Predictor], lfcnz: List[List[List[float]]]):
    SUB_NROW, SUB_NCOL = 3, 2
    nframe = len(lfcnz[0])
    nlayer = len(lfcnz)
    artery = MyScheduler.get_artery(s_dag)
    candidates = [at.id for at in artery if len(at.ancients) == 1]  # MyScheduler中的候选主干节点
    show_cads = False
    if nlayer != len(artery):  # 主干节点数!=总层数，说明有分叉
        show_cads = True
        print(f"DAG的主干节点: {candidates}")
    org_ucps = np.array([snd.out_size[0] * snd.out_size[1] * snd.out_size[2] for snd in s_dag])*4/1024/1024  # MB
    accuracy, reduction = [], []  # 各帧预测的平均精度, 各帧压缩减少的数据比例均值
    reduction_mx = 0  # 各帧中最多可以减少的数据传输量比例
    for f in range(nframe):
        if f//(SUB_NROW*SUB_NCOL) > 0 and f%(SUB_NROW*SUB_NCOL) == 0:
            print(f"Frame[{f-SUB_NROW*SUB_NCOL}:{f}] 预测精度均值= {np.average(np.array(accuracy))*100}% ,"
                  f"压缩减少的数据量均值= {np.average(np.array(reduction))*100}% ,"
                  f"单层数据最多可以减少传输的数据量比例= {reduction_mx*100}%")


        # print(f"Frame{f}: 输入数据cnz={lfcnz[0][f]}")
        lcnz_pred = Scheduler.predict_dag(lfcnz[0][f], s_dag, predictors)
        dif_pred = np.array(Scheduler.lcnz2lsz(lcnz_pred, s_dag))*4/1024/1024  # 预测值, MB
        lcnz_gt = [lfcnz[l][f] for l in range(nlayer)]
        dif_gt = np.array(Scheduler.lcnz2lsz(lcnz_gt, s_dag))*4/1024/1024  # 实际值, MB

        acr = np.average(1 - np.abs(dif_pred-dif_gt)/dif_gt)  # 各层精度的均值
        rdc = np.sum(org_ucps-dif_gt)/np.sum(org_ucps)  # 当前帧总共可以减少的数据传输量比例
        rdc_mx = np.max(1 - dif_gt/org_ucps)  # 单层上最多减少的数据传输量比例
        print(f"Frame{f}: 预测精度均值={acr*100}%, 当前帧共可减少{rdc*100}%的数据传输, "
              f"单层数据最多可减少{rdc_mx*100}%的数据传输")
        accuracy.append(acr)
        reduction.append(rdc)
        reduction_mx = max(reduction_mx, rdc_mx)
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.tick_params(labelsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        x=[i for i in range(len(dif_pred))]
        plt.plot(x,org_ucps, '--',linewidth=2,color='magenta',label='No Sparse Encoding',zorder=100) #原始数据
        plt.plot(x,dif_gt, '-',color='darkorange',linewidth=2,label='Using Sparse Encoding',zorder=100) #差值数据实际值
        #plt.plot(x, org_ucps, '-', linewidth=2, color='darkorange', label='No Sparse Encoding', zorder=100)  # 原始数据
        #plt.plot(x, dif_gt, '--', color='magenta', linewidth=2, label='Using Sparse Encoding', zorder=100)  # 差值数据实际值
        #plt.plot(x,dif_pred, '--',  label='差值数据预测值')#差值数据预测值
        plt.fill_between(x, org_ucps, dif_gt, color='b', hatch = '//',alpha=0.5, label='Volume Reduction', zorder=112)
        plt.xlabel('Index of CNN Layer',fontsize=18)
        plt.ylabel('Data Volume (MB)',fontsize=18)
        plt.legend(bbox_to_anchor=(0.25, 0.7),fontsize=16)
        #plt.legend(loc='best',fontsize=16)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(axis="y", linestyle='-.', zorder=0)

        plt.show()


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    NFRAME_SHOW = 7  # 展示数据集中的多少帧

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f".cache/{CNN_NAME}.{RESOLUTION}.sz", 'rb') as file:
        g_s_dag = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.pred", 'rb') as file:
        g_predictors = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as file:
        g_lfcnz = pickle.load(file)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    predict_show(g_s_dag, g_predictors, g_lfcnz)

    CNN_NAME = 'vg16'
    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f".cache/{CNN_NAME}.{RESOLUTION}.sz", 'rb') as file:
        g_s_dag = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.pred", 'rb') as file:
        g_predictors = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as file:
        g_lfcnz = pickle.load(file)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    predict_show(g_s_dag, g_predictors, g_lfcnz)

    CNN_NAME = 'gn'
    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f".cache/{CNN_NAME}.{RESOLUTION}.sz", 'rb') as file:
        g_s_dag = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.pred", 'rb') as file:
        g_predictors = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as file:
        g_lfcnz = pickle.load(file)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    predict_show(g_s_dag, g_predictors, g_lfcnz)


    CNN_NAME = 'rs50'
    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f".cache/{CNN_NAME}.{RESOLUTION}.sz", 'rb') as file:
        g_s_dag = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.pred", 'rb') as file:
        g_predictors = pickle.load(file)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as file:
        g_lfcnz = pickle.load(file)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    predict_show(g_s_dag, g_predictors, g_lfcnz)


