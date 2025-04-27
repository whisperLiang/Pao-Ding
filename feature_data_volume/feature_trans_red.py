import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import ReLU
from core.dag_dnn import DagDNN
from dnn_models.any_dnn_split import prepare_alexnet, prepare_vgg16
from collections import deque

def calculate_volume(fcnz, feature_shape, threshold=0.5):
    """根据公式计算原始体积和压缩体积"""
    orig_volumes = []
    comp_volumes = []

    # 提取特征形状信息
    if len(feature_shape) == 4:
        batch_size, channels, height, width = feature_shape
        total_elements = batch_size * channels * height * width  # 特征的总元素数

        # 原始体积（不使用CSR编码）
        orig_volume = total_elements * 4 / (1024 ** 2)  # 转换为MB
        orig_volumes.append(orig_volume)

        # 压缩体积（逐通道计算）
        csr_volume = 0
        for cnz in fcnz:
            channels_volume = 0
            for nz in cnz:
                # 非零率判断函数 φ(Q_{f,l}^s)
                phi = 0.5 * (np.sign(nz - threshold) + 1)  # 1 if cnz >= threshold, else 0

                # 单通道的压缩体积
                channel_volume = (
                    (2 * height * width * nz + height + 1) * (1 - phi)
                    + height * width * phi
                )
                channels_volume += channel_volume
            csr_volume += channels_volume

        # 平均压缩体积
        csr_volume = csr_volume / len(fcnz)  # 平均每帧的压缩体积
        csr_volume = csr_volume * batch_size * 4 / (1024 ** 2)  # 转换为MB
        comp_volumes.append(csr_volume)
    else:
        # 对于其他形状的特征，直接使用原始体积作为压缩体积
        orig_volume = np.prod(feature_shape) * 4 / (1024 ** 2)  # 转换为MB
        orig_volumes.append(orig_volume)
        comp_volume = orig_volume
        comp_volumes.append(comp_volume)

    return np.array(orig_volumes), np.array(comp_volumes)

def process_layer_data(dag_dnn, lfcnz, olfcnz, threshold=0.5):
    """处理所有层数据，计算原始体积和压缩体积"""
    orig_volumes, comp_volumes = [], []

    for layer, fcnz, ofcnz in zip(dag_dnn.layers, lfcnz, olfcnz):
        feature_shape = layer.outshape  # 获取特征形状
        if isinstance(feature_shape, deque):
            feature_shape = feature_shape.pop()  # 获取最后一个元素

        if isinstance(layer.module, ReLU):
            # ReLU层使用特征残差的非零率
            orig_volume, comp_volume = calculate_volume(fcnz, feature_shape, threshold)
        else:
            # 其他层使用原始特征的非零率
            orig_volume, comp_volume = calculate_volume(ofcnz, feature_shape, threshold)

        orig_volumes.append(orig_volume)
        comp_volumes.append(comp_volume)

    # 将列表展平为一维数组
    return np.concatenate(orig_volumes), np.concatenate(comp_volumes)

def plot_results(layer_indices, original_volumes, compressed_volumes, title):
    """绘制结果图"""
    # 确保输入为一维数组
    original_volumes = np.array(original_volumes).flatten()
    compressed_volumes = np.array(compressed_volumes).flatten()

    # 计算体积减少量
    volume_reduction = original_volumes - compressed_volumes
    total_reduction = np.sum(volume_reduction)  # 总体积减少量

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, original_volumes, 'b--', label='No Sparse Encoding')
    plt.plot(layer_indices, compressed_volumes, 'green', label='Using Sparse Encoding')
    plt.fill_between(layer_indices, compressed_volumes, original_volumes, color='green', alpha=0.1, label='Volume Reduction')

    # 在图像中显示总减少量
    plt.text(0.5, 0.91, f'Total Reduction: {total_reduction:.2f} MB', transform=plt.gca().transAxes, fontsize=12, color='red')

    plt.title(title)
    plt.xlabel('Index of CNN Layer')
    plt.ylabel('Data Volume (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimized_data_volume_combined.png')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    CNN_NAME = 'AlexNet'  # 可选：AlexNet, VGG
    VIDEO_NAME = 'parking'
    RESOLUTION = '480x720'
    NFRAME_TOTAL = 400
    THRESHOLD = 0.4  # 非零率阈值

    cnn_loaders = {'AlexNet': prepare_alexnet, 'VGG': prepare_vgg16}
    dag_dnn = DagDNN(cnn_loaders[CNN_NAME]())

    # 加载原始特征和差值特征
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.o_lfcnz", 'rb') as f:
        olfcnz = pickle.load(f)
    with open(f".cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        lfcnz = pickle.load(f)

    # 处理所有层
    original_volumes, compressed_volumes = process_layer_data(dag_dnn, lfcnz, olfcnz, THRESHOLD)

    # 绘制图像
    layer_indices = np.arange(len(original_volumes))
    plot_results(layer_indices, original_volumes, compressed_volumes, f'Data Volume of {CNN_NAME}')