"""读取一个dataset下的前后两张图像，展示它们在CNN各层输出数据差值的非零占比变化曲线
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms
from torch import nn
from matplotlib import pyplot as plt


def get_ipt(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    print(input_tensor.shape)
    return input_tensor.unsqueeze(0)


def show_sparse(layer_name, raw, dif):
    assert raw.nelement() == dif.nelement()
    r = round(float(torch.count_nonzero(raw) / raw.nelement() * 100), 2)
    d = round(float(torch.count_nonzero(dif) / dif.nelement() * 100), 2)
    print(layer_name, r, '->', d)
    return r, d


if __name__ == '__main__':
    CNN_NAME = 'gn'

    if CNN_NAME == 'ax':
        cnn = models.alexnet(True).features
    elif CNN_NAME == 'vg16':
        cnn = models.vgg16(True).features
    elif CNN_NAME == 'rs50':
        resnet = models.resnet50(True)
        cnn = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                            resnet.avgpool)
    elif CNN_NAME == 'gn':
        gnet = models.googlenet(True)
        cnn = nn.Sequential(gnet.conv1.conv, gnet.conv1.bn, gnet.maxpool1,
                            gnet.conv2.conv, gnet.conv2.bn, gnet.conv3.conv, gnet.conv3.bn, gnet.maxpool2,
                            gnet.inception3a, gnet.inception3b, gnet.maxpool3,
                            gnet.inception4a, gnet.inception4b, gnet.inception4c, gnet.inception4d, gnet.inception4e,
                            gnet.maxpool4, gnet.inception5a, gnet.inception5b, gnet.avgpool, gnet.dropout)
    else:
        raise Exception(f"Unknown CNN name: {CNN_NAME}")
    cnn.eval()
    ft1 = get_ipt("dataset/image1.jpg")
    ft2 = get_ipt("dataset/image2.jpg")
    r_, d_ = show_sparse("input", ft2, ft2 - ft1)
    plt.show()
    raw_nz, dif_nz = [r_], [d_]
    layers = ["input"]
    with torch.no_grad():
        for i in range(len(cnn)):
            ft1 = cnn[i](ft1)
            ft2 = cnn[i](ft2)
            layer_name = cnn[i]._get_name()
            r_, d_ = show_sparse(layer_name, ft2, ft2 - ft1)
            raw_nz.append(r_)
            dif_nz.append(d_)
            layers.append(layer_name)
    if CNN_NAME == 'rs50':
        layers[-4:] = ["layer" + str(i) for i in range(1, 5)]  # ResNet
    print(f"raw_nz: {raw_nz}")
    print(f"dif_nz: {dif_nz}")
    _, ax = plt.subplots()
    ax.set_xticks(list(range(0, len(layers))))
    ax.set_xticklabels(layers, rotation=20)
    plt.plot(raw_nz, 'r*-')
    plt.plot(dif_nz, 'b*-')
    plt.hlines(49.5, 0, len(layers)-1, linestyles='dashed')
    plt.vlines(range(len(layers)), [0]*len(layers), raw_nz, linestyles='dashed')
    plt.show()
