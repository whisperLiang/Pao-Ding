from torch import nn,randn
from torchvision import models

from model_split import DependencyGraph


def prepare_split_model(example_inputs, model: nn.Module) -> DependencyGraph:
    """通过反向传播，生成依赖图"""
    dpg = DependencyGraph().build_dependency(model, example_inputs)
    return dpg


def prepare_alexnet() -> DependencyGraph:
    """准备AlexNet相关参数"""
    alexnet = models.alexnet(True)
    alexnet.eval()
    example_inputs = randn(1,3,224,224)
    return prepare_split_model(example_inputs, alexnet)


def prepare_vgg16() -> DependencyGraph:
    """准备VGG16相关参数"""
    vgg16 = models.vgg16(True)
    vgg16.eval()
    example_inputs = randn(1,3,224,224)
    return prepare_split_model(example_inputs, vgg16)


def prepare_resnet50() -> DependencyGraph:
    """准备resnet50相关参数"""
    resnet50 = models.resnet50(True)
    resnet50.eval()
    example_inputs = randn(1,3,224,224)
    return prepare_split_model(example_inputs, resnet50)

def prepare_resnet18() -> DependencyGraph:
    """准备resnet18相关参数"""
    resnet18 = models.resnet18(True)
    resnet18.eval()
    example_inputs = randn(1,3,224,224)
    return prepare_split_model(example_inputs, resnet18)

def prepare_googlenet() -> DependencyGraph:
    googlenet = models.googlenet(True)
    googlenet.eval()
    example_inputs = randn(1, 3, 224, 224)
    return prepare_split_model(example_inputs, googlenet)

def prepare_efficientnet() -> DependencyGraph:
    effcientnet = models.efficientnet_b0(True)
    effcientnet.eval()
    example_inputs = randn(1, 3, 224, 224)
    return prepare_split_model(example_inputs, effcientnet)

def prepare_convnext() -> DependencyGraph:
    convnext = models.convnext_tiny(True)
    convnext.eval()
    example_inputs = randn(1, 3, 224, 224)
    return prepare_split_model(example_inputs, convnext)

if __name__ == '__main__':
    layer_topo = prepare_vgg16()