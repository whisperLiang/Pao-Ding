# 开发笔记
## 2023.11.1
- [x] 修复了torch.flatten(x, 1)这类ops._ReshapeOp类型的节点
- [x] 功能：将张量从通道维度平展
- [x] vgg分割成功
## 2023.11.5
- [x] 修复了<Node: (_ElementWiseOp_9(AddBackward0))>
- [x] 功能：将两个或者多个张量数据相加
- [x] ResNet测试成功
## 2023.11.8
- [x] 修复了<Node: (_ConcatOp_20(None))>
- [x] 将张量在通道维度进行拼接
- [x] yolov5的Detect模块无法进行自动化处理，因此将整个模块作为一个节点（182--141）
- [x] yolov5测试成功
## 2023.11.17
- [x] mobilenet_v2测试成功
- [x] googlenet测试成功
## 2023.11.19
- [x] 修复了<Node: (_Mean_20)>
- [x] 功能：适应性地池化或者在指定维度进行平均（densenet、mnasnet）
- [x] 完善了torch_split.py文件，45种模型测试成功
## 2023.11.23
- [x] 解决了nn.dropout等模块与其他模块公用一个梯度，需要舍弃nn.dropout，导致理论上无法训练的问题
## 2024.4.24
- [x] modify computational dependency graph function to plot
- [x] mkdir a folder to store the computational dependency graph
