# 开发笔记
## 2023.11.1
- [x] 修复了torch.flatten(x, 1)这类ops._ReshapeOp类型的节点
- [x] 功能：将张量从通道维度平展
- [x] vgg分割成功
## 2023.11.5
- [x] 修复了<Node: (_ElementWiseOp_9(AddBackward0))>
- [x] 功能：将两个或者多个张量数据相加
- [x] ResNet自动分割测试成功
## 2023.11.8
- [x] 修复了<Node: (_ConcatOp_20(None))>
- [x] 将张量在通道维度进行拼接
- [x] yolov5的Detect模块无法进行自动化处理，因此将整个模块作为一个节点（182--141）
- [x] yolov5自动分割测试成功
## 2023.11.17
- [x] mobilenet_v2自动分割测试成功
- [x] googlenet自动分割测试成功
## 2023.11.19
- [x] 修复了<Node: (_Mean_20)>
- [x] 功能：适应性地池化或者在指定维度进行平均（densenet、mnasnet）
- [x] 完善了torch_split.py文件，完成了vgg, resnet, inception, regnet, efficientnet等56种DNN骨架模型自动分割的测试，覆盖率93.33%
## 2023.11.23
- [x] 解决了nn.dropout等模块与其他模块公用一个梯度，需要舍弃nn.dropout，导致理论上无法训练的问题
## 2023.11.26
- [x] 调研detectron2库里面检测和分割模型的提取
- [x] 思考通用有向无环图的模型分割方案
## 2023.12.03
- [x] yolov5中detect_after_split.py实现对yolov5的n,s,m,l,x五种规格的自动分割
- [x] yolov7中yolov7_detect_split.py实现对yolov7的六种规格的自动分割
## 2023.12.10
- [x] 调研比较mmdetection和detectron2
- [x] 理清协同推理中模型分割的技术方案
## 2023.12.17
- [x] 整理边云协同和边缘设备集群推理的相关论文
## 2023.12.24
- [x] [ToDo] mmdetection覆盖失败，因为它将数据预处理打包作为模型的一个模块，暂时搁置，比较容易解决
## 2024.1.1
- [x] [ToDo] 视觉transformer（VIT）覆盖失败，因为torchviosion库的vit很多连续的非模块的操作，不好处理，看看mmdetection的vit
## 2024.1.7
- [x] 绘制了系统的整体架构图
## 2024.1.14
- [x] 阅读了KDD21年顶会论文代码，准备复现。https://github.com/whisperLiang/Auto-Split
## 2024.1.21
- [x] Auto-Split并不适合无损协同推理，设计很多无关的量化知识，放弃
- [x] 复现了DADS，代码太过简单，不能支持视频流等复杂实验。：https://github.com/Tjyy-1223/DADS
## 2024.1.24
- [x] 复现了ResMap，调度策略在遇到resnet这种dag格式时报错，提交了issues。https://github.com/nju-cn/ResMap
## 2024.1.28
- [x] 实现了worker for cross cloud ladder,基于VGG自动提取的DAG计算图拓扑排序实现
## 2024.1.29
- [x] 实现了worker for cross cloud ladder,基于resnet50自动提取的DAG计算图拓扑排序实现
## 2024.1.30
- [x] 实现了trainer for cross cloud ladder
## 2024.2.1
- [x] 在层拓扑排序中添加了输入节点
- [x] 成功实现了scheduler的初始化
- [x] 实现了predictors only for relu, 三种拟合方式
## 2024.2.2
- [x] 实现了master for cross cloud ladder，相对于原始resmap，可以支持DAG格式的调度
## 2024.2.4
- [x] 消除共用梯度引起Trainer出bug的问题
## 2024.2.5
- [x] 树莓派与服务器协同测试成功
## 2024.2.6
- [x] [实验] none_zero_rate_of_one_frame.py：原始特征非零率显示成功
- [x] [实验] none_zero_of_two_frames.py：残差特征非零率显示成功
- [x] [实验] lcnz_show.py: 各层各通道非零率显示成功