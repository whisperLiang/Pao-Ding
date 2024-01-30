import logging
import sys

import torch
from typing import List, Type, Set, Any

from torch import Tensor
from torch.nn import Module

from core.dnn_config import RawLayer, BlockRule, DNNConfig
from core.echarts_util import gen_html


class RawDNN:
    def __init__(self, dnn_config: DNNConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dnn_cfg = dnn_config
        self.layers = self.__make_dag(dnn_config.dnn, dnn_config.block_rules, self.logger)
        self.__visualize_dag(self.layers, f"{dnn_config.name}.html")
        self.logger.info(f"The DAG of {dnn_config.name} has {len(self.layers)} layers, "
                         f"visualized in {dnn_config.name}.html")

    def execute(self, ipt: Any) -> List[Tensor]:
        return self.__execute_dag(self.layers[0], ipt, [None for _ in self.layers])

    @classmethod
    def __make_dag(cls, dnn: Module, block_rules: Set[Type[BlockRule]],
                   logger: logging.Logger = None) -> List[RawLayer]:
        """处理dnn，根据block_rules的规则，生成相应的由RawLayer构成的DAG图，输出使用logger
        :param dnn 要处理的dnn
        :param block_rules 特殊Module的处理规则
        :param logger 使用此logger输出相关信息"""
        if logger is None:
            # 如果没有传入logger，则默认写入stdout
            logger = logging.getLogger('make_dag')
            if not logger.hasHandlers():
                # 因为make_dag这个logger是全局共用的，所以不能重复添加Handler
                logger.addHandler(logging.StreamHandler(sys.stdout))
                logger.setLevel(logging.INFO)
        layers = cls.__make_dag_recursive(dnn, 'dnn', block_rules, logger)
        for layer in layers:
            if isinstance(layer.module, torch.nn.ReLU):
                layer.module.inplace = False  # 确保各节点的输出数据是分开保存的
        return layers

    @classmethod
    def __execute_dag(cls, root: RawLayer, ipt: Any, results: List[Any]) -> List[Any]:
        """从root开始，以input_tensor为输入执行RawLayer组成的DAG，把各layer的计算结果放在results[root.id_]中
        results长度必须与总layer数相同"""
        if results[root.id_] is not None:  # 已经计算过，直接返回
            return results
        if len(root.ac_layers) <= 1:  # root为起始结点或链上结点，直接使用ipt计算
            with torch.no_grad():
                results[root.id_] = root.module(ipt)
        else:  # root有多个前驱，不使用ipt而使用results中的结果
            inputs = []
            for ac_layer in root.ac_layers:
                if results[ac_layer.id_] is not None:  # 不为空，已计算出
                    inputs.append(results[ac_layer.id_])  # 按顺序记录计算结果
                else:  # 这个前驱结点还没计算，root及其后继就先不计算，直接返回
                    return results
            with torch.no_grad():
                results[root.id_] = root.module(*inputs)  # 将所有前驱结果作为输入
        for ds_layer in root.ds_layers:
            results = cls.__execute_dag(ds_layer, results[root.id_], results)
        return results

    @classmethod
    def __make_dag_recursive(cls, dnn: Module, path: str, block_rules: Set[Type[BlockRule]],
                             logger: logging.Logger) -> List[RawLayer]:
        """处理dnn，当前dnn对应的路径为path，生成相应的RawLayer(已彻底展开，均为叶子结点)，RawLayer.id_从0开始编号
        :return 所有RawLayer，按照id顺序排序，确保List中索引与RawLayer.id_相同"""
        if len(list(dnn.children())) <= 0:  # 没有子结点，直接返回
            return [RawLayer(0, dnn, path, [], [])]
        block_rule = None
        for br in block_rules:
            if br.is_target(dnn):
                block_rule = br
                break
        if block_rule is not None:  # dnn属于特殊处理的Block
            children = block_rule.build_dag(dnn)  # 直接子结点，可能一些子结点尚未彻底展开
        else:  # 不属于特殊处理的Block，则默认所有子结点为顺序执行
            children = []
            for name, child_mod in dnn.named_children():
                children.append(RawLayer(len(children), child_mod, name, [], []))  # 这里只填子结点缩写，与block_rule保持一致
            if len(children) > 1:  # 若不止一个，则要建立前后关系
                children[0].ds_layers = [children[1]]
                for i in range(1, len(children)-1):
                    children[i].ac_layers = [children[i - 1]]
                    children[i].ds_layers = [children[i + 1]]
                children[-1].ac_layers = [children[-2]]
        module_end = {}  # 父结点->展开得到的子结点中的出口结点（即子结点列表的末尾元素）
        layers = []
        for child in children:
            child.module_path = path + '.' + child.module_path  # 添加路径前缀
            chd_layers = cls.__make_dag_recursive(child.module, child.module_path, block_rules, logger)
            module_end[child] = chd_layers[-1]
            if len(layers) > 0 and len(chd_layers) > 0:  # 前面有结点且当前存在子结点，chd_layers所有结点的id要向后平移，前后要建立连接
                offset = layers[-1].id_ + 1  # 偏移量
                for chd in chd_layers:
                    chd.id_ += offset
                # 入口RawLayer继承child的前驱，因为前面的child已经不再使用，所以要使用module_end映射到原先child对应的末尾RawLayer
                chd_layers[0].ac_layers.extend([module_end[ac] for ac in child.ac_layers])
                for ac in child.ac_layers:
                    module_end[ac].ds_layers.append(chd_layers[0])
            layers.extend(chd_layers)  # 添加进去
        return layers

    @staticmethod
    def __visualize_dag(layers: List[RawLayer], file_path: str) -> None:
        """使用echarts可视化生成的DAG图，写入dnn_layers.html"""
        e_nodes = [{"name": str(n.id_) + ',' + n.module_type(),
                    "symbolSize": [len(str(n.id_) + ',' + n.module_type()) * 7, 20],  # 这里使用文字长度*7来控制矩形宽度，高度恒为20
                    "tooltip": {"formatter": n.module_path}}  # 鼠标移动上去显示此模块的路径
                   for n in layers]  # 用于向echarts传结点参数，包括结点名称和标识点大小（标识点形状已经在echarts_util中设置为矩形）
        e_links = []  # 用于向echarts传边的参数
        for layer in layers:
            for ds in layer.ds_layers:
                e_links.append({"source": layer.id_, "target": ds.id_})
        gen_html(e_nodes, e_links, file_path)
