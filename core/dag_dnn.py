import logging
import sys

import torch
from typing import List, Any, Dict, Type
from core.predictor import Predictor, MLPPredictor, LNRreluPredictor, MLPsPredictor

from torch import Tensor
from torch.nn import Module
from core.echarts_util import gen_html
from model_split import DependencyGraph, Node
from collections import deque
from torchvision.models.detection.image_list import ImageList
from networkx import DiGraph

class _InputmoduleOp(Module):
    def __init__(self):
        super(_InputmoduleOp, self).__init__()

    def __repr__(self):
        return "_Inputmodule_()"
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class DagDNN:
    def __init__(self, dpg: DependencyGraph):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = dpg.model.__class__.__name__
        self.node2index, self.layers = self._make_layerstopo(dpg, dpg.example_inputs, ignored_blocks=[], logger=self.logger)
        # ToDo: 生成DAG可视化图
        # self.__visualize_dag(self.layers, f"{self.model_name}_strucure.html")
        self.mdl2pred: Dict[Type[Module], Type[Predictor]] \
        = {torch.nn.ReLU: LNRreluPredictor}
        self.logger.info(f"The DAG of DNN has {len(self.layers)} layers, "
                         f"visualized in {self.model_name}_strucure.html")

    def execute(self, ipt: Any) -> List[Tensor]:
        return self.__execute_dag(self.layers, ipt, [None for _ in self.layers], self.node2index)
    
    @classmethod
    def _make_layerstopo(cls, dpg: DependencyGraph, x: Tensor, ignored_blocks=[], logger: logging.Logger = None) -> (Dict[Node, int], List[Node]):
        """根据依赖图，生成层拓扑排序结构
        :param dpg 要处理的依赖图
        :param x 用于生成层次拓扑结构的输入
        :param ignored_blocks 特殊Module的处理规则
        :param logger 使用此logger输出相关信息
        :return node2index 生成的节点到索引的映射
        :return layer_topo 生成的层次拓扑结构"""

        original_res = dpg.model(x)        
        if logger is None:
            # 如果没有传入logger，则默认写入stdout
            logger = logging.getLogger('make_layers_topo')
            if not logger.hasHandlers():
                # 因为make_layers_topo这个logger是全局共用的，所以不能重复添加Handler
                logger.addHandler(logging.StreamHandler(sys.stdout))
                logger.setLevel(logging.INFO)
        
        ignored_blocks_node_list = []
        # Step 1: create node for ignored layers
        if len(ignored_blocks) > 0:
            for ignored_block in ignored_blocks:
                ignored_layers_node = Node(module=ignored_block,
                                        name = dpg._module2name.get(ignored_block, None))
                if 'Detect' in ignored_layers_node.name:
                    ignored_layers_node.indegree = 3
                else:
                    ignored_layers_node.indegree = 1
                
                ignored_blocks_node_list.append(ignored_layers_node)
                for m in ignored_block.modules():
                    ignored_layers_node = Node(module=m,
                                            name = dpg._module2name.get(m, None))
                    ignored_blocks_node_list.append(ignored_layers_node.name)
                    
        # Step 2: Initialize a queue with nodes having in-degree 0
        queue = deque([dpg.head_node])
        if dpg.gradfn2node['None']:
            queue.append(dpg.gradfn2node['None'][0])

        # Step 3: Perform topological sorting
        layer_topo = []
        node2index = {}

        # 创建输入节点
        inputmodule = _InputmoduleOp()
        oinshape = x.shape
        inputnode = Node(
            module=inputmodule,
            grad_fn=None,
            inshape=oinshape,
            outshape=oinshape
        )
        inputnode.add_output([dpg.head_node])
        dpg.head_node.add_input([inputnode])
        layer_topo.append(inputnode)
        node2index[inputnode] = len(layer_topo) - 1

        while queue:
            
            # node = queue.popleft()
            # for correct sequence
            node = queue.pop()

            # Step 4: Update the in-degree of neighbors and enqueue if in-degree becomes 0
            for index, o_nodelist in enumerate(node.outputs):
                if len(o_nodelist) > 1:
                    for ind in range(len(o_nodelist)-1):
                        if [o_nodelist[ind+1]] not in o_nodelist[ind].outputs:
                            o_nodelist[ind].outputs.append([o_nodelist[ind+1]])
                            o_nodelist[ind].outdegree += 1
                            o_nodelist[ind].fixedoutd += 1
                        if [o_nodelist[ind]] not in o_nodelist[ind+1].inputs:
                            o_nodelist[ind+1].inputs.append([o_nodelist[ind]])
                            o_nodelist[ind+1].indegree += 1
                    node.outputs[index] = [o_nodelist[0]] # 更新输出节点列表

                for o_node in o_nodelist:
                    if o_node.name in ignored_blocks_node_list:
                        node.outputs.remove([o_node])
                        node.outputs.append([ignored_blocks_node_list[0]])
                        ignored_blocks_node_list[0].inputs.append([node])
                        ignored_blocks_node_list[0].indegree -= 1
                        if ignored_blocks_node_list[0].indegree == 0:
                            queue.append(ignored_blocks_node_list[0])
                    else:
                        o_node.indegree -= 1
                        not_satisfied = False # 判断当前节点的输入节点是否满足条件
                        for i_nodelist in o_node.inputs:
                            for i_node in i_nodelist:
                                if i_node.indegree > 0 or i_node in queue:
                                    not_satisfied = True
                                    break
                                # 如果遍历到最后一个输入节点，且所有输入节点入度为0，将其加入到队列中
                                if i_node == o_node.inputs[-1][-1] and o_node not in queue:
                                    o_node.indegree = 0
                                    queue.append(o_node)

                            if not_satisfied:
                                break


            # Step 5: forward the layer by layer
            if len(layer_topo) > 0:
                for i_nodelist in node.inputs:
                    if layer_topo[-1] in i_nodelist:
                        break
                    if i_nodelist == node.inputs[-1]:
                        x = node.inputs[0][-1].outresult
            # if len(layer_topo) > 0 and (layer_topo[-1] not in i_nodelist for i_nodelist in node.inputs):
            #     x = node.inputs[0][-1].outresult
            
            # 将输入按序传入到节点中
            x_ = []
            for index, i_nodelist in enumerate(node.inputs):
                if i_nodelist[-1].outdegree > 0:
                    if i_nodelist[-1].outresult is not None:
                        x_.append(i_nodelist[-1].outresult)
                else:
                    x_.append(x)
                # 更新输入节点列表
                if len(i_nodelist) > 1:
                    node.inputs[index] = [i_nodelist[-1]]
            x = node.forward(x_ if len(x_) > 1 else x)

            if isinstance(x, tuple) and isinstance(x[0], ImageList):
                x = x[0].tensors

        
            # 将节点加入到layer_topo中
            layer_topo.append(node)
            node2index[node] = len(layer_topo) - 1
            # 判断当前节点的输出是否是下一个队列节点的输入
            if len(queue) > 0:
                for o_nodelist in node.outputs:
                    if queue[-1] in o_nodelist:
                        node.outdegree -= 1
            # if len(queue) > 0 and queue[-1] in node.outputs:
            #     node.outdegree -= 1
            # 出度大于0保存当前节点的输出结果
            if node.outdegree > 0:
                node.outresult = x
                # node.outresult = x.clone().detach()
            
            # 将当前节点的输入节点的出度减1，如果出度为0，将其输出结果置空
            for i_nodelist in node.inputs:
                if i_nodelist[-1].outdegree > 0 and i_nodelist[-1] != layer_topo[-2]:
                    i_nodelist[-1].outdegree -= 1
                    if i_nodelist[-1].outdegree == 0:
                        i_nodelist[-1].outresult = None
                    

        # Step 6: Check if the graph is a DAG (no cycles) and if the topological order is correct
        if layer_topo[-1] == dpg.tail_node and torch.allclose(original_res, x):
            print("The graph is a DAG, and the topological order is correct.")

        return node2index, layer_topo

    @classmethod
    def _make_dag(cls, layers_topo: List[Node]) -> DiGraph:
        """根据已经拓扑排序生成用DiGraph表示的DAG
        :param layers_topo 拓扑排序后的层次结构
        :return dag 生成的DAG"""
        dag = DiGraph()
        # Add nodes to the DAG
        for node in layers_topo:
            dag.add_node(node)
        
        # Add edges between nodes based on their inputs and outputs
        for node in layers_topo:
            for onodelist in node.outputs:
                for onode in onodelist:
                    dag.add_edge(node, onode)
        
        return dag


    @classmethod
    def __execute_dag(cls, layers: List[Node], ipt: Any, results: List[Any], node2index: Dict[Node, int]) -> List[Any]:
        """从root开始，以input_tensor为输入执行RawLayer组成的DAG，把各layer的计算结果放在results[node2index[root]]中
        results长度必须与总layer数相同"""
        for root in layers:
            if results[node2index[root]] is not None:  # 已经计算过，直接返回
                return results
            if node2index[root] == 0:  # root为起始结点，直接使用ipt计算
                with torch.no_grad():
                    results[node2index[root]] = root.forward(ipt)
            else:  # 不使用ipt而使用results中的结果
                inputs = []
                for inodelist in root.inputs:
                    for node in inodelist:
                        if node in node2index and results[node2index[node]] is not None: # 不为空，已计算出
                            inputs.append(results[node2index[node]])
                with torch.no_grad():
                    results[node2index[root]] = root.forward(inputs[0] if len(inputs) == 1 else inputs)  # 将所有前驱结果作为输入
        return results

    @staticmethod
    def __visualize_dag(layer_topo:List[Node], file_path: str) -> None:
        """使用echarts可视化生成的DAG图，写入dnn_layers.html"""
        e_nodes = [{"name": str(n.id_) + ',' + n.module_type(),
                    "symbolSize": [len(str(n.id_) + ',' + n.module_type()) * 7, 20],  # 这里使用文字长度*7来控制矩形宽度，高度恒为20
                    "tooltip": {"formatter": n.module_path}}  # 鼠标移动上去显示此模块的路径
                   for n in layer_topo]  # 用于向echarts传结点参数，包括结点名称和标识点大小（标识点形状已经在echarts_util中设置为矩形）
        e_links = []  # 用于向echarts传边的参数
        for layer in layer_topo:
            for ds in layer.ds_layers:
                e_links.append({"source": layer.id_, "target": ds.id_})
        gen_html(e_nodes, e_links, file_path)
