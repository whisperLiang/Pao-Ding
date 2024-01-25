from abc import abstractmethod

from dataclasses import dataclass, field
from typing import List, Type, Set, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module

from core.predictor import Predictor, MLPPredictor, LNRPredictor, DRPredictor


class IMData:
    """Intermediate data, 各层产生的中间数据
    注意：RawLayer.module的返回值应为IMData"""
    def __init__(self, data: Any):
        """这里的data为RawLayer.module实际产生的输出数据"""
        self.data = data


class CpsIM(IMData):
    """可以压缩的中间数据，Compressible IMData"""
    @abstractmethod
    def nzr(self) -> float:
        """整体非零占比，Non-Zero Rate"""
        pass

    @abstractmethod
    def __sub__(self, other: 'CpsIM') -> 'CpsIM':
        pass

    @abstractmethod
    def __add__(self, other: 'CpsIM') -> 'CpsIM':
        pass


class UCpsIM(IMData):
    """不可压缩的中间数据，Uncompressible IMData"""
    pass


class TensorIM(CpsIM):
    '''Tensor对应的中间数据类型，每个TensorIM包裹一个Tensor'''
    def __init__(self, data: Tensor):
        super().__init__(data)

    def nzr(self) -> float:
        return float(self.data.count_nonzero()/self.data.nelement())

    def __sub__(self, other: 'TensorIM') -> 'TensorIM':
        return TensorIM(self.data - other.data)

    def __add__(self, other: 'TensorIM') -> 'TensorIM':
        return TensorIM(self.data + other.data)


@dataclass
class RawLayer:
    """用于生成Node，module可能只是一些计算结点的父结点，而不是单个计算结点"""
    id_: int
    module: Module  # module为DNN中的一个模块，不一定是计算结点，而calc一定是计算结点
    module_path: str  # module在整体DNN中的路径
    ds_layers: List['RawLayer']  # 后继结点
    ac_layers: List['RawLayer'] = field(default_factory=list)  # 前驱结点，默认为空列表
    im_type: Type[IMData] = field(default=TensorIM)  # 中间数据类型默认为Tensor

    def module_type(self):
        """本module所属的类名"""
        return type(self.module).__name__

    def __repr__(self):
        return f"RawLayer({self.id_}, {self.module_path}, {[d.id_ for d in self.ds_layers]})"

    def __hash__(self):
        """用作dict的key时，不同RawLayer用该对象的id区分"""
        return hash(id(self))


class InputModule(Module):
    """输入模块，作为DNN计算的唯一入口模块"""
    def forward(self, x: Tensor):
        return x


class ForkModule(Module):
    """Inception这种分叉结构的分叉模块。此类作为接口，继承此类用于标识DAG中一个分叉点
    若在子类中重写forward，则需要注意forward中不能对输入数据进行修改，因为在DNode中会保存输入数据"""
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        # inputs的类型与Module的默认类型保持了一致
        pass


class BasicFork(ForkModule):
    """输出=输入的分叉模块。因为很多分叉结构处理都只需要在前面加一个这样的分叉点，所以实现此类以便使用"""
    def forward(self, x: Any) -> Any:
        return x


class MergeModule(Module):
    """Inception这种分叉结构的汇聚模块。此类作为接口，继承此类用于标识DAG中一个汇聚点
    若在子类中重写forward，则需要注意forward中不能对输入数据进行修改，因为在DNode中会保存输入数据"""
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        # inputs的类型与Module的默认类型保持了一致
        pass


class BlockRule:
    """用于规定需要特殊处理的Module(称为Block)的子结点结构。非Sequential的Module一般都需要使用此类来规定计算过程"""
    @staticmethod
    @abstractmethod
    def is_target(module: Module) -> bool:
        """module是否应该由本规则处理"""
        pass

    @staticmethod
    @abstractmethod
    def build_dag(block: Module) -> List[RawLayer]:
        """给定block，构造出该Module到其子结点的RawLayer结构，返回所有RawLayer，并且按照id顺序排序
        返回列表中，首个必须为唯一的入口结点，末尾必须为唯一的出口结点
        返回的子结点不需要是叶子结点，系统会使用named_children对其子结点作进一步的展开，
        若某些自定义Module希望一次性执行而不被分成多个结点，则应重写(override)该Module的named_children函数
        block中若存在多分支，则分叉点的Module应继承ForkModule，汇聚点的Module应继承MergeModule，以便处理时识别与优化
        ForkModule和MergeModule对于分支的顺序要一致，即ForkModule各后继的顺序和MergeModule各前驱的顺序一致
        RawLayer.id_从0开始计数，各分支内id连续编号，分支之间编号顺序与分支顺序一致，汇聚点id大于前面所有结点的id
        RawLayer.module不能有inplace操作，这会影响Checker中ground-truth(detailed)的计算（因为计算这个的过程中保存了各RawLayer的数据）
        RawLayer.module_path为各模块名称简写，如ipt，conv1
        RawLayer.ac_layers与RawLayer.ds_layers按照数据流动方向填写"""
        pass


class DNNConfig:
    _MDL2PRED: Dict[Type[Module], Type[Predictor]] \
        = {InputModule: DRPredictor,
           BasicFork: DRPredictor,
           torch.nn.Conv2d: MLPPredictor,
           torch.nn.ReLU: LNRPredictor,
           torch.nn.BatchNorm2d: LNRPredictor,
           torch.nn.MaxPool2d: LNRPredictor}

    def __init__(self, name: str, dnn: Module, block_rules: Set[Type[BlockRule]] = None,
                 cust_mdl2pred: Dict[Type[Module], Type[Predictor]] = None):
        self.name = name  # DNN名称
        self.dnn = dnn
        self.block_rules = (block_rules if block_rules is not None else set())
        self.mdl2pred = self._MDL2PRED.copy()
        if cust_mdl2pred is not None:
            self.mdl2pred.update(cust_mdl2pred)
