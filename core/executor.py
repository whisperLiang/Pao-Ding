from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Type, TypeVar, Generic

from torch import Tensor
from torch.nn import Module

from core.dnn_config import InputModule, RawLayer
from core.raw_dnn import RawDNN
from rpc.msg_pb2 import JobMsg


class Node:
    def __init__(self, id_: int, ancients: List[int], descendants: List[int], calc: Module):
        self.id = id_
        if len(ancients) == 0:
            assert isinstance(calc, InputModule), f"{calc} is not InputModule but has no ancient!"
            self.ancients = []
        else:
            self.ancients = ancients  # 前驱结点，并按照分支处理顺序排序
        self.descendants = descendants  # 后继节点
        self.calc = calc  # 直接使用RawLayer中的Module

    @classmethod
    def raw2dag(cls, raw_layers: List[RawLayer]) -> List['Node']:
        """将由RawLayer构成的DAG图转为由Node构成的DAG图
        :param raw_layers 原先的由RawLayer构成的DAG图"""
        dag = []
        for layer in raw_layers:
            ancients = [d.id_ for d in layer.ac_layers]  # 前驱结点（按序排列）
            descendants = [d.id_ for d in layer.ds_layers]  # 后继结点（按序排列）
            dag.append(Node(layer.id_, ancients, descendants, layer.module))
        return dag


@dataclass
class Job:
    """供Executor使用的通用接口"""
    exec_ids: List[int]  # 要执行的这组CNN层的id，按照执行顺序排列
    out_ids: List[int]  # 这组CNN层中输出层的id

    def __init__(self, exec_ids: List[int], out_ids: List[int], id2data: Dict[int, Tensor] = None):
        self.exec_ids, self.out_ids = exec_ids, out_ids

    @property
    @abstractmethod
    def id2data(self) -> Dict[int, Tensor]:
        """CNN层->输出数据"""
        pass

    @id2data.setter
    @abstractmethod
    def id2data(self, value):
        pass

    @classmethod
    @abstractmethod
    def from_msg(cls, job_msg: JobMsg) -> 'Job':
        pass

    @abstractmethod
    def to_msg(self) -> JobMsg:
        pass


T = TypeVar('T', bound=Node)
class Executor(ABC, Generic[T]):
    """抽象类，用于定义执行Job的统一接口"""
    @abstractmethod
    def __init__(self, raw_dnn: RawDNN, node_type: Type[T] = Node):
        pass

    @abstractmethod
    def exec(self, job: Job) -> Dict[int, Tensor]:
        """送入单个任务，执行并得到输出
        :return {node_id: 数据}
        """
        pass

    @abstractmethod
    def dag(self) -> List[T]:
        """返回内部的dag"""
        pass
