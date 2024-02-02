from typing import List, Dict, Optional, Type, TypeVar, Generic

import torch
from torch import Tensor

from core.dag_dnn import DagDNN
from core.executor import Job, Executor
from model_split import Node
from core.util import msg2tensor, tensor2msg
from rpc.msg_pb2 import JobMsg


class ItgJob(Job):
    """一个完整的可直接送入Executor的Job"""

    def __init__(self, exec_ids: List[int], out_ids: List[int], id2opt: Dict[int, Tensor]):
        super().__init__(exec_ids, out_ids)
        self._id2opt = id2opt  # 先前完成的Job得到的输出，node_id->Tensor

    @property
    def id2data(self) -> Dict[int, Tensor]:
        return self._id2opt

    @id2data.setter
    def id2data(self, id2opt):
        self._id2opt = id2opt

    @classmethod
    def from_msg(cls, job_msg: JobMsg) -> 'ItgJob':
        id2opt = {nid: msg2tensor(opt_msg) for nid, opt_msg in job_msg.id2data.items()}
        return cls(job_msg.exec_ids, job_msg.out_ids, id2opt)

    def to_msg(self) -> JobMsg:
        id2opt = {nid: tensor2msg(opt, False) for nid, opt in self._id2opt.items()}
        return JobMsg(exec_ids=self.exec_ids, out_ids=self.out_ids, id2data=id2opt)


class ExNode(Node):
    """保存实际的数据"""
    # noinspection PyMissingConstructor
    def __init__(self, node: Node) -> None:
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化ExNode的所有成员变量
        self.outdegree = self.fixedoutd # 判断何时将self.__finished设置为True
        self.__finished: bool = False  # 当前节点是否未完成

    def init_inputs(self, output: Optional[Tensor]) -> None:
        """设置输入数据"""
        self.execute(output)

    def set_finished(self) -> None:
        self.outdegree -= 1
        if self.outdegree == 0:
            self.__finished = True

    def execute(self, inputs) -> None:
        """inputs为输入，执行并保存输出"""
        assert self.outresult is None and not self.__finished, "output has been set!"
        with torch.no_grad():
            self.outresult = self.forward(inputs)

    def get_output(self) -> Optional[Tensor]:
        return self.outresult

    def finished(self) -> bool:
        """是否已完成"""
        return self.__finished

    def clear(self):
        """回收内存，但仍为finished状态"""
        self.outresult = None

    def reset(self):
        """完全重置，回到初始状态"""
        self.clear()
        self.outdegree = self.fixedoutd
        self.__finished = False


T = TypeVar('T', bound=ExNode)
class ItgExecutor(Executor, Generic[T]):
    """执行一次inference中的一组CNN层。喂进输入，得到输出"""
    def __init__(self, dag_dnn: DagDNN, node_type: Type[T] = ExNode):
        super().__init__(dag_dnn, node_type)
        self.node2index = dag_dnn.node2index
        self.__ex_dag = [node_type(node) for node in dag_dnn.layers]

    def exec(self, job: ItgJob) -> Dict[int, Tensor]:
        """执行给定的Job，得到输出结果"""
        if len(job.exec_ids) == 0:  # 没有要执行的层，直接返回上一个Worker的结果
            return job.id2data
        self.__init_job(job)
        # 执行job，获取输出
        for exec_id in job.exec_ids:
            inputs = []
            for inodelist in self.__ex_dag[exec_id].inputs:
                for node in inodelist:
                    if node in self.node2index and self.__ex_dag[self.node2index[node]].get_output() is not None:
                        inputs.append(self.__ex_dag[self.node2index[node]].get_output())
                        self.__ex_dag[self.node2index[node]].set_finished()

            self.__ex_dag[exec_id].execute(inputs[0] if len(inputs) == 1 else inputs)
            # 内存回收
            for inodelist in self.__ex_dag[exec_id].inputs:
                for node in inodelist:
                    if node in self.node2index:
                        for outnodelist in node.outputs:
                            if all(self.__ex_dag[self.node2index[outnode]].finished() for outnode in outnodelist):
                                self.__ex_dag[self.node2index[node]].clear()

        out = {oid: self.__ex_dag[oid].get_output() for oid in job.out_ids}
        self.__reset()
        return out

    def dag(self) -> List[T]:
        return self.__ex_dag

    def __init_job(self, job: ItgJob) -> None:
        """为job初始化：设置输入数据，并将输入节点的所有前驱标记为finished"""
        # 设置输入节点的数据
        for node_id, output in job.id2data.items():
            self.__ex_dag[node_id].init_inputs(output)

    def __reset(self) -> None:
        """重置所有Node的状态"""
        # 注意：因为job执行前会设置exec_id和id2opt之前的节点
        # 所以这里要重置所有节点，不能只重置job相关的节点
        for e_node in self.__ex_dag:
            e_node.reset()

