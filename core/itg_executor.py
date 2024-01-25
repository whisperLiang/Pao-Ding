from typing import List, Dict, Optional, Type, TypeVar, Generic

import torch
from torch import Tensor

from core.raw_dnn import RawDNN
from core.executor import Node, Job, Executor
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
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化DNode的所有成员变量
        self.__output: Optional[Tensor] = None  # 当前节点的输出数据，所有后继都完成时清空
        self.__finished: bool = False  # 当前节点是否已经完成

    def set_finish(self, output: Optional[Tensor]) -> None:
        """执行此函数的节点不运行execute，只是设置输入以便后继获取
        当output为None时，此节点不会被用到，只是将此节点标记为已完成，以便进行内存回收
        """
        assert self.__output is None and not self.__finished, "not-None or finished node cannot be set!"
        self.__output = output
        self.__finished = True

    def execute(self, *inputs: Tensor) -> None:
        """inputs为输入，执行并保存输出"""
        assert self.__output is None and not self.__finished, "output has been set!"
        with torch.no_grad():
            self.__output = self.calc(*inputs)
        self.__finished = True

    def get_output(self) -> Optional[Tensor]:
        return self.__output

    def finished(self) -> bool:
        """是否已完成"""
        return self.__finished

    def clear(self):
        """回收内存，但仍为finished状态"""
        self.__output = None

    def reset(self):
        """完全重置，回到初始状态"""
        self.clear()
        self.__finished = False


T = TypeVar('T', bound=ExNode)
class ItgExecutor(Executor, Generic[T]):
    """执行一次inference中的一组CNN层。喂进输入，得到输出"""
    def __init__(self, raw_dnn: RawDNN, node_type: Type[T] = ExNode):
        super().__init__(raw_dnn, node_type)
        dag = Node.raw2dag(raw_dnn.layers)
        self.__ex_dag = [node_type(node) for node in dag]

    def exec(self, job: ItgJob) -> Dict[int, Tensor]:
        """执行给定的Job，得到输出结果"""
        if len(job.exec_ids) == 0:  # 没有要执行的层，直接返回上一个Worker的结果
            return job.id2data
        self.__init_job(job)
        # 执行job，获取输出
        for exec_id in job.exec_ids:
            # print(f"exec layer{exec_id}")
            inputs = [self.__ex_dag[ds].get_output() for ds in self.__ex_dag[exec_id].ancients]
            self.__ex_dag[exec_id].execute(*inputs)
            # 内存回收
            for ac in self.__ex_dag[exec_id].ancients:
                if all(self.__ex_dag[ds].finished() for ds in self.__ex_dag[ac].descendants):
                    self.__ex_dag[ac].clear()
        out = {oid: self.__ex_dag[oid].get_output() for oid in job.out_ids}
        self.__reset()
        return out

    def dag(self) -> List[T]:
        return self.__ex_dag

    def __init_job(self, job: ItgJob) -> None:
        """为job初始化：设置输入数据，并将输入节点的所有前驱标记为finished"""
        # 设置输入节点的数据
        for node_id, output in job.id2data.items():
            self.__ex_dag[node_id].set_finish(output)
        # 标记前驱已完成。设置完再标记的目的是防止前面的输入节点被重复设置
        for node_id in job.id2data.keys():
            self.__finish_ancients(node_id)

    def __finish_ancients(self, node_id: int) -> None:
        """递归将node_id的前驱标记成finished。node_id此时应该已经为finished"""
        for ac in self.__ex_dag[node_id].ancients:
            if not self.__ex_dag[ac].finished():
                self.__ex_dag[ac].set_finish(None)
                self.__finish_ancients(ac)

    def __reset(self) -> None:
        """重置所有Node的状态"""
        # 注意：因为job执行前会设置exec_id和id2opt之前的节点
        # 所以这里要重置所有节点，不能只重置job相关的节点
        for e_node in self.__ex_dag:
            e_node.reset()

