from typing import Dict, List, Type, TypeVar, Generic

from torch import Tensor

from core.itg_executor import ItgExecutor, ItgJob, Job, Executor, ExNode
from core.util import msg2tensor, tensor2msg
from rpc.msg_pb2 import JobMsg
from core.raw_dnn import RawDNN


class InCache:
    def __init__(self) -> None:
        self.__id2opt = {}  # 上一帧的数据

    def update(self, id2dif: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """更新内部数据并返回更新后的值，Dict中的key以id2dif为准
        若self.__id2opt中没有某key但id2dif中有，则 新的id2opt[该key]=dif
        若self.__id2opt中有某key但id2dif中没有，则 新的id2opt[该key]不存在
        """
        id2opt = {}
        for nid, dif in id2dif.items():
            if nid not in self.__id2opt:
                id2opt[nid] = dif
            else:
                id2opt[nid] = self.__id2opt[nid] + dif
        self.__id2opt = id2opt
        return self.__id2opt


class OutCache:
    def __init__(self) -> None:
        self.__id2opt = {}  # 上一帧的数据

    def diff(self, id2opt: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """更新内部数据并返回 (最新数据-之前数据)
        若旧数据中没有某key但新数据中有，则 id2dif[该key]=新opt
        若旧数据中有某key但新数据中没有，则 id2dif[该key]不存在
        """
        id2dif = {}
        for nid, opt in id2opt.items():
            if nid not in self.__id2opt:
                id2dif[nid] = opt
            else:
                id2dif[nid] = opt - self.__id2opt[nid]
        self.__id2opt = id2opt
        return id2dif

    def get(self) -> Dict[int, Tensor]:
        return self.__id2opt


class DifJob(Job):
    def __init__(self, exec_ids: List[int], out_ids: List[int], id2dif: Dict[int, Tensor]):
        super().__init__(exec_ids, out_ids)
        self._id2dif: Dict[int, Tensor] = id2dif  # dif为(后一帧-前一帧)，node_id->Tensor

    def __repr__(self):
        return f"DifJob(exec_ids={self.exec_ids}, out_ids={self.out_ids}, " \
               f"id2dif={ {n: ts.shape for n, ts in self._id2dif.items()} })"

    @property
    def id2data(self) -> Dict[int, Tensor]:
        return self._id2dif

    @id2data.setter
    def id2data(self, id2dif):
        self._id2dif = id2dif

    @classmethod
    def from_msg(cls, job_msg: JobMsg) -> 'DifJob':
        id2dif = {nid: msg2tensor(dif_msg) for nid, dif_msg in job_msg.id2data.items()}
        return DifJob(job_msg.exec_ids, job_msg.out_ids, id2dif)

    def to_msg(self) -> JobMsg:
        id2dif = {nid: tensor2msg(dif) for nid, dif in self._id2dif.items()}
        return JobMsg(exec_ids=self.exec_ids, out_ids=self.out_ids, id2data=id2dif)

    def to_itg(self, in_cache: InCache) -> ItgJob:
        """借助InCache，把DifJob转成完整的ItgJob，更新InCache"""
        return ItgJob(self.exec_ids, self.out_ids, in_cache.update(self._id2dif))

    def clear(self) -> None:
        """清空自己的全部字段"""
        self.exec_ids.clear()
        self._id2dif.clear()
        self.out_ids.clear()


T = TypeVar('T', bound=ExNode)
class DifExecutor(Executor, Generic[T]):
    """内部缓存上次的执行结果，输入DifJob，得到输出
    DifJob必须为 这次数据-上次数据"""

    def __init__(self, raw_dnn: RawDNN, node_type: Type[T] = ExNode):
        super().__init__(raw_dnn, node_type)
        self.__itg_extor = ItgExecutor(raw_dnn, node_type)
        self.__in_cache = InCache()  # DifJob中上一帧输入的缓存，获得输入时更新
        self.__out_cache = OutCache()  # DifJob中上一帧输出的缓存，获得输出时更新

    def exec(self, dif_job: DifJob) -> Dict[int, Tensor]:
        """输入DifJob，输出Dif
        :param dif_job 输入节点的dif 这一帧-上一帧
        :return 输出节点的dif 这一帧-上一帧
        """
        job = dif_job.to_itg(self.__in_cache)
        id2opt = self.__itg_extor.exec(job)
        return self.__out_cache.diff(id2opt)

    def last_out(self) -> Dict[int, Tensor]:
        """获取最新一次运行的原始输出结果（不是Dif）"""
        return self.__out_cache.get()

    def dag(self) -> List[T]:
        return self.__itg_extor.dag()
