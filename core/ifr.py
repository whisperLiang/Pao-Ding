from dataclasses import dataclass
from typing import Dict, List, Type

from torch import Tensor

from core.executor import Job
from rpc.msg_pb2 import IFRMsg, WkJobMsg


@dataclass
class WkJob:
    worker_id: int
    job: Job

    @classmethod
    def from_msg(cls, wj_msg: WkJobMsg, job_type: Type[Job]) -> 'WkJob':
        return cls(wj_msg.worker_id, job_type.from_msg(wj_msg.job_msg))

    def to_msg(self) -> WkJobMsg:
        return WkJobMsg(worker_id=self.worker_id, job_msg=self.job.to_msg())


@dataclass
class IFR:
    """一个帧对应的Worker执行计划，在Worker之间按照顺序传递
    当一帧到来时，Master生成每个Worker相应Job的exec_ids和out_ids
    每次传递时，把已完成的Job所有字段置为空，设置下一个Job的id2opt
    """
    id: int  # 帧号
    wk_jobs: List[WkJob]  # 按照执行顺序排列，至少有一个，不会为空

    @classmethod
    def from_msg(cls, ifr_msg: IFRMsg, job_type: Type[Job]) -> 'IFR':
        return cls(ifr_msg.id, [WkJob.from_msg(wj_msg, job_type) for wj_msg in ifr_msg.wk_jobs])

    def to_msg(self) -> IFRMsg:
        return IFRMsg(id=self.id, wk_jobs=[job.to_msg() for job in self.wk_jobs])

    def is_final(self) -> bool:
        """当前WkJob是最后一个，后者当前WkJob之后所有WkJob都是空任务，就返回True"""
        return len(self.wk_jobs) == 1 or all(len(wj.job.exec_ids)==0 for wj in self.wk_jobs[1:])

    def switch_next(self, id2data: Dict[int, Tensor]) -> None:
        assert not self.is_final()
        self.wk_jobs.pop(0)
        self.wk_jobs[0].job.id2data = id2data