from typing import List, Type, Dict, Any

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job
from core.ifr import WkJob
from core.itg_executor import ItgJob
from core.predictor import NZPred
from master.scheduler import SizedNode, G1Scheduler


class OSCScheduler(G1Scheduler):
    """One-Side Chain Scheduler，所有任务要么全都在边缘端(Worker0)，要么全都在云端(最后一个Worker)"""
    def __init__(self, s_dag: List[SizedNode], nzpred: NZPred, wk_cap: List[float], wk_bwth: List[float],
                 ly_comp: List[float], job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        self.__s_dag = s_dag
        self.__wk_num = len(wk_cap)
        self.__job_type = job_type
        self.__side = config['side']

    def gen_wk_jobs(self, ifr_id: int, pre_ipt: Tensor, cur_ipt: Tensor) -> List[WkJob]:
        if self.__side == 'edge':
            jobs = [WkJob(0, self.__job_type(list(range(1, len(self.__s_dag))), [self.__s_dag[-1].id], {}))] \
                    + [WkJob(w, self.__job_type([], [], {})) for w in range(1, self.__wk_num)]
        elif self.__side == 'cloud':
            jobs = [WkJob(w, self.__job_type([], [0], {})) for w in range(0, self.__wk_num-1)] \
                    + [WkJob(self.__wk_num-1, self.__job_type(list(range(1, len(self.__s_dag))), [self.__s_dag[-1].id], {}))]
        else:
            raise NotImplementedError()
        if self.__job_type == ItgJob:
            jobs[0].job.id2data[0] = cur_ipt
        elif self.__job_type == DifJob:
            jobs[0].job.id2data[0] = cur_ipt - pre_ipt
        else:
            raise NotImplementedError()
        return jobs
