import logging
from typing import List, Dict, Any, Type

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job, Node
from core.ifr import WkJob
from core.itg_executor import ItgJob
from master.scheduler import SizedNode, G1Scheduler
from trainer.trainer import NZPred


class LBScheduler(G1Scheduler):
    """Load Balance Scheduler"""
    def __init__(self, s_dag: List[SizedNode], nzpred: NZPred,
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__job_type = job_type
        self.__wk_cap = wk_cap
        self.__ly_comp = ly_comp
        wk_lynum = self.split_chain(self.__ly_comp[1:], self.__wk_cap)  # 第0层不需要执行
        self.__lb_wk_elys = self.wk_lynum2layers_chain(1, wk_lynum)
        self.__logger.info(f"elys={self.__lb_wk_elys}")
        self.__lb_wk_olys = [self.elys2olys(elys, s_dag) for elys in self.__lb_wk_elys]
        self.__logger.info(f"olys={self.__lb_wk_olys}")

    def gen_wk_jobs(self, ifr_id: int, pre_ipt: Tensor, cur_ipt: Tensor) -> List[WkJob]:
        jobs = [WkJob(w, self.__job_type(lys, self.__lb_wk_olys[w], {}))
                for w, lys in enumerate(self.__lb_wk_elys)]
        if self.__job_type == ItgJob:
            jobs[0].job.id2data[0] = cur_ipt
        elif self.__job_type == DifJob:
            jobs[0].job.id2data[0] = cur_ipt - pre_ipt
        else:
            raise NotImplementedError()
        return jobs
