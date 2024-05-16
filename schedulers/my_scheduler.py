import copy
import logging
from typing import List, Type, Dict, Any, Callable, Tuple

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job
from model_split import Node
from core.ifr import IFR, WkJob
from core.predictor import NZPred
from master.scheduler import Scheduler
from schedulers.metric import LatencyMetric, Metric
from rpc.stub_factory import MStubFactory
from torch.nn import Conv2d, BatchNorm2d
import time


class MyScheduler(Scheduler):
    """以IFR Group为粒度进行调度"""
    # TODO: 改进！
    def __init__(self, s_dag, nzpred: NZPred,
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any], layers: List[Node], node2index: Dict[Node, int], stb_fct: MStubFactory):
        self.__sdag = s_dag
        self.__layers = layers
        self.__node2index = node2index
        self.__o_lbsz = [sz * 4 for sz in self.lcnz2lsz(nzpred.o_lcnz, s_dag)]
        self.__predictors = nzpred.predictors
        self.__wk_cap = wk_cap
        self.__wk_bwth = wk_bwth
        self.__ly_comp = ly_comp
        assert job_type == DifJob, "This scheduler is only used for DifJob!"
        self.__job_type: Type[Job] = job_type
        self.__gp_size: int = config['gp_size']

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__pre_wk_ilys = [[] for _ in range(len(wk_cap))]  # 各Worker上次运行时的输入层
        self.__fs_cost = []  # 各帧各阶段的预估耗时
        self.__stb_fct = stb_fct

    def group_size(self) -> int:
        """建议的group大小，但实际上可能比这个小"""
        return self.__gp_size

    def fs_cost(self) -> List[List[float]]:
        return self.__fs_cost
    
    def layers_to_recur(self) -> List[int]:
        """返回需要递归搜索的层的id列表"""
        layers_to_recur = []
        for ind, node in enumerate(self.__layers):
            if not isinstance(node.module, (Conv2d, BatchNorm2d)):
                layers_to_recur.append(ind)
        return layers_to_recur

    def gen_ifr_group(self, ifr_cnt: int, pre_ipt: Tensor,
                      ipt_group: List[Tensor], s_ready: List[float] = None) -> List[IFR]:
        """一个group的所有ifr都用同一个执行方案。ifr_cnt为当前group中第一个IFR的id
        注意：这里只考虑链状CNN，只考虑云边协同(只有两个Worker)

        为一组输入生成相应的IFR组
        :param ifr_cnt: 当前group中第一个IFR的id
        :param pre_ipt: 上一帧的输入
        :param ipt_group: 要发出去的所有输入帧，1<=长度<=group_size
        :param s_ready: 根据fs_cost和当前IFR状态给出的各阶段就绪时间的估计。fs_cost不合法则为None
        :return ifr_group ifr_group[i]对应ipt_group[i]
        """
        # 获取各个worker的实时上传带宽
        online_bwths = False
        if online_bwths:
            wk_bwths = [0. for _ in self.__wk_bwth]
            for wid in range(len(self.__wk_bwth)):
                self.__logger.info(f"Getting bandwidth from worker{wid}...")
                wk_bwths[wid] = self.__stb_fct.worker(wid).get_bandwidth()*1024*1024
                self.__logger.info(f"worker{wid} bandwidth: {wk_bwths[wid]}")
            self.__wk_bwth = wk_bwths

        assert len(ipt_group) > 0
        dif_group = [ipt_group[0] - pre_ipt] + [ipt_group[i] - ipt_group[i-1] for i in range(1, len(ipt_group))] # 帧差
        # 观察发现，原始数据也存在一定的稀疏性，但是分布非常集中。对于一个层而言，几乎所有帧的非零占比都在平均值附近
        # 所以这里直接使用平均值作为原始数据的非零率，进而计算原始数据大小
        org_gp_lbsz = [self.__o_lbsz for ipt in ipt_group]
        self.__logger.info(f"start predicting...")
        # 中间特征残差数据传输量
        dif_gp_lbsz = [Scheduler.dif2lbsz(dif, self.__sdag, self.__predictors, org_gp_lbsz[ind]) for ind, dif in enumerate(dif_group)]     
        
        metric = LatencyMetric(self.__ly_comp, self.__wk_cap, self.__wk_bwth,
                               self.__pre_wk_ilys, org_gp_lbsz, dif_gp_lbsz, s_ready)
        
        # 是否进行剪枝搜索
        layers_to_recur = self.layers_to_recur()
        search_btime = time.time()
        prune_recursion = True
        if prune_recursion:
            # self.__logger.info(f"layers_to_recur: {layers_to_recur}")
            opt_wk_elys, opt_cost = self.recur_find_prune([], metric, layers_to_recur)
        else:
            opt_wk_elys, opt_cost = self.recur_find_chain([], metric)
        search_etime = time.time()
        self.__logger.info(f"search time: {search_etime-search_btime}")

        self.__logger.info(f"opt: {opt_wk_elys} => cost={opt_cost}")
        # 预估各阶段耗时
        gp_wk_tran, gp_wk_cmpt = Metric.gp_plan2tran_cmpt_chain(
            [opt_wk_elys]*self.__gp_size, self.__pre_wk_ilys,
            self.__ly_comp, self.__wk_cap, self.__wk_bwth,
            org_gp_lbsz, dif_gp_lbsz)
        nworker = len(self.__wk_cap)
        gs_cost = [[(wk_tran[s//2] if s%2==0 else wk_cmpt[s//2]) for s in range(nworker*2)]
                   for wk_tran, wk_cmpt in zip(gp_wk_tran, gp_wk_cmpt)]
        assert len(self.__fs_cost) == ifr_cnt  # 确保fs_cost中的对应关系正确
        self.__fs_cost.extend(gs_cost)
        # 生成并发送任务
        wk_jobs = [WkJob(w, self.__job_type(lys, self.elys2olys(lys, self.__layers, self.__node2index), {}))
                   for w, lys in enumerate(opt_wk_elys)]
        # 更新缓存情况
        self.__pre_wk_ilys[0] = [0]  # worker0必定接收第0层的输出
        for wk_job in wk_jobs[:-1]:
            # 前一个Worker的输出层，就是后一个Worker的缓存层
            self.__pre_wk_ilys[wk_job.worker_id+1] = wk_job.job.out_ids.copy()
        # Worker0接收到的输入数据必定为dif
        ifr_group = []
        for gi, dif in enumerate(dif_group):
            jobs = copy.copy(wk_jobs)
            jobs[0].job.id2data = {0: dif}
            ifr_group.append(IFR(ifr_cnt + gi, jobs))
        return ifr_group

    @classmethod
    def recur_find_chain(cls, wk_elys: List[List[int]], metric: Metric) -> Tuple[List[List[int]], float]:
        """递归为各个Worker分配任务，根据metric寻找最优分配方案。
        :param wk_elys: 各Worker分配的层，初始为空，每次递归加一个Worker，Worker内各层按照id顺序排列
        :param metric: wk_elys的相应代价，越小越好
        :return 最优的wk_elys, 相应的代价
        """
        wk_num, ly_num = metric.wk_num(), metric.ly_num()
        worker_id = len(wk_elys)  # 当前要分配的Worker的ID
        last_ly = max((ly for lys in wk_elys for ly in lys), default=0)  # 当前Worker前已经完成的层
        if worker_id == wk_num-1:  # 当前要分配的是最后一个Worker
            # 最后一个Worker必须完成区间(last_ly, ly_num)的所有层
            wk_elys.append(list(range(last_ly+1, ly_num)))
            cost = metric([wk_elys]*metric.gp_size())
            res = copy.deepcopy(wk_elys)
            wk_elys.pop()
            return res, cost
        opt_wk_elys, opt_cost = [], float('inf')
        # my_last: 当前Worker完成之后，已经完成的层
        for my_last in range(last_ly, ly_num):  # last_ly表示当前Worker什么都没做，ly_num-1表示完成了剩余所有层
            wk_elys.append(list(range(last_ly+1, my_last+1)))
            cad_wk_elys, cad_cost = cls.recur_find_chain(wk_elys, metric)
            wk_elys.pop()
            if cad_cost < opt_cost:
                opt_wk_elys, opt_cost = cad_wk_elys, cad_cost
        assert opt_cost < float('inf')
        return opt_wk_elys, opt_cost
    
    @classmethod
    def recur_find_prune(cls, wk_elys: List[List[int]], metric: Metric, layers_to_recur: List[int]) -> Tuple[List[List[int]], float]:
        """递归剪枝后指定的层为各个Worker分配任务，根据metric寻找最优分配方案。
        :param wk_elys: 各Worker分配的层，初始为空，每次递归加一个Worker，Worker内各层按照id顺序排列
        :param metric: wk_elys的相应代价，越小越好
        :param layers_to_recur: 要递归的层的id列表
        :return 最优的wk_elys, 相应的代价
        """
        wk_num, ly_num = metric.wk_num(), metric.ly_num()
        worker_id = len(wk_elys)
        last_ly = max((ly for lys in wk_elys for ly in lys), default=0)
        if worker_id == wk_num-1:
            wk_elys.append(list(range(last_ly+1, ly_num)))
            cost = metric([wk_elys]*metric.gp_size())
            res = copy.deepcopy(wk_elys)
            wk_elys.pop()
            return res, cost
        opt_wk_elys, opt_cost = [], float('inf')
        for my_last in layers_to_recur:  # 只处理指定的层
            if my_last < last_ly or my_last >= ly_num:
                continue
            wk_elys.append(list(range(last_ly+1, my_last+1)))
            cad_wk_elys, cad_cost = cls.recur_find_prune(wk_elys, metric, layers_to_recur)
            wk_elys.pop()
            if cad_cost < opt_cost:
                opt_wk_elys, opt_cost = cad_wk_elys, cad_cost
        assert opt_cost < float('inf')
        return opt_wk_elys, opt_cost
