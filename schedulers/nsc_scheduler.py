import logging
import sys
from typing import Any, List, Dict, Type, Tuple

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job
from core.ifr import WkJob
from core.predictor import Predictor
from master.scheduler import SizedNode, G1Scheduler
from trainer.trainer import NZPred


class NSCScheduler(G1Scheduler):
    """Neighborhood Search Chain Scheduler"""
    def __init__(self, s_dag: List[SizedNode], nzpred: NZPred,
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        super().__init__(s_dag, nzpred, wk_cap, wk_bwth, ly_comp, job_type, ifr_num, config)
        self.__logger = logging.getLogger(self.__class__.__name__)
        assert job_type == DifJob, "This scheduler is only used for DifJob!"
        self.__ifr_num = ifr_num
        self.__dag = s_dag
        self.__predictors = nzpred.predictors
        self.__job_type: Type[Job] = job_type
        self.__wk_cap = wk_cap
        self.__ly_comp = ly_comp
        wk_lynum = self.split_chain(self.__ly_comp[1:], self.__wk_cap)  # 第0层不需要执行
        self.__lb_wk_elys = self.wk_lynum2layers_chain(1, wk_lynum)
        self.__logger.debug(f"load balance: {self.__lb_wk_elys}")
        self.__wk_bwth = wk_bwth
        self.__wk_jobs = []  # 调度方案
        self.__pre_wk_ilys = [[] for _ in range(len(wk_cap))]  # 各Worker上次运行时的输入层

    def gen_wk_jobs(self, ifr_id: int, pre_ipt: Tensor, cur_ipt: Tensor) -> List[WkJob]:
        dif_ipt = cur_ipt - pre_ipt
        if ifr_id > 1:  # IFR0和IFR1都要生成调度方案，后面的直接用之前的方案
            self.__wk_jobs[0].job.id2data[0] = dif_ipt  # 更新输入数据
            return self.__wk_jobs
        opt_lbsz = self.dif2lbsz(cur_ipt, self.__dag, self.__predictors)
        dif_lbsz = self.dif2lbsz(dif_ipt, self.__dag, self.__predictors)
        # IFR0的数据量和后面的显著不同，所以要单独考虑
        nframe = (1 if ifr_id == 0 else self.__ifr_num-1)
        wk_layers = self.optimize_chain(self.__lb_wk_elys, self.__pre_wk_ilys, self.__wk_cap, self.__wk_bwth,
                                        opt_lbsz, dif_lbsz, self.__ly_comp, nframe, vis=False, logger=self.__logger)
        jobs = [WkJob(w, self.__job_type(lys, ([lys[-1]] if lys else []), {})) for w, lys in enumerate(wk_layers)]
        jobs[0].job.id2data[0] = dif_ipt
        self.__wk_jobs = jobs
        self.__pre_wk_ilys = [([lys[0]] if len(lys) > 0 else []) for lys in wk_layers]
        return jobs

    @classmethod
    def optimize_chain(cls, lb_wk_elys: List[List[int]], pre_wk_ilys: List[List[int]],
                       wk_cap: List[float], wk_bwth: List[float],
                       opt_lbsz: List[float], dif_lbsz: List[float], ly_comp: List[float], nframe: int,
                       vis: bool = False, logger: logging.Logger = None) -> List[List[int]]:
        """对于链状的CNN，从负载最均衡的方案开始，优化nframe总耗时"""
        if logger is None:
            # 如果没有传入logger，则默认写入stdout
            logger = logging.getLogger('optimize_chain')
            if not logger.hasHandlers():
                # 因为这个logger是全局共用的，所以不能重复添加Handler
                logger.addHandler(logging.StreamHandler(sys.stdout))
                logger.setLevel(logging.DEBUG)
        wk_elys = lb_wk_elys
        cur_cost = cls.estimate_latency_chain(wk_elys, pre_wk_ilys, wk_cap, wk_bwth,
                                              opt_lbsz, dif_lbsz, ly_comp, nframe, vis)
        logger.debug(f"wk_elys={wk_elys}, {nframe} frame avg={cur_cost / nframe} (init)")
        init_wk_tran, wk_tran, _ = cls.plan2costs_chain(wk_elys, pre_wk_ilys, wk_cap, wk_bwth,
                                                        opt_lbsz, dif_lbsz, ly_comp)
        while True:
            mw = 1  # 总传输耗时最大的worker，因为Worker0的传输耗时无法优化，所以从Worker1开始
            for w in range(2, len(wk_elys)):
                # w-1和w中至少一个有layer，不然没法优化
                if init_wk_tran[w] + wk_tran[w] * (nframe - 1) > init_wk_tran[mw] + wk_tran[mw] * (nframe - 1) \
                        and len(wk_elys[w - 1]) + len(wk_elys[w]) > 0:
                    mw = w
            pwk = mw - 1  # mw左边第一个有layer的Worker
            while pwk >= 0 and len(wk_elys[pwk]) == 0:
                pwk -= 1
            # ml = (wk_elys[pwk][-1] if pwk >= 0 else 0)  # mw数据传输对应的CNN层: 如果没找到说明是第0层
            elys = wk_elys[mw - 1] + wk_elys[mw]
            bg_ly, ed_ly = elys[0], elys[-1]
            # 在mw-1和mw的任务中搜索割点，重新对mw-1和mw进行分配，要分配的层为elys，即[bg_ly, ed_ly]闭区间
            best_cost, best_wk_elys = cur_cost, wk_elys
            for l in elys:
                tmp_wk_elys = wk_elys[:mw - 1] + [list(range(bg_ly, l + 1))] \
                              + [list(range(l + 1, ed_ly + 1))] + wk_elys[mw + 1:]
                tmp_cost = cls.estimate_latency_chain(tmp_wk_elys, pre_wk_ilys, wk_cap, wk_bwth,
                                                      opt_lbsz, dif_lbsz, ly_comp, nframe, False)
                if tmp_cost < best_cost:
                    logger.debug(f"wk_elys={tmp_wk_elys}, {nframe} frame avg={tmp_cost / nframe}")
                    best_cost, best_wk_elys = tmp_cost, tmp_wk_elys
                    if vis:  # 只对改进了的方案进行可视化
                        cls.estimate_latency_chain(tmp_wk_elys, pre_wk_ilys, wk_cap, wk_bwth,
                                                   opt_lbsz, dif_lbsz, ly_comp, nframe, True)
            if best_cost < cur_cost:
                cur_cost, wk_elys = best_cost, best_wk_elys
            else:
                break
        if vis:
            from matplotlib import pyplot as plt
            plt.show()  # 显示前后优化的多张图像
        return wk_elys

    @classmethod
    def estimate_latency_chain(cls, wk_elys: List[List[int]], pre_wk_ilys: List[List[int]],
                               wk_cap: List[float], wk_bwth: List[float],
                               opt_lbsz: List[float], dif_lbsz: List[float],
                               ly_comp: List[float], nframe: int, vis: bool = False) -> float:
        """输入计划任务和这个任务会执行的帧数，计算按照此计划任务执行的平均时延。只考虑链状CNN
        :param wk_elys 各worker执行哪些层，wk_elys[w][-1]就是Worker w的输出层
        :param pre_wk_ilys 上一次执行中，各Worker的输入层
        :param wk_cap worker的计算能力
        :param wk_bwth wk_bwth[w]为w-1与w之间的带宽。w=0时为master和worker0之间的带宽
        :param opt_lbsz 各层原始输出数据的大小，单位byte
        :param dif_lbsz 各层差值输出数据的大小，单位byte
        :param ly_comp 各layer的计算量
        :param nframe 总共会执行多少帧
        :param vis 是否对当前方案进行可视化
        :return 预计总耗时(不包括最后一个worker返回master的时延)
        """
        init_wk_tran, wk_tran, wk_cmpt = cls.plan2costs_chain(wk_elys, pre_wk_ilys, wk_cap, wk_bwth,
                                                              opt_lbsz, dif_lbsz, ly_comp)
        dp = cls.simulate_pipeline(init_wk_tran, wk_tran, wk_cmpt, nframe)
        if vis:
            cls.visualize_frames(init_wk_tran, wk_tran, wk_cmpt, dp)
        return dp[-1][-1]

    @classmethod
    def plan2costs_chain(cls, wk_elys: List[List[int]], pre_wk_ilys: List[List[int]],
                         wk_cap: List[float], wk_bwth: List[float],
                         opt_lbsz: List[float], dif_lbsz: List[float], ly_comp: List[float]) \
            -> Tuple[List[float], List[float], List[float]]:
        """输入执行计划，得到各Worker的传输耗时wk_tran和计算耗时wk_cmpt。只考虑链状CNN
        wk_elys 为本次的执行计划
        pre_wk_ilys 为上一次执行中，各Worker的输入层
        opt_lbsz 为各层的原始数据大小
        dif_lbsz 为各层的差值数据大小
        :return 第0帧的传输代价init_wk_tran, 第1帧及其后的传输代价wk_tran, 计算代价wk_cmpt
        """
        fwk = len(wk_elys) - 1  # 最后一个有执行任务的Worker
        while fwk > 0 and len(wk_elys[fwk]) == 0:
            fwk -= 1
        # init_wk_tran[w]：第0帧中，Worker w接收前一个Worker输出数据的传输耗时
        init_wk_tran = [0. for _ in range(len(wk_cap))]  # 长度等于总worker数量
        init_wk_tran[0] = dif_lbsz[0] / wk_bwth[0]  # 第一个Worker无论是否执行第1层，都接收的是差值数据
        for w in range(1, fwk + 1):
            if len(wk_elys[w - 1]) == 0:  # Worker w-1不执行任何计算任务
                init_wk_tran[w] = init_wk_tran[w - 1] * wk_bwth[w - 1] / wk_bwth[w]  # Worker w的传输数据量和前一个Worker(w-1)一样
            else:
                if wk_elys[w - 1][-1] in pre_wk_ilys[w]:  # 输入数据已有缓存
                    bsz = dif_lbsz[wk_elys[w - 1][-1]]
                else:  # 输入数据没有缓存
                    bsz = opt_lbsz[wk_elys[w - 1][-1]]
                init_wk_tran[w] = bsz / wk_bwth[w]  # Worker w-1的输出数据量/带宽
        for w in range(fwk + 1, len(wk_cap)):
            init_wk_tran[w] = 0.
        # wk_tran[w]: 第1帧及其后的单帧中，Worker w完成自己的所有层的计算耗时
        wk_tran = [0. for _ in range(len(wk_cap))]  # 长度等于总worker数量
        wk_tran[0] = dif_lbsz[0] / wk_bwth[0]  # 第一个Worker无论是否执行第1层，都接收的是差值数据
        for w in range(1, fwk + 1):
            if len(wk_elys[w - 1]) == 0:  # Worker w-1不执行任何计算任务
                wk_tran[w] = wk_tran[-1] * wk_bwth[w - 1] / wk_bwth[w]  # Worker w的传输数据量和前一个Worker(w-1)一样
            else:  # 因为第0帧已经执行过一次，所以此时输入数据必定有缓存
                bsz = dif_lbsz[wk_elys[w - 1][-1]]
                wk_tran[w] = bsz / wk_bwth[w]  # Worker w-1的输出数据量/带宽
        for w in range(fwk + 1, len(wk_cap)):
            wk_tran[w] = 0.
        # wk_cmpt[w]：单帧中，Worker w完成自己的所有层的计算耗时
        wk_cmpt = [sum(ly_comp[e] for e in wk_elys[w]) / wk_cap[w] for w in range(len(wk_cap))]
        return init_wk_tran, wk_tran, wk_cmpt

    @classmethod
    def simulate_pipeline(cls, init_wk_tran: List[float], wk_tran: List[float],
                          wk_cmpt: List[float], nframe: int) -> List[List[float]]:
        """输入各Worker的传输耗时wk_tran，计算耗时wk_cmpt，帧数nframe，返回各帧在各Worker上的完成耗时
        注意：返回的fs_dp中的worker数<=len(wk_cmpt)=len(wk_tran)，因为不包括后面的计算量为0的worker
        """
        assert len(init_wk_tran) == len(wk_tran), \
            f"len(init_wk_tran)<{len(init_wk_tran)}> != len(wk_tran)<{len(wk_tran)}>"
        assert len(wk_tran) == len(wk_cmpt), f"len(wk_tran)<{len(wk_tran)}> != len(wk_cmpt)<{len(wk_cmpt)}>"
        act_nwk = len(wk_cmpt)  # 从Worker0开始，到最后一个计算量不为0的Worker，实际总共有几个Worker参与计算
        while act_nwk > 0 and wk_cmpt[act_nwk - 1] == 0:
            act_nwk -= 1
        assert act_nwk > 0, "There is no worker to compute!"
        # dp[f][i]：第f帧第i//2个Worker的 传输完成耗时(i为偶数) / 计算完成耗时(i为奇数)
        # dp[f][2*w]：第f帧第w个Worker的传输完成耗时
        # dp[f][2*w+1]：第f帧第w个Worker的计算完成耗时
        dp = [[0. for _ in range(act_nwk * 2)] for _ in range(nframe)]
        # 第0帧的传输代价为init_wk_tran
        dp[0][0] = init_wk_tran[0]  # 传输完成
        dp[0][1] = init_wk_tran[0] + wk_cmpt[0]  # 计算完成
        for w in range(1, act_nwk):
            dp[0][2 * w] = dp[0][2 * w - 1] + init_wk_tran[w]
            dp[0][2 * w + 1] = dp[0][2 * w] + wk_cmpt[w]
        # 第1帧及其后面的传输代价为wk_tran
        for f in range(1, nframe):
            dp[f][0] = dp[f - 1][0] + wk_tran[0]  # Master发完了上一帧，开始发下一帧
            dp[f][1] = max(dp[f - 1][1], dp[f][0]) + wk_cmpt[0]  # 上一帧的计算完成，且当前帧传输完成，才开始当前帧的计算
            for w in range(1, act_nwk):
                dp[f][2 * w] = max(dp[f - 1][2 * w], dp[f][2 * w - 1]) + wk_tran[w]
                dp[f][2 * w + 1] = max(dp[f - 1][2 * w + 1], dp[f][2 * w]) + wk_cmpt[w]
        return dp

    @classmethod
    def visualize_frames(cls, init_wk_tran: List[float], wk_tran: List[float],
                         wk_cmpt: List[float], fs_dp: List[List[float]]):
        """wk_tran[w], wk_cmpt[w]表示Worker w的传输耗时和计算耗时，先传输后计算
        fs_dp为 simulate_pipeline 的输出，不包括后面没有计算的Worker
        fs_dp[f][s]：第f帧第s个阶段完成时的耗时。s=2*w时表示w传输完成耗时，s=2*w+1时表示w计算完成耗时
        """
        from matplotlib import pyplot as plt
        import matplotlib.colors as mcolors
        act_nwk = len(fs_dp[0]) // 2
        nframe = len(fs_dp)
        fig = plt.figure()
        ax = fig.subplots()
        ax.invert_yaxis()
        ticklabels = ['m->w0', 'w0']
        for w in range(1, len(wk_cmpt)):
            ticklabels.extend([f'{w - 1}->{w}', f'w{w}'])
        plt.yticks(list(range(2 * len(wk_cmpt))), ticklabels)
        colors = list(mcolors.XKCD_COLORS.values())
        for w in range(act_nwk):
            plt.barh(2 * w, init_wk_tran[w], left=fs_dp[0][2 * w] - init_wk_tran[w], color=colors[0])
            plt.barh(2 * w + 1, wk_cmpt[w], left=fs_dp[0][2 * w + 1] - wk_cmpt[w], color=colors[0])
        # 剩余的Worker画空的条形，使得纵轴显示出没有执行的worker
        for w in range(act_nwk, len(wk_cmpt)):
            plt.barh(2 * w, 0)
            plt.barh(2 * w + 1, 0)
        for f in range(1, nframe):
            for w in range(act_nwk):
                plt.barh(2 * w, wk_tran[w], left=fs_dp[f][2 * w] - wk_tran[w], color=colors[f])
                plt.barh(2 * w + 1, wk_cmpt[w], left=fs_dp[f][2 * w + 1] - wk_cmpt[w], color=colors[f])
            # 剩余的Worker画空的条形，使得纵轴显示出没有执行的worker
            for w in range(act_nwk, len(wk_cmpt)):
                plt.barh(2 * w, 0)
                plt.barh(2 * w + 1, 0)
