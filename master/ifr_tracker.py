import copy
import logging
import threading
import time
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, List, Optional

from torch import Tensor

from core.ifr import IFR
from rpc.stub_factory import MStubFactory


@dataclass
class PendingIpt:
    ifr_id: int
    ipt: Optional[Tensor]
    send_time: float


class IFRTracker:
    def __init__(self, ifr_num: int, wk_num: int, pd_num: int, stb_fct: MStubFactory):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__ifr_num = ifr_num
        self.__pd_num = pd_num
        self.__stb_fct = stb_fct

        self.__pd_dct: Dict[int, PendingIpt] = {}  # key为ifr_id
        self.__pd_cv = threading.Condition()  # 确保pd_dct的修改是串行的
        self.__ifr_max = -1  # 已发送的最大IFR的ID

        self.__stg_num = wk_num*2  # 阶段数
        self.__begin_time = 0.
        self.__avg_latency = [-1. for _ in range(ifr_num)]  # 前i+1帧的平均时延 = (第i帧完成时间-第0帧发出时间)/(i+1)
        self.__ifr_latency = [-1. for _ in range(ifr_num)]  # 各IFR时延(完成时间-发出时间)，未完成为-1
        # fs_time[f][s]: 第f帧第s阶段的完成时间
        # fs_time[f][i]: 第f帧第i//2个Worker的 传输完成耗时(i为偶数) / 计算完成耗时(i为奇数)
        # fs_time[f][2*w]: 第f帧第w个Worker的传输完成耗时
        # fs_time[f][2*w+1]: 第f帧第w个Worker的计算完成耗时
        # fs_time[f][-1] 很接近但不等于 ifr_latency[f]
        #   前者是最后一个Worker计算完成耗时，后者是它计算完成后Master收到report_finish的耗时
        self.__fs_time = [[-1. for s in range(self.__stg_num)] for f in range(ifr_num)]
        self.__fst_lock = threading.Lock()  # 确保fs_time的访问是串行的

    def send_group(self, ipt_group: List[Tensor], ifr_group: List[IFR], save_ipt: bool) -> None:
        """异步发送当前IFR组"""
        for i, ifr in enumerate(ifr_group):
            self.__logger.info(f"ready IFR{ifr.id}: "
                               + ', '.join(f'w{wj.worker_id}={wj.job.exec_ids}' for wj in ifr.wk_jobs))
            pd_data = (ipt_group[i] if save_ipt else None)
            pd_ipt = PendingIpt(ifr.id, pd_data, -1)
            # 可能因为pending数量达到上限而阻塞
            with self.__pd_cv:
                while len(self.__pd_dct) >= self.__pd_num:
                    self.__pd_cv.wait()
                self.__pd_dct[pd_ipt.ifr_id] = pd_ipt
                self.__pd_cv.notifyAll()
            self.__logger.info(f"start process IFR{ifr.id}", extra={'trace': True})
            pd_ipt.send_time = time.time()
            self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr)
            assert self.__ifr_max + 1 == ifr.id  # 确保IFR是按照顺序发送的
            self.__ifr_max += 1
            if ifr.id == 0:
                self.__begin_time = pd_ipt.send_time

    def finish_stage(self, ifr_id: int, worker_id: int, sub_stage: int) -> None:
        """标识一个IFR在一个Worker上的一个子阶段完成。
        sub_stage: 按照worker处理顺序排序，0为传输，1为计算"""
        assert 0 <= sub_stage <= 1  # sub_stage目前只包括传输和计算
        stage_id = worker_id*2 + sub_stage
        self.__fs_time[ifr_id][stage_id] = time.time() - self.__begin_time

    def report_finish(self, ifr_id: int) -> Optional[Tensor]:
        with self.__pd_cv:
            # 因为pd_ipt一定在里面，所以不会阻塞
            pd_ipt = self.__pd_dct.pop(ifr_id)
            self.__ifr_latency[ifr_id] = time.time() - pd_ipt.send_time
            self.__avg_latency[ifr_id] = (time.time() - self.__begin_time) / (ifr_id + 1)
            self.__pd_cv.notifyAll()
        with self.__fst_lock:
            if self.__fs_time[ifr_id][-1] < 0:  # 前面的worker提前完成了，最后一个Worker没有执行
                # 从后往前，第一个有时间戳的阶段
                last = next(s_ for s_ in range(self.__stg_num-1, -1, -1) if self.__fs_time[ifr_id][s_] >= 0)
                assert last >= 0
                # 把last后面的阶段完成时间戳全部设置为last的完成时间戳
                for s in range(last+1, self.__stg_num):
                    self.__fs_time[ifr_id][s] = self.__fs_time[ifr_id][last]
        assert all(tm>0 for tm in self.__fs_time[ifr_id]),\
            f"IFR{ifr_id} finished but s_time={self.__fs_time[ifr_id]}"
        self.__logger.info(f"IFR{ifr_id} finished, latency={round(self.__ifr_latency[ifr_id], 2)}s")
        if self.__ifr_latency.count(-1) == 0:  # 所有IFR均完成
            # 注意：因为调度策略可能会变化，所以IFR0可能在w0完成，而IFR1可能在w1完成，从而导致IFR可能不是按序完成的
            # 但是，同一个IFR在worker间的执行顺序是固定的，所以相邻Worker的缓存应该是可以保证一致的
            total = time.time() - self.__begin_time
            self.__logger.info(f"All {self.__ifr_num} IFRs finished, "
                               f"total={round(total, 2)}s, avg={round(total/self.__ifr_num, 2)}s")
            self.__logger.info(f"ifr_latency={self.__ifr_latency}")
            self.__logger.info(f"avg_latency={self.__avg_latency}")
        return pd_ipt.ipt

    def stage_ready_time(self, fs_cost: List[List[float]]) -> Optional[List[float]]:
        """输入各帧各阶段的预估耗时，给出各阶段完成pending这些帧的时间
        :param fs_cost: fs_cost[f][s] 为 第f帧在第s阶段的预计耗时cost。若fs_cost不合法则返回None
        :return pending这些帧各阶段预计的完成耗时，时间从第0帧发出去开始计。若没有pending帧，则从0开始计
        """
        # 合法性检查: 帧数正确，阶段数正确
        if not (len(fs_cost) == self.__ifr_max + 1
                and all(len(s_cost) == self.__stg_num for s_cost in fs_cost)):
            return None

        # 检查第0帧是否已经开始
        with self.__pd_cv:
            fmin, fmax = min(self.__pd_dct, default=-1), max(self.__pd_dct, default=-1)
        if fmin < 0:
            # pd_dct为空，没有pending的IFR
            if self.__begin_time == 0.:  # 第0帧还没开始，起点为0
                begin = 0.
            else:  # 第0帧已经开始了，起点为当前时间
                begin = time.time() - self.__begin_time
            return [begin for _ in range(self.__stg_num)]

        # 有可能IFR9早于IFR8完成，遇到这种情况再具体分析
        assert fmax == self.__ifr_max, f"IFR{self.__ifr_max} finished earlier than IFR{fmax}!"
        with self.__fst_lock:
            # fs_dp 保存所有已发出去的帧的运行状态，从0到fmax
            fs_dp = copy.deepcopy(self.__fs_time[:fmax+1])
        # s_lst 各阶段最后一个已完成的帧号
        s_lst = [-1 for _ in range(self.__stg_num)]
        for s in range(self.__stg_num):
            lst_f = -1  # -1表示没有已完成的帧
            for f in range(fmax, -1, -1):
                if fs_dp[f][s] >= 0:
                    lst_f = f
                    break
            s_lst[s] = lst_f
        # first_s(s_time) 找到s_time中第一个小于零的元素的索引
        first_s = lambda s_time: next(s_ for s_ in range(len(s_time)) if s_time[s_] < 0)
        # 先填fmin
        if fmin == 0:  # 第0帧在处理中
            # 先填fst_s
            fst_s = first_s(fs_dp[0])  # 第一个没有完成的阶段
            if fst_s == 0:  # 第0阶段完成时间 = 本身耗时
                fs_dp[fmin][fst_s] = fs_cost[fmin][fst_s]
            else:  # 上一阶段真实完成时间 + 本身耗时
                fs_dp[fmin][fst_s] = fs_dp[fmin][fst_s-1] + fs_cost[fmin][fst_s]
            # 再填fst_s后面
            for s in range(fst_s+1, self.__stg_num):
                fs_dp[fmin][s] = fs_dp[fmin][s-1] + fs_cost[fmin][s]
        else:  # fmin-1已经完成, 可以用fs_dp[fmin-1]
            fst_s = first_s(fs_dp[fmin])
            if fst_s == 0:  # 第0阶段耗时: fmin-1完成之后就可以开始了
                fs_dp[fmin][fst_s] = fs_dp[fmin-1][fst_s] + fs_cost[fmin][fst_s]
            else:
                fs_dp[fmin][fst_s] = max(fs_dp[fmin-1][fst_s], fs_dp[fmin][fst_s-1]) \
                                     + fs_cost[fmin][fst_s]
            for s in range(fst_s+1, self.__stg_num):
                fs_dp[fmin][s] = max(fs_dp[fmin-1][s], fs_dp[fmin][s-1]) + fs_cost[fmin][s]
        # 填fmin+1到fmax
        for f in range(fmin+1, fmax+1):
            fst_s = first_s(fs_dp[f])
            if fst_s == 0:  # 第0阶段完成时间：fmin完成时间 + 本身耗时
                fs_dp[f][fst_s] = fs_dp[f-1][fst_s] + fs_cost[f][fst_s]
            else:  # 需要等f-1和fst_s-1
                fs_dp[f][fst_s] = max(fs_dp[f-1][fst_s], fs_dp[f][fst_s-1]) + fs_cost[f][fst_s]
            for s in range(1, self.__stg_num):
                fs_dp[f][s] = max(fs_dp[f-1][s], fs_dp[f][s-1]) + fs_cost[f][s]
        # from matplotlib import pyplot as plt
        # # 可视化fs_dp
        # self.visualize_frames(0, fmax, fs_dp, s_lst)
        # plt.show()
        return fs_dp[fmax]

    @classmethod
    def visualize_frames(cls, fbegin: int, fend: int, fs_dp: List[List[float]],
                         s_lst: List[int]):
        """可视化fs_dp闭区间[fbegin, fend]之间的帧，fs_dp从第0帧开始
        s_lst[s] 第s阶段最新已完成的帧，-1表示没有完成的帧
        fs_dp[f][s]表示第f帧第s阶段完成时的时间。s=2*w时表示w传输完成耗时，s=2*w+1时表示w计算完成耗时
        """
        print(f"fs_dp[{fbegin}:{fend+1}]={fs_dp[fbegin:fend+1]}")
        print(f"s_lst={s_lst}")
        from matplotlib import pyplot as plt
        import matplotlib.colors as mcolors
        nstage = len(fs_dp[fbegin])
        assert nstage % 2 == 0
        nworker = nstage // 2
        fig = plt.figure()
        ax = fig.subplots()
        ax.invert_yaxis()
        ticklabels = ['m->w0', 'w0']
        for w in range(1, nworker):
            ticklabels.extend([f'w{w-1}->w{w}', f'w{w}'])
        assert len(ticklabels) == nstage, f"len({ticklabels})!={nstage}!"
        plt.yticks(list(range(nstage)), ticklabels)
        colors = list(mcolors.XKCD_COLORS.values())
        # bg_time为第fbegin帧第0阶段的开始时间
        bg_time = (0 if fbegin == 0 else fs_dp[fbegin-1][0])
        # 绘制第fbegin帧
        plt.barh(0, fs_dp[fbegin][0]-bg_time, left=bg_time, color=colors[0])
        for s in range(1, nstage):
            plt.barh(s, fs_dp[fbegin][s]-fs_dp[fbegin][s-1], left=fs_dp[fbegin][s-1], color=colors[0])
        # 绘制fbegin+1到fend的帧
        for f in range(fbegin+1, fend+1):
            plt.barh(0, fs_dp[f][0]-fs_dp[f-1][0], left=fs_dp[f-1][0], color=colors[f-fbegin])
            for s in range(1, nstage):
                left = max(fs_dp[f-1][s], fs_dp[f][s-1])
                plt.barh(s, fs_dp[f][s]-left, left=left, color=colors[f-fbegin])
        # 标识实际和估计之间的分割线
        for s in range(nstage):
            if s_lst[s] >= 0:
                plt.plot([fs_dp[s_lst[s]][s], fs_dp[s_lst[s]][s]], [s-.4, s+.4], 'b--')
