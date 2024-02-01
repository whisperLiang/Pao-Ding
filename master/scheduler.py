from abc import abstractmethod
from fractions import Fraction
from typing import List, Optional, Tuple, Dict, Any, Type

import torch
from torch import Tensor

from core.executor import Node, Job
from core.ifr import WkJob, IFR
from core.itg_executor import ExNode, ItgExecutor, ItgJob
from core.predictor import Predictor, NZPred
from core.dag_dnn import DagDNN


class _SizingExNode(ExNode):
    """在Executor中运行，以获取输出数据大小的信息"""
    def __init__(self, node: Node):
        super().__init__(node)
        self.out_size = None

    def execute(self, inputs) -> None:
        super().execute(inputs)
        self.out_size = tuple(self.get_output().shape)


class SizedNode(Node):
    """根据输出数据大小和稀疏率预测模型进行初始化"""
    def __init__(self, se_node: _SizingExNode):
        super().__init__(se_node)
        self.out_size: Tuple[int, int, int] = se_node.out_size  # (通道数, 行数, 列数)
        R, C = se_node.out_size[-2:]
        self.nz_thres = (1 - 1/C - 1/(R*C))/2  # 一个通道的非零占比<nz_thres时，稀疏压缩的数据传输量更少

    @classmethod
    def raw2dag_sized(cls, dag_dnn: DagDNN, frame_size: Tuple[int, int]):
        """使用DagDNN和指定的帧大小，初始化保存有输出数据大小的DAG图"""
        itg_extor = ItgExecutor(dag_dnn, _SizingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = ItgJob(list(range(len(dag_dnn.layers))), [len(dag_dnn.layers)-1], dag_dnn.node2index, {0: ipt})
        itg_extor.exec(job)
        return [[cls(se_node).out_size, cls(se_node).nz_thres] for se_node in itg_extor.dag()]


class Scheduler:
    @abstractmethod
    def __init__(self, s_dag: List[SizedNode], nzpred: NZPred,
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        pass

    @abstractmethod
    def group_size(self) -> int:
        """建议的group大小"""

    def fs_cost(self) -> List[List[float]]:
        """返回各帧各个阶段的耗时估计。返回值不会被修改
        :return fs_cost: fs_cost[f][s]为第f帧第s阶段的耗时估计
        s=2*w时表示w传输耗时，s=2*w+1时表示w计算耗时 (w为Worker的ID)
        返回的数组fs_cost需满足如下条件：
            1. len(fs_cost) = 当前Scheduler已发出去的总帧数
            2. len(fs_cost[f]) = worker数*2
        否则gen_ifr_group的s_ready参数将为None
        """
        return []

    @abstractmethod
    def gen_ifr_group(self, ifr_cnt: int, pre_ipt: Tensor,
                      ipt_group: List[Tensor], s_ready: List[float] = None) -> List[IFR]:
        """为一组输入生成相应的IFR组
        :param ifr_cnt: 当前group中第一个IFR的id
        :param pre_ipt: 上一帧的输入
        :param ipt_group: 要发出去的所有输入帧，1<=长度<=group_size
        :param s_ready: 根据fs_cost和当前IFR状态给出的各阶段就绪时间的估计。fs_cost不合法则为None
        :return ifr_group ifr_group[i]对应ipt_group[i]
        """

    @classmethod
    def split_chain(cls, ly_comp: List[float], wk_cap: List[float]) -> List[int]:
        """按照Worker的计算能力，对链状的CNN进行切割使得各Worker耗时相近，返回切割点（切割点属于前一个Worker）
        :param ly_comp: ly_comp[l]为第l层的计算量，即baseline的worker运行耗时
        :param wk_cap: 按照Worker执行顺序，各Worker的相对计算能力，其中一个worker的计算能力为1，作为baseline
        :return: 各Worker执行的层数
        """
        assert len(ly_comp) > 0, "The number of layers is 0!"
        assert len(wk_cap) > 0, "There is no worker!"
        ly_comp_acc = [0. for _ in range(len(ly_comp) + 1)]  # ly_comp的累积值: ly_comp_acc[l] = sum(ly_comp[:l])
        for l in range(len(ly_comp)):
            ly_comp_acc[l + 1] = ly_comp_acc[l] + ly_comp[l]
        total_comp, total_cap = sum(ly_comp), sum(wk_cap)
        wk_comp = [cap / total_cap * total_comp for cap in wk_cap]  # 各worker应该分得的总计算量
        wk_comp_acc = [0. for _ in range(len(wk_comp) + 1)]  # wk_comp的累积值: wk_comp_acc[l] = sum(wk_comp[:l])
        for w in range(len(wk_comp)):
            wk_comp_acc[w + 1] = wk_comp_acc[w] + wk_comp[w]
        # 按照Worker的执行顺序，每个Worker及其前驱执行的最后一个层,
        #   若w-1和w的层相同则表示w没有执行任何层，取值-1表示没有执行任何一个层
        wk_ly = []
        lycnt = 0  # 上一个worker执行的最后一个层+1
        for w, acc in enumerate(wk_comp_acc[1:-1]):  # acc[0]不对应任何worker；不考虑最后一个worker，因为它肯定是最后一个层
            ly = -100  # ly可能为-1，此时表示当前没有执行任何一个层
            while ly == -100:
                if lycnt == len(ly_comp_acc):  # 所有层都被前面的worker分配掉了
                    ly = len(ly_comp_acc) - 1
                elif ly_comp_acc[lycnt] <= acc <= ly_comp_acc[lycnt + 1]:  # lycnt处于边界上
                    if acc - ly_comp_acc[lycnt] <= ly_comp_acc[lycnt + 1] - acc:
                        ly = lycnt - 1
                    else:
                        ly = lycnt
                        lycnt += 1
                else:
                    lycnt += 1
            wk_ly.append(ly)
        wk_ly.append(len(ly_comp) - 1)
        # wk_ly转成wk_lynum，表示各Worker执行的层数
        wk_lynum = [wk_ly[0] + 1] + [wk_ly[w] - wk_ly[w - 1] for w in range(1, len(wk_ly))]
        return wk_lynum

    @classmethod
    def wk_lynum2layers_chain(cls, begin_layer: int, wk_lynum: List[int]) -> List[List[int]]:
        """根据各Worker执行层数，从begin_layer开始，按照执行顺序为Worker分配具体执行的层。只考虑链状CNN
        :param begin_layer: Worker的任务从第几层开始，包括begin_layer
        :param wk_lynum: 各worker的层数
        :return: 各worker具体执行哪几层
        """
        ly_cnt = begin_layer
        wk_layers = []
        for lynum in wk_lynum:
            wk_layers.append(list(range(ly_cnt, ly_cnt + lynum)))
            ly_cnt += lynum
        return wk_layers

    @classmethod
    def dif2lbsz(cls, dif_ipt: Tensor, s_dag: List[SizedNode], predictors: List[Predictor]):
        """根据输入差值，估计各层输出差值的数据量
        :return 各层的差值数据量lbsz
        """
        cnz = [float(chan.count_nonzero() / chan.nelement()) for chan in dif_ipt[0]]
        lcnz = cls.predict_dag(cnz, s_dag, predictors)
        lsz = cls.lcnz2lsz(lcnz, s_dag)
        return [sz * 4 for sz in lsz]

    @classmethod
    def predict_dag(cls, ipt_cnz: List[float], dag: List[Node], predictors: List[Predictor]) -> List[List[float]]:
        """根据输入数据与上一帧的非零占比，预测DAG各个节点输出数据与上一帧的非零占比"""
        assert len(dag) == len(predictors)
        results = [[] for _ in range(len(dag))]
        results[0] = predictors[0].predict([ipt_cnz])
        for d in dag[0].descendants:
            cls._predict_dag(d, results, dag, predictors)
        return results

    @classmethod
    def lcnz2lsz(cls, lcnz: List[List[float]], s_dag: List[SizedNode]) -> List[float]:
        """对各层，根据通道的非零占比计算出输出数据总元素个数"""
        lsz = []
        for l in range(len(s_dag)):
            size = 0
            try:
                H, R, C = s_dag[l][0][-3:]
            except:
                R, C = s_dag[l][0][-2:]
                H = 1
            for c in range(H):
                p = lcnz[l][c]
                if p < s_dag[l][1]:
                    size += 2 * R * C * p + R + 1
                else:
                    size += R * C
            lsz.append(size)
        return lsz

    @classmethod
    def elys2olys(cls, elys: List[int], dag: List[Node]) -> List[int]:
        """执行elys这些层，应该给出哪些层的输出数据"""
        olys = []  # 输出层
        elyset = set(elys)
        for ly in elys:
            # 如果 ly没有后继 或者 有的后继不在elys中，那么就把ly加入到olys
            if len(dag[ly].descendants) == 0 or any(ds not in elyset for ds in dag[ly].descendants):
                olys.append(ly)
        return olys

    @classmethod
    def _predict_dag(cls, node_id: int, res_lcnz: List[List[float]],
                     dag: List[Node], predictors: List[Predictor]) -> None:
        """模仿core.dag_dnn.DagDNN.__execute_dag
        res_lcnz的每个元素必须初始化为空列表
        """
        if len(res_lcnz[node_id]) > 0:
            return
        acnz = []
        for aid in dag[node_id].ancients:
            if len(res_lcnz[aid]) > 0:
                acnz.append(res_lcnz[aid])
            else:
                return
        res_lcnz[node_id] = predictors[node_id].predict(acnz)
        for d in dag[node_id].descendants:
            cls._predict_dag(d, res_lcnz, dag, predictors)


class G1Scheduler(Scheduler):
    """group_size为1的调度器，用于兼容以前写的Scheduler"""
    def group_size(self) -> int:
        return 1

    def gen_ifr_group(self, ifr_cnt: int, pre_ipt: Tensor,
                      ipt_group: List[Tensor], s_ready: List[float] = None) -> List[IFR]:
        assert len(ipt_group) == 1
        return [IFR(ifr_cnt, self.gen_wk_jobs(ifr_cnt, pre_ipt, ipt_group[0]))]

    @abstractmethod
    def gen_wk_jobs(self, ifr_id: int, pre_ipt: Tensor, cur_ipt: Tensor) -> List[WkJob]:
        """为当前的输入帧生成IFR
        :param ifr_id IFR序号
        :param pre_ipt 上一帧的输入数据
        :param cur_ipt 当前帧的输入数据
        :return wk_jobs 作为IFR.wk_jobs
        """
