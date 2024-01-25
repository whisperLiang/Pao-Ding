import logging
import time
from queue import Queue
from typing import Dict, Any, List, Tuple, Type
from threading import Thread, Condition

import torch
from torch import Tensor
import tqdm

from core.dif_executor import DifExecutor
from core.executor import Node, Executor
from core.ifr import IFR
from core.itg_executor import ItgExecutor, ExNode, ItgJob
from core.util import cached_func, ActTimer
from core.raw_dnn import RawDNN
from rpc.stub_factory import WStubFactory


class Worker(Thread):
    """以pipeline的方式执行Job"""
    def __init__(self, id_: int, raw_dnn: RawDNN, frame_size: Tuple[int, int], check: bool,
                 executor_type: Type[Executor], stb_fct: WStubFactory, config: Dict[str, Any]) -> None:
        super().__init__(daemon=True)
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__id = id_
        self.__check = check
        self.__cv = Condition()
        self.__executor: Executor = executor_type(raw_dnn)
        self.__ex_queue: Queue[IFR] = Queue()  # 执行的任务队列
        self.__stb_fct = stb_fct
        self.__logger.info(f"Worker{self.__id} profiling...")
        self.__costs = []
        # TODO: 缓存文件名包括hostname
        costs = cached_func(f"w{id_}.{raw_dnn.dnn_cfg.name}.{frame_size[0]}x{frame_size[1]}.cst", self.profile_dnn_cost,
                            raw_dnn, frame_size, config['prof_niter'], logger=self.__logger)
        self.__logger.info(f"layer_costs={costs}")
        with self.__cv:
            self.__costs = costs
            self.__cv.notifyAll()

    def id(self):
        return self.__id

    def run(self) -> None:
        # last_ifr_id = -1
        while True:
            ifr = self.__ex_queue.get()
            if ifr.id < 0:
                break
            self.__logger.debug(f"get IFR{ifr.id}")
            # TODO: 当Worker执行顺序不保证时，这里应该检查ifr中的数据为dif还是itg，dif需要一致性，而itg无需一致性
            # assert ifr.id == last_ifr_id + 1, "IFR sequence is inconsistent, DifJob cannot be executed!"
            assert len(ifr.wk_jobs) > 0, "IFR has finished, cannot be executed!"
            assert ifr.wk_jobs[0].worker_id == self.__id, \
                f"IFR(wk={ifr.wk_jobs[0].worker_id}) should not appear in Worker{self.__id}!"
            self.__logger.info(f"executing IFR{ifr.id}: {ifr.wk_jobs[0].job.exec_ids}")
            self.__logger.info(f"start execute IFR{ifr.id}", extra={'trace': True})
            id2data = self.__executor.exec(ifr.wk_jobs[0].job)
            self.__logger.info(f"finish execute IFR{ifr.id}", extra={'trace': True})
            self.__stb_fct.master().finish_stage(ifr.id, self.__id, 1)
            # last_ifr_id = ifr.id
            # IFR已经处于最终状态，则直接发给Master
            if ifr.is_final():
                self.__logger.info(f"IFR{ifr.id} finished")
                if self.__check:
                    if isinstance(self.__executor, DifExecutor):
                        result = next(iter(self.__executor.last_out().values()))
                    elif isinstance(self.__executor, ItgExecutor):
                        result = next(iter(id2data.values()))
                    else:
                        raise NotImplementedError()
                else:
                    result = None
                self.__stb_fct.master().report_finish(ifr.id, result)
            else:
                ifr.switch_next(id2data)
                self.__stb_fct.worker().new_ifr(ifr)

    def new_ifr(self, ifr: IFR) -> None:
        self.__ex_queue.put(ifr)
        self.__stb_fct.master().finish_stage(ifr.id, self.__id, 0)

    def stop(self) -> None:
        self.__stb_fct.stop()
        self.__ex_queue.put(IFR(-1, []))  # 用特殊IFR标识退出

    def layer_cost(self) -> List[float]:
        """执行配置文件指定的CNN，返回每层耗时"""
        with self.__cv:
            while len(self.__costs) == 0:
                self.__cv.wait()
            self.__logger.debug("got layer costs")
            return self.__costs

    class _TimingExNode(ExNode):
        def __init__(self, node: Node):
            super().__init__(node)
            self.cost = 0

        def execute(self, *inputs: Tensor) -> None:
            begin = time.time()
            super().execute(*inputs)
            self.cost = time.time() - begin

    @classmethod
    def profile_dnn_cost(cls, raw_dnn: RawDNN, frame_size: Tuple[int, int], niter: int) -> List[float]:
        itg_extor = ItgExecutor(raw_dnn, cls._TimingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = ItgJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        layer_cost = [0 for _ in range(len(raw_dnn.layers))]
        for _ in tqdm.tqdm(range(niter)):
            itg_extor.exec(job)
            for pnode in itg_extor.dag():
                layer_cost[pnode.id] += pnode.cost
        for l in range(len(raw_dnn.layers)):
            layer_cost[l] /= niter
        return layer_cost
