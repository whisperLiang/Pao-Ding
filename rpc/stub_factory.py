import logging
import pickle
import threading
from queue import Queue
from typing import Dict, Any, List, Optional, Callable

import grpc
from torch import Tensor

from core.ifr import IFR
from core.predictor import NZPred
from core.util import SerialTimer, timed_rpc, tensor2msg
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import Req, FinishMsg, LayerCostMsg, NZPredMsg, StageMsg

MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
GRPC_OPTIONS=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
]


class AsyncClient(threading.Thread):
    """异步执行RPC，仅适用于不需要返回值的函数"""
    def __init__(self):
        super().__init__(daemon=True)
        self._que = Queue()
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        while True:
            func, args = self._que.get()
            if func == self.stop:
                break
            self._logger.debug(f"{func.__name__}({args}) calling...")
            func(*args)
            self._logger.debug(f"{func.__name__}({args}) finished")

    def call_async(self, func: Callable, *args) -> None:
        self._que.put((func, args))
        self._logger.debug(f"{func.__name__}({args}) enqueued")

    @classmethod
    def stop(cls) -> None:
        """此函数仅用于标识退出，不做任何事情"""


class MasterStub:
    def __init__(self, stg_rev_que: 'Queue[StageMsg]', fsh_rev_que: 'Queue[FinishMsg]', aclient: AsyncClient):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._stg_rev_que = stg_rev_que
        self._fsh_rev_que = fsh_rev_que
        self._client = aclient

    def finish_stage(self, ifr_id: int, worker_id: int, sub_stage: int) -> None:
        self._client.call_async(self._finish_stage, ifr_id, worker_id, sub_stage)

    def _finish_stage(self, ifr_id: int, worker_id: int, sub_stage: int) -> None:
        self._stg_rev_que.put(StageMsg(ifr_id=ifr_id, worker_id=worker_id, sub_stage=sub_stage))

    def report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        self._client.call_async(self._report_finish, ifr_id, tensor4d)

    def _report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        self._fsh_rev_que.put(self._encode_finish(ifr_id, tensor4d))

    def _encode_finish(self, ifr_id: int, tensor4d: Tensor = None) -> FinishMsg:
        self._logger.info(f"start encode IFR{ifr_id}-finished", extra={'trace': True})
        if tensor4d is None:
            msg = FinishMsg(ifr_id=ifr_id)
        else:
            msg = FinishMsg(ifr_id=ifr_id, arr3d=tensor2msg(tensor4d))
        self._logger.info(f"finish encode IFR{ifr_id}-finished", extra={'trace': True})
        self._logger.info(f"start transmit IFR{ifr_id}-finished", extra={'trace': True})
        return msg


class WorkerStub:
    def __init__(self, id_, channel: grpc.Channel, aclient: AsyncClient):
        self._name = f'worker{id_}'  # 请求目标的名称
        self._stub = msg_pb2_grpc.WorkerStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = aclient

    def new_ifr(self, ifr: IFR) -> None:
        self._client.call_async(self._new_ifr, ifr)

    def _new_ifr(self, ifr: IFR) -> None:
        self._logger.info(f"start encode IFR{ifr.id}", extra={'trace': True})
        msg = ifr.to_msg()
        self._logger.info(f"finish encode IFR{ifr.id}", extra={'trace': True})
        self._logger.info(f"start transmit IFR{ifr.id}", extra={'trace': True})
        timed_rpc(self._stub.new_ifr, msg, self._name, 's', self._logger)

    def layer_cost(self) -> List[float]:
        msg = timed_rpc(self._stub.layer_cost, Req(), self._name, 'r', self._logger)
        with SerialTimer(SerialTimer.SType.LOAD, LayerCostMsg, self._logger):
            costs = pickle.loads(msg.costs)
        return costs


class TrainerStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.TrainerStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_nzpred(self) -> NZPred:
        msg = timed_rpc(self._stub.get_nzpred, Req(), 'trainer', 'r', self._logger)
        with SerialTimer(SerialTimer.SType.LOAD, NZPredMsg, self._logger):
            return pickle.loads(msg.nzpred)


class MStubFactory:
    def __init__(self, config: Dict[str, Any]):
        wk_num = len(config['port']['worker'])
        net_config = config['net']
        self.wk_chan: List[Optional[grpc.Channel]] = [None for _ in range(wk_num)]
        for route, addr in net_config.items():
            if route.startswith('m->w'):
                self.wk_chan[int(route.replace('m->w', ''))] = grpc.insecure_channel(addr, options=GRPC_OPTIONS)
        assert all(chan is not None for chan in self.wk_chan)
        self.trn_chan = grpc.insecure_channel(net_config['m->t'], options=GRPC_OPTIONS)
        self.aclient = AsyncClient()
        self.aclient.start()

    def worker(self, worker_id: int) -> WorkerStub:
        return WorkerStub(worker_id, self.wk_chan[worker_id], self.aclient)

    def trainer(self) -> TrainerStub:
        return TrainerStub(self.trn_chan)

    def stop(self) -> None:
        self.aclient.call_async(AsyncClient.stop)


class WStubFactory:
    """Worker的Stub工厂，用于创建Worker的Stub
    用于创建并管理与工作线程相关的gRPC stubs（存根）以及异步客户端。"""
    def __init__(self, id_: int, stg_rev_que: 'Queue[StageMsg]', fsh_rev_que: 'Queue[FinishMsg]', config: Dict[str, Any]):
        wk_num = len(config['port']['worker'])
        net_config = config['net']
        self.id = id_
        self.nwk_chan = None
        if id_ < wk_num - 1:
            self.nwk_chan = grpc.insecure_channel(net_config[f'w{id_}->w{id_+1}'], options=GRPC_OPTIONS)
        self.stg_rev_que = stg_rev_que
        self.fsh_rev_que = fsh_rev_que
        # 发出去的RPC都要用aclient发送，以确保同一个Worker上IFR是按序处理的
        self.aclient = AsyncClient()
        self.aclient.start()

    def worker(self) -> WorkerStub:
        assert self.nwk_chan is not None
        return WorkerStub(self.id+1, self.nwk_chan, self.aclient)

    def master(self) -> MasterStub:
        return MasterStub(self.stg_rev_que, self.fsh_rev_que, self.aclient)

    def stop(self) -> None:
        self.stg_rev_que.put(StageMsg(ifr_id=-1))
        self.fsh_rev_que.put(FinishMsg(ifr_id=-1))  # 使用ifr_id=-1标识运行结束
        self.aclient.call_async(AsyncClient.stop)  # 使用AsyncClient.stop标识运行结束
