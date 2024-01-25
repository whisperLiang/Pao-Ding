import logging
import pickle
from queue import Queue
from concurrent import futures
from typing import Dict, Any

import grpc

from core.ifr import IFR
from core.raw_dnn import RawDNN
from core.util import SerialTimer
from rpc.msg_pb2 import IFRMsg, Rsp, Req, LayerCostMsg, FinishMsg, StageMsg
from rpc import msg_pb2_grpc
from rpc.stub_factory import WStubFactory, GRPC_OPTIONS
from worker.worker import Worker


class WorkerServicer(msg_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id: int, config: Dict[str, Any]):
        self.job_type = config['job']
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stg_rev_que: 'Queue[StageMsg]' = Queue()
        self.fsh_rev_que: 'Queue[FinishMsg]' = Queue()
        self.worker = Worker(worker_id, RawDNN(config['dnn_loader']()), config['frame_size'], config['check'],
                             config['executor'], WStubFactory(worker_id, self.stg_rev_que, self.fsh_rev_que, config),
                             config['worker'])
        self.worker.start()
        self.__serve(str(config['port']['worker'][worker_id]))

    def new_ifr(self, ifr_msg: IFRMsg, context: grpc.ServicerContext) -> Rsp:
        self.logger.info(f"finish transmit IFR{ifr_msg.id}", extra={'trace': True})
        self.logger.info(f"start decode IFR{ifr_msg.id}", extra={'trace': True})
        ifr = IFR.from_msg(ifr_msg, self.job_type)
        self.logger.info(f"finish decode IFR{ifr_msg.id}", extra={'trace': True})
        self.worker.new_ifr(ifr)
        return Rsp()

    def layer_cost(self, req: Req, context: grpc.ServicerContext) -> LayerCostMsg:
        costs = self.worker.layer_cost()
        with SerialTimer(SerialTimer.SType.DUMP, LayerCostMsg, self.logger):
            return LayerCostMsg(costs=pickle.dumps(costs))

    def finish_stage_rev(self, req: Req, context: grpc.ServicerContext) -> FinishMsg:
        while 1:
            stage_msg = self.stg_rev_que.get()
            if stage_msg.ifr_id < 0:
                return
            yield stage_msg

    def report_finish_rev(self, req: Req, context: grpc.ServicerContext) -> FinishMsg:
        while 1:
            finish_msg = self.fsh_rev_que.get()
            if finish_msg.ifr_id < 0:
                return
            yield finish_msg

    def __serve(self, port: str):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=GRPC_OPTIONS)
        msg_pb2_grpc.add_WorkerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.logger.info("start serving...")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            self.logger.info(f"Ctrl-C received, exit")
            self.worker.stop()
            server.stop(.5)
