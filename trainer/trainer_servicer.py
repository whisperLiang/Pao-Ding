import logging
import pickle
from concurrent import futures
from typing import Any, Dict

import grpc

from core.raw_dnn import RawDNN
from core.util import SerialTimer
from rpc.msg_pb2 import Req, NZPredMsg
from rpc import msg_pb2_grpc
from rpc.stub_factory import GRPC_OPTIONS
from trainer.trainer import Trainer


class TrainerServicer(msg_pb2_grpc.TrainerServicer):
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trainer = Trainer(RawDNN(config['dnn_loader']()), config['video_path'],
                               config['frame_size'], config['trainer'])
        self.trainer.start()
        self.__serve(str(config['port']['trainer']))

    def get_nzpred(self, request: Req, context: grpc.ServicerContext) -> NZPredMsg:
        nzpred = self.trainer.get_nzpred()
        with SerialTimer(SerialTimer.SType.DUMP, NZPredMsg, self.logger):
            return NZPredMsg(nzpred=pickle.dumps(nzpred))

    def __serve(self, port: str):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=GRPC_OPTIONS)
        msg_pb2_grpc.add_TrainerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.logger.info("start serving...")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            self.logger.info(f"Ctrl-C received, exit")
            server.stop(.5)
