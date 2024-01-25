import logging
import threading
from typing import Dict, Any

import grpc

from core.raw_dnn import RawDNN
from core.util import SerialTimer, msg2tensor
from master.master import Master
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import FinishMsg, Req, StageMsg
from rpc.stub_factory import MStubFactory, GRPC_OPTIONS


class MasterServicer:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.wk_addr = ['' for _ in range(len(config['port']['worker']))]
        for rt, addr in config['net'].items():
            if rt.startswith('m->w'):
                self.wk_addr[int(rt.replace('m->w', ''))] = addr
        self.master = Master(len(config['port']['worker']), RawDNN(config['dnn_loader']()),
                             config['video_path'], config['frame_size'], config['job'], config['check'],
                             MStubFactory(config), config['master'])
        threads = [threading.Thread(target=self.__report_finish_rev, args=(addr,), daemon=True) for addr in self.wk_addr]
        threads += [threading.Thread(target=self.__finish_stage_rev, args=(addr,), daemon=True) for addr in self.wk_addr]
        for thread in threads:
            thread.start()
        try:
            self.master.run()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info(f"Ctrl-C received, exit")

    def __report_finish_rev(self, addr: str) -> None:
        stub = msg_pb2_grpc.WorkerStub(grpc.insecure_channel(addr, options=GRPC_OPTIONS))
        response = stub.report_finish_rev(Req())
        for finish_msg in response:
            self.__report_finish(finish_msg)

    def __report_finish(self, finish_msg: FinishMsg) -> None:
        self.logger.info(f"finish process IFR{finish_msg.ifr_id}", extra={'trace': True})
        if len(finish_msg.arr3d.arr2ds) == 0:
            tensor = None
        else:
            with SerialTimer(SerialTimer.SType.LOAD, FinishMsg, self.logger):
                tensor = msg2tensor(finish_msg.arr3d)
        self.master.report_finish(finish_msg.ifr_id, tensor)

    def __finish_stage_rev(self, addr: str) -> None:
        stub = msg_pb2_grpc.WorkerStub(grpc.insecure_channel(addr, options=GRPC_OPTIONS))
        response = stub.finish_stage_rev(Req())
        for stage_msg in response:
            self.__finish_stage(stage_msg)

    def __finish_stage(self, stage_msg: StageMsg) -> None:
        self.logger.debug(f"IFR{stage_msg.ifr_id}: "
                          f"worker{stage_msg.worker_id} sub{stage_msg.sub_stage} finished")
        self.master.finish_stage(stage_msg.ifr_id, stage_msg.worker_id, stage_msg.sub_stage)
