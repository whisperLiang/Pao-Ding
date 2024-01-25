import logging
import os
import pickle
import sys
import time
from enum import Enum, auto
from typing import Callable, Any, Type

import numpy as np
import torch
from google.protobuf.message import Message
from scipy.sparse import csr_matrix
from torch import Tensor

from rpc.msg_pb2 import Arr3dMsg, Arr2dMsg


def msg2tensor(arr3d: Arr3dMsg) -> Tensor:
    tensor3ds = []
    for arr2d in arr3d.arr2ds:
        if arr2d.sparse:
            tensor3ds.append(torch.as_tensor(pickle.loads(arr2d.data).toarray()).unsqueeze(0))
        else:
            tensor3ds.append(torch.as_tensor(pickle.loads(arr2d.data)).unsqueeze(0))
    return torch.cat(tensor3ds).unsqueeze(0)


def tensor2msg(tensor4d: Tensor, sparse: bool = True) -> Arr3dMsg:
    """tensor4d：要序列化的数据；sparse=True表示尝试使用稀疏表示，否则不使用稀疏表示"""
    if not sparse:
        arr3d = Arr3dMsg()
        for mtrx2d in tensor4d.numpy()[0]:
            arr3d.arr2ds.add(sparse=False, data=pickle.dumps(mtrx2d))
        return arr3d
    if tensor4d.shape[2] > tensor4d.shape[3]:  # 行数>列数时，应该用CSC
        logger = logging.getLogger('tensor2msg')
        logger.warning(f"shape={tensor4d.shape}. nrow>ncol, CSC is recommended, instead of CSR!")
    arr3d = Arr3dMsg()
    for mtrx2d in tensor4d.numpy()[0]:
        arr2d = Arr2dMsg()
        arr2d.sparse = (np.count_nonzero(mtrx2d)*2+mtrx2d.shape[0]+1 < mtrx2d.size)
        if arr2d.sparse:
            arr2d.data = pickle.dumps(csr_matrix(mtrx2d))
        else:
            arr2d.data = pickle.dumps(mtrx2d)
        arr3d.arr2ds.append(arr2d)
    return arr3d


def cached_func(file_name: str, func: Callable, *args,
                prefix: str = '.cache', logger: logging.Logger = None) -> Any:
    """对于执行耗时较长的函数，将其运行结果用pickle序列化，缓存在.cache目录下
    :param file_name 缓存文件名称，通过检查该文件名是否存在确定是否执行func
    :param func 要执行的函数
    :param args 函数的参数
    :param prefix 缓存文件的相对路径，默认为当前执行路径下的.cache目录
    :param logger 写入的logger，默认logger名为func的函数名，写入stdout
    :return 函数的结果
    """
    if logger is None:
        # 如果没有传入logger，则默认写入stdout
        logger = logging.getLogger(func.__name__)
        if not logger.hasHandlers():
            # 因为这个logger是全局共用的，所以不能重复添加Handler
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.DEBUG)
    file_path = prefix + '/' + file_name
    if os.path.isfile(file_path):
        logger.debug(f"{file_name} exists, loading...")
        with open(file_path, 'rb') as cfile:
            return pickle.load(cfile)
    else:
        if not os.path.isdir('.cache'):
            os.mkdir('.cache')
        logger.debug(f"{file_name} not exists, generating...")
        data = func(*args)
        logger.debug(f"{file_name} generated, writing...")
        with open(file_path, 'wb') as cfile:
            pickle.dump(data, cfile)
        return data


class Timer:
    def __init__(self):
        self._begin = 0
        self._cost = 0

    def __enter__(self):
        self._begin = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cost = time.time() - self._begin

    def cost(self):
        return self._cost


class ActTimer(Timer):
    def __init__(self, act_name: str, logger: logging.Logger):
        super().__init__()
        self._act_name = act_name
        self._logger = logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self._logger.debug(f"{self._act_name} costs {round(self.cost(), 2)}s")


class SerialTimer(ActTimer):
    class SType(Enum):
        DUMP = auto()
        LOAD = auto()

    def __init__(self, s_type: SType, obj_type: Type, logger: logging.Logger):
        act_name = ('Dumping' if s_type == self.SType.DUMP else 'Loading')
        super().__init__(f"{act_name} {obj_type.__name__}", logger)


def timed_rpc(rpc_func: Callable, req_msg: Message, dest: str, mode: str, logger: logging.Logger) -> Message:
    """对整个rpc计时，rpc_func应该只有发送或接收明显耗时，mode为s表示发送，r表示接收"""
    with Timer() as timer:
        rsp_msg = rpc_func(req_msg)
    if 's' in mode:
        mb_size = req_msg.ByteSize() / 1024 / 1024  # 单位MB
        # 计算网速时，添加极小量避免本地模拟时出现除零异常
        logger.debug(f"Sending {req_msg.__class__.__name__} to {dest} costs {round(timer.cost(), 2)}s, "
                     f"size={round(mb_size, 2)}MB, speed={round(mb_size / (timer.cost() + 1e-6), 2)}MB/s")
    if 'r' in mode:
        mb_size = rsp_msg.ByteSize() / 1024 / 1024  # 单位MB
        logger.debug(f"Getting {rsp_msg.__class__.__name__} from {dest} costs {round(timer.cost(), 2)}s, "
                     f"size={round(mb_size, 2)}MB, speed={round(mb_size / (timer.cost() + 1e-6), 2)}MB/s")
    return rsp_msg
