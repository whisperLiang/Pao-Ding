from typing import List

import cv2
import numpy as np
import scipy.sparse
import torch

from core.itg_executor import ItgExecutor, ItgJob
from core.raw_dnn import RawDNN
from core.util import tensor2msg, msg2tensor
from dnn_models.chain import prepare_alexnet, prepare_vgg19
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from unit_tests.common import get_ipt_from_video


def _run_jobs(raw_dnn: RawDNN, wk2jobs: List[ItgJob]):
    cap = cv2.VideoCapture(f'../media/road.mp4')
    ipt = get_ipt_from_video(cap)
    # 注意：第0层必须是InputModule
    wk2jobs[0].id2data[0] = ipt
    executor = ItgExecutor(raw_dnn)
    ex_out = {}
    for wk, cur_job in enumerate(wk2jobs):
        id2opt = executor.exec(cur_job)
        if wk + 1 < len(wk2jobs):
            wk2jobs[wk + 1].id2data = id2opt
        else:
            ex_out = id2opt
    # 直接执行RawLayer，以检查正确性
    results = raw_dnn.execute(ipt)
    assert len(ex_out) == 1, "There is more than 1 output node!"
    assert torch.allclose(results[-1], next(iter(ex_out.values())), atol=1e-6), \
        f"max_err={torch.max(torch.abs(results[-1]-next(iter(ex_out.values()))))}"


def test_alexnet():
    wk2jobs = [ItgJob(list(range(1, 5)), [4], {}),
               ItgJob(list(range(5, 10)), [9], {}),
               ItgJob(list(range(10, 14)), [13], {})]
    _run_jobs(RawDNN(prepare_alexnet()), wk2jobs)


def test_alexnet_empty():
    wk2jobs = [ItgJob(list(range(1, 7)), [6], {}),
               ItgJob([], [6], {}),
               ItgJob(list(range(7, 14)), [13], {})]
    _run_jobs(RawDNN(prepare_alexnet()), wk2jobs)


def test_vgg19():
    wk2jobs = [ItgJob(list(range(1, 10)), [9], {}),
               ItgJob(list(range(10, 20)), [19], {}),
               ItgJob(list(range(20, 38)), [37], {})]
    _run_jobs(RawDNN(prepare_vgg19()), wk2jobs)


def test_googlenet():
    wk2jobs = [ItgJob(list(range(1, 55)) + list(range(55, 57)) + list(range(59, 63)) + list(range(65, 68)),
                      [56, 62, 67, 55], {}),
               ItgJob(list(range(57, 59)) + list(range(63, 65)) + list(range(68, 71)) + list(range(71, 118))
                      + list(range(118, 122)) + list(range(122, 124)) + list(range(128, 130)) + list(range(134, 135)),
                      [121, 123, 129, 134], {}),
               ItgJob(list(range(124, 128)) + list(range(130, 134)) + list(range(135, 203)), [202], {})]
    _run_jobs(RawDNN(prepare_googlenet()), wk2jobs)


def test_resnet50():
    wk2jobs = [ItgJob(list(range(1, 55)), [49, 54], {}),
               ItgJob(list(range(55, 110)), [101, 109], {}),
               ItgJob(list(range(110, 173)), [172], {})]
    _run_jobs(RawDNN(prepare_resnet50()), wk2jobs)


#----- 以下对RPC数据转换进行测试 -----#

def test_tensor_serialize():
    """测试Tensor数据使用Arr3dMsg进行序列化，不压缩"""
    # 测试数据为360p：360行*480列
    send = torch.from_numpy(np.array(
        [scipy.sparse.random(360, 480, .495, dtype=np.single).A for _ in range(16)])).unsqueeze(0)
    msg = tensor2msg(send, False)
    recv = msg2tensor(msg)
    assert torch.allclose(send, recv)


def test_job_msg():
    """测试ItgJob使用JobMsg进行序列化"""
    cap = cv2.VideoCapture(f'../media/road.mp4')
    ipt = get_ipt_from_video(cap)
    send_job = ItgJob([1, 2, 3], [4], {0: ipt})
    msg = send_job.to_msg()
    recv_job = ItgJob.from_msg(msg)
    assert send_job.exec_ids == recv_job.exec_ids
    assert send_job.out_ids == recv_job.out_ids
    assert torch.allclose(send_job.id2data[0], recv_job.id2data[0])
