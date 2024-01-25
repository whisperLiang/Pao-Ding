from typing import List

import cv2
import torch
from torch import Tensor
import numpy as np
import scipy.sparse

from core.dif_executor import DifExecutor, DifJob
from core.raw_dnn import RawDNN
from core.util import msg2tensor, tensor2msg
from dnn_models.resnet import prepare_resnet50
from unit_tests.common import get_ipt_from_video

#----- 以下对DifExecutor执行正确性进行测试 -----#

def _co_execute(frame_id: int, dif_extors: List[DifExecutor],
                wk2job: List[DifJob], benchmarks: List[Tensor]):
    for wk, d_job in enumerate(wk2job):
        id2dif = dif_extors[wk].exec(d_job)
        id2opt = dif_extors[wk].last_out()
        for oid in d_job.out_ids:
            assert torch.allclose(id2opt[oid], benchmarks[oid], atol=1e-5), \
                f"Frame{frame_id} Worker{wk} layer{oid} max_err={torch.max(torch.abs(id2opt[oid]-benchmarks[oid]))}"
        if wk + 1 < len(wk2job):
            wk2job[wk + 1].id2data = id2dif


def test_fixed_jobs():
    """当DifJob固定时，测试多个DifExecutor是否可以正常协同，观察误差"""
    raw_dnn = RawDNN(prepare_resnet50())
    dif_extors = [DifExecutor(raw_dnn) for _ in range(3)]  # 分别对应3个Worker
    cap = cv2.VideoCapture(f'../media/road.mp4')
    cur = get_ipt_from_video(cap)
    dif = cur
    for fid in range(5):
        # 直接执行RawLayer，以检查正确性
        results = raw_dnn.execute(cur)
        wk2job = [DifJob(list(range(1, 55)), [49, 54], {0: dif}),
                  DifJob(list(range(55, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})]
        _co_execute(fid, dif_extors, wk2job, results)
        nxt = get_ipt_from_video(cap)
        dif = nxt - cur
        cur = nxt


def test_fixed_jobs_empty():
    """当DifJob固定但其中有空任务时，测试多个DifExecutor是否可以正常协同，观察误差"""
    raw_dnn = RawDNN(prepare_resnet50())
    dif_extors = [DifExecutor(raw_dnn) for _ in range(3)]  # 分别对应3个Worker
    cap = cv2.VideoCapture(f'../media/road.mp4')
    cur = get_ipt_from_video(cap)
    dif = cur
    for fid in range(5):
        # 直接执行RawLayer，以检查正确性
        results = raw_dnn.execute(cur)
        wk2job = [DifJob(list(range(1, 55)), [49, 54], {0: dif}),
                  DifJob([], [49, 54], {}),
                  DifJob(list(range(55, 173)), [172], {})]
        _co_execute(fid, dif_extors, wk2job, results)
        nxt = get_ipt_from_video(cap)
        dif = nxt - cur
        cur = nxt


def test_var_jobs():
    """当DifJob会变时，测试多个DifExecutor是否可以正常协同，观察误差"""
    raw_dnn = RawDNN(prepare_resnet50())
    dif_extors = [DifExecutor(raw_dnn) for _ in range(3)]  # 分别对应3个Worker
    cap = cv2.VideoCapture(f'../media/road.mp4')
    cur = get_ipt_from_video(cap)
    dif = cur
    for fid in range(4):
        # 直接执行RawLayer，以检查正确性
        results = raw_dnn.execute(cur)
        plans = [
                 # 初始计划
                 [DifJob(list(range(1, 55)), [49, 54], {0: dif}),
                  DifJob(list(range(55, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})],
                 # Worker0-1边界变化
                 [DifJob(list(range(1, 44)) + [46], [43, 46], {0: dif}),
                  DifJob([44, 45] + list(range(47, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})],
                 # Worker1-2边界变化
                 [DifJob(list(range(1, 44)) + [46], [43, 46], {0: dif}),
                  DifJob([44, 45] + list(range(47, 106)), [101, 105], {}),
                  DifJob(list(range(106, 173)), [172], {})],
                 # Worker0-1,1-2边界都变化了
                 [DifJob(list(range(1, 37)), [36], {0: dif}),
                  DifJob(list(range(37, 111)), [110], {}),
                  DifJob(list(range(111, 173)), [172], {})]
                 ]
        wk2job = plans[fid]
        _co_execute(fid, dif_extors, wk2job, results)
        nxt = get_ipt_from_video(cap)
        dif = nxt - cur
        cur = nxt

#----- 以下对RPC数据转换进行测试 -----#

def test_tensor_compress():
    """测试Tensor数据使用Arr3dMsg进行压缩"""
    # 测试数据为360p：360行*480列
    send = torch.from_numpy(np.array(
        [scipy.sparse.random(360, 480, .495, dtype=np.single).A for _ in range(16)])).unsqueeze(0)
    msg = tensor2msg(send)
    org, cps = send.numpy().nbytes/1024/1024, msg.ByteSize()/1024/1024
    assert cps <= org, "Compressed data is too large!"
    recv = msg2tensor(msg)
    assert torch.allclose(send, recv)


def test_job_msg():
    """测试DifJob使用JobMsg进行序列化"""
    cap = cv2.VideoCapture(f'../media/road.mp4')
    ipt = get_ipt_from_video(cap)
    send_job = DifJob([1, 2, 3], [4], {0: ipt})
    msg = send_job.to_msg()
    recv_job = DifJob.from_msg(msg)
    assert send_job.exec_ids == recv_job.exec_ids
    assert send_job.out_ids == recv_job.out_ids
    assert torch.allclose(send_job.id2data[0], recv_job.id2data[0])
