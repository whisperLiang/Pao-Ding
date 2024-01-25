"""研究各CNN层输出的3维矩阵中，各通道的稀疏程度
数据格式LFCNZ：data[层l][帧f][通道c] = (帧f-上一帧)在层l输出数据中通道c的非零占比nz
数据格式O_LFCNZ：data[层l][帧f][通道c] = 帧f在层l输出数据中通道c的非零占比nz
"""
import pickle
from typing import Tuple, List

import cv2
import tqdm
from torch import Tensor

from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.master import Master
from trainer.trainer import Trainer


def _tensor_cnz(tensor4d: Tensor) -> List[float]:
    return [float(chan.count_nonzero() / chan.nelement()) for chan in tensor4d[0]]


def gen_original(raw_dnn: RawDNN, video_path: str,
                 frame_num: int, frame_size: Tuple[int, int]) -> List[List[List[float]]]:
    """生成原始数据对应各通道的非零占比，即o_lfcnz"""
    cap = cv2.VideoCapture(video_path)
    o_lfcnz = [[[] for f in range(frame_num)] for l in raw_dnn.layers]
    for f in tqdm.tqdm(range(frame_num), f"collecting o_lfcnz"):
        opts = raw_dnn.execute(Master.get_ipt_from_video(cap, frame_size))
        lcnz = [_tensor_cnz(ts) for ts in opts]
        for l in range(len(lcnz)):
            o_lfcnz[l][f] = lcnz[l]
    return o_lfcnz


if __name__ == '__main__':
    CNN_NAME_set = ['ax','vg16', 'gn','rs50']
    VIDEO_NAME_set=['campus','parking','road']
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    ORIGINAL_= [True,False]  # False为差值数据LFCNZ，True为原始数据OLFCNZ

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    for CNN_NAME in CNN_NAME_set:
        for VIDEO_NAME in VIDEO_NAME_set:
            for ORIGINAL in ORIGINAL_:
                g_raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
                g_frame_size = tuple(map(int, RESOLUTION.split('x')))[:2]
                g_video_path = f'../media/{VIDEO_NAME}.mp4'
                write_path = f"dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.{'o_' if ORIGINAL else ''}lfcnz"

                if ORIGINAL:
                    # noinspection PyTypeChecker
                    result = gen_original(g_raw_dnn, g_video_path, NFRAME_TOTAL, g_frame_size)
                else:
                    # noinspection PyTypeChecker
                    result = Trainer.collect_lfcnz(g_raw_dnn, g_video_path, NFRAME_TOTAL, g_frame_size)
                with open(write_path, 'wb') as file:
                    pickle.dump(result, file)
                print(f'{write_path} has been written')
