import dataclasses
import logging
import os
from threading import Condition, Thread
from typing import Tuple, List, Any, Dict

import cv2
import tqdm
import time
from torch import Tensor

from core.util import cached_func
from core.healthy_check import set_health_status
from master.master import Master
from core.predictor import Predictor, NZPred
from core.dag_dnn import DagDNN


class Trainer(Thread):
    """运行在较高性能和较大内存的PC上，收集数据并训练稀疏率预测模型"""

    def __init__(self, dag_dnn: DagDNN, video_path: str, frame_size: Tuple[int, int], config: Dict[str, Any]):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__dag_dnn = dag_dnn
        self.__video_path = video_path
        self.__frame_size = frame_size
        self.__frame_num = config['frame_num']
        self.__cv = Condition()
        self.__predictors = []
        self.__o_lcnz = []

    def run(self) -> None:
        dnn_name = self.__dag_dnn.model_name
        vid_name = os.path.basename(self.__video_path).split('.')[0]  # 只保留文件名，去掉拓展名
        frm_size = f"{self.__frame_size[0]}x{self.__frame_size[1]}"
        trn_numb = str(self.__frame_num)
        data_name = dnn_name + '.' + vid_name + '.' + frm_size + '.' + trn_numb
        self.__logger.info("collecting O_LFCNZ data...")
        o_lfcnz = cached_func(data_name + '.o_lfcnz', self.collect_olfcnz, self.__dag_dnn, self.__video_path,
                            self.__frame_num, self.__frame_size, logger=self.__logger)
        self.__logger.info("computing O_LCNZ data...")
        o_lcnz = cached_func(data_name + '.o_lcnz', self.avg_lcnz, o_lfcnz, logger=self.__logger)
        self.__logger.info("collecting LFCNZ data...")
        lfcnz = cached_func(data_name + '.lfcnz', self.collect_lfcnz, self.__dag_dnn, self.__video_path,
                            self.__frame_num, self.__frame_size, logger=self.__logger)
        self.__logger.info("training predictors...")
        b_time = time.time()
        predictors = cached_func(data_name + '.pred', self.train_logrelupredictors,
                                 self.__dag_dnn, lfcnz, logger=self.__logger)
        e_time = time.time()
        self.__logger.info(f"training time: {e_time - b_time:.2f}s")
        set_health_status(True)
        self.__logger.info("train finished, predictors are ready")
        with self.__cv:
            self.__o_lcnz = o_lcnz
            self.__predictors = predictors
            self.__cv.notifyAll()

    def get_nzpred(self) -> NZPred:
        self.__logger.debug("trying to get predictors...")
        with self.__cv:
            while len(self.__predictors) == 0:
                self.__cv.wait()
            self.__logger.debug("got predictors")
            return NZPred(self.__o_lcnz, self.__predictors)

    @classmethod
    def collect_olfcnz(cls, dag_dnn: DagDNN, video_path: str,
                      frame_num: int, frame_size: Tuple[int, int]) -> List[List[List[float]]]:
        """收集数据格式O_LFCNZ：data[层l][帧f][通道c] = 帧f在层l输出数据中通道c的非零占比nz"""
        cap = cv2.VideoCapture(video_path)
        o_lfcnz = [[[] for f in range(frame_num)] for l in dag_dnn.layers]
        for f in tqdm.tqdm(range(frame_num), f"collecting o_lfcnz"):
            opts = dag_dnn.execute(Master.get_ipt_from_video(cap, frame_size))
            lcnz = [cls._layer_cnz(ts) for ts in opts]
            for l in range(len(lcnz)):
                o_lfcnz[l][f] = lcnz[l]
        return o_lfcnz

    @classmethod
    def avg_lcnz(cls, o_lfcnz: List[List[List[float]]]) -> List[List[float]]:
        """根据原始数据的lfcnz数据，计算各层的cnz均值，以此作为对任意帧在各层输出数据cnz的估计
        注意：可以这样估计是因为 根据对AlexNet, VGG16, GoogLeNet, ResNet50的观察，所有帧各层原始数据的整体非零占比都集中在均值附近
        """
        nframe = len(o_lfcnz[0])
        lcnz = [[] for _ in o_lfcnz]
        for l, fcnz in enumerate(o_lfcnz):
            nchan = len(fcnz[0])
            lcnz[l] = [sum(fcnz[f][c] for f in range(nframe))/nframe for c in range(nchan)]
        return lcnz

    @classmethod
    def collect_lfcnz(cls, dag_dnn: DagDNN, video_path: str,
                      frame_num: int, frame_size: Tuple[int, int]) -> List[List[List[float]]]:
        """收集LFCNZ数据：data[层l][帧f][通道c] = (帧f-上一帧)在层l输出数据中通道c的非零占比nz"""
        cap = cv2.VideoCapture(video_path)
        lst_results = []
        lfcnz = [[] for _ in dag_dnn.layers]
        for _ in tqdm.tqdm(range(frame_num)):
            cur_results = dag_dnn.execute(Master.get_ipt_from_video(cap, frame_size))
            lcnz = cls._dif_lcnz(lst_results, cur_results)
            for l in range(len(lcnz)):
                lfcnz[l].append(lcnz[l])
            lst_results = cur_results
        return lfcnz
    
    @classmethod
    def _lfcnz2lfnz(cls, lfcnz: List[List[List[float]]]) -> List[List[float]]:
        """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
        return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]

    # @classmethod
    # def train_predictors(cls, dag_dnn: DagDNN, lfcnz: List[List[List[float]]]) -> Dict[int, Predictor]:
    #     predictors = {}
    #     for l in tqdm.tqdm(range(len(dag_dnn.layers))):
    #         layer = dag_dnn.layers[l]
    #         layer_type = layer.module.__class__
    #         if layer_type in dag_dnn.mdl2pred:
    #             predictor = dag_dnn.mdl2pred[layer_type](layer.module)
    #             afcnz = []
    #             for inodelist in layer.inputs:
    #                 for node in inodelist:
    #                     if node in dag_dnn.node2index:
    #                         afcnz.append(lfcnz[dag_dnn.node2index[node]])

    #             predictor.fit(afcnz, lfcnz[l])
    #             predictors[l] = predictor
    #     return predictors
    
    # @classmethod
    # def train_relupredictors(cls, dag_dnn: DagDNN, lfcnz: List[List[List[float]]]) -> Dict[int, Predictor]:
    #     """
    #     根据原始图片的残差非零率预测Relu层残差非零率
    #     """
    #     predictors = {}
    #     for l in tqdm.tqdm(range(len(dag_dnn.layers))):
    #         layer = dag_dnn.layers[l]
    #         layer_type = layer.module.__class__
    #         if layer_type in dag_dnn.mdl2pred:
    #             predictor = dag_dnn.mdl2pred[layer_type](layer.module)
    #             predictor.fit([lfcnz[0]], lfcnz[l])
    #             predictors[l] = predictor
    #     return predictors
    
    @classmethod
    def train_logrelupredictors(cls, dag_dnn: DagDNN, lfcnz: List[List[List[float]]]) -> Dict[int, Predictor]:
        """
        根据原始图片的残差非零率预测Relu层残差非零率
        """
        predictors = {}
        arelu_l = 0
        lfnz = cls._lfcnz2lfnz(lfcnz)
        for l in tqdm.tqdm(range(len(dag_dnn.layers))):
            layer = dag_dnn.layers[l]
            layer_type = layer.module.__class__
            if layer_type in dag_dnn.mdl2pred:
                predictor = dag_dnn.mdl2pred[layer_type](layer.module)
                predictor.fit(lfnz[arelu_l], lfnz[l])
                predictors[l] = predictor
                arelu_l = l
        return predictors

    @classmethod
    def _layer_cnz(cls, tensor4d: Tensor) -> List[float]:
        return [float(chan.count_nonzero() / chan.nelement()) for chan in tensor4d[0]]

    @classmethod
    def _dif_lcnz(cls, last_results: List[Tensor], cur_results: List[Tensor]) -> List[List[float]]:
        if len(last_results) == 0:
            return [cls._layer_cnz(res) for res in cur_results]
        dif_results = []
        for i in range(len(last_results)):
            dif_results.append(cls._layer_cnz(cur_results[i] - last_results[i]))
        return dif_results
