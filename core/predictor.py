"""对特定CNN和视频进行profile，获得各层对于输出数据中非零占比的预测
这里的代码要保存较多中间数据，内存占用较多，应该在PC上进行
"""
import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import List

import numpy as np
import torch
from scipy.optimize import curve_fit


class Predictor(ABC):
    """每个CNN层对应一个Predictor"""
    def __init__(self, module: torch.nn.Module):
        pass

    @abstractmethod
    def fit(self, afnz: List[float], fnz: List[float]) -> 'Predictor':
        """使用afnz和fcnz训练模型
        :param afnz 输入数据，前驱第f帧与第f-1帧差值非零占比
        :param fnz 输出数据，第f帧与第f-1帧差值非零占比
        """
        return self

    @abstractmethod
    def predict(self, anz: float) -> float:
        """对于给定输入数据的稀疏率，给出输出数据稀疏率的预测
        :param anz 前驱的nz按照输入顺序排序
        :return pre_nz 输出数据的非零占比
        """
        pass


@dataclasses.dataclass
class NZPred:
    o_lcnz: List[List[float]]
    predictors: List[Predictor]


class LOGreluPredictor(Predictor):
    """使用log函数(log)进行预测"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.popt = None
        self.pcov = None

    def fit(self, afnz: List[float], fnz: List[float]) -> 'LOGreluPredictor':
        X, y = np.array(afnz), np.array(fnz)
        func = lambda x, k, p, r: (k*p*np.exp(r*x))/(k+p*(np.exp(r*x)-1))  # logistic函数
        self.popt, self.pcov = curve_fit(func, X, y, maxfev=50000)
        return self

    def predict(self, anz: float) -> float:
        func = lambda x, k, p, r: (k*p*np.exp(r*x))/(k+p*(np.exp(r*x)-1))  # logistic函数
        pre_nz = func(anz, *self.popt)
        return pre_nz