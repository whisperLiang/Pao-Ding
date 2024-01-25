"""对特定CNN和视频进行profile，获得各层对于输出数据中非零占比的预测
这里的代码要保存较多中间数据，内存占用较多，应该在PC上进行
"""
import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import List

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class Predictor(ABC):
    """每个CNN层对应一个Predictor"""
    def __init__(self, module: torch.nn.Module):
        pass

    @abstractmethod
    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'Predictor':
        """使用afcnz和fcnz训练模型
        :param afcnz 输入数据，afcnz[a][f][c]=第a个前驱第f帧第c个通道的非零占比
        :param fcnz 输出数据，fcnz[f][c]=第f帧第c个通道的非零占比
        """
        return self

    @abstractmethod
    def predict(self, acnz: List[List[float]]) -> List[float]:
        """对于给定输入数据的稀疏率，给出输出数据稀疏率的预测
        :param acnz 多个前驱的cnz按照输入顺序排序, acnz[c]为输入数据中第c个通道的非零占比
        :return cnz cnz[c]为输出数据中第c个通道的非零占比
        """
        pass


@dataclasses.dataclass
class NZPred:
    o_lcnz: List[List[float]]
    predictors: List[Predictor]


class MLPPredictor(Predictor):
    """使用多层感知机进行预测，相应层只有一个前驱"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.regr = MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500)

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'MLPPredictor':
        assert len(afcnz) == 1
        self.regr.fit(afcnz[0], fcnz)
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        return self.regr.predict([acnz[0]])[0]


class MLPsPredictor(Predictor):
    """使用多层感知机进行预测，相应层只有一个前驱"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.regrs: List[MLPRegressor] = []

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'MLPsPredictor':
        assert len(afcnz) == 1
        nchan = len(fcnz[0])
        self.regrs = [MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500) for _ in range(nchan)]
        X, y = np.array(afcnz[0]), np.array(fcnz)
        for c, regr in enumerate(self.regrs):
            regr.fit(X, y[:, c])
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        # 使用map以加快速度
        return list(map(partial(self.mlps, np.array([acnz[0]])), self.regrs))

    @staticmethod
    def mlps(cnz: np.array, regr: MLPRegressor) -> float:
        return regr.predict(cnz)[0]


class LNRPredictor(Predictor):
    """使用线性函数(LiNeaR)，对每个通道分别进行预测
    输入输出通道数必须相同，相应层只有一个前驱"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.regrs: List[LinearRegression] = []

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'LNRPredictor':
        assert len(afcnz) == 1
        X, y = np.array(afcnz[0]), np.array(fcnz)
        self.regrs = [LinearRegression() for _ in range(X.shape[1])]
        for c, regr in enumerate(self.regrs):
            regr.fit(X[:, c].reshape(-1, 1), y[:, c])
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        # 使用map以加快速度
        return list(map(self.linear, self.regrs, acnz[0]))

    @staticmethod
    def linear(regr: LinearRegression, nz: float) -> float:
        return regr.coef_[0]*nz + regr.intercept_


class DRPredictor(Predictor):
    """DiRect：InputModule, BasicFork"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'DRPredictor':
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        return acnz[0]
