from cnn.Layers.BaseLayer import BaseLayer
from cnn.Layers.BaseNeuron import BaseNeuron
import numpy as np


class LinearNet(BaseLayer, BaseNeuron):
    def __init__(self, w, b):
        self._fwdcache = None
        self._W = w
        self._b = b
        self._dw = None
        self._db = None

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, newValue):
        self._W = newValue

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, newValue):
        self._b = newValue

    @property
    def dw(self):
        return self._dw

    @dw.setter
    def dw(self, newValue):
        self._dw = newValue

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, newValue):
        self._db = newValue

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self, X):
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        x2 = np.reshape(X, (N, D))
        self.cache = X
        return (np.dot(x2, self.W) + self.b)

    def backward(self, dout):
        x, w, b = self.cache, self.W, self.b
        N = x.shape[0]
        D = np.prod(x.shape[1:])
        x_reshaped = np.reshape(x, (N, -1))
        dx = np.dot(dout, w.T)
        self.dw = np.dot(x_reshaped.T, dout)
        self.db = np.dot(dout.T, np.ones(N))
        dx = np.reshape(dx, x.shape)
        return dx

    @staticmethod
    def generateWeightsAndBias(D, H, weight_scale):
        """
        :param D:
        :param H:
        :param weight_scale:
        :return:
        """
        np.random.seed(seed=123)
        w = weight_scale * np.random.randn(D, H)
        b = np.zeros(H)
        return w, b

    def printDimensions(self):
        print("W:{}\tb:{}".format(self.W.shape, self.b.shape))
