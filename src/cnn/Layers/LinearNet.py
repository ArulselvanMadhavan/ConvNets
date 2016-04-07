from cnn.Layers.BaseLayer import BaseLayer
import numpy as np


class LinearNet(BaseLayer):
    def __init__(self, w, b):
        self._fwdcache = None
        self.W = w
        self.b = b

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
        dw = np.dot(x_reshaped.T, dout)
        db = np.dot(dout.T, np.ones(N))
        dx = np.reshape(dx, x.shape)
        return dx, dw, db

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
