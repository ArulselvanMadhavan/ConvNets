from cnn.Layers.BaseLayer import BaseLayer
import numpy as np


class LinearNet(BaseLayer):
    def __init__(self):
        self._fwdcache = None

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self, X, W, b):
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        x2 = np.reshape(X, (N, D))
        self.cache = (X, W, b)
        return np.dot(x2, W) + b

    def backward(self, dout):
        x, w, b = self.cache
        N = x.shape[0]
        D = np.prod(x.shape[1:])
        x_reshaped = np.reshape(x, (N, -1))
        dx = np.dot(dout, w.T)
        dw = np.dot(x_reshaped.T, dout)
        db = np.dot(dout.T, np.ones(N))
        dx = np.reshape(dx, x.shape)
        return dx, dw, db
