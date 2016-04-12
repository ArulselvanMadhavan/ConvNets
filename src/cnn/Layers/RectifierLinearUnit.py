from cnn.Layers.BaseLayer import BaseLayer
import numpy as np


class RectifierLinearUnit(BaseLayer):
    def __init__(self):
        self._fwdcache = None

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        dx, x = None, self.cache
        dx = np.array(dout, copy=True)
        dx[x <= 0] = 0
        return dx
