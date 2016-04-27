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
        self.mask = X < 0
        return np.where(X < 0, 0.01 * X, X)

    def backward(self, dout):
        dout[self.mask] = 0.01
        return dout
