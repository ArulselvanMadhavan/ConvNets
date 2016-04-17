import numpy as np
from cnn.Layers.BaseLayer import BaseLayer

class tanH(BaseLayer):
    def __init__(self):
        self._fwdcache = None

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self,X):
        self.cache = X
        return np.tanh(X)

    def backward(self, dout):
        dx = (1 - np.tanh(self.cache) ** 2) * dout
        return dx