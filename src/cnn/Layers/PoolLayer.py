from cnn.Layers.BaseLayer import BaseLayer
import numpy as np


class PoolLayer(BaseLayer):
    def __init__(self, W, stride):
        self._fwdcache = None
        self._W = W
        self.stride = stride


    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, newValue):
        self._W = newValue

    def forward(self, x):
        N, C, img_height, img_width = x.shape
        F, C, HH, WW = self.W.shape
        stride = self.stride
        H_out = 1 + (img_height  - HH) // stride
        W_out = 1 + (img_width  - WW) // stride

        x_col = np.zeros((N, F, H_out, W_out))
        y_col = np.zeros((N, F, H_out, W_out)) # for storing max index
        for img in range(N):
            for filterId in range(F):
                w_start = 0
                for width in range(W_out):
                    w_end = w_start + WW
                    h_start = 0
                    for height in range(H_out):
                        h_end = h_start + HH
                        xin = img[:, w_start:w_end, h_start:h_end]
                        max_value = xin.np.amax
                        max_index = xin.np.argmax
                        x_col[img][filterId][width][height] = max_value
                        y_col[img][filterId][width][height] = max_index
                        h_start += stride
                    w_start += stride

        self.cache = x
        return x_col


        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        dx, x = None, self.cache
        dx = np.array(dout, copy=True)
        dx[x <= 0] = 0
        return dx
