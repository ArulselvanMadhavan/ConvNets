import numpy as np
from cnn.Layers.BaseLayer import BaseLayer
from cnn.Layers.BaseNeuron import BaseNeuron

try:
    from cnn.Layers.image import col2im
except ImportError:
    print(ImportError.msg)
    print('Build the cython extension using the command-> python setup_image.py build_ext --inplace')


class FastConvLayer(BaseLayer, BaseNeuron):
    def __init__(self, W, b, stride, pad):
        """
        :param W: Weight Matrix
        :param b: Bias Vector
        :param stride:
        :param pad:
        :return:
        """
        self._fwdcache = None
        self._W = W
        self._b = b
        self._dw = None
        self._db = None
        self.stride = stride
        self.pad = pad

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

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, FH, FW = self.W.shape
        stride, pad = self.stride, self.pad

        assert (W + 2 * pad - FW) % stride == 0
        assert (H + 2 * pad - FH) % stride == 0

        p = pad
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        H += 2 * pad
        W += 2 * pad
        out_h = (H - FH) // stride + 1
        out_w = (W - FW) // stride + 1


        shape = (C, FH, FW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                                   shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * FH * FW, N * out_h * out_w)

        res = self.W.reshape(F, -1).dot(x_cols) + self.b.reshape(-1, 1)

        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)

        out = np.ascontiguousarray(out)

        self.cache = (x, x_cols)
        return out

    def backward(self, dout):
        x, x_cols = self.cache

        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        _, _, out_h, out_w = dout.shape

        self.db = np.sum(dout, axis=(0, 2, 3))

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        self.dw = dout_reshaped.dot(x_cols.T).reshape(self.W.shape)

        dx_cols = self.W.reshape(F, -1).T.dot(dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im(dx_cols, N, C, H, W, HH, WW, self.pad, self.stride)
        return dx

    @staticmethod
    def generateWeightsAndBias(weight_dim, weight_scale):
        W = np.random.normal(loc=0.0, scale=weight_scale, size=weight_dim)
        b = np.zeros(weight_dim[0])
        return W, b
