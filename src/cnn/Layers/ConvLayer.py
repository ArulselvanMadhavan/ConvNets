from cnn.Layers.BaseLayer import BaseLayer
from cnn.Layers.BaseNeuron import BaseNeuron
import numpy as np


class ConvLayer(BaseLayer, BaseNeuron):
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
        """
        X - I/p image
        N - Number of images
        C - RGB Channel - 3rd dimension of an image
        H - Height
        W - Width
        HH - Filter Height
        WW - Filter Width

        :param x:
        :return:
        """
        N, C, img_height, img_width = x.shape
        F, C, HH, WW = self.W.shape
        stride = self.stride
        pad = self.pad
        assert (img_height + 2 * pad - HH) % stride == 0
        assert (img_width + 2 * pad - WW) % stride == 0
        H_out = 1 + (img_height + 2 * pad - HH) // stride
        W_out = 1 + (img_width + 2 * pad - WW) // stride

        x_col = np.zeros((N, F, H_out, W_out))
        for img in range(N):
            x_img = x[img]
            npad = ((0, 0), (pad, pad), (pad, pad))
            x_img_padded = np.pad(x_img, npad, mode='constant', constant_values=0.0)
            for filterId in range(F):
                kernel_3d = self.W[filterId]
                bias = self.b[filterId]
                w_start = 0
                for width in range(W_out):
                    w_end = w_start + WW
                    h_start = 0
                    for height in range(H_out):
                        h_end = h_start + HH
                        xin = x_img_padded[:, w_start:w_end, h_start:h_end]
                        x_col[img][filterId][width][height] = np.sum(xin * kernel_3d) + bias
                        h_start += stride
                    w_start += stride
        self.cache = x
        return x_col

    def backward(self, dout):
        """
        dx,dw,db - gradients of their respective variables
        :param dout:
        :return:
        """
        x, w, b, stride, pad = self.cache, self.W, self.b, self.stride, self.pad
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)
        N, C, img_height, img_width = x.shape
        F, C, HH, WW = w.shape
        _, _, H_out, W_out = dout.shape
        npad = ((0, 0), (pad, pad), (pad, pad))
        for image in range(N):
            dx_pad = np.pad(dx[image, :, :, :], npad, mode='constant', constant_values=0.0)
            x_pad = np.pad(x[image, :, :, :], npad, mode='constant', constant_values=0.0)
            for filterId in range(F):
                kernel_nd = w[filterId]
                for ww in range(W_out):
                    for hh in range(H_out):
                        h_start = hh * stride
                        h_end = h_start + HH
                        w_start = ww * stride
                        w_end = w_start + WW
                        dx_pad[:, w_start:w_end, h_start:h_end] += kernel_nd * dout[image, filterId, ww, hh]
                        dw[filterId, :, :, :] += x_pad[:, w_start:w_end, h_start:h_end] * dout[image, filterId, ww, hh]
                db[filterId] += np.sum(dout[image][filterId])
            dx[image, :, :, :] = dx_pad[:, pad:-pad, pad:-pad]
        self.dw = dw
        self.db = db
        return dx

    @staticmethod
    def generateWeightsAndBias(weight_dim, weight_scale):
        W = np.random.normal(loc=0.0,scale=weight_scale,size=weight_dim)
        b = np.zeros(weight_dim[0])
        return W, b
