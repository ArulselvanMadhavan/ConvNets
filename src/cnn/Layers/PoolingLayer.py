import numpy as np
from cnn.Layers.BaseLayer import BaseLayer


class PoolingLayer(BaseLayer):
    def __init__(self):
        self._fwdcache = None

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self, X, ph=2, pw=2, stride=2):
        """
        Do a forward
        :param X:
        :param ph: Height of pooling block
        :param pw: Width of pooling block
        :param stride: Stride of pooling block
        bh - block height
        bw - block width
        w_start - starting width of window
         w_end - ending width of window
         h_start - starting height of window
         h_end - ending height of window
        :return:
        """

        #For the purposes of this project we always assume that ph,pw,stride will be equal
        # and is always a square
        assert ph == pw == stride
        N, C, H, W = X.shape
        H_out = (H - ph) / stride + 1
        W_out = (W - pw) / stride + 1
        out = np.zeros((N, C, H_out, W_out))
        indices = np.zeros((N, C, H_out, W_out), dtype=int)
        for image in range(int(N)):
            for channel in range(int(C)):
                for bh in range(int(H_out)):
                    for bw in range(int(W_out)):
                        w_start = bw * stride
                        w_end = w_start + pw
                        h_start = bh * stride
                        h_end = h_start + ph
                        #Select the window
                        window = X[image, channel, h_start:h_end, w_start:w_end]
                        #Store the max index
                        max_index = np.argmax(window)
                        #get the value of max index
                        out[image, channel, bh, bw] = window[
                            np.unravel_index(max_index, (h_end - h_start, w_end - w_start))]
                        indices[image, channel, bh, bw] = max_index
        self.cache = (X, ph, pw, stride, indices)
        return out

    def backward(self, dout):
        """
        Do a reverse of the sequence of operations in forward
        :param dout:
        :return:
        """
        x, ph, pw, stride, indices = self.cache
        N, C, H_out, W_out = dout.shape
        dx = np.zeros_like(x)
        for image in range(N):
            for channel in range(C):
                for bh in range(H_out):
                    for bw in range(W_out):
                        w_start = bw * stride
                        w_end = w_start + pw
                        h_start = bh * stride
                        h_end = h_start + ph
                        ind_pos = np.unravel_index(indices[image, channel, bh, bw], (h_end - h_start, w_end - w_start))
                        window = dx[image, channel, h_start:h_end, w_start:w_end]
                        window[ind_pos] = dout[image, channel, bh, bw]
                        dx[image, channel, h_start:h_end, w_start:w_end] = window
        return dx

    # def forward(self, x):
    #     N, C, img_height, img_width = x.shape
    #     F, C, HH, WW = self.W.shape
    #     stride = self.stride
    #     H_out = 1 + (img_height  - HH) // stride
    #     W_out = 1 + (img_width  - WW) // stride
    #
    #     x_col = np.zeros((N, F, H_out, W_out))
    #     y_col = np.zeros((N, F, H_out, W_out)) # for storing max index
    #     for img in range(N):
    #         for filterId in range(F):
    #             w_start = 0
    #             for width in range(W_out):
    #                 w_end = w_start + WW
    #                 h_start = 0
    #                 for height in range(H_out):
    #                     h_end = h_start + HH
    #                     xin = img[:, w_start:w_end, h_start:h_end]
    #                     max_value = xin.np.amax
    #                     max_index = xin.np.argmax
    #                     x_col[img][filterId][width][height] = max_value
    #                     y_col[img][filterId][width][height] = max_index
    #                     h_start += stride
    #                 w_start += stride
    #
    #     self.cache = x
    #     self._maxIndexCache = y_col
    #     return x_col
