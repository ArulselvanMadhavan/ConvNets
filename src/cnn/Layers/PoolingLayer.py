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

    def forward2(self, X, ph=2, pw=2, stride=2):
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

    def backward2(self, dout):
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

    def forward(self, X, ph=2, pw=2, stride=2):
        N, C, H, W = X.shape
        assert ph == pw == stride, 'Invalid pool params'
        assert H % ph == 0
        assert W % pw == 0
        X_reshaped = X.reshape(N, C, H / ph, ph,
                               W / pw, pw)
        # print("Shape:{}\nReshaped:{}".format(X_reshaped.shape,X_reshaped))
        out = X_reshaped.max(axis=3).max(axis=4)
        self.cache = (X, X_reshaped, out)
        return out

    def backward(self, dout):
        x, x_reshaped, out = self.cache
        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)
        return dx