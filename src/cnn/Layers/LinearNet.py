from cnn.Layers.BaseLayer import BaseLayer
import numpy as np


class LinearNet(BaseLayer):
    def __init__(self):
        pass

    @staticmethod
    def forward(X=None, W=None, b=None):
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        x2 = np.reshape(X, (N, D))
        out = np.dot(x2, W) + b
        cache = (X, W, b)
        return (out, cache)

    @staticmethod
    def backward(dout=None, cache=None):
        x, w, b = cache
        N = x.shape[0]
        D = np.prod(x.shape[1:])
        x_reshaped = np.reshape(x, (N, -1))
        dx = np.dot(dout, w.T)
        dw = np.dot(x_reshaped.T, dout)
        db = np.dot(dout.T, np.ones(N))
        dx = np.reshape(dx, x.shape)
        return dx, dw, db
