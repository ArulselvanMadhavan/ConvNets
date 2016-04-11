from cnn.Layers.BaseLayer import BaseLayer
from cnn.Layers.BaseNeuron import BaseNeuron
import numpy as np
from cnn.Layers.LinearNet import LinearNet
from cnn.Layers.RectifierLinearUnit import RectifierLinearUnit


class Linear_Rect(BaseLayer, BaseNeuron):
    def __init__(self, w, b):
        self._fwdcache = None
        self.lin_obj = LinearNet(w, b)

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    @property
    def W(self):
        return self.lin_obj.W

    @W.setter
    def W(self, newValue):
        self.lin_obj.W = newValue

    @property
    def b(self):
        return self.lin_obj.b

    @b.setter
    def b(self, newValue):
        self.lin_obj.b = newValue

    @property
    def dw(self):
        return self.lin_obj.dw

    @dw.setter
    def dw(self, newValue):
        self.lin_obj.dw = newValue

    @property
    def db(self):
        return self.lin_obj.db

    @db.setter
    def db(self, newValue):
        self.lin_obj.db = newValue

    def forward(self, x):
        out = self.lin_obj.forward(x)  # x is cached inside this object. w,b - Refer to w and b.
        self.rect_obj = RectifierLinearUnit()  # out is cached inside this object
        return self.rect_obj.forward(out)

    def backward(self, dout):
        dx = self.rect_obj.backward(dout)
        out = self.lin_obj.backward(dx)
        return out

    def generateWeightsAndBias(*args, **kwargs):
        return LinearNet.generateWeightsAndBias(*args, **kwargs)

    def printW(self):
        print(self.W)

    def printDimensions(self):
        print("W:{}\tb:{}".format(self.lin_obj.W.shape, self.lin_obj.b.shape))
