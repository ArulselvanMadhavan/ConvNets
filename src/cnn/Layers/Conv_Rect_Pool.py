import numpy as np
from cnn.Layers.BaseLayer import BaseLayer
from cnn.Layers.RectifierLinearUnit import RectifierLinearUnit
from cnn.Layers.BaseNeuron import BaseNeuron
from cnn.Layers.ConvLayer import ConvLayer
from cnn.Layers.FastConvLayer import FastConvLayer
from cnn.Layers.PoolingLayer import PoolingLayer

class Conv_Rect_Pool(BaseLayer, BaseNeuron):
    def __init__(self, w, b, stride=1, pad=1, ph=2, pw=2, pstride=2):
        self._fwdcache = None
        self.stride = stride
        self.pad = pad
        self.ph = ph
        self.pw = pw
        self.pstride = pstride
        self._dw = None
        self._db = None
        self.conv_obj = FastConvLayer(w,b,stride,pad)
        self.rect_obj = RectifierLinearUnit()
        self.pool_obj = PoolingLayer()

    @property
    def W(self):
        return self.conv_obj.W

    @W.setter
    def W(self, newValue):
        self.conv_obj.W = newValue

    @property
    def b(self):
        return self.conv_obj.b

    @b.setter
    def b(self, newValue):
        self.conv_obj.b = newValue

    @property
    def dw(self):
        # return self._dw
        return self.conv_obj.dw #Relay to conv obj

    @dw.setter
    def dw(self, newValue):
        self.conv_obj.dw = newValue
        # self._dw = newValue #Relay to conv obj

    @property
    def db(self):
        # return self._db
        return self.conv_obj.db #Relay to conv obj

    @db.setter
    def db(self, newValue):
        # self._db = newValue
        self.conv_obj.db = newValue #Relay to conv obj

    @property
    def cache(self):
        return self._fwdcache

    @cache.setter
    def cache(self, newvalue):
        self._fwdcache = newvalue

    def forward(self, x):
        out = self.conv_obj.forward(x)
        out = self.rect_obj.forward(out)
        out = self.pool_obj.forward(out)
        return out

    def backward(self, dout):
        ds = self.pool_obj.backward(dout)
        da = self.rect_obj.backward(ds)
        dx = self.conv_obj.backward(da)
        return dx

    def generateWeightsAndBias(*args, **kwargs):
        return ConvLayer.generateWeightsAndBias(*args, **kwargs)

    def getDownsampledHeight(self, H):
        return (H - self.ph) / self.pstride + 1

    def getDownsampledWidth(self, W):
        return (W - self.pw) / self.pstride + 1

    def printDimensions(self):
        print("W:{}\tb:{}".format(self.W.shape, self.b.shape))
