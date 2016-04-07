from cnn.Layers.BaseLayer import BaseLayer
from abc import ABCMeta, abstractmethod, abstractproperty


class BaseNeuron(metaclass=ABCMeta):
    @abstractproperty
    def W(self):
        return "Should never return the cache output from Base class"

    @W.setter
    def W(self, newvalue):
        return

    @abstractproperty
    def b(self):
        pass

    @b.setter
    def b(self, newValue):
        pass

    @abstractproperty
    def dw(self):
        pass

    @dw.setter
    def dw(self, newValue):
        pass

    @abstractproperty
    def db(self):
        pass

    @db.setter
    def db(self, newValue):
        pass
