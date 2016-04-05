from abc import abstractmethod, ABCMeta, abstractproperty


class BaseLayer(metaclass=ABCMeta):

    @abstractproperty
    def cache(self):
        return "Should never return the cache output from Base class"

    @cache.setter
    def cache(self, newvalue):
        return

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass
