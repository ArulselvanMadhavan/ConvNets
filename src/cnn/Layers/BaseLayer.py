from abc import abstractmethod, ABCMeta, abstractproperty


class BaseLayer(metaclass=ABCMeta):
    @abstractproperty
    def output(self):
        return "Should never return the output from Base class"

    @output.setter
    def output(self, out):
        return

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
