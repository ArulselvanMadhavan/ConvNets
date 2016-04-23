from abc import ABCMeta, abstractmethod, abstractproperty


class BaseGradientUpdate(metaclass=ABCMeta):
    @abstractmethod
    def update(self, x, dx):
        pass

    @abstractmethod
    def decay(self,lr_decay):
        pass
