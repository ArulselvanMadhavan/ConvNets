from abc import abstractmethod, ABCMeta


class BaseLayer(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def forward(**opts):
        pass

    @staticmethod
    @abstractmethod
    def backward(**opts):
        pass
