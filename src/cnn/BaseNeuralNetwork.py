from abc import ABCMeta, abstractmethod, abstractproperty

class BaseNeuralNetwork(metaclass=ABCMeta):
    @abstractmethod
    def loss(self,X,y):
        pass