from abc import abstractmethod, abstractproperty, ABCMeta

class AbstractLoss(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def compute_loss_and_gradient(x, y):
        """
        Compute the loss
        :param x: Final score
        :param y: Correct score
        :return: loss and gradient
        """
        pass
