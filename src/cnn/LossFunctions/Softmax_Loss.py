import numpy as np
from cnn.LossFunctions.AbstractLoss import AbstractLoss

class Softmax_Loss(AbstractLoss):
    @staticmethod
    def compute_loss_and_gradient(x,y):
        """

        :param x:
        :param y:
        :return:
        """
        print("Softmax loss function should be placed here")