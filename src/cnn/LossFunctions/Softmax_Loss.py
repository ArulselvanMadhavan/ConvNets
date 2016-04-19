import numpy as np
from cnn.LossFunctions.AbstractLoss import AbstractLoss


class Softmax_Loss(AbstractLoss):
    @staticmethod
    def compute_loss_and_gradient(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx
