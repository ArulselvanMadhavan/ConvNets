from cnn.GradientUpdates.BaseGradientUpdate import BaseGradientUpdate
import numpy as np


class StochasticGradientUpdate(BaseGradientUpdate):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, x, dx):
        x -= self.learning_rate * dx
        return x

    def decay(self,lr_decay):
        self.learning_rate *= lr_decay