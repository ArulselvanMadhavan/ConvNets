#Reference: https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L768
from cnn.GradientUpdates.BaseGradientUpdate import BaseGradientUpdate
import numpy as np

class Adam(BaseGradientUpdate):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None,t=0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.m = m
        self.v = v

    def decay(self,lr_decay):
        self.learning_rate *= lr_decay

    def update(self, x, dx):
        if self.m is None:
            self.m = np.zeros_like(x)
        if self.v is None:
            self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dx ** 2
        mt_hat = self.m / (1 - (self.beta1) ** self.t)
        vt_hat = self.v / (1 - (self.beta2) ** self.t)
        next_x = x - self.learning_rate * mt_hat / (np.sqrt(vt_hat + self.epsilon))
        return next_x
