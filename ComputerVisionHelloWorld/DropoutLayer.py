import numpy as np
from layer import Layer

class DropoutLayer(Layer):
    def __init__(self, p):    
        self.p = p
        self.is_training = True
        super().__init__()

    def _forward_core(self, X):
        if self.is_training:
            mask = np.random.rand(*X.shape) > self.p
            mask = mask / (1 - self.p)
            Z = X * mask # elementwise multiplication by mask and scalar
            self.mask = mask
            return Z, Z
        return X, X        

    def _backward_core(self, dL_dA, Z, A_prev):
        dL_dA_prev = dL_dA * self.mask
        grad_W = None
        grad_B = None
        return dL_dA_prev, grad_W, grad_B

    def update_weights_and_biases(self, learn_rate, grad_W, grad_B):
        pass  # No weights to update