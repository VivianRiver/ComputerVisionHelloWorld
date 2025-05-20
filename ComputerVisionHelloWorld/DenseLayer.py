import numpy as np
from layer import Layer

class DenseLayer(Layer):
    def __init__(self, W, b, activation, activation_der):
        self.W = W
        self.b = b
        self.activation = activation
        self.activation_der = activation_der
        super().__init__()

    def _forward_core(self, X):
        # n_s number of samples
        # n_f number of features
        # n_n number of neurons
        # X: n_s * n_f rows of inputs, columns of features
        # W: n_n * n_f rows of neuron, columns of weights per feature
        # b: n_n * 1 columns vector of biases
        WT = self.W.T
        # WT: n_f * n_n rows of weights per feature, colunmns of neurons
        Z = X @ WT + self.b;
        # Z: rows of outputs per input, columns of neurons?
        A = self.activation(Z)
        # A: activations of Z computer element-wise
        return Z, A

    def _backward_core(self, dL_dA, Z, A_prev):
        # dA_dZ element-wise activation function derivative
        dA_dZ = self.activation_der(Z) 
        # dL_dZ element-wise multiplication
        dL_dZ = dL_dA * dA_dZ
        grad_W = dL_dZ.T @ A_prev / A_prev.shape[0]
        grad_B = np.mean(dL_dZ, axis=0, keepdims=True)
        return dL_dZ, grad_W, grad_B

    def update_weights_and_biases(self, learn_rate, grad_W, grad_b):
        self.W -= learn_rate * grad_W
        self.b -= learn_rate * grad_b