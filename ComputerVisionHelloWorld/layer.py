from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward_pass(self, X):
        pass

    @abstractmethod
    def backward_pass(self, dL_dA, Z, A_prev):
        pass

    @abstractmethod
    def update_weights_and_biases(self, learn_rate, grad_W, grad_b):
        pass