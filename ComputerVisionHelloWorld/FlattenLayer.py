from layer import Layer

class FlattenLayer(Layer):
    def forward_pass(self, X):
        self.input_shape = X.shape  # Save for backward pass
        Z = X.reshape(X.shape[0], -1)
        return Z, Z  # Z is also the activation here

    def backward_pass(self, dL_dA, Z, A_prev):
        dL_dA_prev = dL_dA.reshape(self.input_shape)
        grad_W = None
        grad_B = None
        return dL_dA_prev, grad_W, grad_B

    def update_weights_and_biases(self, learn_rate, grad_W, grad_B):
        pass  # No weights to update