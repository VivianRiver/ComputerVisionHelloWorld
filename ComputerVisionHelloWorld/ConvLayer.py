import numpy as np
from layer import Layer

class ConvLayer(Layer):
    def __init__(self, W, b, activation, activation_der):
        # W.shape = (n_filters, kernel_height, kernel_width, input_channels)
        self.W = W
        # b.shape = (n_filters,)
        self.b = b
        self.activation = activation
        self.activation_der = activation_der
        return    

    def forward_pass(self, X):
        # Input shape: (n_s, H_in, W_in, in_channels)
        n_s, H_in, W_in, in_channels = X.shape
        n_f, K_h, K_w, _ = self.W.shape
        stride = self.stride

        # Output dimensions
        H_out = (H_in - K_h) // stride + 1
        W_out = (W_in - K_w) // stride + 1

        # Allocate output tensor
        Z = np.zeros((n_s, H_out, W_out, n_f))

        # Convolution
        for b in range(n_s):                 # for each image in the batch
            for f in range(n_f):             # for each filter
                for i in range(H_out):       # slide vertically
                    for j in range(W_out):   # slide horizontally
                        vert_start = i * stride
                        vert_end = vert_start + K_h
                        horiz_start = j * stride
                        horiz_end = horiz_start + K_w
                    
                        patch = X[b, vert_start:vert_end, horiz_start:horiz_end, :]
                        kernel = self.W[f]
                        bias = self.b[f]
                    
                        conv = np.sum(patch * kernel) + bias
                        Z[b, i, j, f] = conv

        # Apply activation
        A = self.activation(Z)
        return Z, A

    def backward_pass(self, dL_dA, Z, A_prev):
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