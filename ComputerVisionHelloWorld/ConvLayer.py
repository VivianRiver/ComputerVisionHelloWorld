import numpy as np
from layer import Layer

class ConvLayer(Layer):
    def __init__(self, W, b, activation, activation_der, stride):
        # W.shape = (n_filters, kernel_height, kernel_width, input_channels)
        self.W = W
        # b.shape = (n_filters,)
        self.b = b
        self.activation = activation
        self.activation_der = activation_der

        self.stride = stride

        super().__init__()

    def _forward_core(self, X):                
        if X.ndim < 4:
            X = X.reshape(-1, 28, 28, 1)

        n_s, H_in, W_in, in_channels = X.shape
        n_f, K_h, K_w, _ = self.W.shape
        stride = self.stride

        H_out = (H_in - K_h) // stride + 1
        W_out = (W_in - K_w) // stride + 1

        # Save shape for backward
        self.X_shape = X.shape
        self.output_shape = (n_s, H_out, W_out, n_f)

        # im2col: extract patches
        X_col = np.lib.stride_tricks.as_strided(
            X,
            shape=(n_s, H_out, W_out, K_h, K_w, in_channels),
            strides=(
                X.strides[0],
                stride * X.strides[1],
                stride * X.strides[2],
                X.strides[1],
                X.strides[2],
                X.strides[3]
            )
        )

        # Reshape: (n_s, H_out, W_out, K_h, K_w, C) → (n_s * H_out * W_out, K_h * K_w * C)
        X_col = X_col.reshape(-1, K_h * K_w * in_channels)

        # Reshape filters: (n_f, K_h, K_w, C) → (n_f, K_h * K_w * C)
        W_col = self.W.reshape(n_f, -1)

        # Compute: (n_s * H_out * W_out, K_h * K_w * C) @ (K_h * K_w * C, n_f).T → (n_s * H_out * W_out, n_f)
        Z = X_col @ W_col.T + self.b

        # Reshape Z to output shape: (n_s, H_out, W_out, n_f)
        Z = Z.reshape(n_s, H_out, W_out, n_f)

        A = self.activation(Z)
        return Z, A

    def _backward_core(self, dL_dA, Z, A_prev):                      
        if A_prev.ndim == 2:
            A_prev = A_prev.reshape(-1, 28, 28, 1)
        
        n_s, H_in, W_in, in_channels = A_prev.shape
        n_f, K_h, K_w, _ = self.W.shape
        stride = self.stride

        H_out = (H_in - K_h) // stride + 1
        W_out = (W_in - K_w) // stride + 1

        # Flatten the input using the same im2col strategy
        X_col = np.lib.stride_tricks.as_strided(
            A_prev,
            shape=(n_s, H_out, W_out, K_h, K_w, in_channels),
            strides=(
                A_prev.strides[0],
                stride * A_prev.strides[1],
                stride * A_prev.strides[2],
                A_prev.strides[1],
                A_prev.strides[2],
                A_prev.strides[3]
            )
        ).reshape(-1, K_h * K_w * in_channels)

        # Compute dL/dZ = dL/dA * dA/dZ
        dA_dZ = self.activation_der(Z)
        dL_dZ = dL_dA * dA_dZ          # shape: (n_s, H_out, W_out, n_f)
        dL_dZ_flat = dL_dZ.reshape(-1, n_f)

        # Gradient w.r.t. filters
        grad_W = dL_dZ_flat.T @ X_col
        grad_W = grad_W.reshape(n_f, K_h, K_w, in_channels)

        # Gradient w.r.t. bias
        grad_B = np.sum(dL_dZ_flat, axis=0)

        # Backprop to input
        W_col = self.W.reshape(n_f, -1)
        dL_dA_prev_col = dL_dZ_flat @ W_col  # shape: (n_s * H_out * W_out, K_h * K_w * in_channels)

        # Initialize empty gradient for input
        dL_dA_prev = np.zeros((n_s, H_in, W_in, in_channels))

        # Map patches back into the input gradient
        col = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = dL_dA_prev_col[col::H_out*W_out, :].reshape(n_s, K_h, K_w, in_channels)
                vert_start = i * stride
                vert_end = vert_start + K_h
                horiz_start = j * stride
                horiz_end = horiz_start + K_w
                dL_dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += patch
                col += 1

        return dL_dA_prev, grad_W / n_s, grad_B / n_s

    def update_weights_and_biases(self, learn_rate, grad_W, grad_b):
        self.W -= learn_rate * grad_W
        self.b -= learn_rate * grad_b