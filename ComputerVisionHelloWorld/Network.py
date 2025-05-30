import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, X, Y, loss_function, loss_der, epoch_count, batch_size, learn_rate):
        for layer in self.layers:
            layer.is_training = True
        
        X_batches, Y_batches = self.batch_samples(X, Y, batch_size)
        for epoch in range(epoch_count):
            batch_number = 0
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                batch_number += 1
                # print(batch_number)
                zValues, aValues = self.forward_pass(X_batch)
                oA = aValues[-1]
                loss_der_value = oA - Y_batch  # predicted probabilities minus true one-hot labels        
                grad_W, grad_B = self.backward_pass(X_batch, loss_der_value, zValues, aValues)        
                self.update_weights_and_biases(learn_rate, grad_W, grad_B)
    
            if 1 == 1: #epoch % 100 == 0 or epoch == epoch_count - 1:
                _, aValues = self.forward_pass(X)
                oA = aValues[-1]
                # mse_loss = mse(oA, Y)
                # print(f"epoch {epoch} | MSE: {np.mean(mse_loss):.10f}")  
                loss = loss_function(oA, Y)
                print(f"epoch {epoch} | CE: {np.mean(loss):.10f}")  
        for layer in self.layers:
            layer.is_training = False
        return;

    def optimize_sample(self, X, Y, loss_function, loss_der, epoch_count, batch_size, learn_rate):
        X_optimized = X.copy()
        for epoch in range(epoch_count):
            zValues, aValues = self.forward_pass(X_optimized)
            oA = aValues[-1]
            # loss_der_value = oA - Y  # predicted probabilities minus true one-hot labels        
            # grad_X = self.backward_pass_input_optimization(X_optimized, loss_der_value, zValues, aValues)
            grad_X = self.backward_pass_input_optimization(X_optimized, zValues, aValues, target_class_index=0)  # 'A' is index 0

            X_optimized -= learn_rate * grad_X

            if epoch % 1000 == 0 or epoch == epoch_count - 1:
                # _, aValues = self.forward_pass(X_optimized)
                # oA = aValues[-1]                
                # loss = loss_function(oA, Y)
                # print(f"epoch {epoch} | CE: {np.mean(loss):.10f}")  
                logits = zValues[-1]                
                predicted_class = np.argmax(logits)
                print(f"epoch {epoch} | logit for 'A': {logits[0, 0]:.6f} | predicted: {predicted_class}")

        return X_optimized

    @staticmethod
    def batch_samples(X, Y, batch_size):
        n_batches = np.ceil(len(Y) / batch_size)    
        X_batches = np.array_split(X, n_batches)
        Y_batches = np.array_split(Y, n_batches)
        return X_batches, Y_batches

    def forward_pass(self, X):
        zValues = []
        aValues = []
        a = X
        for layer in self.layers:
            z, a = layer.forward_pass(a)
            zValues.append(z)
            aValues.append(a)
        return zValues, aValues

    def backward_pass(self, X, loss_der_value, zValues, aValues):
        # Output layer gradients    
        # grad_oW = (loss_der_value.T @ aValues[1]) / X.shape[0]
        # grad_oB = np.mean(loss_der_value, axis=0, keepdims=True)
        # grad_W = [grad_oW]
        # grad_B = [grad_oB]
        # dL_dA = loss_der_value

        grad_W = []
        grad_B = []

        last = len(self.layers) - 1
        Z = zValues[last]
        A_prev = aValues[last - 1] if last > 0 else X

        dL_dA, grad_oW, grad_oB = self.layers[last].backward_pass(loss_der_value, Z, A_prev)

        grad_W.insert(0, grad_oW)
        grad_B.insert(0, grad_oB)

        # Iterate over the layers in reverse order, but skip the output layer, the last one, as we accounted for it above.
        # for layer in layers[:-1][::-1]
        layer_count = len(self.layers)
        self.layers[layer_count - 1].print_all = False
        for i in range(layer_count)[:-1][::-1]:            
            this_layer_z_value = zValues[i]
            prev_layer_input = aValues[i - 1] if i > 0 else X
                    
            dL_dA, grad_hW, grad_hB = self.layers[i].backward_pass(dL_dA, this_layer_z_value, prev_layer_input)            

            grad_W.insert(0, grad_hW)
            grad_B.insert(0, grad_hB)

        return grad_W, grad_B
        
    def backward_pass_input_optimization(self, X, zValues, aValues, target_class_index):
        last = len(self.layers) - 1
        Z = zValues[last]
        A_prev = aValues[last - 1] if last > 0 else X
        
        # dL_dA = self.layers[last].backward_pass_input_optimization(loss_der_value, Z, A_prev)        
        dL_dZ = np.zeros_like(Z)
        dL_dZ[0, target_class_index] = -1
        dL_dA = self.layers[last].backward_pass_input_optimization(dL_dZ, Z, A_prev)

        # Iterate over the layers in reverse order, but skip the output layer, the last one, as we accounted for it above.
        # for layer in layers[:-1][::-1]
        layer_count = len(self.layers)
        self.layers[layer_count - 1].print_all = False
        for i in range(layer_count)[:-1][::-1]:            
            this_layer_z_value = zValues[i]
            prev_layer_input = aValues[i - 1] if i > 0 else X                    
            dL_dA = self.layers[i].backward_pass_input_optimization(dL_dA, this_layer_z_value, prev_layer_input)

        return dL_dA

    def update_weights_and_biases(self, learn_rate, grad_W, grad_B):
        layer_count = len(self.layers)
        for i in range(layer_count):
            self.layers[i].update_weights_and_biases(learn_rate, grad_W[i], grad_B[i])