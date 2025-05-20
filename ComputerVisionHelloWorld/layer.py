from abc import ABC, abstractmethod

class Layer(ABC):   
    def __init__(self):
        self.shapeLogger = ShapeLogger()
        self.print_all = True

    @abstractmethod
    def _forward_core(self, X):
        pass

    def forward_pass(self, X):        
        Z, A = self._forward_core(X)        
        if (self.print_all):
            self.shapeLogger.log_forward(self.__class__.__name__, X.shape, A.shape)
            self.shapeLogger.display_forward()
        return Z, A
    
    @abstractmethod
    def _backward_core(self, dL_dA, Z, A_prev):
        pass

    def backward_pass(self, dL_dA, Z, A_prev):
        dL_dZ, grad_W, grad_B = self._backward_core(dL_dA, Z, A_prev)        
        if (self.print_all):
            self.shapeLogger.log_backward(self.__class__.__name__, dL_dA.shape, Z.shape, A_prev.shape)
            self.shapeLogger.display_backward()
            # turn off printing after the first backward_pass, which we assume happens after the forward pass
            # so that printing happens only for first epoch
            self.print_all = False
        return dL_dZ, grad_W, grad_B

    @abstractmethod
    def update_weights_and_biases(self, learn_rate, grad_W, grad_b):
        pass

class ShapeLogger:
    def __init__(self):
        self.forward_shapes = []
        self.backward_shapes = []            

    def log_forward(self, layer_name, input_shape, output_shape):
        self.forward_shapes.append((layer_name, input_shape, output_shape))

    def log_backward(self, layer_name, dL_dA_shape, Z_shape, A_prev_shape):
        self.backward_shapes.append((layer_name, dL_dA_shape, Z_shape, A_prev_shape))

    def display_forward(self):
        print("\n📈 FORWARD PASS SHAPES:")
        for name, inp, out in self.forward_shapes:
            print(f"Layer {name:<20} | Input: {inp} → Output: {out}")

    def display_backward(self):
        print("\n🔁 BACKWARD PASS SHAPES:")
        for name, dLdA, Z, Aprev in self.backward_shapes:
            print(f"Layer {name:<20} | dL/dA: {dLdA} | Z: {Z} | A_prev: {Aprev}")
        print("\n")