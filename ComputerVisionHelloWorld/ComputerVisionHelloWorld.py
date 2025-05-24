# "Original", total MSE after 1000 epochs:
# MSE =  .0069460356

# After replacing hidden layer activation with tanh and output layer activation with sigmoid
# MSE = .0201235843

# After replacing MSE loss function with BCE loss function in training
# MSE = .0069460356

# With Xavier initialization,
# MSE = .0018702731 

# With batch size of 64,
# MSE = .0000004627

# Training on letters A, B, C, D, E, F, O, X
# With 5 hidden layers
# CE = 0.0000650340
# Match Rate = 94%

# Training on letters A, B, C, D, E, F, G, H, O, X
# With 5 hidden layers
# CE = 0.1163045366
# Match Rate = 92%

# Training on letters A, B, C, D, E, F, G, H, I, J, O, X
# With 5 hidden layers, 200 epochs
# CE = 0.1368096434
# Match Rate = 91%
# Interestingly, not only does it level out here, but if training continues, it fails catastrophically at about epoch 800 and match rate drops to 0.

# Training on letters A, B, C, D, E, F, G, H, I, J, K, L, O, X
# With 5 hidden layers, 200 epochs
# CE = 0.1685344727
# Match Rate = 87.5%

# Training on letters A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, X
# With 5 hidden layers, 200 epochs
# CE = 0.2050403917
# Match Rate = 86.7%

# Training on letters A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, X
# With 5 hidden layers, 40 epochs
# CE = 0.2607291410
# Match Rate = 86.6%

# Training on letters A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, X
# With 5 hidden layers, 10 epochs
# CE = 0.3425823625
# Match Rate = 85.3%

from turtle import backward
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

from DenseLayer import DenseLayer
from DropoutLayer import DropoutLayer
from Network import Network

from emnist import Emnist
from BrightenAugmentor import BrightenAugmentor

from LetterDedications import letters
import layer

np.random.seed(42)

def tanh(x):
    return np.tanh(x)        

def tanh_der(x):
    return 1 - np.tanh(x) ** 2        

def relu(x):
    return np.maximum(0, x)        

def relu_der(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(z):
    # Step 1: Stablize the inputs by subtracting the max per sample
    z_stable = z - np.max(z, axis=1, keepdims=True)
    # Step 2: Exponentiate
    exp_z = np.exp(z_stable)
    # Step 3: Normalize by the sum of exponentials
    softmax_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax_output


def mse(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

def mse_der(y_pred, y_true):
    return y_pred - y_true

def bce(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_der(y_pred, y_true):
    return (y_pred - y_true) / y_pred / (1 - y_pred)

def cross_entropy_loss(y_pred, y_true):
    # Clip y_pred to avoid log(0)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

def load_image_as_input_vector(path):
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    array = img_array.flatten().reshape(1, -1)
    return array

def init_network(n_f, n_h1n, n_h2n, n_on):
    # initialize weights and biases
    # h1W holds the weights of the n_hn neurons in the first hidden layer.
    h1Limit = np.sqrt(6 / (n_f + n_h1n))
    h1W = np.random.randn(n_h1n, n_f) * h1Limit
    h1B = np.zeros((1, n_h1n))
    hLayer1 = DenseLayer(h1W, h1B, relu, relu_der)    

    # h2W holds the weights of the n_hn neurons in the second hidden layer.
    h2Limit = np.sqrt(6 / (n_f + n_h2n))
    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    hLayer2 = DenseLayer(h2W, h2B, relu, relu_der)

    # dropoutLayer1 = DropoutLayer(0.05)

    # h3W holds the weights of the n_hn neurons in the second hidden layer.
    # use same parameters as layer 2
    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    hLayer3 = DenseLayer(h2W, h2B, tanh, tanh_der)

    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    hLayer4 = DenseLayer(h2W, h2B, tanh, tanh_der)
    
    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    hLayer5 = DenseLayer(h2W, h2B, tanh, tanh_der)

    # oW holds the weights of the n_on neuron in the output layer.    
    oLimit = np.sqrt(6 / (n_on + n_h2n))
    oW = np.random.randn(n_on, n_h2n) * oLimit
    oB = np.zeros((1, n_on))
    oLayer = DenseLayer(oW, oB, softmax, lambda Z: np.ones_like(Z))    

    layers = [
        hLayer1,
        hLayer2,
        # dropoutLayer1,
        hLayer3,
        hLayer4,
        hLayer5,
        oLayer]

    network = Network(layers)

    # === Loss Functions ===
    def cross_entropy(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred), axis=1)

    def cross_entropy_derivative(y_pred, y_true):
        return y_pred - y_true

    return network, cross_entropy, cross_entropy_derivative

emnist = Emnist(letters, "c:\\temp\\emnist\\emnist-letters-train-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-train-labels-idx1-ubyte.gz", augment=True)


X_emnist, Y_emnist = emnist.X, emnist.Y

n_f = 784
n_h1n = 32
n_h2n = 32
n_on = emnist.letter_count

indices = np.arange(X_emnist.shape[0])
np.random.shuffle(indices)
X = X_emnist[indices]
Y = Y_emnist[indices]

network, loss_func, loss_der = init_network(n_f, n_h1n, n_h2n, n_on)
epoch_count = 10
batch_size = 64
learn_rate = 0.05

network.train(X, Y, cross_entropy_loss, cross_entropy_derivative, epoch_count, batch_size, learn_rate)

match_count = 0
mismatch_count = 0
total_count = 0
mismatches = []
def check_result(input_vector, expected):        
    global match_count, mismatch_count, total_count  # <-- add this line
    _, aValues = network.forward_pass(input_vector)
    oA = aValues[-1]    
    prediction = rounded = (oA >= 0.5).astype(int)        
    if (np.array_equal(expected.flatten(), prediction.flatten())):
        match_count += 1        
    else:
        mismatch_count += 1        
        expected_label = np.argmax(expected)
        predicted_label = np.argmax(oA)
        mismatches.append((input_vector, expected_label, predicted_label))
        
    total_count += 1    

emnist_test = Emnist(letters, "c:\\temp\\emnist\\emnist-letters-test-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-test-labels-idx1-ubyte.gz", augment=False)
X_test, Y_test = emnist_test.X, emnist_test.Y
for x, y in zip(X_test, Y_test):
    check_result(x, y)

print(f"Match: {match_count}")
print(f"Mismatch: {mismatch_count}")
print(f"Total: {total_count}")
print(f"Match Rate {match_count * 100.0 / total_count:.4f}%")

fig, axes = plt.subplots(16, 16, figsize=(10, 10))
axes = axes.flatten()

class_names = emnist.letters  # Adjust for your classes

for ax, (img, true_label, pred_label) in zip(axes, mismatches):
    img_corrected = img.reshape(28, 28).T  # ← just this
    ax.imshow(img_corrected, cmap='gray')
    ax.axis('off')
    ax.set_title(f"T: {class_names[true_label]}, P: {class_names[pred_label]}")

plt.tight_layout()
plt.show()

# for x in ["x", "o", "x1", "o1", "x2", "o2", "x3", "o3", "x4", "o4", "a1", "b1"]:
#     print()
#     print(x)
#     input_vector = load_image_as_input_vector(f"c:\\temp\\{x}.bmp")
#     check_result(input_vector, "X")