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

from turtle import backward
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

import emnist
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

def tanh_der(x):
    return 1 - np.tanh(x) ** 2 

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

def batch_samples(X, Y, batch_size):
    n_batches = np.ceil(len(Y) / batch_size)    
    X_batches = np.array_split(X, n_batches)
    Y_batches = np.array_split(Y, n_batches)
    return X_batches, Y_batches

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
    hLayer1 = layer.layer(h1W, h1B, tanh, tanh_der)    

    # h2W holds the weights of the n_hn neurons in the second hidden layer.
    h2Limit = np.sqrt(6 / (n_f + n_h2n))
    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    hLayer2 = layer.layer(h2W, h2B, tanh, tanh_der)

    # oW holds the weights of the n_on neuron in the output layer.    
    oLimit = np.sqrt(6 / (n_on + n_h2n))
    oW = np.random.randn(n_on, n_h2n) * oLimit
    oB = np.zeros((1, n_on))
    oLayer = layer.layer(oW, oB, softmax, mse_der)

    layers = [hLayer1, hLayer2, oLayer]

    # return hW, hB, oW, oB, tanh, tanh_der, softmax, cross_entropy_loss, cross_entropy_derivative
    return layers, cross_entropy_loss, cross_entropy_derivative

def forward_pass(layers, X):
     zValues = []
     aValues = []
     a = X
     for layer in layers:
         z, a = layer.forward_pass(a)
         zValues.append(z)
         aValues.append(a)
     return zValues, aValues

def backward_pass(layers, X, loss_der_value, zValues, aValues):
    # Output layer gradients    
    grad_oW = (loss_der_value.T @ aValues[1]) / X.shape[0]
    grad_oB = np.mean(loss_der_value, axis=0, keepdims=True)

    grad_W = [grad_oW]
    grad_B = [grad_oB]

    prev_der_value = loss_der_value

    # Iterate over the layers in reverse order, but skip the output layer, the last one, as we accounted for it above.
    # for layer in layers[:-1][::-1]
    layer_count = len(layers)
    for i in range(layer_count)[:-1][::-1]:
        next_layer_weights = layers[i + 1].W        
        this_layer_z_value = zValues[i]
        prev_layer_input = aValues[i - 1] if i > 0 else X
        
        dL_dA = prev_der_value @ next_layer_weights
        dL_dZ, grad_hW, grad_hB = layers[i].backward_one_layer(dL_dA, this_layer_z_value, prev_layer_input)

        prev_der_value = dL_dZ

        grad_W.insert(0, grad_hW)
        grad_B.insert(0, grad_hB)

    return grad_W, grad_B
        
def update_weights_and_biases(layers, learn_rate, grad_W, grad_B):
    layers[2].update_weights_and_biases(learn_rate, grad_W[2], grad_B[2])
    layers[1].update_weights_and_biases(learn_rate, grad_W[1], grad_B[1])
    layers[0].update_weights_and_biases(learn_rate, grad_W[0], grad_B[0])

n_f = 784
n_h1n = 32
n_h2n = 32
n_on = 8

batch_size = 64
epoch_count = 200
learn_rate = 0.05
X_emnist, Y_emnist = emnist.load_emnist_letters("c:\\temp\\emnist\\emnist-letters-train-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-train-labels-idx1-ubyte.gz")
indices = np.arange(X_emnist.shape[0])
np.random.shuffle(indices)
X = X_emnist[indices]
Y = Y_emnist[indices]
X_batches, Y_batches = batch_samples(X, Y, batch_size)

a_count = Y[Y[:,0] == 1].shape[0]
b_count = Y[Y[:,1] == 1].shape[0]
c_count = Y[Y[:,2] == 1].shape[0]
d_count = Y[Y[:,3] == 1].shape[0]
e_count = Y[Y[:,4] == 1].shape[0]
f_count = Y[Y[:,5] == 1].shape[0]
o_count = Y[Y[:,6] == 1].shape[0]
x_count = Y[Y[:,7] == 1].shape[0]

print(a_count)
print(b_count)
print(c_count)
print(d_count)
print(e_count)
print(f_count)
print(o_count)
print(x_count)

layers, loss_func, loss_der = init_network(n_f, n_h1n, n_h2n, n_on)
for epoch in range(epoch_count):
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        zValues, aValues = forward_pass(layers, X_batch)
        oA = aValues[-1]
        loss_der_value = oA - Y_batch  # predicted probabilities minus true one-hot labels        
        grad_W, grad_B = backward_pass(layers, X_batch, loss_der_value, zValues, aValues)        
        update_weights_and_biases(layers, learn_rate, grad_W, grad_B)
    
    if epoch % 100 == 0 or epoch == epoch_count -1:
        _, aValues = forward_pass(layers, X)
        oA = aValues[-1]
        mse_loss = mse(oA, Y)
        print(f"epoch {epoch} | MSE: {np.mean(mse_loss):.10f}")  
        loss = loss_func(oA, Y)
        print(f"epoch {epoch} | CE: {np.mean(loss):.10f}")  

match_count = 0
mismatch_count = 0
total_count = 0
mismatches = []
def check_result(input_vector, expected):        
    global match_count, mismatch_count, total_count  # <-- add this line
    _, aValues = forward_pass(layers, input_vector)
    oA = aValues[-1]    
    prediction = rounded = (oA >= 0.5).astype(int)        
    if (np.array_equal(expected.flatten(), prediction.flatten())):
        match_count += 1        
    else:
        mismatch_count += 1
        class_names = ["A", "B", "C", "D", "O", "X"]  # Adjust for your classes
        expected_label = np.argmax(expected)
        predicted_label = np.argmax(oA)
        mismatches.append((input_vector, expected_label, predicted_label))
        
    total_count += 1    

X_test, Y_test = emnist.load_emnist_letters("c:\\temp\\emnist\\emnist-letters-test-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-test-labels-idx1-ubyte.gz")
for x, y in zip(X_test, Y_test):
    check_result(x, y)

print(f"Match: {match_count}")
print(f"Mismatch: {mismatch_count}")
print(f"Total: {total_count}")

fig, axes = plt.subplots(16, 16, figsize=(10, 10))
axes = axes.flatten()

class_names = ["A", "B", "C", "D", "E", "F", "O", "X"]  # Adjust for your classes

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