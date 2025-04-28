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

import gzip
from PIL import Image, ImageOps
import numpy as np

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

def load_emnist_letters_abox(images_path, labels_path, max_samples=None):
    # Load labels
    with gzip.open(labels_path, 'rb') as lbpath:
        _ = int.from_bytes(lbpath.read(4), 'big')  # magic number
        num_labels = int.from_bytes(lbpath.read(4), 'big')
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    # Load images
    with gzip.open(images_path, 'rb') as imgpath:
        _ = int.from_bytes(imgpath.read(4), 'big')  # magic number
        num_images = int.from_bytes(imgpath.read(4), 'big')
        rows = int.from_bytes(imgpath.read(4), 'big')
        cols = int.from_bytes(imgpath.read(4), 'big')
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    # Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0

    # EMNIST-letters labels are 1-indexed: 1='a', ..., 26='z'
    # We want 'o' = 15 and 'x' = 24
    # We now expand this to include 'a' = 1 and 'b' = 2
    mask_a = labels == 1 # 'a'
    mask_b = labels == 2 # 'b'
    mask_x = labels == 24  # 'x'
    mask_o = labels == 15  # 'o'

    #a_images = np.empty((0, *images.shape[1:]))
    a_images = images[mask_a]
    #b_images = np.empty((0, *images.shape[1:]))
    b_images = images[mask_b]
    x_images = images[mask_x]
    o_images = images[mask_o]

# Create one-hot encoded labels
    a_labels = np.tile(np.array([[1, 0, 0, 0]]), (a_images.shape[0], 1))  # 'a'
    b_labels = np.tile(np.array([[0, 1, 0, 0]]), (b_images.shape[0], 1))  # 'b'
    o_labels = np.tile(np.array([[0, 0, 1, 0]]), (o_images.shape[0], 1))  # 'o'
    x_labels = np.tile(np.array([[0, 0, 0, 1]]), (x_images.shape[0], 1))  # 'x'

    # Combine and shuffle
    combined_images = np.concatenate((a_images, b_images, x_images, o_images), axis=0)
    combined_labels = np.concatenate((a_labels, b_labels, x_labels, o_labels), axis=0)

    if max_samples:
        combined_images = combined_images[:max_samples]
        combined_labels = combined_labels[:max_samples]

    indices = np.arange(combined_images.shape[0])
    np.random.shuffle(indices)

    combined_images = combined_images[indices]
    combined_labels = combined_labels[indices]

    # Flatten images for use in feedforward net
    X = combined_images.reshape(combined_images.shape[0], -1)
    Y = combined_labels

    return X, Y

def batch_samples(X, Y, batch_size):
    n_batches = np.ceil( Y.size / batch_size)    
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

def init_network(n_s, n_f, n_h1n, n_h2n, n_on):
    # initialize weights and biases
    # h1W holds the weights of the n_hn neurons in the first hidden layer.
    h1Limit = np.sqrt(6 / (n_f + n_h1n))
    h1W = np.random.randn(n_h1n, n_f) * h1Limit
    h1B = np.zeros((1, n_h1n))
    # h2W holds the weights of the n_hn neurons in the second hidden layer.
    h2Limit = np.sqrt(6 / (n_f + n_h2n))
    h2W = np.random.randn(n_h2n, n_h1n) * h2Limit
    h2B = np.zeros((1, n_h2n))
    # oW holds the weights of the n_on neuron in the output layer.    
    oLimit = np.sqrt(6 / (n_on + n_h2n))
    oW = np.random.randn(n_on, n_h2n) * oLimit
    oB = np.zeros((1, n_on))
    # return hW, hB, oW, oB, tanh, tanh_der, softmax, cross_entropy_loss, cross_entropy_derivative
    return h1W, h1B, h2W, h2B, oW, oB, tanh, tanh_der, softmax, cross_entropy_loss, cross_entropy_derivative

def forward_one_layer(X, W, b, activation):
    # n_s number of samples
    # n_f number of features
    # n_n number of neurons
    # X: n_s * n_f rows of inputs, columns of features
    # W: n_n * n_f rows of neuron, columns of weights per feature
    # b: n_n * 1 columns vector of biases
    WT = W.T
    # WT: n_f * n_n rows of weights per feature, colunmns of neurons
    Z = X @ WT + b;
    # Z: rows of outputs per input, columns of neurons?
    A = activation(Z)
    # A: activations of Z computer element-wise
    return Z, A

def backward_one_layer(dL_dA, Z, A_prev, activation_der):
    # dA_dZ element-wise activation function derivative
    dA_dZ = activation_der(Z) 
    # dL_dZ element-wise multiplication
    dL_dZ = dL_dA * dA_dZ
    grad_W = dL_dZ.T @ A_prev / A_prev.shape[0]
    grad_B = np.mean(dL_dZ, axis=0, keepdims=True)
    return dL_dZ, grad_W, grad_B                 

def forward_pass(X, h1W, h1B, h2W, h2B, oW, oB, hActivation, oActivation):
    h1Z, h1A = forward_one_layer(X, h1W, h1B, hActivation) #h Activation is tanh for now
    h2Z, h2A = forward_one_layer(h1A, h2W, h2B, hActivation) #h Activation is tanh for now
    oZ, oA = forward_one_layer(h2A, oW, oB, oActivation) # oActivation is softmax for now
    return h1Z, h1A, h2Z, h2A, oZ, oA

n_s = 1000
n_f = 784
n_h1n = 32
n_h2n = 32
n_on = 4

batch_size = 64
epoch_count = 1000
learn_rate = 0.05
X_emnist, Y_emnist = load_emnist_letters_abox("c:\\temp\\emnist\\emnist-letters-train-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-train-labels-idx1-ubyte.gz")
indices = np.arange(X_emnist.shape[0])
np.random.shuffle(indices)
X = X_emnist[indices]
Y = Y_emnist[indices]
X_batches, Y_batches = batch_samples(X, Y, batch_size)

a_count = Y[Y[:,0] == 1].shape[0]
b_count = Y[Y[:,1] == 1].shape[0]
o_count = Y[Y[:,2] == 1].shape[0]
x_count = Y[Y[:,3] == 1].shape[0]

print(a_count)
print(b_count)
print(o_count)
print(x_count)

h1W, h1B, h2W, h2B, oW, oB, hActivation, hActivation_der, oActivation, loss_func, loss_der = init_network(n_s, n_f, n_h1n, n_h2n, n_on)
for epoch in range(epoch_count):    
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        h1Z, h1A, h2Z, h2A, oZ, oA = forward_pass(X_batch, h1W, h1B, h2W, h2B, oW, oB, hActivation, oActivation)        

        # Output layer gradient (simple now!)
        dL_doZ = oA - Y_batch  # predicted probabilities minus true one-hot labels
        grad_oW = dL_doZ.T @ h2A / X_batch.shape[0]
        grad_oB = np.mean(dL_doZ, axis=0, keepdims=True)

        # # output layer gradient
        # dL_doA = loss_der(oA, Y_batch)
        # dL_doZ, grad_oW, grad_oB = backward_one_layer(dL_doA, oZ, hA, oActivation_der)
        
        # hidden layer gradient
        dL_dh2A =  dL_doZ @ oW
        dL_dh2Z, grad_h2W, grad_h2B = backward_one_layer(dL_dh2A, h2Z, h1A, hActivation_der)

        dL_dh1A = dL_dh2A @ h2W
        dL_dh1Z, grad_h1W, grad_h1B = backward_one_layer(dL_dh1A, h1Z, X_batch, hActivation_der)

        oW -= learn_rate * grad_oW
        oB -= learn_rate * grad_oB
        h2W -= learn_rate * grad_h2W
        h2B -= learn_rate * grad_h2B
        h1W -= learn_rate * grad_h1W
        h1B -= learn_rate * grad_h1B
    
    if epoch % 100 == 0 or epoch == epoch_count -1:
        h1Z, h1A = forward_one_layer(X, h1W, h1B, hActivation)
        h2Z, h2A = forward_one_layer(h1A, h2W, h2B, hActivation)
        oZ, oA = forward_one_layer(h2A, oW, oB, oActivation)
        mse_loss = mse(oA, Y)
        print(f"epoch {epoch} | MSE: {np.mean(mse_loss):.10f}")  
        loss = loss_func(oA, Y)
        print(f"epoch {epoch} | CE: {np.mean(loss):.10f}")  

def check_result(input_vector, expected):        
    _, _, _, _, _, oA = forward_pass(input_vector, h1W, h1B, h2W, h2B, oW, oB, hActivation, oActivation)    
    print(oA)
    prediction = rounded = (oA >= 0.5).astype(int)    
    print(prediction)
    # print(f"Expected {expected}, Actual {actual}")    



for x in ["x", "o", "x1", "o1", "x2", "o2", "x3", "o3", "x4", "o4", "a1", "b1"]:
    print()
    print(x)
    input_vector = load_image_as_input_vector(f"c:\\temp\\{x}.bmp")
    check_result(input_vector, "X")