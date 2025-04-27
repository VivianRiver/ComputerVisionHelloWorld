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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

def mse_der(y_pred, y_true):
    return y_pred - y_true

def bce(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_der(y_pred, y_true):
    return (y_pred - y_true) / y_pred / (1 - y_pred)

def load_emnist_letters_xo(images_path, labels_path, max_samples=None):
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
    mask_x = labels == 24  # 'x'
    mask_o = labels == 15  # 'o'

    x_images = images[mask_x]
    o_images = images[mask_o]

    x_labels = np.zeros((x_images.shape[0], 1))  # 0 = x
    o_labels = np.ones((o_images.shape[0], 1))   # 1 = o

    # Combine and shuffle
    combined_images = np.concatenate((x_images, o_images), axis=0)
    combined_labels = np.concatenate((x_labels, o_labels), axis=0)

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

def get_samples(n_s):
    def draw_X():
        img = np.zeros((28, 28))
        # draw an X using diagonal lines
        for i in range(28):
            img[i, i] = 1.0
            img[i, 27 - i] = 1.0
        return img
        
    def draw_O():
        img = np.zeros((28, 28))
        # draw an O
        cx, cy = 13.5, 13.5  # center of the grid (float helps symmetry)
        r = 10  # reasonable radius (not touching edges)
        theta = np.linspace(0, 2 * np.pi, 100)
        x_vals = cx + r * np.cos(theta)
        y_vals = cy + r * np.sin(theta)
        x_idx = np.round(x_vals).astype(int)
        y_idx = np.round(y_vals).astype(int)
        img[y_idx, x_idx] = 1.0  # y is the row, x is the column
        return img

    def add_noise(img, scale=0.1):
        noise = np.random.normal(0, scale, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0.0, 1.0)            
    
    def shift_image(img, dx=2, dy=2):
        shifted = np.roll(img, shift=(np.random.randint(-dy, dy + 1),
                                      np.random.randint(-dx, dx + 1)), axis=(0, 1))
        return shifted

    X_data = []
    Y_data = []
    for _ in range(n_s // 2):
        image_X = draw_X()
        image_X = shift_image(image_X)
        image_X = add_noise(image_X)
        flat_X = image_X.flatten()
        X_data.append(flat_X)
        Y_data.append(0)
    for _ in range(n_s // 2):
        image_O = draw_O()
        image_O = shift_image(image_O)
        image_O = add_noise(image_O)
        flat_O = image_O.flatten()
        X_data.append(flat_O)
        Y_data.append(1)    
    X = np.array(X_data) # shape: (1000, 784)    
    Y = np.array(Y_data).reshape(-1, 1) # shape: (1000, 1)
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

def init_network(n_s, n_f, n_n):    
    # initialize weights and biases
    # hW holds the weights of the two neurons in the hidden layer.  They receive a 2d input, and so have two weights and one scalar bias each
    hLimit = np.sqrt(2 / (n_f + n_n))
    hW = np.random.randn(n_n, n_f) * hLimit
    hB = np.zeros((1, n_n))
    # hW holds the weights of the one neuron in the output layer.  It receive a 2d input, and so have two weights and one scalar bias
    # o1 is the sole neuron in the output layer.  It receives a 2d input, and so has two weights and one scalar bias
    oLimit = np.sqrt(2 / (1 + n_n))
    oW = np.random.randn(1, n_n) * oLimit
    oB = np.zeros((1, 1))
    return hW, hB, oW, oB, tanh, tanh_der, sigmoid, sigmoid_der, bce, bce_der

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

n_s = 1000
n_f = 784
n_n = 16

batch_size = 64
epoch_count = 1000
learn_rate = 0.1
X_synth, Y_synth = get_samples(n_s)
X_emnist, Y_emnist = load_emnist_letters_xo("c:\\temp\\emnist\\emnist-letters-train-images-idx3-ubyte.gz", "c:\\temp\\emnist\\emnist-letters-train-labels-idx1-ubyte.gz")
X_combined = np.concatenate((X_emnist, X_synth), axis=0)
Y_combined = np.concatenate((Y_emnist, Y_synth), axis=0)
indices = np.arange(X_combined.shape[0])
np.random.shuffle(indices)
X = X_combined[indices]
Y = Y_combined[indices]
X_batches, Y_batches = batch_samples(X, Y, batch_size)

hW, hB, oW, oB, hActivation, hActivation_der, oActivation, oActivation_der, loss_func, loss_der = init_network(n_s, n_f, n_n)
for epoch in range(epoch_count):    
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        hZ, hA = forward_one_layer(X_batch, hW, hB, hActivation)
        oZ, oA = forward_one_layer(hA, oW, oB, oActivation)

        # output layer gradient
        dL_doA = loss_der(oA, Y_batch)
        dL_doZ, grad_oW, grad_oB = backward_one_layer(dL_doA, oZ, hA, oActivation_der)
        # hidden layer gradient
        dL_dhA =  dL_doZ @ oW
        dL_dhZ, grad_hW, grad_hB = backward_one_layer(dL_dhA, hZ, X_batch, hActivation_der)

        oW -= learn_rate * grad_oW
        oB -= learn_rate * grad_oB
        hW -= learn_rate * grad_hW
        hB -= learn_rate * grad_hB
    
    if epoch % 100 == 0 or epoch == epoch_count -1:
        hZ, hA = forward_one_layer(X, hW, hB, hActivation)
        oZ, oA = forward_one_layer(hA, oW, oB, oActivation)
        mse_loss = mse(oA, Y)
        print(f"epoch {epoch} | MSE: {np.mean(mse_loss):.10f}")  

def check_result(input_vector, expected):        
    _, hA = forward_one_layer(input_vector, hW, hB, hActivation)
    _, oA = forward_one_layer(hA, oW, oB, oActivation)
    prediction = oA[0][0]
    actual = "O" if prediction > 0.5 else "X"
    print(prediction)
    print(f"Expected {expected}, Actual {actual}")    

print()
print("x")
input_vector = load_image_as_input_vector("c:\\temp\\x.bmp")
check_result(input_vector, "X")

print()
print("o")
input_vector = load_image_as_input_vector("c:\\temp\\o.bmp")
check_result(input_vector, "O")

print()
print("x1")
input_vector = load_image_as_input_vector("c:\\temp\\x1.bmp")
check_result(input_vector, "X")

print()
print("o1")
input_vector = load_image_as_input_vector("c:\\temp\\o1.bmp")
check_result(input_vector, "O")

print()
print("x2")
input_vector = load_image_as_input_vector("c:\\temp\\x2.bmp")
check_result(input_vector, "X")

print()
print("o2")
input_vector = load_image_as_input_vector("c:\\temp\\o2.bmp")
check_result(input_vector, "O")

print()
print("x3")
input_vector = load_image_as_input_vector("c:\\temp\\x3.bmp")
check_result(input_vector, "X")

print()
print("o3")
input_vector = load_image_as_input_vector("c:\\temp\\o3.bmp")
check_result(input_vector, "O")

print()
print("x4")
input_vector = load_image_as_input_vector("c:\\temp\\x4.bmp")
check_result(input_vector, "X")

print()
print("o4")
input_vector = load_image_as_input_vector("c:\\temp\\o4.bmp")
check_result(input_vector, "O")