from PIL import Image, ImageOps
import numpy as np

np.random.seed(42)

def tanh(x):
    # return np.tanh(x)
    return 1/2 * np.tanh(x) + 1/2

def tanh_der(x):
    # return 1 - np.tanh(x) ** 2
    return 1/2 * (1 - np.tanh(x) ** 2)

def loss_func(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

def loss_der(y_pred, y_true):
    return y_pred - y_true

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
    hW = np.random.randn(n_n, n_f)
    hB = np.zeros((1, n_n))
    # hW holds the weights of the one neuron in the output layer.  It receive a 2d input, and so have two weights and one scalar bias
    # o1 is the sole neuron in the output layer.  It receives a 2d input, and so has two weights and one scalar bias
    oW = np.random.randn(1, n_n)
    oB = np.zeros((1, 1))
    return hW, hB, oW, oB

def forward_one_layer(X, W, b):
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
    A = tanh(Z)
    # A: activations of Z computer element-wise
    return Z, A

def backward_one_layer(dL_dA, Z, A_prev):
    # dA_dZ element-wise activation function derivative
    dA_dZ = tanh_der(Z) 
    # dL_dZ element-wise multiplication
    dL_dZ = dL_dA * dA_dZ
    grad_W = dL_dZ.T @ A_prev / A_prev.shape[0]
    grad_B = np.mean(dL_dZ, axis=0, keepdims=True)
    return dL_dZ, grad_W, grad_B                 

n_s = 1000
n_f = 784
n_n = 16

epoch_count = 10000
learn_rate = 0.1
X, Y = get_samples(n_s)
hW, hB, oW, oB = init_network(n_s, n_f, n_n)
for epoch in range(epoch_count):
    loss = 0    
    hZ, hA = forward_one_layer(X, hW, hB)
    oZ, oA = forward_one_layer(hA, oW, oB)

    # output layer gradient
    dL_doA = loss_der(oA, Y)
    dL_doZ, grad_oW, grad_oB = backward_one_layer(dL_doA, oZ, hA)
    # hidden layer gradient
    dL_dhA =  dL_doZ @ oW
    dL_dhZ, grad_hW, grad_hB = backward_one_layer(dL_dhA, hZ, X)    

    oW -= learn_rate * grad_oW
    oB -= learn_rate * grad_oB
    hW -= learn_rate * grad_hW
    hB -= learn_rate * grad_hB

    if epoch % 200 == 0:        
        print(f"epoch {epoch} | total loss: {loss: .4f}")

# for i in range(X.shape[0]):
i = 0
print(f"Input: {X[i]} | Expected: {Y[i][0]} | Actual: {oA[i][0]:.3f}")    
i = 500
print(f"Input: {X[i]} | Expected: {Y[i][0]} | Actual: {oA[i][0]:.3f}")    


print("Analyzing X")
input_vector = load_image_as_input_vector("c:\\temp\\x.bmp")
_, hA = forward_one_layer(input_vector, hW, hB)
_, oA = forward_one_layer(hA, oW, oB)
prediction = oA[0][0]
print(prediction)
if prediction > 0.5:
    print("Predicted: O")
else:
    print("Predicted: X")

print("Analyzing O")
input_vector = load_image_as_input_vector("c:\\temp\\o.bmp")
_, hA = forward_one_layer(input_vector, hW, hB)
_, oA = forward_one_layer(hA, oW, oB)
prediction = oA[0][0]
print(prediction)
if prediction > 0.5:
    print("Predicted: O")
else:
    print("Predicted: X")

print("Analyzing X handwriting")
input_vector = load_image_as_input_vector("c:\\temp\\x1.bmp")
_, hA = forward_one_layer(input_vector, hW, hB)
_, oA = forward_one_layer(hA, oW, oB)
prediction = oA[0][0]
print(prediction)
if prediction > 0.5:
    print("Predicted: O")
else:
    print("Predicted: X")

print("Analyzing O handwriting")
input_vector = load_image_as_input_vector("c:\\temp\\o1.bmp")
_, hA = forward_one_layer(input_vector, hW, hB)
_, oA = forward_one_layer(hA, oW, oB)
prediction = oA[0][0]
print(prediction)
if prediction > 0.5:
    print("Predicted: O")
else:
    print("Predicted: X")