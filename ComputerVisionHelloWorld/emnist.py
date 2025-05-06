import numpy as np
import gzip

def load_emnist_letters(images_path, labels_path, max_samples=None):
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
    mask_c = labels == 3 # 'c'
    mask_d = labels == 4 # 'd'
    mask_e = labels == 3 # 'e'
    mask_f = labels == 4 # 'f'
    mask_x = labels == 24  # 'x'
    mask_o = labels == 15  # 'o'

    #a_images = np.empty((0, *images.shape[1:]))
    a_images = images[mask_a]
    #b_images = np.empty((0, *images.shape[1:]))
    b_images = images[mask_b]
    c_images = images[mask_c]
    d_images = images[mask_d]
    e_images = images[mask_e]
    f_images = images[mask_f]
    x_images = images[mask_x]
    o_images = images[mask_o]

# Create one-hot encoded labels
    a_labels = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0, 0]]), (a_images.shape[0], 1))  # 'a'
    b_labels = np.tile(np.array([[0, 1, 0, 0, 0, 0, 0, 0]]), (b_images.shape[0], 1))  # 'b'
    c_labels = np.tile(np.array([[0, 0, 1, 0, 0, 0, 0, 0]]), (c_images.shape[0], 1))  # 'c'
    d_labels = np.tile(np.array([[0, 0, 0, 1, 0, 0, 0, 0]]), (d_images.shape[0], 1))  # 'd'
    e_labels = np.tile(np.array([[0, 0, 0, 0, 1, 0, 0, 0]]), (c_images.shape[0], 1))  # 'e'
    f_labels = np.tile(np.array([[0, 0, 0, 0, 0, 1, 0, 0]]), (d_images.shape[0], 1))  # 'f'
    o_labels = np.tile(np.array([[0, 0, 0, 0, 0, 0, 1, 0]]), (o_images.shape[0], 1))  # 'o'
    x_labels = np.tile(np.array([[0, 0, 0, 0, 0, 0, 0, 1]]), (x_images.shape[0], 1))  # 'x'

    # Combine and shuffle
    combined_images = np.concatenate((a_images, b_images, c_images, d_images, e_images, f_images, x_images, o_images), axis=0)
    combined_labels = np.concatenate((a_labels, b_labels, c_labels, d_labels, e_labels, f_labels, x_labels, o_labels), axis=0)

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