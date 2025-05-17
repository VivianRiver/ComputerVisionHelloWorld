import numpy as np
import gzip

class Emnist:
    def __init__(self, images_path, labels_path, max_samples=None):        
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

        # letters = ['a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'o', 'x']
        letters = ['o', 'x']
        letter_count = len(letters)

        combined_images = []
        combined_labels = []

        for i in range(letter_count):
            l = letters[i]
            alpha_position = ord(l) - ord('a') + 1
            mask = labels == alpha_position
            letter_images = images[mask]
            one_hot = self.one_hot(letter_count, i)
            letter_labels = np.tile(one_hot, (letter_images.shape[0], 1))
            combined_images.append(letter_images)
            combined_labels.append(letter_labels)
            print(f"{l}: position in alphabet: {alpha_position}, {len(letter_images)} images")

        combined_images = np.concatenate(combined_images)
        combined_labels = np.concatenate(combined_labels)

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

        self.X = X
        self.Y = Y
        self.letter_count = letter_count
    def one_hot(self, length, index):
        array = np.zeros(length, dtype=int)
        array[index] = 1
        return array