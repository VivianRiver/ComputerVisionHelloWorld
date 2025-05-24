import numpy as np
import gzip
from BrightenAugmentor import BrightenAugmentor
from RotationAugmentor import RotateAugmentor

class Emnist:
    def __init__(self, letters, images_path, labels_path, augment = False, max_samples=None):        
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
        # images = images.astype(np.float32) / 255.0
        
        #letters = ['o', 'x']
        letter_count = len(letters)                

        # TODO: Create Image loader class to encapsulate loading and augmenting" images
        augmentors = []
        
        if (augment):
            #darkeners = [BrightenAugmentor(0, i) for i in range(-5, 0)]
            #brighteners = [BrightenAugmentor(0, i) for i in range(0, 5)]
            #augmentors = darkeners + brighteners            
            #augmentors = [BrightenAugmentor(0, 3)]
            rotators = [RotateAugmentor(0, i) for i in range(-9, 15, 4)]
            augmentors = rotators            

        combined_images = []
        combined_labels = []

        for i in range(letter_count):
            l = letters[i]
            alpha_position = ord(l) - ord('a') + 1
            mask = labels == alpha_position
            
            # Here, we get the images of the particular letter from EMNIST.
            original_letter_images = images[mask]
            # Now, we "augment" the images by making slight modifications to each one and adding the modified images.            
            augmented_letter_images = np.array([augmentor.augment(image) for augmentor in augmentors for image in original_letter_images])            
            # letter_labels = np.tile(one_hot, (letter_images.shape[0], 1))  
                      
            letter_images = np.concatenate([original_letter_images, augmented_letter_images], axis=0) if augment else original_letter_images
            # combined_labels.append(letter_labels)            
            one_hot = self.one_hot(letter_count, i)
            letter_labels = np.tile(one_hot, (len(letter_images), 1))

            combined_images.append(letter_images)
            combined_labels.append(letter_labels)
            print(f"{l}: position in alphabet: {alpha_position}, {len(original_letter_images)} EMNIST images, with augmentation: {len(letter_images)} images")

        combined_images = np.concatenate(combined_images)
        combined_images = combined_images.astype(np.float32) / 255.0
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
        self.letters = letters
        self.letter_count = letter_count
    def one_hot(self, length, index):
        array = np.zeros(length, dtype=int)
        array[index] = 1
        return array