import numpy as np
from scipy.ndimage import rotate
from EmnistAugmentor import EmnistAugmentor

class RotateAugmentor(EmnistAugmentor):
    def __init__(self, background_color, degrees):
        self.background_color = background_color
        self.degrees = degrees
        super().__init__()

    def _augment_core(self, image):
        rotated = rotate(image, angle=self.degrees, reshape=False, mode="constant", cval=self.background_color)
        return rotated