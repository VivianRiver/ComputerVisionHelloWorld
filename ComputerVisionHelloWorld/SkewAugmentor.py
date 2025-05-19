import numpy as np
from skimage.transform import AffineTransform, warp
from EmnistAugmentor import EmnistAugmentor

class SkewAugmentor(EmnistAugmentor):
    def __init__(self, background_color, shear):
        self.background_color = background_color
        self.shear = shear
        super().__init__()

    def _augment_core(self, image):
        transform = AffineTransform(shear=self.shear)
        sheared = warp(image, inverse_map=transform.inverse, mode="constant", cval=self.background_color)
        sheared *= 255.0
        sheared = np.clip(sheared, 0, 255)
        sheared = sheared.astype(np.uint8)  # optional but usually wise
        return sheared