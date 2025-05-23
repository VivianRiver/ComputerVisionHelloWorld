import numpy as np
from EmnistAugmentor import EmnistAugmentor

class BrightenAugmentor(EmnistAugmentor):
    def __init__(self, background_color, bits):
        self.background_color = background_color,
        self.bits = bits
        super().__init__()

    def _augment_core(self, image):
        M = image.copy()
        unchanged = (M == self.background_color)
        adjusted = M + self.bits
        adjusted = np.clip(adjusted, 0, 255)
        M[~unchanged] = adjusted[~unchanged]
        return M