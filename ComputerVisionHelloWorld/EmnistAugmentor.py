import numpy as np
from abc import ABC, abstractmethod

class EmnistAugmentor(ABC):
    def augment(self, image):
        assert (image.shape == (28, 28))
        return self._augment_core(image)
    
    @abstractmethod
    def _augment_core(self, image):
        pass
        