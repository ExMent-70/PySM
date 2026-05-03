from abc import ABC, abstractmethod
import numpy as np
from typing import List


class MultiModalModel(ABC):

    @abstractmethod
    def encode_images(self, images: List[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def shutdown(self):
        pass