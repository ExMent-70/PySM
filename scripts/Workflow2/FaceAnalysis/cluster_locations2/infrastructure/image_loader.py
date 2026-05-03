import numpy as np
import cv2
from pathlib import Path
from typing import Optional


class ImageLoader:
    def __init__(self, input_size: tuple[int, int]):
        self.input_size = input_size

    def load(self, path: Path) -> Optional[np.ndarray]:
        try:
            data = path.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        img = (img - mean) / std
        return np.expand_dims(img.transpose(2, 0, 1), axis=0)