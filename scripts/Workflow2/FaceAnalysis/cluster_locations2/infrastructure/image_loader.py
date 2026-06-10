import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class ImageLoader:
    def load(self, path: Path) -> Optional[np.ndarray]:
        try:
            data = path.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
