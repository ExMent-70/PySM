import numpy as np
from pathlib import Path

from .clip_model_loader import ClipModelLoader
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class ClipModel:
    def __init__(
        self,
        model_loader: ClipModelLoader,
        tokenizer_path: Path,
        input_size=(224, 224),
    ):
        self.image_encoder = ImageEncoder(model_loader, input_size)
        self.text_encoder = TextEncoder(model_loader, tokenizer_path, input_size)
        self._closed = False

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        return self.image_encoder.encode(images)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.text_encoder.encode(texts)

    def similarity(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
        return img_emb @ txt_emb.T

    def shutdown(self):
        if self._closed:
            return
        self.image_encoder.model.shutdown()
        self._closed = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
