import numpy as np
from pathlib import Path

from .clip_model_loader import ClipModelLoader
from .clip_tokenizer import ClipTokenizerWrapper


class TextEncoder:
    def __init__(self, model_loader: ClipModelLoader, tokenizer_path: Path, input_size=(224, 224)):
        self.model = model_loader
        self.tokenizer_path = tokenizer_path
        self.input_size = tuple(input_size)
        self.tokenizer: ClipTokenizerWrapper | None = None

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = ClipTokenizerWrapper(self.tokenizer_path)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_tokenizer()

        input_ids, attention_mask = self.tokenizer.tokenize(texts)

        # ВАЖНО: приведение типов
        input_ids = input_ids.astype(np.int64)
        attention_mask = attention_mask.astype(np.int64)

        feed = {}

        for key, name in self.model.inputs.items():
            if key == "ids":
                feed[name] = input_ids
            elif key == "mask":
                feed[name] = attention_mask
            elif key == "pixel":
                # Заглушка для image branch
                width, height = self.input_size
                feed[name] = np.zeros((len(texts), 3, height, width), dtype=np.float32)

        out = self.model.session.run(
            [self.model.outputs["text"]],
            feed,
        )[0]

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms
