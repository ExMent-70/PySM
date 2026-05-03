import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pathlib import Path
from .models import ImageEmbedding, ResolvedImage
from ..infrastructure.image_loader import ImageLoader
from ..infrastructure.model_loader import ModelLoader


class EmbeddingService:
    def __init__(self, model: ModelLoader, loader: ImageLoader):
        self.model = model
        self.loader = loader

    def _safe_norm(self, x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1
        return x / n

    def embed(self, items: List[ResolvedImage], workers: int) -> List[ImageEmbedding]:
        def process(item):
            img = self.loader.load(item.input_path)
            if img is None:
                return None
            tensor = self.loader.preprocess(img)
            return item.original_path, tensor

        tensors = []
        paths = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(process, items):
                if r:
                    p, t = r
                    paths.append(p)
                    tensors.append(t)

        if not tensors:
            return []

        batch = np.vstack(tensors)
        dummy = np.zeros((batch.shape[0], 77), dtype=np.int64)

        out = self.model.session.run(
            [self.model.outputs["image"]],
            {
                self.model.inputs["pixel"]: batch,
                self.model.inputs["ids"]: dummy,
                self.model.inputs["mask"]: dummy,
            },
        )[0]

        out = self._safe_norm(out)

        return [
            ImageEmbedding(path=paths[i], vector=out[i].flatten())
            for i in range(len(paths))
        ]