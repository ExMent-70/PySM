import numpy as np
from scipy.spatial.distance import cdist
from typing import List

from .models import ImageEmbedding


class ClassificationService:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def classify(
        self,
        image_embeddings: List[ImageEmbedding],
        text_embeddings: np.ndarray,
    ) -> np.ndarray:

        if not image_embeddings:
            return np.array([])

        matrix = np.vstack([e.vector for e in image_embeddings])

        sim = 1 - cdist(matrix, text_embeddings, metric="cosine")

        idx = np.argmax(sim, axis=1)
        score = np.max(sim, axis=1)

        return np.where(score >= self.threshold, idx, -1)