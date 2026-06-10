from dataclasses import dataclass
from typing import List

import numpy as np

from .models import ImageEmbedding


@dataclass(slots=True)
class ClassificationResult:
    labels: np.ndarray
    scores: np.ndarray


class ClassificationService:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def _safe_normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def classify_with_scores(
        self,
        image_embeddings: List[ImageEmbedding],
        text_embeddings: np.ndarray,
    ) -> ClassificationResult:

        if not image_embeddings:
            return ClassificationResult(
                labels=np.array([], dtype=int),
                scores=np.array([], dtype=float),
            )

        if text_embeddings.size == 0:
            return ClassificationResult(
                labels=np.full(len(image_embeddings), -1, dtype=int),
                scores=np.zeros(len(image_embeddings), dtype=float),
            )

        image_matrix = np.vstack([e.vector for e in image_embeddings])
        image_matrix = self._safe_normalize(image_matrix)
        text_matrix = self._safe_normalize(text_embeddings)

        sim = image_matrix @ text_matrix.T

        idx = np.argmax(sim, axis=1)
        score = np.max(sim, axis=1)
        labels = np.where(score >= self.threshold, idx, -1)

        return ClassificationResult(labels=labels, scores=score)

    def classify(
        self,
        image_embeddings: List[ImageEmbedding],
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        return self.classify_with_scores(image_embeddings, text_embeddings).labels
