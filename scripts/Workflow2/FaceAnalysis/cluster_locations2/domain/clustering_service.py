import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
from .models import ImageEmbedding


class ClusteringService:
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples

    def run(self, data: List[ImageEmbedding]) -> np.ndarray:
        if not data:
            return np.array([])

        matrix = np.vstack([d.vector for d in data])
        return DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(matrix)