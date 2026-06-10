from typing import Literal

import numpy as np


CacheMode = Literal["use", "refresh", "off"]
VALID_CACHE_MODES = {"use", "refresh", "off"}


def validate_cache_mode(cache_mode: str):
    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError(f"Unknown cache_mode: {cache_mode}")


def validate_embeddings_shape(
    embeddings: np.ndarray,
    expected_count: int,
    source: str,
    kind: str,
):
    if embeddings.ndim != 2:
        raise ValueError(f"{source} {kind} embeddings must be a 2D matrix")
    if embeddings.shape[0] != expected_count:
        raise ValueError(
            f"{source} {kind} embeddings count mismatch: "
            f"expected {expected_count}, got {embeddings.shape[0]}"
        )
