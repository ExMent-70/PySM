from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class ResolvedImage:
    input_path: Path
    original_path: Path


@dataclass(slots=True)
class ImageEmbedding:
    path: Path
    vector: np.ndarray