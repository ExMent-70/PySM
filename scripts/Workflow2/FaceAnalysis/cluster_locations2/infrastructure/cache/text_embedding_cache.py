import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .common import validate_cache_mode, validate_embeddings_shape


class TextEmbeddingCache:
    """
    Cache for text embeddings.

    Prompt order is part of the cache identity because classification labels are
    prompt indexes. Reordering identical prompt strings must invalidate the cache.
    """

    def __init__(
        self,
        cache_dir: Path,
        model_path: Path | None = None,
        tokenizer_path: Path | None = None,
        model_fingerprint: Dict[str, Any] | None = None,
    ):
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_fingerprint = model_fingerprint or {
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
        }

        self.emb_path = self.cache_dir / "embeddings.npy"
        self.meta_path = self.cache_dir / "meta.json"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_meta(self, prompts: List[str]) -> Dict[str, Any]:
        ordered_prompts = list(prompts)
        prompts_hash = hashlib.md5(
            json.dumps(ordered_prompts, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        return {
            "schema_version": 2,
            "model_fingerprint": self.model_fingerprint,
            "prompts": ordered_prompts,
            "prompts_hash": prompts_hash,
        }

    def _is_meta_valid(self, prompts: List[str]) -> bool:
        if not self.meta_path.exists():
            return False

        try:
            with self.meta_path.open("r", encoding="utf-8") as f:
                old_meta = json.load(f)
        except Exception:
            return False

        return old_meta == self._build_meta(prompts)

    def _save_meta(self, prompts: List[str]):
        tmp_path = self.meta_path.with_name(f"{self.meta_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._build_meta(prompts), f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.meta_path)

    def _save_embeddings(self, embeddings: np.ndarray):
        tmp_path = self.emb_path.with_name(f"{self.emb_path.name}.tmp")
        with tmp_path.open("wb") as f:
            np.save(f, embeddings)
        tmp_path.replace(self.emb_path)

    def _load_embeddings(self, prompts: List[str]) -> np.ndarray | None:
        try:
            embeddings = np.load(self.emb_path)
            validate_embeddings_shape(embeddings, len(prompts), "Cached", "text")
            return embeddings
        except Exception:
            return None

    def get_or_compute(
        self,
        prompts: List[str],
        compute_fn,
        cache_mode: str = "use",
    ) -> np.ndarray:
        validate_cache_mode(cache_mode)

        if cache_mode == "use" and self.emb_path.exists() and self._is_meta_valid(prompts):
            embeddings = self._load_embeddings(prompts)
            if embeddings is not None:
                return embeddings

        embeddings = compute_fn(prompts)
        validate_embeddings_shape(embeddings, len(prompts), "Computed", "text")

        if cache_mode != "off":
            self._save_embeddings(embeddings)
            self._save_meta(prompts)

        return embeddings
