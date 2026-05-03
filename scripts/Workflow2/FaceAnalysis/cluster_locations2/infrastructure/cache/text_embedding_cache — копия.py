import json
import hashlib
from pathlib import Path

import numpy as np


class TextEmbeddingCache:
    """
    Production cache for text embeddings.

    Ключ зависит от:
    - prompts
    - пути к ONNX модели
    - пути к tokenizer

    Хранение:
        <data_dir>/_Cache/
            text_emb_<hash>.npy
            text_emb_<hash>.json
    """
    def __init__(
        self,
        cache_dir: Path,
        model_path: Path,
        tokenizer_path: Path,
    ):
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.emb_path = self.cache_dir / "embeddings.npy"
        self.meta_path = self.cache_dir / "meta.json"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # META
    # =========================

    def _build_meta(self, prompts):
        normalized = sorted(prompts)
        prompts_hash = hashlib.md5(
            json.dumps(prompts, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return {
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "prompts_hash": prompts_hash,
        }

    def _is_meta_valid(self, prompts):
        if not self.meta_path.exists():
            return False

        with self.meta_path.open("r", encoding="utf-8") as f:
            old_meta = json.load(f)

        return old_meta == self._build_meta(prompts)

    def _save_meta(self, prompts):
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._build_meta(prompts), f, indent=2)

    # =========================
    # MAIN
    # =========================

    def get_or_compute(self, prompts, compute_fn):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # если кеш валиден → вернуть
        if self.emb_path.exists() and self._is_meta_valid(prompts):
            return np.load(self.emb_path)

        # иначе пересчитать полностью
        embeddings = compute_fn(prompts)

        np.save(self.emb_path, embeddings)
        self._save_meta(prompts)

        return embeddings