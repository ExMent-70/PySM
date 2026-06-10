import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .common import validate_cache_mode, validate_embeddings_shape


class ImageEmbeddingCache:
    """
    Session-level cache for image embeddings.

    The cache is intentionally coarse-grained: if the session manifest changes,
    all image embeddings are recomputed. This is simpler and safer for stable
    photo sessions than partial sync_state-based updates.
    """

    def __init__(
        self,
        cache_dir: Path,
        model_fingerprint: Dict[str, Any],
        input_size=(224, 224),
        use_originals=True,
        mask_suffix=None,
    ):
        self.cache_dir = cache_dir
        self.model_fingerprint = model_fingerprint
        self.input_size = tuple(input_size)
        self.use_originals = use_originals
        self.mask_suffix = mask_suffix

        self.emb_path = self.cache_dir / "embeddings.npy"
        self.idx_path = self.cache_dir / "index.json"
        self.meta_path = self.cache_dir / "manifest.json"

    def _file_signature(self, item: Dict[str, Any]) -> Dict[str, Any]:
        input_path = Path(item["input_path"])
        original_path = Path(item["original_path"])

        def stat_payload(path: Path) -> Dict[str, Any]:
            if not path.exists():
                return {"path": str(path), "exists": False}
            stat = path.stat()
            return {
                "path": str(path),
                "exists": True,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }

        return {
            "name": item["name"],
            "input": stat_payload(input_path),
            "original": stat_payload(original_path),
        }

    def _build_manifest(self, file_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        file_signatures = [self._file_signature(item) for item in file_items]
        payload = {
            "schema_version": 2,
            "model_fingerprint": self.model_fingerprint,
            "input_size": list(self.input_size),
            "use_originals": self.use_originals,
            "mask_suffix": self.mask_suffix,
            "files": file_signatures,
        }
        manifest_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payload["manifest_hash"] = manifest_hash
        return payload

    def _load_manifest(self) -> Dict[str, Any] | None:
        if not self.meta_path.exists():
            return None
        try:
            with self.meta_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _is_cache_valid(self, file_items: List[Dict[str, Any]]) -> bool:
        if not self.emb_path.exists() or not self.idx_path.exists():
            return False
        return self._load_manifest() == self._build_manifest(file_items)

    def _clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_all(
        self,
        file_names: List[str],
        file_items: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ):
        validate_embeddings_shape(embeddings, len(file_names), "Computed", "image")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.emb_path, embeddings)

        index = {name: i for i, name in enumerate(file_names)}
        with self.idx_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._build_manifest(file_items), f, indent=2, ensure_ascii=False)

    def _load_ordered(self, file_names: List[str]) -> np.ndarray:
        embeddings = np.load(self.emb_path)
        with self.idx_path.open("r", encoding="utf-8") as f:
            index = json.load(f)

        validate_embeddings_shape(embeddings, len(index), "Cached", "image")

        missing = [name for name in file_names if name not in index]
        if missing:
            raise KeyError(f"Cache index is missing files: {missing[:5]}")

        return np.vstack([embeddings[index[name]] for name in file_names])

    def get_or_compute(
        self,
        file_names: List[str],
        file_items: List[Dict[str, Any]],
        compute_fn,
        cache_mode: str = "use",
    ) -> np.ndarray:
        validate_cache_mode(cache_mode)

        if cache_mode == "use" and self._is_cache_valid(file_items):
            return self._load_ordered(file_names)

        embeddings = compute_fn(file_names)
        validate_embeddings_shape(embeddings, len(file_names), "Computed", "image")

        if cache_mode != "off":
            self._clear_cache()
            self._save_all(file_names, file_items, embeddings)

        return embeddings
