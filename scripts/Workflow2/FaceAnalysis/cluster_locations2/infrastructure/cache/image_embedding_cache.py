import json
import shutil
from pathlib import Path

import numpy as np


class ImageEmbeddingCache:
    """
    Production cache для image embeddings.

    Гарантии:
    - консистентность с sync_state.json
    - защита от изменения файлов
    - защита от изменения параметров pipeline
    - контроль порядка файлов
    """
    def __init__(
        self,
        cache_dir: Path,
        sync_state_path: Path,
        data_dir: Path,
        model_path: Path,
        input_size=(224, 224),
        use_originals=True,
        mask_suffix=None,
    ):
        self.cache_dir = cache_dir
        self.sync_state_path = sync_state_path
        self.data_dir = data_dir

        self.model_path = model_path
        self.input_size = input_size
        self.use_originals = use_originals
        self.mask_suffix = mask_suffix

        self.emb_path = self.cache_dir / "embeddings.npy"
        self.idx_path = self.cache_dir / "index.json"
        self.meta_path = self.cache_dir / "meta.json"

    # =========================
    # META
    # =========================
    def _get_state_mtime(self):
        if not self.sync_state_path.exists():
            return 0.0
        return self.sync_state_path.stat().st_mtime    
    

    def _build_meta(self):
        return {
            "model_path": str(self.model_path),
            "input_size": list(self.input_size),
            "use_originals": self.use_originals,
            "mask_suffix": self.mask_suffix,
            "state_mtime": self._get_state_mtime(),  # ← ВАЖНО
        }

    def _is_meta_valid(self):
        if not self.meta_path.exists():
            return False

        with self.meta_path.open("r", encoding="utf-8") as f:
            old_meta = json.load(f)

        return old_meta == self._build_meta()

    def _save_meta(self):
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._build_meta(), f, indent=2)

    def _clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # IO
    # =========================

    def _save_all(self, file_names, embeddings):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.emb_path, embeddings)

        index = {name: i for i, name in enumerate(file_names)}
        with self.idx_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def _load_cache(self):
        embeddings = np.load(self.emb_path)

        with self.idx_path.open("r", encoding="utf-8") as f:
            index = json.load(f)

        cache_map = {name: embeddings[i] for name, i in index.items()}
        return embeddings, cache_map

    # =========================
    # SYNC STATE (read-only)
    # =========================

    def _load_sync_state(self):
        if not self.sync_state_path.exists():
            return {}

        with self.sync_state_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # =========================
    # PATH
    # =========================

    def _resolve_path(self, name):
        # ⚠️ предполагается что name — полный путь или относительный от data_dir
        return self.data_dir / "JPG" / name

    # =========================
    # DIFF
    # =========================

    def _detect_changes(self, file_names, sync_state):
        changed = []
        unchanged = []

        for name in file_names:
            path = self._resolve_path(name)

            if name not in sync_state or not path.exists():
                changed.append(name)
                continue

            stat = path.stat()
            old = sync_state[name]

            if stat.st_mtime != old["mtime"] or stat.st_size != old["size"]:
                changed.append(name)
            else:
                unchanged.append(name)

        return changed, unchanged

    # =========================
    # MERGE
    # =========================

    def _merge(self, file_names, cached_map, new_map):
        result = []

        for name in file_names:
            if name in new_map:
                result.append(new_map[name])
            else:
                result.append(cached_map[name])

        return np.vstack(result)

    # =========================
    # MAIN
    # =========================

    def get_or_compute(self, file_names, compute_fn):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. meta check
        if not self._is_meta_valid():
            self._clear_cache()

        has_cache = self.emb_path.exists() and self.idx_path.exists()

        # 2. если кеша нет → полный расчёт
        if not has_cache:
            embeddings = compute_fn(file_names)
            self._save_all(file_names, embeddings)
            self._save_meta()
            return embeddings

        # 3. загрузка кеша
        embeddings, cached_map = self._load_cache()

        # 4. sync_state
        sync_state = self._load_sync_state()

        # 5. diff
        changed, _ = self._detect_changes(file_names, sync_state)

        # 6. если ничего не изменилось → ВАЖНО: модель не трогаем
        if not changed:
            return embeddings

        # 7. считаем только изменённые
        new_matrix = compute_fn(changed)
        new_map = {name: new_matrix[i] for i, name in enumerate(changed)}

        # 8. merge
        final_embeddings = self._merge(file_names, cached_map, new_map)

        # 9. сохранить
        self._save_all(file_names, final_embeddings)
        self._save_meta()

        return final_embeddings