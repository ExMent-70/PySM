# analize/cluster_faces/run_cluster_faces.py

# --- Блок 1: Импорты и настройка путей ---
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import toml
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from _common.json_data_manager import JsonDataManager
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    JsonDataManager = None
    ConfigResolver = None
    tqdm = lambda x, **kwargs: x


# --- Блок 2: Настройка логирования и вспомогательные функции ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def construct_clustering_paths() -> Dict[str, Optional[Path]]:
    """
    Формирует пути для кластеризации на основе переменных контекста PySM.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical("Ошибка: Скрипт запущен без окружения PySM, автоматическое формирование путей невозможно.")
        return {"data_dir": None, "children_file": None}

    # Получаем переменные из контекста
    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")
    children_file_name = pysm_context.get("wf_children_file_name")

    if not all([session_path_str, session_name, photo_session, children_file_name]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_... ) не найдены.")
        return {"data_dir": None, "children_file": None}

    base_path = Path(session_path_str) / session_name
    
    # Формируем пути
    data_dir = base_path / "Output" / f"Analysis_{photo_session}"
    children_file = base_path / f"{photo_session}_{children_file_name}"
    
    return {"data_dir": data_dir, "children_file": children_file}


# --- Блок 3: Конфигурация и Pydantic-модели ---
# ==============================================================================
class DbscanConfig(BaseModel):
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "cosine"

class HdbscanConfig(BaseModel):
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    metric: str = "cosine"
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False

class PortraitClusteringConfig(BaseModel):
    dbscan: DbscanConfig = Field(default_factory=DbscanConfig)
    hdbscan: HdbscanConfig = Field(default_factory=HdbscanConfig)

class ClusteringConfig(BaseModel):
    portrait: PortraitClusteringConfig = Field(default_factory=PortraitClusteringConfig)

class MatchingConfig(BaseModel):
    match_threshold: float = 0.5
    use_auto_threshold: bool = False
    percentile: int = 10

class AppConfig(BaseModel):
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)

class ConfigManager:
    def __init__(self, config_path: Path):
        self.config = toml.load(config_path)
        AppConfig(**self.config) # Валидация
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys: value = value[key]
            return value
        except KeyError:
            return default

# --- Блок 4: Классы-обработчики ---
# ==============================================================================
class EmbeddingLoader:
    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = embeddings_dir

    def load(self, data_type: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        npy_path = self.embeddings_dir / f"{data_type}_embeddings.npy"
        idx_path = self.embeddings_dir / f"{data_type}_index.json"
        if not npy_path.exists() or not idx_path.exists():
            logger.warning(f"Файлы для '{data_type}' не найдены в {self.embeddings_dir}")
            return None, None
        try:
            embeddings = np.load(npy_path)
            with idx_path.open("r", encoding="utf-8") as f:
                index_data = json.load(f)
            if data_type == "group":
                index_data = {
                    f"{key.split('::')[0]}::{int(key.split('::')[1])}": val
                    for key, val in index_data.items()
                }
            return embeddings, index_data
        except Exception as e:
            logger.error(f"Ошибка загрузки эмбеддингов для '{data_type}': {e}", exc_info=True)
            return None, None

class ClusterManager:
    def __init__(self, config_manager: ConfigManager, algorithm: str, children_list: List[str]):
        self.config = config_manager
        self.algorithm = algorithm.lower()
        self.children_list = children_list
        if self.algorithm == 'hdbscan' and not HDBSCAN_AVAILABLE:
            logger.warning("Библиотека hdbscan не найдена. Алгоритм принудительно изменен на 'dbscan'.")
            self.algorithm = 'dbscan'

    def run(self, data_dir: Path):
        logger.debug(f"Запуск кластеризации с алгоритмом: {self.algorithm.upper()}")
        
        embed_loader = EmbeddingLoader(data_dir / "_Embeddings")
        portrait_embeds, portrait_index = embed_loader.load("portrait")
        group_embeds, group_index = embed_loader.load("group")
        
        json_manager = JsonDataManager(
            portrait_json_path=data_dir / "info_portrait_faces.json",
            group_json_path=data_dir / "info_group_faces.json"
        )
        if not json_manager.load_data(): return

        if portrait_embeds is None or not portrait_index:
            logger.warning("Нет портретных эмбеддингов для кластеризации. Процесс завершен.")
            return

        labels = self._cluster_portraits(portrait_embeds)
        _clusters_by_label, cluster_to_child = self._assign_names_and_update_json(labels, portrait_index, json_manager)

        if group_embeds is not None and group_index:
            self._match_group_faces(portrait_embeds, labels, group_embeds, group_index, cluster_to_child, json_manager)
        else:
            logger.info("Групповые эмбеддинги не найдены, сопоставление пропущено.")
            
        json_manager.save_data()
        self._save_match_results(json_manager, data_dir)
        logger.info("Кластеризация и сопоставление успешно завершены.")

    def _cluster_portraits(self, embeddings: np.ndarray) -> np.ndarray:
        params = self.config.get(f"clustering.portrait.{self.algorithm}", {})
        logger.info(f"Параметры кластеризации для алгоритма {self.algorithm.upper()}: <i>{params}</i>")

        if self.algorithm == 'dbscan':
            clusterer = DBSCAN(**params)
        elif self.algorithm == 'hdbscan':
            if 'min_samples' in params:
                params['min_cluster_size'] = params.pop('min_samples')
            clusterer = hdbscan.HDBSCAN(**params)
        else:
            logger.error(f"Неподдерживаемый алгоритм: {self.algorithm}")
            return np.array([-1] * len(embeddings))

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        logger.info(f"Создано <b>{n_clusters}</b> портретных кластера. Шумовых точек: <b>{n_noise}</b><br>")
        return labels

    def _assign_names_and_update_json(self, labels: np.ndarray, index: Dict, manager: JsonDataManager) -> Tuple[Dict, Dict]:
        index_to_filename = {v: k for k, v in index.items()}
        clusters: Dict[int, List[str]] = {}
        for i, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters: clusters[label_int] = []
            clusters[label_int].append(index_to_filename[i])
        
        valid_clusters = {lbl: files for lbl, files in clusters.items() if lbl != -1}
        sorted_labels = sorted(
            valid_clusters.keys(), 
            key=lambda lbl: min([int(f.split('-')[-1].split('.')[0]) for f in valid_clusters[lbl] if f.split('-')[-1].split('.')[0].isdigit()] or [float('inf')])
        )
        
        cluster_to_child: Dict[int, str] = {}
        for i, label in enumerate(sorted_labels):
            name = self.children_list[i] if i < len(self.children_list) else f"Unknown_{label}"
            cluster_to_child[label] = name

        for i, label in enumerate(labels):
            filename = index_to_filename[i]
            label_int = int(label)
            child_name = cluster_to_child.get(label_int, "Noise")
            manager.update_face(filename, 0, {"cluster_label": label_int if label_int != -1 else None, "child_name": child_name})
            
        return clusters, cluster_to_child

    def _match_group_faces(self, p_embeds, p_labels, g_embeds, g_index, c_to_c, manager):
        centroids = {lbl: p_embeds[p_labels == lbl].mean(axis=0) for lbl in c_to_c.keys()}
        if not centroids:
            logger.warning("Не удалось вычислить центроиды, сопоставление отменено.")
            return

        centroid_labels = list(centroids.keys())
        centroid_matrix = np.array(list(centroids.values()))
        
        metric = self.config.get(f"clustering.portrait.{self.algorithm}.metric", "cosine")
        distances = cdist(g_embeds, centroid_matrix, metric=metric)
        
        best_match_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        match_threshold = self.config.get("matching.match_threshold", 0.5)
        
        index_to_key = {v: k for k, v in g_index.items()}
        for i in tqdm(range(len(g_embeds)), desc="Сопоставление групповых лиц"):
            filename, face_idx_str = index_to_key[i].split('::')
            face_idx = int(face_idx_str)
            
            best_label = centroid_labels[best_match_indices[i]]
            min_dist = min_distances[i]
            
            update_data = {}
            if min_dist < match_threshold:
                update_data = {
                    "matched_portrait_cluster_label": int(best_label),
                    "matched_child_name": c_to_c.get(best_label),
                    "match_distance": float(min_dist)
                }
            else:
                update_data = {
                    "matched_portrait_cluster_label": None,
                    "matched_child_name": "No Match",
                    "match_distance": float(min_dist)
                }
            manager.update_face(filename, face_idx, update_data)
            
    def _save_match_results(self, manager: JsonDataManager, data_dir: Path):
        portrait_to_group = {}
        for filename, data in manager.portrait_data.items():
            face = data.get("faces", [{}])[0]
            label = face.get("cluster_label")
            if label is None or label == -1: continue
            if str(label) not in portrait_to_group:
                portrait_to_group[str(label)] = {"child_name": face.get("child_name"), "group_photos": {}}

        for filename, data in manager.group_data.items():
            for face in data.get("faces", []):
                label = face.get("matched_portrait_cluster_label")
                if label is None: continue
                dist = face.get("match_distance")
                if str(label) in portrait_to_group:
                    if filename not in portrait_to_group[str(label)]["group_photos"]:
                        portrait_to_group[str(label)]["group_photos"][filename] = []
                    portrait_to_group[str(label)]["group_photos"][filename].append(dist)

        for label_data in portrait_to_group.values():
            photos_list = []
            for fname, dists in label_data["group_photos"].items():
                photos_list.append({"filename": fname, "min_distance": min(d for d in dists if d is not None) if any(d is not None for d in dists) else None, "num_faces": len(dists)})
            label_data["group_photos"] = sorted(photos_list, key=lambda x: x['min_distance'] or float('inf'))

        with (data_dir / "matches_portrait_to_group.json").open("w", encoding="utf-8") as f:
            json.dump(portrait_to_group, f, indent=2, ensure_ascii=False)
        logger.info("- сопоставление портретных и групповых фотографий: <i>matches_portrait_to_group.json</i><br>")

# --- Блок 5: Точка входа и `main` ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кластеризация и сопоставление лиц.")
    default_config_path = Path(__file__).parent / "config.toml"
    parser.add_argument("--a_cf_config_file", default=str(default_config_path))
    parser.add_argument("--a_cf_algorithm", choices=['dbscan', 'hdbscan'], default='dbscan')
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()

def main():
    cli_config = get_config()
    
    paths = construct_clustering_paths()
    data_dir = paths.get("data_dir")
    children_file_path = paths.get("children_file")
    
    if not data_dir or not children_file_path:
        # Сообщение об ошибке уже выведено в construct_clustering_paths
        sys.exit(1)
        
    config_path = Path(cli_config.a_cf_config_file)
    
    if not data_dir.is_dir():
        logger.critical(f"Папка с данными не найдена: {data_dir}"); sys.exit(1)
    if not config_path.is_file():
        logger.critical(f"Файл конфигурации не найден: {config_path}"); sys.exit(1)

    children_list = []
    if children_file_path.is_file():
        with children_file_path.open("r", encoding="utf-8") as f:
            children_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Загружен список имен из {children_file_path.name}: <b>{len(children_list)}</b> имен.")
    else:
        logger.warning(f"Файл со списком имен не найден: {children_file_path}. Имена будут сгенерированы.")
        
    config_manager = ConfigManager(config_path)
    cluster_manager = ClusterManager(config_manager, cli_config.a_cf_algorithm, children_list)
    cluster_manager.run(data_dir)
    
    if IS_MANAGED_RUN and pysm_context:
        pysm_context.log_link(url_or_path=str(data_dir), text="Открыть папку с обработанными данными (файлы JSON)<br>")
        print(" ", file=sys.stderr)

if __name__ == "__main__":
    main()