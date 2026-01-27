# analize/cluster_portraits/cluster_portraits.py
"""
Выполняет кластеризацию лиц на портретных фотографиях.

Рефакторинг v2026.01.24:
- Удалена зависимость от config.toml.
- Параметры передаются через CLI с префиксами db_ и hdb_.
- Разделена ответственность (Clusterer, NameResolver).
"""

# --- Блок 1: Импорты и настройка окружения ---
import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.cluster import DBSCAN

# Настройка системного пути
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    pass

from _common.json_data_manager import JsonDataManager
from _common._shared import EmbeddingLoader
from pysm_lib.pysm_context import ConfigResolver

# Попытка импорта HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# --- Блок 2: Логирование ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Блок 3: Компоненты бизнес-логики ---

class NameResolver:
    """
    Отвечает за загрузку списка имен и их назначение кластерам.
    """

    @staticmethod
    def load_names_from_file(file_path: Path) -> List[str]:
        """
        Загружает список имен из файла.
        Реализует логику fallback: если файл не найден, ищет children.txt в той же папке.
        """
        target_file = file_path
        
        # 1. Проверка основного файла
        if not target_file.is_file():
            logger.warning(f"Файл имен не найден: {target_file}")
            # 2. Попытка найти children.txt рядом
            fallback = target_file.parent / "children.txt"
            if fallback.is_file():
                logger.info(f"Используется резервный файл имен: {fallback.name}")
                target_file = fallback
            else:
                logger.error("Файл с именами не найден. Кластеры будут именоваться как Unknown.")
                return []

        # 3. Чтение
        try:
            with target_file.open("r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            logger.info(f"Загружено имен: {len(names)}")
            return names
        except Exception as e:
            logger.error(f"Ошибка чтения файла имен: {e}")
            return []

    @staticmethod
    def resolve_cluster_names(
        labels: np.ndarray, 
        index_map: Dict[str, int], 
        children_names: List[str]
    ) -> Dict[int, str]:
        """
        Сопоставляет ID кластера с именем ребенка на основе хронологии.
        Кластеры сортируются по времени появления первого фото.
        """
        # Инверсия индекса: id -> filename
        id_to_file = {v: k for k, v in index_map.items()}
        
        # Группировка файлов по кластерам
        clusters: Dict[int, List[str]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                clusters[int(label)].append(id_to_file.get(idx, ""))

        # Функция для сортировки файлов (извлечение номера из IMG_001.jpg)
        def get_sort_key(filename: str) -> int:
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else float('inf')

        # Функция определения времени старта кластера
        def get_cluster_start_time(cluster_id: int) -> int:
            files = clusters[cluster_id]
            if not files:
                return float('inf')
            return min(get_sort_key(f) for f in files)

        # Сортировка ID кластеров по хронологии
        sorted_ids = sorted(clusters.keys(), key=get_cluster_start_time)

        # Присвоение имен
        name_map = {}
        for i, cluster_id in enumerate(sorted_ids):
            if i < len(children_names):
                name_map[cluster_id] = children_names[i]
            else:
                name_map[cluster_id] = f"Unknown_Cluster_{cluster_id}"
        
        return name_map


class FaceClusterer:
    """
    Фасад для алгоритмов кластеризации.
    """

    def __init__(self, config: argparse.Namespace, arg_prefix: str = "a_cp_"):
        self.config = config
        self.prefix = arg_prefix

    def run(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        Запускает выбранный алгоритм кластеризации.
        """
        algo_name = getattr(self.config, f"{self.prefix}algorithm")
        
        if algo_name == "dbscan":
            return self._run_dbscan(embeddings)
        elif algo_name == "hdbscan":
            return self._run_hdbscan(embeddings)
        else:
            logger.error(f"Неизвестный алгоритм: {algo_name}")
            return None

    def _run_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        eps = getattr(self.config, f"{self.prefix}db_eps")
        min_samples = getattr(self.config, f"{self.prefix}db_min_samples")
        # Используем общую метрику
        metric = getattr(self.config, f"{self.prefix}metric")

        logger.info(f"Запуск DBSCAN (eps={eps}, min_samples={min_samples}, metric={metric})")
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clusterer.fit_predict(embeddings)
        self._log_stats(labels)
        return labels

    def _run_hdbscan(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        if not HDBSCAN_AVAILABLE:
            logger.error("Библиотека 'hdbscan' не установлена.")
            return None

        min_cluster_size = getattr(self.config, f"{self.prefix}hdb_min_cluster_size")
        # Используем общую метрику
        metric = getattr(self.config, f"{self.prefix}metric")
        cluster_selection_epsilon = getattr(self.config, f"{self.prefix}hdb_cluster_selection_epsilon")
        min_samples = getattr(self.config, f"{self.prefix}hdb_min_samples", None)
        
        logger.info(
            f"Запуск HDBSCAN (min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, metric={metric})"
        )

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=False
            )
            labels = clusterer.fit_predict(embeddings)
            self._log_stats(labels)
            return labels
        except Exception as e:
            logger.critical(f"Ошибка выполнения HDBSCAN: {e}")
            return None

    def _log_stats(self, labels: np.ndarray):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        logger.info(f"Результат: Кластеров: {n_clusters}, Шум: {n_noise}")


# --- Блок 4: Конфигурация CLI ---

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кластеризация портретов.")
    p = "a_cp_"  # prefix
    
    # Основные пути
    parser.add_argument(f"--{p}portrait_json", type=str, required=True, help="Путь к info_portrait_faces.json")
    parser.add_argument(f"--{p}names_file", type=str, required=True, help="Файл со списком имен")
    
    # Общие настройки
    parser.add_argument(f"--{p}algorithm", choices=['dbscan', 'hdbscan'], default='dbscan', help="Алгоритм кластеризации")
    parser.add_argument(f"--{p}metric", type=str, default="cosine", help="Метрика расстояния (cosine/euclidean)")

    # Параметры DBSCAN
    parser.add_argument(f"--{p}db_eps", type=float, default=0.25, help="DBSCAN: Epsilon")
    parser.add_argument(f"--{p}db_min_samples", type=int, default=3, help="DBSCAN: Min Samples")
    # Параметр db_metric УДАЛЕН

    # Параметры HDBSCAN
    parser.add_argument(f"--{p}hdb_min_cluster_size", type=int, default=2, help="HDBSCAN: Min Cluster Size")
    parser.add_argument(f"--{p}hdb_min_samples", type=int, default=None, help="HDBSCAN: Min Samples (Optional)")
    parser.add_argument(f"--{p}hdb_cluster_selection_epsilon", type=float, default=0.0, help="HDBSCAN: Cluster Selection Epsilon")
    # Параметр hdb_metric УДАЛЕН

    return ConfigResolver(parser).resolve_all()


# --- Блок 5: Точка входа ---

def main():
    try:
        # 1. Конфигурация
        cli_config = get_config()
        arg_prefix = "a_cp_"
        
        portrait_json_path = Path(getattr(cli_config, f"{arg_prefix}portrait_json"))
        names_file_path = Path(getattr(cli_config, f"{arg_prefix}names_file"))
        
        # Вычисляем путь к эмбеддингам
        embeddings_dir = portrait_json_path.parent / "_Embeddings"

        logger.info(f"JSON портретов: {portrait_json_path}")
        logger.info(f"Файл имен: {names_file_path}")

        # 2. Загрузка данных
        # Эмбеддинги
        embed_loader = EmbeddingLoader(embeddings_dir)
        embeddings, index_map = embed_loader.load("portrait")
        
        if embeddings is None:
            logger.error("Не удалось загрузить эмбеддинги. Работа прервана.")
            sys.exit(1)

        # JSON метаданные
        json_manager = JsonDataManager(portrait_json_path=portrait_json_path)
        if not json_manager.load_data():
            logger.error("Не удалось загрузить JSON данных.")
            sys.exit(1)

        # Список имен
        children_names = NameResolver.load_names_from_file(names_file_path)

        # 3. Кластеризация
        clusterer = FaceClusterer(cli_config, arg_prefix)
        labels = clusterer.run(embeddings)
        
        if labels is None:
            logger.error("Кластеризация не удалась.")
            sys.exit(1)

        # 4. Сопоставление имен
        cluster_names = NameResolver.resolve_cluster_names(labels, index_map, children_names)
        
        # 5. Применение изменений
        # Инвертируем карту индексов для быстрого поиска
        index_to_filename = {v: k for k, v in index_map.items()}

        updates_count = 0
        for i, label in enumerate(labels):
            filename = index_to_filename.get(i)
            if not filename:
                continue
                
            label_int = int(label)
            
            if label_int == -1:
                child_name = "Noise"
                final_label = None
            else:
                child_name = cluster_names.get(label_int, f"Cluster_{label_int}")
                final_label = label_int

            # Обновляем лицо (портреты всегда index 0)
            update_data = {
                "cluster_label": final_label,
                "child_name": child_name
            }
            json_manager.update_face(filename, 0, update_data, data_type="portrait")
            updates_count += 1

        # 6. Сохранение
        json_manager.save_data()
        logger.info(f"Успешно обновлено {updates_count} записей.")

    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()