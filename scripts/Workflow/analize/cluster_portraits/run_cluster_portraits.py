# analize/cluster_portraits/cluster_portraits.py
"""
Выполняет кластеризацию лиц на портретных фотографиях.

Скрипт группирует схожие лица в кластеры, соответствующие уникальным
людям, и присваивает этим кластерам имена из предоставленного списка.
"""

# --- Блок 1: Импорты и настройка окружения ---
# ==============================================================================
import argparse
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

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
from _common._shared import ConfigManager, EmbeddingLoader
from pysm_lib.pysm_context import ConfigResolver

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# --- Блок 2: Настройка логирования и утилиты ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def extract_sort_key(filename: str) -> int:
    """
    Извлекает числовой индекс из имени файла для хронологической сортировки.
    Пример: 'IMG_0045.jpg' -> 45.
    Если число не найдено, возвращает бесконечность (в конец списка).
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')


# --- Блок 3: Основной класс-обработчик ---
# ==============================================================================
class PortraitClusteringProcess:
    """
    Инкапсулирует логику кластеризации портретов и присвоения имен.
    """

    def __init__(self, config_manager: ConfigManager, algorithm: str, children_list: List[str]):
        self.config = config_manager
        self.algorithm = algorithm.lower()
        self.children_list = children_list

        if self.algorithm == 'hdbscan' and not HDBSCAN_AVAILABLE:
            logger.warning("Hdbscan не найден. Используется 'dbscan'.")
            self.algorithm = 'dbscan'

    def run(
        self,
        embeddings: np.ndarray,
        index: Dict[str, int],
        json_manager: JsonDataManager,
    ) -> bool:
        """Запускает пайплайн кластеризации."""
        logger.info(f"Запуск кластеризации: {self.algorithm.upper()}")

        # 1. Кластеризация
        labels = self._cluster_embeddings(embeddings)
        if labels is None:
            return False

        # 2. Генерация маппинга имен
        cluster_names = self._generate_cluster_names(labels, index)

        # 3. Применение изменений
        self._apply_updates(labels, index, cluster_names, json_manager)

        logger.info("Кластеризация завершена.")
        return True

    def _cluster_embeddings(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Выполняет кластеризацию."""
        params = self.config.get(f"clustering.portrait.{self.algorithm}", {})
        logger.info(f"Параметры: <i>{params}</i>")

        try:
            if self.algorithm == 'dbscan':
                clusterer = DBSCAN(**params)
            elif self.algorithm == 'hdbscan':
                clusterer = hdbscan.HDBSCAN(**params)
            else:
                logger.error(f"Неизвестный алгоритм: {self.algorithm}")
                return None

            labels = clusterer.fit_predict(embeddings)
            
            # Статистика
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            logger.info(f"Кластеров: <b>{n_clusters}</b>. Шум: <b>{n_noise}</b><br>")
            
            return labels
        except Exception as e:
            logger.critical(f"Ошибка внутри алгоритма кластеризации: {e}", exc_info=True)
            return None

    def _generate_cluster_names(self, labels: np.ndarray, index: Dict[str, int]) -> Dict[int, str]:
        """
        Определяет имена для каждого кластера на основе хронологии появления.
        """
        # Инвертируем индекс для быстрого поиска имени файла по ID эмбеддинга
        index_to_filename = {v: k for k, v in index.items()}
        
        # Группируем файлы по кластерам
        clusters: Dict[int, List[str]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:
                clusters[int(label)].append(index_to_filename[i])

        # Сортируем кластеры по времени появления первого файла в них
        # Используем безопасную функцию extract_sort_key
        def get_cluster_start_time(cluster_id):
            files = clusters[cluster_id]
            if not files: return float('inf')
            # Находим минимальный номер файла в кластере
            return min(extract_sort_key(f) for f in files)

        sorted_cluster_ids = sorted(clusters.keys(), key=get_cluster_start_time)

        # Сопоставляем ID кластера с именем из списка
        cluster_name_map: Dict[int, str] = {}
        for i, cluster_id in enumerate(sorted_cluster_ids):
            if i < len(self.children_list):
                cluster_name_map[cluster_id] = self.children_list[i]
            else:
                cluster_name_map[cluster_id] = f"Unknown_Cluster_{cluster_id}"
        
        return cluster_name_map

    def _apply_updates(
        self, 
        labels: np.ndarray, 
        index: Dict[str, int], 
        cluster_names: Dict[int, str],
        manager: JsonDataManager
    ) -> None:
        """Обновляет данные в менеджере."""
        index_to_filename = {v: k for k, v in index.items()}
        
        for i, label in enumerate(labels):
            filename = index_to_filename[i]
            label_int = int(label)
            
            if label_int == -1:
                child_name = "Noise"
                final_label = None
            else:
                child_name = cluster_names.get(label_int, f"Cluster_{label_int}")
                final_label = label_int

            update_data = {
                "cluster_label": final_label,
                "child_name": child_name,
            }
            # Обновляем лицо с индексом 0 (портреты всегда одно лицо)
            manager.update_face(filename, 0, update_data, data_type="portrait")


# --- Блок 4: Конфигурация и Аргументы ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кластеризация портретов.")
    arg_prefix = "a_cp_"
    
    parser.add_argument(f"--{arg_prefix}portrait_json", type=str, required=True)
    parser.add_argument(f"--{arg_prefix}names_file", type=str, required=True)
    parser.add_argument(f"--{arg_prefix}config", type=str, default="config.toml")
    parser.add_argument(f"--{arg_prefix}algorithm", choices=['dbscan', 'hdbscan'], default='dbscan')

    return ConfigResolver(parser).resolve_all()


# --- Блок 5: Точка входа ---
# ==============================================================================
def main():
    cli_config = get_config()
    arg_prefix = "a_cp_"

    try:
        # 1. Пути
        portrait_json_path = Path(getattr(cli_config, f"{arg_prefix}portrait_json"))
        names_file_path = Path(getattr(cli_config, f"{arg_prefix}names_file"))
        config_path = Path(getattr(cli_config, f"{arg_prefix}config"))
        embeddings_dir = portrait_json_path.parent / "_Embeddings"

        # 2. Загрузка данных
        config_manager = ConfigManager(config_path)
        embed_loader = EmbeddingLoader(embeddings_dir)

        portrait_embeds, portrait_index = embed_loader.load("portrait")
        if portrait_embeds is None:
            logger.warning("Эмбеддинги не найдены. Выход.")
            sys.exit(0)

        json_manager = JsonDataManager(portrait_json_path=portrait_json_path)
        if not json_manager.load_data():
            sys.exit(1)

        # Чтение имен (UTF-8)
        try:
            with names_file_path.open("r", encoding="utf-8") as f:
                children_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Загружено имен: <b>{len(children_list)}</b>")
        except FileNotFoundError:
            logger.error(f"Файл имен не найден: {names_file_path}")
            sys.exit(1)

        # 3. Запуск процесса
        algorithm = getattr(cli_config, f"{arg_prefix}algorithm")
        process = PortraitClusteringProcess(config_manager, algorithm, children_list)
        
        success = process.run(portrait_embeds, portrait_index, json_manager)

        # 4. Сохранение (через общий менеджер)
        if success:
            json_manager.save_data()
            print(" ", file=sys.stderr)
        else:
            sys.exit(1)

    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()