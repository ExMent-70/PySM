# analize/cluster_portraits/cluster_portraits.py
"""
Выполняет кластеризацию лиц на портретных фотографиях.

Скрипт группирует схожие лица в кластеры, соответствующие уникальным
людям, и присваивает этим кластерам имена из предоставленного списка.
Он автоматически определяет путь к файлам эмбеддингов на основе
расположения входного JSON-файла.

Является первым этапом в двухэтапном процессе анализа. На выходе
обновляет `info_portrait_faces.json`, обогащая его данными о кластерах.

Предназначен для работы исключительно в управляемой среде PySM.
"""

# --- Блок 1: Импорты и настройка окружения ---
# ==============================================================================
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

# Настройка системного пути для доступа к библиотекам проекта
try:
    current_script_path = Path(__file__).resolve()
    # analize/cluster_portraits/ -> analize/
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    # __file__ не определен, например, в интерактивной среде.
    pass

# Импорт обязательных модулей проекта и PySM
from _common.json_data_manager import JsonDataManager
from _common._shared import ConfigManager, EmbeddingLoader
from pysm_lib.pysm_context import ConfigResolver

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# --- Блок 2: Настройка логирования ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Блок 3: Основной класс-обработчик ---
# ==============================================================================
class PortraitClusteringProcess:
    """
    Инкапсулирует всю логику кластеризации портретов и присвоения имен.
    """

    def __init__(self, config_manager: ConfigManager, algorithm: str, children_list: List[str]):
        """
        Инициализирует процесс кластеризации.

        Args:
            config_manager: Экземпляр ConfigManager с параметрами.
            algorithm: Название алгоритма ('dbscan' или 'hdbscan').
            children_list: Список имен для присвоения кластерам.
        """
        self.config = config_manager
        self.algorithm = algorithm.lower()
        self.children_list = children_list

        if self.algorithm == 'hdbscan' and not HDBSCAN_AVAILABLE:
            logger.warning(
                "Библиотека hdbscan не найдена. "
                "Алгоритм принудительно изменен на 'dbscan'."
            )
            self.algorithm = 'dbscan'

    def run(
        self,
        embeddings: np.ndarray,
        index: Dict[str, int],
        json_manager: JsonDataManager,
    ) -> bool:
        """
        Запускает полный конвейер кластеризации и обновления метаданных.

        Args:
            embeddings: Массив эмбеддингов портретных лиц.
            index: Словарь {имя_файла: индекс} для эмбеддингов.
            json_manager: Менеджер JSON-данных для обновления.

        Returns:
            True в случае успеха, иначе False.
        """
        logger.info(
            "Запуск кластеризации портретов с алгоритмом: "
            f"{self.algorithm.upper()}"
        )

        # Шаг 1: Группировка эмбеддингов в кластеры.
        labels = self._cluster_embeddings(embeddings)
        if labels is None:
            return False

        # Шаг 2: Присвоение имен кластерам и обновление данных в памяти.
        self._assign_names_and_update_json(labels, index, json_manager)

        logger.info("Кластеризация и присвоение имен успешно завершены.")
        return True

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray | None:
        """
        Выполняет кластеризацию эмбеддингов с помощью выбранного алгоритма.

        Args:
            embeddings: Массив векторов лиц для кластеризации.

        Returns:
            Массив меток кластеров для каждого вектора или None в случае ошибки.
        """
        params = self.config.get(f"clustering.portrait.{self.algorithm}", {})
        logger.info(
            f"Параметры кластеризации для {self.algorithm.upper()}: <i>{params}</i>"
        )

        if self.algorithm == 'dbscan':
            clusterer = DBSCAN(**params)
        elif self.algorithm == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(**params)
        else:
            logger.error(f"Неподдерживаемый алгоритм: {self.algorithm}")
            return None

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        logger.info(
            f"Создано <b>{n_clusters}</b> портретных кластера. "
            f"Шумовых точек: <b>{n_noise}</b><br>"
        )
        return labels

    def _assign_names_and_update_json(
        self, labels: np.ndarray, index: Dict, manager: JsonDataManager
    ) -> None:
        """
        Присваивает имена кластерам и обновляет данные в JsonDataManager.

        Args:
            labels: Массив меток кластеров, полученный от кластеризатора.
            index: Словарь {имя_файла: индекс_эмбеддинга}.
            manager: Менеджер JSON для обновления.
        """
        index_to_filename = {v: k for k, v in index.items()}
        clusters: Dict[int, List[str]] = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(index_to_filename[i])

        # Сортировка кластеров для последовательного присвоения имен.
        valid_clusters = {lbl: files for lbl, files in clusters.items() if lbl != -1}
        sorted_labels = sorted(
            valid_clusters.keys(),
            key=lambda lbl: min(
                [
                    int(f.split('-')[-1].split('.')[0])
                    for f in valid_clusters[lbl]
                    if f.split('-')[-1].split('.')[0].isdigit()
                ]
                or [float('inf')]
            ),
        )

        cluster_to_child: Dict[int, str] = {}
        for i, label in enumerate(sorted_labels):
            name = (
                self.children_list[i]
                if i < len(self.children_list)
                else f"Unknown_{label}"
            )
            cluster_to_child[label] = name

        # Обновление данных для каждого лица в JsonDataManager.
        for i, label in enumerate(labels):
            filename = index_to_filename[i]
            label_int = int(label)
            child_name = cluster_to_child.get(label_int, "Noise")

            update_data = {
                "cluster_label": label_int if label_int != -1 else None,
                "child_name": child_name,
            }
            manager.update_face(filename, 0, update_data, data_type="portrait")


# --- Блок 4: Обработка аргументов ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Определяет и парсит аргументы скрипта с помощью PySM ConfigResolver.

    Returns:
        Объект Namespace с распарсенными аргументами.
    """
    parser = argparse.ArgumentParser(description="Кластеризация портретных фотографий.")
    arg_prefix = "a_cp_"

    # (ИЗМЕНЕНО) Аргумент --a_cp_embeddings_dir УДАЛЕН.
    parser.add_argument(
        f"--{arg_prefix}portrait_json",
        type=str,
        default="",
        help="Путь к файлу info_portrait_faces.json",
    )
    parser.add_argument(
        f"--{arg_prefix}names_file",
        type=str,
        default="",
        help="Путь к .txt файлу со списком имен",
    )
    parser.add_argument(
        f"--{arg_prefix}config",
        type=str,
        default="config.toml",
        help="Путь к файлу конфигурации config.toml",
    )
    parser.add_argument(
        f"--{arg_prefix}algorithm",
        choices=['dbscan', 'hdbscan'],
        default='dbscan',
        help="Алгоритм кластеризации",
    )

    return ConfigResolver(parser).resolve_all()


# --- Блок 5: Точка входа ---
# ==============================================================================
def main():
    """
    Главная функция-оркестратор.
    """
    cli_config = get_config()
    arg_prefix = "a_cp_"

    try:
        # Шаг 1: Получение и преобразование путей
        portrait_json_path = Path(getattr(cli_config, f"{arg_prefix}portrait_json"))
        names_file_path = Path(getattr(cli_config, f"{arg_prefix}names_file"))
        config_path = Path(getattr(cli_config, f"{arg_prefix}config"))

        # Шаг 2: (НОВАЯ ЛОГИКА) Динамическое вычисление пути к эмбеддингам
        embeddings_dir = portrait_json_path.parent / "_Embeddings"
        logger.info(f"Используются эмбеддинги из: {embeddings_dir}")

        # Шаг 3: Загрузка всех необходимых данных
        config_manager = ConfigManager(config_path)
        embed_loader = EmbeddingLoader(embeddings_dir)

        portrait_embeds, portrait_index = embed_loader.load("portrait")
        if portrait_embeds is None or portrait_index is None:
            logger.warning(
                f"Портретные эмбеддинги не найдены в {embeddings_dir}. "
                "Процесс завершен."
            )
            sys.exit(0)

        json_manager = JsonDataManager(portrait_json_path=portrait_json_path)
        if not json_manager.load_data():
            sys.exit(1)

        with names_file_path.open("r", encoding="utf-8") as f:
            children_list = [line.strip() for line in f if line.strip()]
        logger.info(
            f"Загружен список из <b>{len(children_list)}</b> имен "
            f"из файла {names_file_path.name}."
        )

        # Шаг 4: Запуск процесса кластеризации
        algorithm = getattr(cli_config, f"{arg_prefix}algorithm")
        cluster_process = PortraitClusteringProcess(
            config_manager, algorithm, children_list
        )
        success = cluster_process.run(
            portrait_embeds, portrait_index, json_manager
        )

        # Шаг 5: Сохранение результатов
        if success:
            json_manager.save_data()
            print(" ", file=sys.stderr)  # Визуальный отступ в логе PySM
        else:
            logger.error("Процесс кластеризации завершился с ошибкой.")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"Произошла непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)


# --- Блок 6: Защита точки входа ---
# ==============================================================================
if __name__ == "__main__":
    main()