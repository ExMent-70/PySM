# analize/match_group_faces/match_group_faces.py
"""
Идентифицирует лица на групповых фотографиях путем их сопоставления
с ранее созданными и именованными портретными кластерами.

Ключевая особенность этого скрипта заключается в его способности работать
с данными из разных фотосъемок. Он автоматически определяет путь к файлам
эмбеддингов (.npy) на основе расположения соответствующих JSON-файлов,
что позволяет использовать один набор эталонных портретов для анализа
множества различных наборов групповых фотографий.

Скрипт предназначен для работы исключительно в управляемой среде PySM.
"""

# --- Блок 1: Импорты и настройка окружения ---
# ==============================================================================
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cdist

# Настройка системного пути для доступа к библиотекам проекта
try:
    current_script_path = Path(__file__).resolve()
    # analize/match_group_faces/ -> analize/
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    # __file__ не определен, например, в интерактивной среде.
    # Предполагаем, что пути уже настроены.
    pass

# Импорт обязательных модулей проекта и PySM
from _common.json_data_manager import JsonDataManager
from _common._shared import ConfigManager, EmbeddingLoader
from pysm_lib.pysm_context import ConfigResolver
from pysm_lib.pysm_progress_reporter import tqdm


# --- Блок 2: Настройка логирования ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Блок 3: Основной класс-обработчик ---
# ==============================================================================
class GroupMatchingProcess:
    """
    Инкапсулирует всю логику сопоставления лиц на групповых фотографиях
    с эталонными портретными кластерами.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Инициализирует процесс, извлекая необходимые параметры из конфигурации.

        Args:
            config_manager: Экземпляр ConfigManager с загруженными настройками.
        """
        self.config = config_manager
        self.match_threshold = self.config.get("matching.match_threshold", 0.5)
        logger.info(
            "Порог для сопоставления (match_threshold): "
            f"<b>{self.match_threshold}</b>"
        )


    def run(
        self,
        p_embeds: np.ndarray,
        p_index: Dict[str, int],
        g_embeds: Optional[np.ndarray],
        g_index: Optional[Dict[str, int]],
        manager: JsonDataManager,
        output_dir: Path,
    ) -> bool:
        """
        Запускает полный конвейер сопоставления.

        Args:
            p_embeds: Массив эмбеддингов эталонных портретных лиц.
            p_index: Словарь {имя_файла: индекс} для портретных эмбеддингов.
            g_embeds: Массив эмбеддингов лиц с групповых фото для анализа.
            g_index: Словарь {имя_файла::индекс_лица: индекс} для групповых эмбеддингов.
            manager: Экземпляр JsonDataManager с загруженными метаданными.
            output_dir: Директория для сохранения итогового отчета.

        Returns:
            True в случае успешного завершения, иначе False.
        """
        # Шаг 1: Создание "галереи эталонов" из портретов.
        centroids, cluster_to_name = self._calculate_centroids(
            p_embeds, p_index, manager
        )
        if not centroids:
            logger.warning(
                "Не удалось вычислить центроиды (нет валидных кластеров). "
                "Отчет будет содержать только имена кластеров."
            )

        # Шаг 2: Сопоставление, только если есть групповые данные.
        if g_embeds is not None and g_index is not None and centroids:
            self._match_faces(g_embeds, g_index, centroids, cluster_to_name, manager)
        else:
            logger.info("Групповые эмбеддинги отсутствуют, этап сопоставления пропущен.")

        # Шаг 3: Формирование и сохранение итогового отчета (выполняется всегда).
        self._save_match_report(manager, output_dir)

        logger.info("Сопоставление групповых лиц успешно завершено.")
        return True

    def _calculate_centroids(
        self, p_embeds: np.ndarray, p_index: Dict, manager: JsonDataManager
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """
        Вычисляет центроиды (средние векторы) для каждого портретного кластера.

        Этот метод является сердцем создания "галереи эталонов". Он связывает
        метаданные из JSON (имя кластера, имя файла) с фактическими векторами
        лиц через индексный файл.

        Args:
            p_embeds: Массив всех портретных эмбеддингов.
            p_index: Словарь {имя_файла: индекс_в_массиве_эмбеддингов}.
            manager: Менеджер JSON с данными о портретах.

        Returns:
            Кортеж из двух словарей:
            - {id_кластера: усредненный_вектор_np.ndarray}
            - {id_кластера: имя_человека_str}
        """
        labels_by_cluster = defaultdict(list)
        cluster_to_name: Dict[int, str] = {}

        for filename, data in manager.portrait_data.items():
            face = data.get("faces", [{}])[0]
            label = face.get("cluster_label")

            if label is not None and label != -1:
                embedding_idx = p_index.get(filename)
                if embedding_idx is not None:
                    labels_by_cluster[label].append(embedding_idx)
                    if label not in cluster_to_name:
                        cluster_to_name[label] = face.get(
                            "child_name", f"Unknown_{label}"
                        )

        centroids = {
            label: p_embeds[indices].mean(axis=0)
            for label, indices in labels_by_cluster.items()
            if indices
        }
        logger.info(
            f"Вычислено <b>{len(centroids)}</b> центроидов для сопоставления."
        )
        return centroids, cluster_to_name

    def _match_faces(
        self,
        g_embeds: np.ndarray,
        g_index: Dict,
        centroids: Dict,
        cluster_to_name: Dict,
        manager: JsonDataManager,
    ):
        """
        Находит для каждого лица с группового фото наиболее похожий эталон.

        Args:
            g_embeds: Массив эмбеддингов групповых лиц.
            g_index: Индекс для групповых эмбеддингов.
            centroids: Словарь {id_кластера: эталонный_вектор}.
            cluster_to_name: Словарь {id_кластера: имя_человека}.
            manager: Менеджер JSON для обновления данных в памяти.
        """
        if not centroids:
            return

        centroid_labels = list(centroids.keys())
        centroid_matrix = np.array(list(centroids.values()))
        metric = self.config.get("clustering.portrait.dbscan.metric", "cosine")

        # Вычисляем матрицу расстояний: строки - групповые лица, столбцы - эталоны.
        distances = cdist(g_embeds, centroid_matrix, metric=metric)

        # Находим индекс и значение минимального расстояния для каждой строки.
        best_match_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        index_to_key = {v: k for k, v in g_index.items()}
        for i in tqdm(range(len(g_embeds)), desc="Сопоставление групповых лиц"):
            filename, face_idx_str = index_to_key[i].split("::")
            face_idx = int(face_idx_str)

            best_label = centroid_labels[best_match_indices[i]]
            min_dist = min_distances[i]

            if min_dist < self.match_threshold:
                update_data = {
                    "matched_portrait_cluster_label": int(best_label),
                    "matched_child_name": cluster_to_name.get(best_label, "Unknown"),
                    "match_distance": float(min_dist),
                }
            else:
                update_data = {
                    "matched_portrait_cluster_label": None,
                    "matched_child_name": "No Match",
                    "match_distance": float(min_dist),
                }
            manager.update_face(filename, face_idx, update_data, data_type="group")


    def _save_match_report(self, manager: JsonDataManager, output_dir: Path):
        """
        Формирует и сохраняет сводный JSON-отчет о результатах сопоставления.

        Отчет имеет структуру: {id_кластера: {child_name, group_photos: [...]}}.
        Включает все портретные кластеры, даже если для них нет совпадений.

        Args:
            manager: Менеджер JSON с обновленными данными.
            output_dir: Папка для сохранения файла отчета.
        """
        report = defaultdict(
            lambda: {"child_name": "N/A", "group_photos": defaultdict(list)}
        )

        for data in manager.portrait_data.values():
            face = data.get("faces", [{}])[0]
            label, name = face.get("cluster_label"), face.get("child_name")
            if label is not None and label != -1:
                report[str(label)]["child_name"] = name

        for filename, data in manager.group_data.items():
            for face in data.get("faces", []):
                label = face.get("matched_portrait_cluster_label")
                if label is not None:
                    dist = face.get("match_distance")
                    report[str(label)]["group_photos"][filename].append(dist)

        # Преобразование в финальный, более чистый формат JSON
        final_report = {}
        for label, data in report.items():
            photos_list = [
                {
                    "filename": fname,
                    "min_distance": min(d for d in dists if d is not None),
                    "num_faces": len(dists),
                }
                for fname, dists in data["group_photos"].items()
                if dists
            ]
            
            # Запись для кластера создается всегда.
            # Если совпадений нет, 'group_photos' будет пустым списком.
            final_report[label] = {
                "child_name": data["child_name"],
                "group_photos": sorted(
                    photos_list, key=lambda x: x.get("min_distance") or float("inf")
                ),
            }

        output_path = output_dir / "matches_portrait_to_group.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        logger.info(
            "- сопоставление портретных и групповых фотографий: "
            f"<i>{output_path.name}</i><br>"
        )


# --- Блок 4: Обработка аргументов ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Определяет и парсит аргументы скрипта с помощью PySM ConfigResolver.

    Returns:
        Объект Namespace с распарсенными аргументами.
    """
    parser = argparse.ArgumentParser(description="Сопоставление лиц на групповых фото.")
    arg_prefix = "a_mg_"

    parser.add_argument(
        f"--{arg_prefix}portrait_json",
        type=str,
        default="",
        help="Путь к обработанному info_portrait_faces.json (эталон)",
    )
    parser.add_argument(
        f"--{arg_prefix}group_json",
        type=str,
        default="",
        help="Путь к info_group_faces.json для анализа",
    )
    parser.add_argument(
        f"--{arg_prefix}config",
        type=str,
        default="config.toml",
        help="Путь к файлу config.toml",
    )

    return ConfigResolver(parser).resolve_all()


# --- Блок 5: Точка входа ---
# ==============================================================================
def main():
    """
    Главная функция-оркестратор.

    Выполняет шаги:
    1. Получение конфигурации.
    2. Вычисление путей и загрузка данных.
    3. Запуск процесса сопоставления.
    4. Сохранение результатов.
    """
    cli_config = get_config()
    arg_prefix = "a_mg_"
    try:
        # Шаг 1: Получение и преобразование путей
        portrait_json_path = Path(getattr(cli_config, f"{arg_prefix}portrait_json"))
        group_json_path = Path(getattr(cli_config, f"{arg_prefix}group_json"))
        # Папка для сохранения отчета теперь вычисляется на основе пути к групповому JSON
        output_dir = group_json_path.parent
        config_path = Path(getattr(cli_config, f"{arg_prefix}config"))

        # Шаг 2: Динамическое вычисление путей к эмбеддингам
        portrait_embeddings_dir = portrait_json_path.parent / "_Embeddings"
        group_embeddings_dir = group_json_path.parent / "_Embeddings"
        logger.info(
            f"Используются эмбеддинги для портретов из: {portrait_embeddings_dir}"
        )
        logger.info(f"Используются эмбеддинги для групп из: {group_embeddings_dir}")

        # Шаг 3: Загрузка всех необходимых данных
        config_manager = ConfigManager(config_path)

        portrait_loader = EmbeddingLoader(portrait_embeddings_dir)
        p_embeds, p_index = portrait_loader.load("portrait")

        group_loader = EmbeddingLoader(group_embeddings_dir)
        g_embeds, g_index = group_loader.load("group")

        if p_embeds is None or p_index is None:
            logger.error(f"Эталонные эмбеддинги не найдены в {portrait_embeddings_dir}.")
            sys.exit(1)
        
        # Убираем прерывание выполнения, если групповых эмбеддингов нет
        if g_embeds is None or g_index is None:
            logger.warning(f"Групповые эмбеддинги не найдены в {group_embeddings_dir}. Сопоставление не будет производиться.")
            # Скрипт продолжит выполнение, g_embeds и g_index будут None

        json_manager = JsonDataManager(
            portrait_json_path=portrait_json_path, group_json_path=group_json_path
        )
        if not json_manager.load_data():
            sys.exit(1)

        # Шаг 4: Запуск процесса сопоставления
        matching_process = GroupMatchingProcess(config_manager)
        success = matching_process.run(
            p_embeds, p_index, g_embeds, g_index, json_manager, output_dir
        )

        # Шаг 5: Сохранение результатов
        if success:
            # Сохраняем только обновленный group_json, оставляя portrait_json нетронутым.
            json_manager.portrait_json_path = None
            json_manager.save_data()
            print(" ", file=sys.stderr)
        else:
            logger.error("Процесс сопоставления завершился с ошибкой.")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"Произошла непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)


# --- Блок 6: Защита точки входа ---
# ==============================================================================
if __name__ == "__main__":
    main()