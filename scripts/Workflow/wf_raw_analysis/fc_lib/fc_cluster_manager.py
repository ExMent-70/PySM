# fc_lib/fc_cluster_manager.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re  # Импорт re для Regex

import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from scipy.spatial.distance import cdist
from tqdm import tqdm
import os
import psutil

from .fc_config import ConfigManager
from .fc_messages import get_message
from .fc_json_data_manager import JsonDataManager
from .fc_file_manager import FileManager

# Импортируем общую функцию загрузки эмбеддингов
from .fc_utils import load_embeddings_and_indices

logger = logging.getLogger(__name__)


class ClusterManager:
    """Класс для кластеризации, сопоставления лиц и обновления JSON данных."""

    def __init__(self, config: ConfigManager, json_manager: JsonDataManager):
        """
        Инициализирует менеджер кластеризации.
        """
        self.config = config
        self.json_manager = json_manager
        self.output_path = Path(self.config.get("paths", "output_path"))
        # Папка, где хранятся файлы эмбеддингов и индексов
        self.embeddings_dir = self.output_path / "_Embeddings"
        # Папка с исходными файлами (для file_mgr)
        self.folder_path = Path(self.config.get("paths", "folder_path"))
        # FileManager теперь используется только для сохранения matches
        self.file_mgr = FileManager(config, json_manager)
        self._load_children_list()
        # Regex для поиска 4 цифр подряд в имени файла
        self.filename_number_pattern = re.compile(r"(\d{4})")

    def _load_children_list(self):
        """Загружает список имен детей из файла, указанного в конфигурации."""
        children_file_path = self.config.get("paths", "children_file", default=None)
        self.children_list: List[str] = []
        if children_file_path:
            children_file = Path(children_file_path)  # Путь уже должен быть абсолютным
            if children_file.is_file():
                try:
                    with children_file.open("r", encoding="utf-8") as f:
                        self.children_list = [
                            line.strip() for line in f if line.strip()
                        ]
                    logger.info(
                        f"Загружен список детей из {children_file.resolve()}: {len(self.children_list)} имен."
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка чтения файла со списком детей {children_file.resolve()}: {e}"
                    )
            else:
                logger.warning(
                    f"Файл со списком детей не найден или не является файлом: {children_file.resolve()}"
                )
        else:
            logger.warning(
                "Путь к файлу со списком детей ('paths.children_file') не указан в конфигурации."
            )

    # Метод _get_data_from_json удален, т.к. эмбеддинги загружаются из файлов

    def _cluster_embeddings(
        self, embeddings_array: np.ndarray, cluster_type: str
    ) -> np.ndarray:
        """
        Выполняет кластеризацию для заданного набора эмбеддингов.
        (Логика без изменений)
        """
        if embeddings_array.size == 0:
            logger.warning(f"Нет эмбеддингов для кластеризации типа '{cluster_type}'.")
            return np.array([])

        logger.info(
            f"Кластеризация {embeddings_array.shape[0]} эмбеддингов типа '{cluster_type}'..."
        )
        config_clustering = self.config.get("clustering", cluster_type)
        algorithm = config_clustering.get("algorithm", "HDBSCAN")
        metric = config_clustering.get("metric", "cosine")
        labels = np.array([-1] * len(embeddings_array))

        with tqdm(
            total=1, desc=f"{cluster_type.capitalize()} Clustering ({algorithm})"
        ) as pbar:
            try:
                if algorithm == "DBSCAN":
                    eps = config_clustering.get("eps", 0.5)
                    min_samples = config_clustering.get("min_samples", 5)
                    logger.info(
                        f"Используется DBSCAN с eps={eps}, min_samples={min_samples}, metric={metric}"
                    )
                    db = DBSCAN(
                        eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1
                    )
                    labels = db.fit_predict(embeddings_array)
                elif algorithm == "HDBSCAN":
                    min_cluster_size = config_clustering.get("min_samples", 5)
                    min_samples_hdbscan = config_clustering.get(
                        "min_samples_param", None
                    )
                    cluster_selection_epsilon = config_clustering.get(
                        "eps", 0.0
                    )  # eps используется как cluster_selection_epsilon в HDBSCAN
                    allow_single_cluster = config_clustering.get(
                        "allow_single_cluster", False
                    )

                    logger.info(
                        f"Используется HDBSCAN с min_cluster_size={min_cluster_size}, "
                        f"min_samples={min_samples_hdbscan if min_samples_hdbscan is not None else 'auto'}, "
                        f"metric={metric}, cluster_selection_epsilon={cluster_selection_epsilon}, "
                        f"allow_single_cluster={allow_single_cluster}"
                    )

                    # Проверка и адаптация параметров для HDBSCAN
                    if len(embeddings_array) <= min_cluster_size:
                        logger.warning(
                            f"Количество {cluster_type} эмбеддингов ({len(embeddings_array)}) <= min_cluster_size ({min_cluster_size}). Все точки могут быть помечены как шум (-1)."
                        )
                    if min_samples_hdbscan is not None and min_samples_hdbscan >= len(
                        embeddings_array
                    ):
                        logger.warning(
                            f"Параметр min_samples ({min_samples_hdbscan}) для HDBSCAN >= кол-ва точек ({len(embeddings_array)}). Уменьшен до {max(1, len(embeddings_array) - 1)}."
                        )
                        min_samples_hdbscan = max(1, len(embeddings_array) - 1)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples_hdbscan,
                        metric=metric,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        allow_single_cluster=allow_single_cluster,
                        core_dist_n_jobs=-1,
                    )
                    labels = clusterer.fit_predict(embeddings_array)
                else:
                    logger.error(
                        get_message(
                            "ERROR_UNSUPPORTED_CLUSTERING_ALGORITHM",
                            algorithm=algorithm,
                        )
                    )
                    return np.array([])

                pbar.update(1)
            except Exception as e:
                logger.error(
                    f"Ошибка во время кластеризации {cluster_type} ({algorithm}): {e}",
                    exc_info=True,
                )
                return np.array([])  # Возвращаем пустой массив при ошибке

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for label in labels if label == -1)
        logger.info(
            get_message("INFO_CLUSTERS_FOUND", n_clusters=n_clusters, n_noise=n_noise)
        )
        return labels

    def _determine_cluster_order(
        self, clusters_dict: Dict[int, List[str]]
    ) -> List[Tuple[int, int]]:
        """Определяет порядок кластеров на основе 4-значных номеров в именах файлов."""
        cluster_order = []
        logger.info(
            "Определение порядка кластеров по 4-значным номерам в именах файлов..."
        )
        for label, filenames in clusters_dict.items():
            if not filenames:
                continue
            min_number = float("inf")
            found_number = False
            for fn in filenames:
                stem = Path(fn).stem
                # Ищем все 4-значные числа
                matches = self.filename_number_pattern.findall(stem)
                if matches:
                    try:
                        # Берем первое найденное число
                        num = int(matches[0])
                        min_number = min(min_number, num)
                        found_number = True
                        break  # Достаточно одного номера
                    except ValueError:
                        logger.warning(
                            f"Ошибка преобразования '{matches[0]}' в число для файла {fn}"
                        )
                        continue

            if found_number:
                cluster_order.append((min_number, label))
                logger.debug(
                    f"Кластер {label}: найден минимальный номер файла {min_number}."
                )
            else:
                # Логируем только если не нашли номер ни в одном файле кластера
                logger.warning(
                    f"Не удалось извлечь 4-значные номера файлов для определения порядка кластера {label}: {filenames[:5]}... Кластер будет в конце."
                )
                cluster_order.append((float("inf"), label))  # Помещаем в конец

        cluster_order.sort(key=lambda x: x[0])  # Сортируем по номеру файла
        ordered_labels = [label for _, label in cluster_order]
        logger.info(
            f"Определен порядок кластеров для присвоения имен: {ordered_labels}"
        )
        return cluster_order

    def _calculate_centroids(
        self, embeddings_array: np.ndarray, labels_array: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Вычисляет центроиды для каждого кластера.
        (Логика без изменений)
        """
        centroids: Dict[int, np.ndarray] = {}
        if embeddings_array.size == 0 or labels_array.size == 0:
            logger.warning(
                "Пустые эмбеддинги или метки переданы в _calculate_centroids."
            )
            return centroids
        if embeddings_array.shape[0] != labels_array.shape[0]:
            logger.error(
                f"Несоответствие размеров в _calculate_centroids: эмбеддинги {embeddings_array.shape[0]}, метки {labels_array.shape[0]}."
            )
            return centroids  # Не можем посчитать

        unique_labels = np.unique(labels_array)
        # Игнорируем шум (-1) при вычислении центроидов
        valid_labels = [int(lbl) for lbl in unique_labels if lbl != -1]

        logger.info(f"Вычисление центроидов для {len(valid_labels)} кластеров...")
        for label in valid_labels:
            cluster_mask = labels_array == label
            cluster_embeddings = embeddings_array[cluster_mask]
            if cluster_embeddings.size > 0:
                calculated_centroid = np.mean(cluster_embeddings, axis=0).astype(
                    np.float32
                )
                centroids[label] = calculated_centroid
                logger.debug(
                    f"Центроид для кластера {label} вычислен (из {len(cluster_embeddings)} точек)."
                )
            else:
                # Эта ситуация не должна возникать, если label взят из unique_labels
                logger.warning(
                    f"Для кластера {label} не найдено эмбеддингов при вычислении центроидов (это странно)."
                )

        logger.info(f"Вычислено центроидов: {len(centroids)}")
        return centroids

    def _cluster_portraits_and_update_json(
        self, portrait_embeddings_array: np.ndarray, portrait_index: Dict[str, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], Dict[int, str]]:
        """
        Кластеризует портреты, обновляет JsonDataManager, вычисляет центроиды.
        """
        portrait_centroids = {}
        portrait_clusters_by_label: Dict[int, List[str]] = {}  # {label: [filename,...]}
        cluster_to_child: Dict[int, str] = {}  # {label: child_name}

        if (
            portrait_embeddings_array is None
            or portrait_embeddings_array.size == 0
            or not portrait_index
        ):
            logger.info("Нет портретных эмбеддингов или индекса для кластеризации.")
            return portrait_centroids, portrait_clusters_by_label, cluster_to_child

        # 1. Кластеризация
        portrait_labels_array = self._cluster_embeddings(
            portrait_embeddings_array, "portrait"
        )
        if portrait_labels_array.size != portrait_embeddings_array.shape[0]:
            logger.error(
                "Ошибка кластеризации портретов. Используются метки шума (-1)."
            )
            portrait_labels_array = np.array([-1] * portrait_embeddings_array.shape[0])

        # 2. Группировка файлов по меткам
        index_to_filename = {v: k for k, v in portrait_index.items()}
        for emb_idx, label in enumerate(portrait_labels_array):
            label_int = int(label)
            filename = index_to_filename.get(emb_idx)
            if filename:
                if label_int not in portrait_clusters_by_label:
                    portrait_clusters_by_label[label_int] = []
                portrait_clusters_by_label[label_int].append(filename)
            # else: logger.warning(...) # Логировалось при загрузке индекса

        # 3. Определение порядка и присвоение имен
        valid_clusters_temp = {
            lbl: files for lbl, files in portrait_clusters_by_label.items() if lbl != -1
        }
        if not valid_clusters_temp:
            logger.info(
                get_message("INFO_NO_PORTRAIT_CLUSTERS")
                + " Все портреты помечены как шум."
            )
        else:
            cluster_order = self._determine_cluster_order(valid_clusters_temp)
            unknown_counter = 1
            for i, (min_number, label) in enumerate(cluster_order):
                assigned_name = None
                if i < len(self.children_list):
                    assigned_name = self.children_list[i]
                    logger.info(
                        f"Кластеру {label} (min_num={min_number}) присвоено имя: '{assigned_name}'"
                    )
                else:
                    assigned_name = f"Unknown_{unknown_counter}"
                    unknown_counter += 1
                    logger.info(
                        f"Не хватило имен в списке, кластеру {label} (min_num={min_number}) присвоено имя: {assigned_name}"
                    )
                cluster_to_child[label] = (
                    assigned_name  # Сохраняем сопоставление метки и имени
                )

        # 4. Обновление JsonDataManager
        logger.info(
            "Обновление JSON данных для портретов метками кластеров и именами..."
        )
        update_count = 0
        fail_count = 0
        for emb_idx, label in enumerate(portrait_labels_array):
            filename = index_to_filename.get(emb_idx)
            if not filename:
                continue

            label_int = int(label)
            child_name = "Noise"
            cluster_label_to_save = None
            if label_int != -1:
                cluster_label_to_save = label_int
                child_name = cluster_to_child.get(
                    label_int, f"Unknown_{label_int}"
                )  # Берем присвоенное имя

            success = self.json_manager.update_face(
                filename,
                0,  # face_idx = 0 для портретов
                {"cluster_label": cluster_label_to_save, "child_name": child_name},
            )
            if success:
                update_count += 1
            else:
                fail_count += 1
        logger.info(
            get_message(
                "INFO_JSON_PORTRAIT_UPDATE",
                update_count=update_count,
                fail_count=fail_count,
            )
        )

        # 5. Вычисление центроидов
        portrait_centroids = self._calculate_centroids(
            portrait_embeddings_array, portrait_labels_array
        )

        return portrait_centroids, portrait_clusters_by_label, cluster_to_child

    def _match_group_to_portraits(
        self,
        portrait_centroids: Dict[int, np.ndarray],
        group_embeddings_array: np.ndarray,
        match_threshold: float,  # Принимаем порог
    ) -> Tuple[Dict[int, List[int]], np.ndarray]:
        """
        Сопоставляет групповые эмбеддинги с портретными центроидами.
        Возвращает словарь {portrait_label: [group_emb_indices]} и матрицу расстояний.
        """
        matches_indices: Dict[int, List[int]] = {}
        all_distances = np.array([])

        if (
            not portrait_centroids
            or group_embeddings_array is None
            or group_embeddings_array.size == 0
        ):
            logger.info("Нет центроидов или групповых эмбеддингов для сопоставления.")
            return matches_indices, all_distances

        metric = self.config.get("clustering", "portrait", {}).get("metric", "cosine")
        logger.info(f"Сопоставление с порогом {match_threshold:.4f} (metric={metric}).")

        logger.info(
            f"Вычисление матрицы расстояний ({len(portrait_centroids)} портр. x {group_embeddings_array.shape[0]} групп.)..."
        )
        try:
            centroid_values = list(portrait_centroids.values())
            centroid_labels = list(portrait_centroids.keys())
            if not centroid_values:
                logger.warning("Нет валидных центроидов для cdist.")
                return matches_indices, all_distances
            # Вычисляем матрицу: строки - центроиды, столбцы - групповые эмбеддинги
            all_distances = cdist(
                centroid_values, group_embeddings_array, metric=metric
            )
        except Exception as e:
            logger.error(
                f"Ошибка при вычислении cdist (metric={metric}): {e}", exc_info=True
            )
            return matches_indices, all_distances

        if all_distances.size > 0:
            logger.info(f"Поиск совпадений ниже порога {match_threshold:.4f}...")
            # Итерируем по строкам матрицы (каждая строка = расстояния от одного центроида)
            for i, portrait_label in enumerate(centroid_labels):
                if i < all_distances.shape[0]:
                    dist_row = all_distances[i]
                    # Находим индексы столбцов (group_emb_idx), где расстояние < порога
                    matched_group_indices = np.where(dist_row < match_threshold)[
                        0
                    ].tolist()
                    if matched_group_indices:
                        matches_indices[portrait_label] = matched_group_indices
                        logger.debug(
                            f"Кластер {portrait_label}: найдено {len(matched_group_indices)} групповых лиц."
                        )
                else:
                    logger.error(
                        f"Ошибка индексации матрицы расстояний для метки {portrait_label} (индекс {i}, форма {all_distances.shape})."
                    )
        else:
            logger.warning("Матрица расстояний пуста, сопоставление не проведено.")

        logger.info(get_message("INFO_MATCHES_FOUND", count=len(matches_indices)))
        return matches_indices, all_distances

    def _update_group_json_with_matches(
        self,
        group_index: Dict[Tuple[str, int], int],
        distances: np.ndarray,
        portrait_labels_ordered: List[int],
        cluster_to_child: Dict[int, str],
        match_threshold: float,
    ) -> None:
        """
        Обновляет записи для групповых лиц в JsonDataManager информацией о совпадениях.
        """
        if distances.size == 0 or not group_index:
            logger.info("Нет данных для обновления группового JSON совпадениями.")
            return
        if distances.shape[0] != len(portrait_labels_ordered):
            logger.error(
                f"Несоответствие distances.shape[0] ({distances.shape[0]}) и len(portrait_labels_ordered) ({len(portrait_labels_ordered)})."
            )
            return

        logger.debug("Обновление группового JSON информацией о сопоставлениях...")
        update_count = 0
        fail_count = 0
        num_group_faces = distances.shape[1]
        group_emb_idx_to_key: Dict[int, Tuple[str, int]] = {
            v: k for k, v in group_index.items()
        }

        if num_group_faces != len(group_emb_idx_to_key):
            logger.error(
                f"Несоответствие кол-ва эмбеддингов ({num_group_faces}) и индекса ({len(group_emb_idx_to_key)})."
            )
            # Продолжаем, но можем пропустить некоторые обновления

        for group_emb_idx in tqdm(range(num_group_faces), desc="Обновление group JSON"):
            group_key = group_emb_idx_to_key.get(group_emb_idx)
            if not group_key:
                # logger.warning(f"Не найден ключ для индекса группового эмбеддинга {group_emb_idx}.")
                continue
            group_filename, face_idx_in_file = group_key

            # Получаем столбец расстояний до этого группового лица
            if group_emb_idx >= distances.shape[1]:
                logger.error(
                    f"Индекс group_emb_idx={group_emb_idx} вне диапазона столбцов distances ({distances.shape[1]})."
                )
                continue
            distances_to_group_face = distances[:, group_emb_idx]

            if distances_to_group_face.size == 0:
                continue

            best_match_row_index = np.argmin(distances_to_group_face)
            min_distance = float(distances_to_group_face[best_match_row_index])

            if best_match_row_index >= len(portrait_labels_ordered):
                logger.error(
                    f"Ошибка индексации portrait_labels_ordered: index={best_match_row_index}, len={len(portrait_labels_ordered)}"
                )
                continue

            matched_portrait_label = portrait_labels_ordered[best_match_row_index]

            update_payload = {}
            if min_distance < match_threshold:
                matched_child_name = cluster_to_child.get(
                    matched_portrait_label, f"Unknown_{matched_portrait_label}"
                )
                update_payload = {
                    "matched_portrait_cluster_label": int(matched_portrait_label),
                    "matched_child_name": matched_child_name,
                    "match_distance": min_distance,
                }
                logger.debug(
                    f"{group_filename}[{face_idx_in_file}]: Match Cls {matched_portrait_label} ('{matched_child_name}'), dist={min_distance:.4f}"
                )
            else:
                update_payload = {
                    "matched_portrait_cluster_label": None,
                    "matched_child_name": "No Match",
                    "match_distance": min_distance,
                }
                # logger.debug(f"{group_filename}[{face_idx_in_file}]: No Match (min_dist={min_distance:.4f})")

            if not self.json_manager.update_face(
                group_filename, face_idx_in_file, update_payload
            ):
                fail_count += 1
            else:
                update_count += 1
        logger.info(
            f"Обновлено {update_count} записей в групповом JSON. Ошибок: {fail_count}."
        )

    def run_clustering_and_matching(self) -> bool:
        """
        Выполняет полный цикл кластеризации, сопоставления и обновления JSON.
        """
        print("")
        print("<b>Запуск кластеризации и сопоставления</b>")

        success = True
        try:
            memory = psutil.virtual_memory()
            logger.debug(
                get_message(
                    "INFO_MEMORY_USAGE",
                    percent=memory.percent,
                    available=memory.available / (1024 * 1024),
                    total=memory.total / (1024 * 1024),
                )
            )
        except Exception as mem_e:
            logger.warning(f"Не удалось получить инфо о памяти: {mem_e}")

        logger.info("Загрузка JSON данных (метаданные)...")
        if not self.json_manager.load_data():
            logger.error("Критическая ошибка: Не удалось загрузить JSON.")
            return False

        logger.debug("Загрузка эмбеддингов и индексов...")
        portrait_embeddings_array, portrait_index = load_embeddings_and_indices(
            self.embeddings_dir, "portrait", "ClusterManager"
        )
        group_embeddings_array, group_index = load_embeddings_and_indices(
            self.embeddings_dir, "group", "ClusterManager"
        )

        # Инициализация переменных
        portrait_centroids, portrait_clusters_by_label, cluster_to_child = {}, {}, {}
        matches_indices, all_distances = {}, np.array([])
        calculated_match_threshold = self.config.get(
            "matching", "match_threshold", default=0.5
        )  # Default

        # Кластеризация портретов
        if portrait_embeddings_array is not None and portrait_index:
            print("")
            print("<b>Кластеризация портретов</b>")
            portrait_centroids, portrait_clusters_by_label, cluster_to_child = (
                self._cluster_portraits_and_update_json(
                    portrait_embeddings_array, portrait_index
                )
            )
            logger.debug("Сохранение JSON после обновления портретных данных...")
            if not self.json_manager.save_data():
                logger.error("Ошибка сохранения JSON.")
                success = False
        else:
            logger.info("Пропуск кластеризации портретов (нет данных).")

        # Вычисление порога сопоставления
        if (
            self.config.get("matching", "use_auto_threshold", default=False)
            and portrait_centroids
            and group_embeddings_array is not None
        ):
            logger.info("--- Вычисление порога сопоставления ---")
            try:
                percentile = self.config.get("matching", "percentile", default=10)
                metric = self.config.get("clustering", "portrait", {}).get(
                    "metric", "cosine"
                )
                valid_centroid_values = list(portrait_centroids.values())
                if valid_centroid_values and group_embeddings_array.size > 0:
                    temp_distances = cdist(
                        valid_centroid_values, group_embeddings_array, metric=metric
                    )
                    if temp_distances.size > 0:
                        calculated_match_threshold = np.percentile(
                            temp_distances.flatten(), percentile
                        )
                        logger.info(
                            f"Авто-порог ({metric}, {percentile}%): {calculated_match_threshold:.4f}"
                        )
                    else:
                        logger.warning("Матрица расстояний пуста для авто-порога.")
                else:
                    logger.warning("Нет данных для расчета авто-порога.")
            except Exception as e:
                logger.error(
                    f"Ошибка расчета авто-порога: {e}. Исп. default {calculated_match_threshold:.4f}."
                )
        else:
            logger.debug(
                f"Исп. порог сопоставления из конфига: {calculated_match_threshold:.4f}"
            )

        # Сопоставление
        if portrait_centroids and group_embeddings_array is not None:
            print("")
            print("<b>Сопоставление портретов с групповыми фотографиями</b>")

            matches_indices, all_distances = self._match_group_to_portraits(
                portrait_centroids, group_embeddings_array, calculated_match_threshold
            )
        else:
            logger.info("Пропуск сопоставления (нет данных).")

        # Обновление группового JSON
        if all_distances.size > 0 and group_index and portrait_centroids:
            portrait_labels_ordered = list(portrait_centroids.keys())
            self._update_group_json_with_matches(
                group_index,
                all_distances,
                portrait_labels_ordered,
                cluster_to_child,
                calculated_match_threshold,
            )
            logger.debug("Сохранение JSON после обновления групповых данных...")
            if not self.json_manager.save_data():
                logger.error("Ошибка сохранения JSON.")
                success = False
        else:
            logger.info("Пропуск обновления группового JSON (нет данных).")

        # Сохранение файлов сопоставлений
        logger.debug("--- Сохранение файлов сопоставлений ---")
        try:
            matches_filenames = {}  # {label: [group_filename,...]}
            if matches_indices and group_index:
                group_emb_idx_to_key: Dict[int, Tuple[str, int]] = {
                    v: k for k, v in group_index.items()
                }
                for label, indices in matches_indices.items():
                    filenames = sorted(
                        list(
                            set(
                                group_emb_idx_to_key[idx][0]
                                for idx in indices
                                if idx in group_emb_idx_to_key
                            )
                        )
                    )
                    if filenames:
                        matches_filenames[label] = filenames

            matches_csv_path = self.output_path / "matches.csv"
            self.file_mgr.save_matches_to_csv(matches_filenames, matches_csv_path)

            can_save_detailed_matches = bool(
                group_embeddings_array is not None
                and portrait_centroids
                and group_index
                and portrait_clusters_by_label
            )
            if can_save_detailed_matches:
                # group_file_lookup нужен для методов сохранения FileManager
                group_file_lookup: Dict[str, List[Tuple[int, int]]] = {}
                for (fname, fidx), emb_idx in group_index.items():
                    if fname not in group_file_lookup:
                        group_file_lookup[fname] = []
                    group_file_lookup[fname].append((emb_idx, fidx))

                group_to_portrait_path = (
                    self.output_path / "matches_group_to_portrait.json"
                )
                self.file_mgr.save_group_to_portrait_matches(
                    matches=matches_filenames,
                    portrait_centroids=portrait_centroids,
                    group_embeddings=group_embeddings_array,
                    group_file_lookup=group_file_lookup,
                    portrait_clusters=portrait_clusters_by_label,
                    match_threshold=calculated_match_threshold,
                    output_file=group_to_portrait_path,
                )
                portrait_to_group_path = (
                    self.output_path / "matches_portrait_to_group.json"
                )
                self.file_mgr.save_portrait_to_group_matches(
                    matches=matches_filenames,
                    portrait_centroids=portrait_centroids,
                    group_embeddings=group_embeddings_array,
                    group_file_lookup=group_file_lookup,
                    portrait_clusters=portrait_clusters_by_label,
                    match_threshold=calculated_match_threshold,
                    output_file=portrait_to_group_path,
                )
                logger.info(
                    "Json файлы сопоставления (Group to Portrait) и (Portrait to Group) сохранены"
                )
                
            else:
                logger.warning(
                    "Недостаточно данных для сохранения детальных JSON сопоставлений."
                )
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении файлов сопоставлений: {e}", exc_info=True
            )
            success = False

        logger.info(
            f"- Кластеризация и сопоставление завершены {'с ошибками' if not success else 'успешно'}"
        )
        return success
