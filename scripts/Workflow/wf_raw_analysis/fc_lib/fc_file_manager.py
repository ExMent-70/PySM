# fc_lib/fc_file_manager.py

from pathlib import Path
import logging
import csv
import shutil
from typing import Dict, List, Tuple, Set, Optional  # Добавили Optional
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import os

# --- Обновленные импорты ---
from .fc_config import ConfigManager
from .fc_messages import get_message

# Используем fc_utils.save_json
from .fc_utils import save_json
from .fc_json_data_manager import JsonDataManager

# --- Конец обновленных импортов ---

logger = logging.getLogger(__name__)


# Функция-обертка run_file_moving без изменений
def run_file_moving(config: ConfigManager, json_manager: JsonDataManager) -> None:
    """
    Запускает перемещение/копирование файлов в папки кластеров,
    если это включено в конфигурации.
    """
    if not config.get("task", "move_files_to_claster", default=False):
        logger.info(get_message("INFO_MOVE_FILES_TO_CLUSTERS_DISABLED"))
        return

    logger.info("=" * 10 + " Запуск перемещения/копирования файлов " + "=" * 10)
    file_manager = FileManager(config, json_manager)

    # Проверяем, нужно ли загружать JSON данные
    if not json_manager.portrait_data and not json_manager.group_data:
        # Пытаемся загрузить, если еще не загружено
        if not json_manager.load_data():
            logger.error(
                "Не удалось загрузить JSON данные. Перемещение/копирование файлов отменено."
            )
            return
        # Повторная проверка после загрузки
        if not json_manager.portrait_data and not json_manager.group_data:
            logger.info("Нет данных в JSON файлах для перемещения/копирования.")
            return

    # Запускаем перемещение
    file_manager.move_files()
    logger.info("=" * 10 + " Перемещение/копирование файлов завершено " + "=" * 10)


class FileManager:
    """Класс для управления файлами и сохранения результатов."""

    def __init__(self, config: ConfigManager, json_manager: JsonDataManager):
        """
        Инициализирует менеджер файлов.
        """
        self.config = config
        self.json_manager = json_manager
        self.output_path = Path(self.config.get("paths", "output_path"))
        # Определяем папки для поиска исходников
        self.original_folder_path = Path(self.config.get("paths", "folder_path"))
        self.processed_jpg_path = self.output_path / "_JPG"
        # Папка для эмбеддингов (не используется для перемещения, но полезна для контекста)
        self.embeddings_dir = self.output_path / "_Embeddings"

        # Список папок, где искать файлы для перемещения/копирования
        self.source_search_paths: List[Path] = []
        if self.original_folder_path.is_dir():
            self.source_search_paths.append(self.original_folder_path)
        else:
            # Логгер может быть еще не полностью настроен, используем print для надежности
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Исходная папка {self.original_folder_path} не найдена или не является директорией."
            )
            # Логируем также через logger, если он уже работает
            logger.warning(
                f"Исходная папка {self.original_folder_path} не найдена или не является директорией."
            )

        # Добавляем папку _JPG, только если она существует и включено сохранение JPEG
        if self.config.get("processing", "save_jpeg", False):
            if self.processed_jpg_path.is_dir():
                self.source_search_paths.append(self.processed_jpg_path)
            else:
                logger.warning(
                    f"Папка {self.processed_jpg_path.resolve()} не найдена, хотя save_jpeg=true. Поиск в ней производиться не будет."
                )
        # Логируем итоговый список папок для поиска
        logger.info(
            f"Папки для поиска файлов для перемещения/копирования: {[p.resolve() for p in self.source_search_paths]}"
        )

    def save_matches_to_csv(
        self, matches: Dict[int, List[str]], output_file: Path
    ) -> None:
        """Сохраняет сопоставления кластеров в CSV."""
        if not matches:
            logger.info("Нет данных сопоставлений для сохранения в CSV.")
            return
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Portrait Cluster", "Group Photos"])
                sorted_matches = sorted(matches.items(), key=lambda item: item[0])
                for portrait_label, group_filenames in sorted_matches:
                    writer.writerow(
                        [
                            f"Cluster {portrait_label}",
                            "; ".join(sorted(group_filenames)),
                        ]
                    )
            logger.info(
                get_message(
                    "INFO_MATCHES_SAVED",
                    count=len(matches),
                    output_file=output_file.resolve(),
                )
            )
        except Exception as e:
            logger.error(
                f"Ошибка сохранения CSV файла сопоставлений {output_file.resolve()}: {e}"
            )

    # Метод теперь принимает match_threshold и group_file_lookup
    def save_group_to_portrait_matches(
        self,
        matches: Dict[int, List[str]],
        portrait_centroids: Dict[int, np.ndarray],
        group_embeddings: np.ndarray,
        group_file_lookup: Dict[str, List[Tuple[int, int]]],  # Принимаем lookup
        portrait_clusters: Dict[int, List[str]],
        match_threshold: float,  # Принимаем порог
        output_file: Path,
    ) -> None:
        """Сохраняет связи групповых фото с портретными кластерами в JSON."""
        if (
            not matches
            or not portrait_centroids
            or group_embeddings.size == 0
            or not group_file_lookup
            or not portrait_clusters
        ):
            logger.info(
                "Недостаточно данных для сохранения matches_group_to_portrait.json."
            )
            return

        logger.debug(get_message("INFO_PREPARING_MATCHES") + " (Group to Portrait)")
        group_to_portrait = {}
        metric = self.config.get("clustering", "portrait", {}).get("metric", "cosine")

        sorted_portrait_labels = sorted(matches.keys())

        for portrait_label in sorted_portrait_labels:
            if portrait_label < 0:
                continue
            if portrait_label not in portrait_centroids:
                logger.warning(
                    f"Нет центроида для портретного кластера {portrait_label}. Пропуск."
                )
                continue

            portrait_centroid = portrait_centroids[portrait_label]
            portrait_filenames = sorted(portrait_clusters.get(portrait_label, []))
            group_files_in_match = matches[portrait_label]

            for group_file in group_files_in_match:
                if group_file not in group_file_lookup:
                    logger.warning(
                        f"Групповой файл {group_file} из matches не найден в group_file_lookup."
                    )
                    continue

                indices_in_group = group_file_lookup[group_file]
                embedding_indices = [idx for idx, _ in indices_in_group]
                if not embedding_indices:
                    continue

                # Проверяем, что индексы не выходят за пределы массива group_embeddings
                max_required_index = max(embedding_indices)
                if max_required_index >= group_embeddings.shape[0]:
                    logger.error(
                        f"Индекс эмбеддинга {max_required_index} вне допустимого диапазона ({group_embeddings.shape[0]}) для файла {group_file}. Пропуск."
                    )
                    continue

                group_file_embeddings = group_embeddings[embedding_indices]

                try:
                    distances = cdist(
                        [portrait_centroid], group_file_embeddings, metric=metric
                    )[0]
                except Exception as e:
                    logger.error(
                        f"Ошибка расчета cdist для {group_file} и кластера {portrait_label}: {e}"
                    )
                    continue

                num_faces_matched = sum(1 for d in distances if d < match_threshold)
                min_distance = float(np.min(distances)) if distances.size > 0 else None

                if group_file not in group_to_portrait:
                    group_to_portrait[group_file] = []
                group_to_portrait[group_file].append(
                    {
                        "portrait_cluster": int(portrait_label),
                        "num_faces_matched": num_faces_matched,
                        "min_distance": min_distance,
                        "portrait_files": portrait_filenames,
                    }
                )

        for group_file in group_to_portrait:
            group_to_portrait[group_file].sort(key=lambda x: x["portrait_cluster"])
        save_json(group_to_portrait, output_file, "INFO_GROUP_TO_PORTRAIT_SAVED")

    # Метод теперь принимает match_threshold и group_file_lookup
    def save_portrait_to_group_matches(
        self,
        matches: Dict[int, List[str]],
        portrait_centroids: Dict[int, np.ndarray],
        group_embeddings: np.ndarray,
        group_file_lookup: Dict[str, List[Tuple[int, int]]],  # Принимаем lookup
        portrait_clusters: Dict[int, List[str]],
        match_threshold: float,  # Принимаем порог
        output_file: Path,
    ) -> None:
        """Сохраняет связи портретных кластеров с групповыми фото в JSON."""
        if (
            not matches
            or not portrait_centroids
            or group_embeddings.size == 0
            or not group_file_lookup
            or not portrait_clusters
        ):
            logger.info(
                "Недостаточно данных для сохранения matches_portrait_to_group.json."
            )
            return

        logger.debug(get_message("INFO_PREPARING_MATCHES") + " (Portrait to Group)")
        portrait_to_group = {}
        metric = self.config.get("clustering", "portrait", {}).get("metric", "cosine")

        sorted_portrait_labels = sorted(matches.keys())

        for portrait_label in sorted_portrait_labels:
            if portrait_label < 0:
                continue
            if portrait_label not in portrait_centroids:
                continue

            portrait_centroid = portrait_centroids[portrait_label]
            portrait_filenames = sorted(portrait_clusters.get(portrait_label, []))
            group_files_in_match = matches[portrait_label]

            group_files_info = []
            total_faces_matched = 0
            all_min_distances = []

            for group_file in sorted(group_files_in_match):
                if group_file not in group_file_lookup:
                    continue

                indices_in_group = group_file_lookup[group_file]
                embedding_indices = [idx for idx, _ in indices_in_group]
                if not embedding_indices:
                    continue

                # Проверка индексов
                max_required_index = max(embedding_indices)
                if max_required_index >= group_embeddings.shape[0]:
                    logger.error(
                        f"Индекс эмбеддинга {max_required_index} вне допустимого диапазона ({group_embeddings.shape[0]}) для файла {group_file} при сохранении portrait->group. Пропуск."
                    )
                    continue

                group_file_embeddings = group_embeddings[embedding_indices]

                try:
                    distances = cdist(
                        [portrait_centroid], group_file_embeddings, metric=metric
                    )[0]
                except Exception as e:
                    logger.error(
                        f"Ошибка расчета cdist для {group_file} и кластера {portrait_label} при сохранении portrait->group: {e}"
                    )
                    continue

                num_faces_matched = sum(1 for d in distances if d < match_threshold)
                min_distance = float(np.min(distances)) if distances.size > 0 else None

                total_faces_matched += num_faces_matched
                if min_distance is not None:
                    all_min_distances.append(min_distance)

                group_files_info.append(
                    {
                        "group_file": group_file,
                        "num_faces_matched": num_faces_matched,
                        "min_distance": min_distance,
                    }
                )

            average_min_distance = (
                float(np.mean(all_min_distances)) if all_min_distances else None
            )
            portrait_to_group[str(portrait_label)] = {
                "portrait_files": portrait_filenames,
                "group_files_matched": group_files_info,
                "total_faces_matched_in_groups": total_faces_matched,
                "average_min_distance": average_min_distance,
            }

        save_json(portrait_to_group, output_file, "INFO_PORTRAIT_TO_GROUP_SAVED")

    # Логика поиска в нескольких источниках
    def move_files(self) -> None:
        """Перемещает или копирует файлы в папки на основе данных из JsonDataManager и настроек [moving]."""
        moving_config = self.config.get("moving", default={})
        move_operation = moving_config.get("move_or_copy_files", False)
        extensions_to_action: Set[str] = set(
            moving_config.get("file_extensions_to_action", [])
        )

        action_log_name = "Перемещение" if move_operation else "Копирование"
        logger.info(f"{action_log_name} файлов в папки кластеров...")

        if not self.source_search_paths:
            logger.error(
                "Нет доступных папок для поиска исходных файлов. Перемещение/копирование отменено."
            )
            return
        logger.info(
            f"Поиск файлов будет производиться в: {[p.resolve() for p in self.source_search_paths]}"
        )

        if not extensions_to_action:
            logger.warning(
                "Список расширений 'moving.file_extensions_to_action' пуст. Файлы не будут обработаны."
            )
            return
        logger.info(
            f"Будут обработаны файлы с расширениями: {sorted(list(extensions_to_action))}"
        )

        portrait_filenames = self.json_manager.get_all_filenames("portrait")
        group_filenames = self.json_manager.get_all_filenames("group")

        if not portrait_filenames and not group_filenames:
            logger.info("Нет файлов для перемещения/копирования.")
            return

        tasks = []
        for filename in portrait_filenames:
            face_data = self.json_manager.get_face(filename, 0)
            if not face_data:
                logger.warning(
                    f"Не найдены данные для портретного файла {filename}. Пропуск."
                )
                continue
            cluster_label = face_data.get("cluster_label")
            child_name = face_data.get("child_name")
            folder_name = "00-Noise"
            if cluster_label is not None:
                cluster_label_str = f"{cluster_label:02d}"
                name_part = (
                    child_name
                    if child_name and child_name not in ["Noise", "Unknown"]
                    else f"Cluster_{cluster_label_str}"
                )
                if child_name and child_name.startswith(f"{cluster_label_str}-"):
                    name_part = child_name[len(cluster_label_str) + 1 :]
                name_part = name_part.replace("Unknown_", "")  # Убираем префикс Unknown
                folder_name = (
                    f"{cluster_label_str}-{name_part}"
                    if name_part
                    else cluster_label_str
                )
            elif child_name == "Noise":
                folder_name = "00-Noise"
            elif child_name and child_name.startswith("Unknown"):
                folder_name = f"00-{child_name}"
            tasks.append((filename, folder_name))

        group_folder_name = "00-group_photos"
        for filename in group_filenames:
            tasks.append((filename, group_folder_name))

        if not tasks:
            logger.warning("Нет файлов для обработки после формирования задач.")
            return

        max_workers = self.config.get(
            "processing", "max_workers", default=os.cpu_count() or 4
        )
        max_workers = min(
            max_workers, self.config.get("processing", "max_workers_limit", default=16)
        )
        logger.info(
            f"Используется {max_workers} потоков для {action_log_name.lower()} файлов"
        )
        errors_count = 0

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="FileMoveWorker"
        ) as executor:
            futures = [
                executor.submit(
                    self._move_single_set,
                    filename,
                    folder_name,
                    move_operation,
                    extensions_to_action,
                )
                for filename, folder_name in tasks
            ]
            for future in tqdm(
                futures, total=len(tasks), desc=f"{action_log_name} файлов"
            ):
                try:
                    if not future.result():
                        errors_count += 1
                except Exception as e:
                    # Лог ошибки теперь более информативен, но не показывает имя файла
                    # Добавим имя файла в лог ошибки
                    task_args = next(
                        (
                            t
                            for t in tasks
                            if futures[future]
                            == executor.submit(
                                self._move_single_set,
                                t[0],
                                t[1],
                                move_operation,
                                extensions_to_action,
                            )
                        ),
                        None,
                    )
                    filename_for_log = task_args[0] if task_args else "Неизвестный файл"
                    logger.error(
                        f"Критическая ошибка в потоке {action_log_name.lower()} файла '{filename_for_log}': {e}",
                        exc_info=True,
                    )
                    errors_count += 1

        logger.info(f"{action_log_name} файлов завершено. Ошибок: {errors_count}.")

    # Метод ищет в нескольких папках
    def _move_single_set(
        self,
        filename: str,
        folder_name: str,
        move_operation: bool,
        extensions: Set[str],
    ) -> bool:
        """
        Перемещает или копирует файлы с тем же stem, что и filename,
        и с расширениями из списка extensions, найденные в source_search_paths,
        в папку назначения.
        """
        file_action = shutil.move if move_operation else shutil.copy2
        action_name = "Перемещение" if move_operation else "Копирование"
        action_name_log = "Перемещен" if move_operation else "Скопирован"

        destination_dir = self.output_path / folder_name
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Не удалось создать папку назначения {destination_dir.resolve()}: {e}"
            )
            return False

        file_stem = Path(filename).stem
        files_processed_count = 0
        operation_successful = True
        found_files_to_process: Dict[Path, Path] = {}

        try:
            for source_dir in self.source_search_paths:
                if not source_dir.is_dir():
                    continue  # Пропускаем несуществующие папки

                logger.debug(
                    f"Поиск файлов с основой '{file_stem}' в {source_dir.resolve()}..."
                )
                try:  # Добавим обработку ошибок доступа к папке
                    file_iterator = source_dir.iterdir()
                except OSError as list_err:
                    logger.error(
                        f"Ошибка доступа к папке {source_dir.resolve()}: {list_err}"
                    )
                    continue  # Пропускаем эту папку

                for src_file in file_iterator:
                    if (
                        src_file.is_file()
                        and src_file.stem == file_stem
                        and src_file.suffix.lower() in extensions
                    ):
                        dst_file = destination_dir / src_file.name
                        resolved_src_path = src_file.resolve()

                        is_duplicate = False
                        # Проверяем по исходному пути
                        if any(
                            existing_src.resolve() == resolved_src_path
                            for existing_src in found_files_to_process.keys()
                        ):
                            is_duplicate = True
                        # Проверяем по пути назначения (на случай файлов с одинаковым именем из разных источников)
                        elif dst_file in found_files_to_process.values():
                            # Найден файл с таким же именем назначения. Какой оставить?
                            # Оставляем тот, который нашли первым (из первой папки в source_search_paths).
                            # Это дает приоритет файлам из original_folder_path над _JPG, если имена совпадают.
                            logger.warning(
                                f"Файл {src_file.name} из {source_dir.resolve()} будет пропущен, т.к. файл с таким же именем назначения {dst_file.resolve()} уже запланирован к обработке из другого источника."
                            )
                            is_duplicate = True

                        if not is_duplicate:
                            found_files_to_process[src_file] = dst_file
                            logger.debug(
                                f"Запланирован файл для обработки: {src_file.resolve()} -> {dst_file.resolve()}"
                            )

            if not found_files_to_process:
                logger.warning(
                    f"Не найдено файлов для {action_name.lower()} с основой имени '{file_stem}' и расширениями {extensions} в {self.source_search_paths}"
                )
                return True  # Нет ошибок, просто нечего делать

            logger.debug(
                f"Начинаем {action_name.lower()} {len(found_files_to_process)} файлов для основы '{file_stem}' в {destination_dir.resolve()}"
            )
            for src_file, dst_file in found_files_to_process.items():
                try:
                    if dst_file.exists():
                        # Убрали предупреждение о перезаписи, т.к. оно может быть слишком частым
                        # logger.warning(f"Файл {dst_file.name} уже существует в {destination_dir}. Перезапись при {action_name.lower()}.")
                        pass  # Просто перезаписываем молча или можно добавить проверку на идентичность файлов
                    # --- Добавлена проверка, что исходный файл не совпадает с целевым ---
                    if src_file.resolve() == dst_file.resolve():
                        logger.debug(
                            f"Исходный и целевой пути совпадают для {src_file.name}, {action_name.lower()} не требуется."
                        )
                        files_processed_count += 1  # Считаем как обработанный
                        continue
                    # --- Конец проверки ---

                    file_action(src_file, dst_file)
                    logger.debug(
                        get_message(
                            "INFO_FILE_MOVED",
                            action=action_name_log,
                            filename=src_file.name,
                            cluster_name=folder_name,
                        )
                    )
                    files_processed_count += 1
                except Exception as e:
                    logger.error(
                        get_message(
                            "WARNING_FILE_ACTION_FAILED",
                            action=action_name.lower(),
                            filename=src_file.name,
                            destination=folder_name,
                            exc=e,
                        )
                    )
                    operation_successful = False  # Отмечаем ошибку

            if operation_successful:
                logger.debug(
                    f"Успешно {action_name_log.lower()} {files_processed_count} файлов для основы '{file_stem}'"
                )
            else:
                logger.error(
                    f"При {action_name.lower()} файлов для основы '{file_stem}' возникли ошибки."
                )

            return operation_successful

        except Exception as e:
            logger.error(
                f"Критическая ошибка при поиске или {action_name.lower()} файлов для основы '{file_stem}': {e}",
                exc_info=True,
            )
            return False
