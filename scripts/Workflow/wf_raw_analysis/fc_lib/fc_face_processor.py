# fc_lib/fc_face_processor.py

import os
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Импорты из проекта
from .fc_config import ConfigManager
from .fc_messages import get_message
from .fc_image_loader import ImageLoader
from .fc_face_detector import FaceDetector
from .fc_json_data_manager import JsonDataManager
from .fc_onnx_manager import ONNXModelManager
from .face_data_processor_interface import FaceDataProcessorInterface

# Импортируем утилиту для сохранения JSON индекса
from .fc_utils import save_json

logger = logging.getLogger(__name__)


class FaceProcessor:
    """
    Управляет процессом обнаружения и анализа лиц в изображениях
    различных форматов (RAW, JPEG, PSD). Сохраняет эмбеддинги отдельно.
    """

    # Блок 1: Изменение конструктора __init__
    def __init__(
        self,
        config: ConfigManager,
        json_manager: JsonDataManager,
        face_data_processors: List[FaceDataProcessorInterface],
        # --- НОВЫЙ АРГУМЕНТ ---
        onnx_manager: ONNXModelManager,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Инициализирует FaceProcessor.
        """
        self.config = config
        self.json_manager = json_manager
        self.face_data_processors = face_data_processors
        # --- НОВЫЙ АТРИБУТ ---
        self.onnx_manager = onnx_manager
        self.progress_callback = progress_callback
        
        self.output_dir = Path(self.config.get("paths", "output_path"))
        self.embeddings_dir = self.output_dir / "_Embeddings"
        self.folder_path = Path(self.config.get("paths", "folder_path"))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Папка для сохранения эмбеддингов: {self.embeddings_dir.resolve()}")
        except OSError as e:
            logger.error(f"Не удалось создать папку для эмбеддингов {self.embeddings_dir.resolve()}: {e}. Эмбеддинги не будут сохранены!")
            self.embeddings_dir = None
            
        self.loader = ImageLoader(config)
        self.original_shapes: Dict[str, Tuple[int, int]] = {}
        self.detector: Optional[FaceDetector] = None

    # Блок 2: Изменение метода _initialize_detector_if_needed
    def _initialize_detector_if_needed(self):
        """Инициализирует FaceDetector, если он еще не создан."""
        if self.detector is None:
            logger.debug("Инициализация FaceDetector (ленивая)...")
            print("PYSM_CONSOLE_BLOCK_START")
            # --- ИЗМЕНЕНИЕ: Передаем onnx_manager в FaceDetector ---
            self.detector = FaceDetector(
                self.config, self.face_data_processors, self.onnx_manager
            )
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            print("PYSM_CONSOLE_BLOCK_END")
            logger.info("FaceDetector успешно инициализирован (ленивая инициализация).")

    def process_single_image(
        self, file_path: Path
    ) -> Tuple[
        Path,
        Optional[List[Dict[str, Any]]],
        Optional[List[np.ndarray]],
        Optional[Tuple[int, int]],
    ]:
        """
        Обрабатывает одно изображение: загружает, определяет лица, анализирует.
        Возвращает кортеж: (путь, список_данных_лиц_без_эмбеддингов | None, список_эмбеддингов | None, размеры_после_загрузки | None).
        """
        if self.detector is None:
            # Эта проверка дублируется с collect_and_process_images, но для надежности
            try:
                self._initialize_detector_if_needed()
            except Exception as init_err:
                logger.critical(
                    f"Критическая ошибка инициализации FaceDetector в потоке для {file_path.name}: {init_err}",
                    exc_info=True,
                )
                return file_path, None, None, None  # Не можем обработать
            if self.detector is None:  # Если инициализация не удалась
                logger.error(
                    f"FaceDetector не инициализирован (ошибка при попытке) для {file_path.name}."
                )
                return file_path, None, None, None

        try:
            proc_config = self.config.get("processing", default={})
            raw_ext = set(proc_config.get("raw_extensions", []))
            psd_ext = set(proc_config.get("psd_extensions", []))
            jpg_ext = {".jpg", ".jpeg"}
            file_suffix_lower = file_path.suffix.lower()
            if file_suffix_lower in raw_ext:
                file_type = "raw"
            elif file_suffix_lower in psd_ext:
                file_type = "psd"
            elif file_suffix_lower in jpg_ext:
                file_type = "jpg"
            else:
                file_type = "unknown"

            if file_type == "unknown":
                logger.warning(f"Пропуск файла с неизвестным типом: {file_path.name}")
                return file_path, None, None, None

            img = self.loader.load_image(file_path, file_type)
            if img is None:
                # Ошибка уже залогирована в ImageLoader
                return file_path, None, None, None

            # Получаем face_data_list (без эмбеддингов) И face_embeddings_list
            face_data_list, face_embeddings_list, shape_after_loader = (
                self.detector.detect_and_analyze(file_path, img)
            )

            # Возвращаем все три компонента + путь
            return file_path, face_data_list, face_embeddings_list, shape_after_loader

        except Exception as e:
            # Логируем ошибку, возникшую именно на этом этапе (в потоке)
            logger.error(
                f"Исключение при обработке файла {file_path.name} в process_single_image: {e}",
                exc_info=True,
            )
            return file_path, None, None, None  # Возвращаем None для всех результатов

    def collect_and_process_images(self) -> bool:
        """
        Собирает и обрабатывает изображения, сохраняя данные лиц в JSON,
        а эмбеддинги в отдельные .npy файлы.
        """
        logger.info(
            get_message("INFO_PROCESSING_START", folder_path=self.folder_path.resolve())
        )

        proc_config = self.config.get("processing", default={})
        image_type_selected = proc_config.get("select_image_type", "RAW").upper()
        extensions_map = {
            "RAW": set(proc_config.get("raw_extensions", [])),
            "JPEG": {".jpg", ".jpeg"},
            "PSD": set(proc_config.get("psd_extensions", [])),
        }
        allowed_extensions = extensions_map.get(image_type_selected)
        if not allowed_extensions:
            logger.error(
                f"Не удалось определить расширения для типа '{image_type_selected}'."
            )
            return False


        try:
            # Используем self.folder_path для поиска исходных файлов
            image_files = [
                f
                for f in self.folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in allowed_extensions
            ]
        except Exception as e:
            logger.error(f"Ошибка сканирования папки {self.folder_path.resolve()}: {e}")
            return False
        if not image_files:
            logger.warning(
                get_message("WARNING_NO_IMAGES", folder_path=self.folder_path.resolve())
            )
            return True  # Считаем успехом, если нечего обрабатывать

        sorted_list = sorted(list(allowed_extensions))
        count=len(image_files)

        logger.info(f"Режим обработки {image_type_selected} {sorted_list}. Найдено  {count} изображений для обработки")

        # Инициализация детектора ПЕРЕД циклом потоков
        try:
            self._initialize_detector_if_needed()
            if self.detector is None:
                raise RuntimeError("FaceDetector is None after initialization attempt.")
        except Exception as detector_init_err:
            logger.critical(
                f"Критическая ошибка инициализации FaceDetector: {detector_init_err}",
                exc_info=True,
            )
            return False  # Не можем продолжать без детектора

        max_workers = proc_config.get("max_workers", os.cpu_count() or 4)
        max_workers = min(max_workers, proc_config.get("max_workers_limit", 16))
        # Очищаем данные в JsonDataManager перед началом
        self.json_manager.clear_data("all")
        self.original_shapes.clear()
        logger.info(f"Используется {max_workers} потоков для обработки изображений.")
        processing_failed = False
        processed_count = 0
        group_count = 0
        portrait_count = 0
        total_files = len(image_files)

        # Структуры для хранения эмбеддингов и их индексов
        all_portrait_embeddings: List[np.ndarray] = []
        portrait_embedding_index: Dict[
            str, int
        ] = {}  # filename -> index in all_portrait_embeddings
        all_group_embeddings: List[np.ndarray] = []
        group_embedding_index: Dict[
            Tuple[str, int], int
        ] = {}  # (filename, face_idx) -> index in all_group_embeddings

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ImageWorker"
        ) as executor:
            futures = {
                executor.submit(self.process_single_image, file): file
                for file in image_files
            }
            processed_futures = 0

            for future in tqdm(
                as_completed(futures), total=total_files, desc="Обработка изображений"
            ):
                file_path = futures[future]
                face_data_list = None
                face_embeddings_list = None
                shape_after_loader = None
                processed_futures += 1

                try:
                    # Получаем все результаты из потока
                    _, face_data_list, face_embeddings_list, shape_after_loader = (
                        future.result()
                    )
                    # Логирование результата (можно настроить уровень)
                    logger.debug(
                        f"Результат для {file_path.name}: "
                        f"Data OK={face_data_list is not None}, "
                        f"Embeddings OK={face_embeddings_list is not None} (count={len(face_embeddings_list) if face_embeddings_list else 0}), "
                        f"Shape OK={shape_after_loader is not None}"
                    )
                    # Обновляем прогресс после успешного получения результата
                    if self.progress_callback:
                        try:
                            self.progress_callback(processed_futures, total_files)
                        except Exception as cb_err:
                            logger.warning(f"Ошибка progress_callback: {cb_err}")

                except Exception as e:
                    logger.error(
                        f"Ошибка в потоке обработки для '{file_path.name}': {e}",
                        exc_info=True,
                    )
                    processing_failed = True
                    # Обновляем прогресс даже при ошибке
                    if self.progress_callback:
                        try:
                            self.progress_callback(processed_futures, total_files)
                        except Exception as cb_err:
                            logger.warning(
                                f"Ошибка progress_callback (при ошибке потока): {cb_err}"
                            )
                    continue  # Переходим к следующему future

                # Проверяем полученные результаты
                if shape_after_loader is None:
                    processing_failed = True
                    logger.warning(
                        f"Не удалось получить размеры для {file_path.name} (вероятно, ошибка загрузки/обработки). Пропуск."
                    )
                    continue
                if not face_data_list or not face_embeddings_list:
                    # Логируем, даже если лиц/эмбеддингов нет (это не ошибка обработки)
                    logger.info(
                        f"Лица или эмбеддинги не найдены в файле: {file_path.name}"
                    )
                    continue
                if len(face_data_list) != len(face_embeddings_list):
                    logger.error(
                        f"Несоответствие кол-ва лиц ({len(face_data_list)}) и эмбеддингов ({len(face_embeddings_list)}) для {file_path.name}. Пропуск файла."
                    )
                    processing_failed = True
                    continue

                # Сохраняем данные в JSON (без эмбеддингов)
                self.original_shapes[file_path.name] = shape_after_loader
                # face_data_list уже не содержит эмбеддинги

                file_data_to_save = {
                    "filename": file_path.name,
                    "faces": face_data_list,  # Список словарей без эмбеддингов
                    "original_shape": shape_after_loader,
                }
                processed_count += 1
                is_portrait = len(face_data_list) == 1
                # Добавляем данные в JsonDataManager
                self.json_manager.add_file_data(
                    file_path.name, file_data_to_save, is_portrait=is_portrait
                )

                # Собираем эмбеддинги и их индексы для последующего сохранения
                if is_portrait:
                    portrait_count += 1
                    # Для портрета берем первый (и единственный) эмбеддинг
                    current_embedding_index = len(all_portrait_embeddings)
                    all_portrait_embeddings.append(face_embeddings_list[0])
                    portrait_embedding_index[file_path.name] = current_embedding_index
                    logger.debug(
                        f"Добавлен портретный эмбеддинг для {file_path.name} с индексом {current_embedding_index}"
                    )
                else:
                    group_count += 1
                    # Для группы итерируем по всем эмбеддингам
                    for face_idx, embedding in enumerate(face_embeddings_list):
                        current_embedding_index = len(all_group_embeddings)
                        all_group_embeddings.append(embedding)
                        group_embedding_index[(file_path.name, face_idx)] = (
                            current_embedding_index
                        )
                        logger.debug(
                            f"Добавлен групповой эмбеддинг для {file_path.name}[{face_idx}] с индексом {current_embedding_index}"
                        )

            # --- Конец цикла по futures ---

        # Логируем итоги обработки
        if processing_failed:
            logger.error(
                "Во время обработки изображений возникли ошибки (см. лог выше)."
            )
        logger.info(
            f"Обработано файлов с лицами: {processed_count} (Портретов: {portrait_count}, Групповых: {group_count})."
        )
        logger.debug("Данные (без эмбеддингов) добавлены в JsonDataManager (в памяти).")

        # Сохранение эмбеддингов и индексов в файлы
        save_embeddings_success = True
        if self.embeddings_dir:  # Проверяем, была ли создана папка
            logger.debug(
                f"Сохранение файлов эмбеддингов в {self.embeddings_dir.resolve()}..."
            )
            try:
                # Сохраняем портретные эмбеддинги
                if all_portrait_embeddings:
                    portrait_npy_path = self.embeddings_dir / "portrait_embeddings.npy"
                    portrait_idx_path = self.embeddings_dir / "portrait_index.json"
                    np.save(
                        portrait_npy_path,
                        np.array(all_portrait_embeddings, dtype=np.float32),
                    )
                    # Используем fc_utils.save_json
                    save_json(
                        portrait_embedding_index,
                        portrait_idx_path,
                        "INFO_EMBEDDING_INDEX_SAVED",
                    )
                    logger.debug(
                        f"Портретные эмбеддинги ({len(all_portrait_embeddings)}) и индекс сохранены."
                    )
                else:
                    logger.info("Нет портретных эмбеддингов для сохранения.")

                # Сохраняем групповые эмбеддинги
                if all_group_embeddings:
                    group_npy_path = self.embeddings_dir / "group_embeddings.npy"
                    group_idx_path = self.embeddings_dir / "group_index.json"
                    # Конвертируем ключ-кортеж в строку для JSON
                    group_embedding_index_str_keys = {
                        f"{fname}::{fidx}": idx
                        for (fname, fidx), idx in group_embedding_index.items()
                    }
                    np.save(
                        group_npy_path, np.array(all_group_embeddings, dtype=np.float32)
                    )
                    save_json(
                        group_embedding_index_str_keys,
                        group_idx_path,
                        "INFO_EMBEDDING_INDEX_SAVED",
                    )
                    logger.debug(
                        f"Групповые эмбеддинги ({len(all_group_embeddings)}) и индекс сохранены."
                    )
                else:
                    logger.info("Нет групповых эмбеддингов для сохранения.")

            except Exception as e:
                logger.error(
                    f"Ошибка при сохранении файлов эмбеддингов в {self.embeddings_dir.resolve()}: {e}",
                    exc_info=True,
                )
                save_embeddings_success = False
        else:
            logger.error(
                "Папка для эмбеддингов не была создана или недоступна. Эмбеддинги не сохранены."
            )
            save_embeddings_success = False  # Считаем это ошибкой этапа

        # Сброс детектора для освобождения памяти GPU/CPU
        if self.detector is not None:
            logger.info("Очистка детектора лиц...")
            try:
                # Дополнительно проверяем наличие сессий анализатора атрибутов, если он есть
                if hasattr(self.detector, "face_data_processors"):
                    for processor in self.detector.face_data_processors:
                        if hasattr(processor, "sessions"):
                            del processor.sessions  # Пытаемся удалить сессии ONNX
                if hasattr(self.detector, "analyzer") and self.detector.analyzer:
                    del self.detector.analyzer  # Удаляем анализатор insightface
                del self.detector  # Удаляем сам детектор
                self.detector = None
                gc.collect()  # Принудительный сбор мусора
                logger.info("Детектор лиц и связанные ресурсы очищены.")
            except Exception as del_err:
                logger.warning(f"Не удалось полностью очистить детектор: {del_err}")

        # Итоговый успех зависит от обработки И сохранения эмбеддингов
        return not processing_failed and save_embeddings_success

    def process_folder(self) -> bool:
        """Полный цикл обработки папки: сбор данных и их сохранение."""
        run_image_analysis = self.config.get(
            "task", "run_image_analysis_and_clustering", True
        )
        if not run_image_analysis:
            logger.info(
                "Этап 1 (Обработка изображений) пропущен согласно конфигурации."
            )
            return True  # Считаем успешным пропуском

        try:
            print("")
            print(
                "<b>Обработка изображений, сбор данных и сохранение эмбеддингов</b>"
            )
            # collect_and_process_images теперь включает сохранение эмбеддингов
            collect_success = self.collect_and_process_images()
            if not collect_success:
                logger.error(
                    "Этап 1 (Обработка изображений / Сохранение эмбеддингов) завершился с ошибками."
                )
                return False  # Прерываем выполнение, если что-то пошло не так
            # Сохраняем JSON после успешного завершения Шага 1
            if not self.json_manager.save_data():
                logger.error("Ошибка при сохранении JSON данных (без эмбеддингов).")
                return False  # Ошибка сохранения JSON критична
            return True  # Возвращаем True, т.к. collect_success был True

        except Exception as e:
            logger.critical(
                f"Критическая ошибка в FaceProcessor.process_folder: {e}", exc_info=True
            )
            # Пытаемся очистить детектор даже при ошибке
            if self.detector is not None:
                try:
                    del self.detector
                    self.detector = None
                    gc.collect()
                except:
                    pass
            return False
