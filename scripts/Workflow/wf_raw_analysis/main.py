# main.py
import os
import logging
import sys
from pathlib import Path

# --- ИЗМЕНЕНИЕ: Уточняем импорт Handler ---
from logging import Handler
from typing import List, Optional, Any, Union, Callable, Type, ForwardRef, cast

# --- КОНЕЦ ИЗМЕНЕНИЯ ---
import traceback

# Импорты вашего проекта
from fc_lib.fc_config import ConfigManager
from fc_lib.fc_json_data_manager import JsonDataManager
from fc_lib.fc_face_processor import FaceProcessor
from fc_lib.fc_cluster_manager import ClusterManager
from fc_lib.fc_onnx_manager import ONNXModelManager
from fc_lib.fc_messages import get_message

# Импортируем модули с функциями-обертками
from fc_lib import fc_keypoints, fc_xmp_utils, fc_file_manager, fc_report
from fc_lib.processor_registry import initialize_face_data_processors
from fc_lib.face_data_processor_interface import FaceDataProcessorInterface

# Инициализируем логгер на уровне модуля
logger = logging.getLogger(__name__)


# --- ФУНКЦИЯ НАСТРОЙКИ ЛОГИРОВАНИЯ ---
def setup_logging(
    log_level_str: str = "INFO",
    log_file: Optional[Union[str, Path]] = "face_processing.log",
    # --- ИЗМЕНЕНИЕ: Используем корректный тип logging.Handler ---
    gui_log_handler: Optional[Handler] = None,
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
):
    """Настраивает корневой логгер Python."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)
    #log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format = "%(levelname)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    # Очищаем все предыдущие обработчики для "чистой" настройки
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
            except Exception as e:
                # Используем print, т.к. логгер еще не настроен
                print(f"Warning: Could not close handler {handler}: {e}")
            root_logger.removeHandler(handler)

    root_logger.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=log_datefmt)
    handlers: List[Handler] = []

    # 1. Файловый обработчик (если указан путь)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Не удалось настроить файловый логгер ({log_file}): {e}")

    # 2. Обработчик для вывода в консоль (sys.stdout или sys.stderr)
    # Этот вывод будет перехвачен PySM
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    # 3. Обработчик для GUI (если передан)
    if gui_log_handler:
        # Убедимся, что у GUI-обработчика есть форматтер
        if not gui_log_handler.formatter:
            gui_log_handler.setFormatter(formatter)
        handlers.append(gui_log_handler)

    # Добавляем все собранные обработчики в корневой логгер
    for handler in handlers:
        root_logger.addHandler(handler)
        
    # --- НОВЫЙ БЛОК: Перехват стандартных предупреждений Python ---
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    # Можно установить уровень ниже, чтобы видеть все, или выше, чтобы скрыть
    warnings_logger.setLevel(logging.ERROR) 
    # --- КОНЕЦ НОВОГО БЛОКА ---        

    # Понижаем уровень логгеров от шумных библиотек
    logging.getLogger("insightface").setLevel(logging.ERROR)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("rawpy").setLevel(logging.ERROR)
    logging.getLogger("psd_tools").setLevel(logging.ERROR)

    logging.info(
        f"Уровень логирования: {log_level_str}. Файл: {log_file if log_file else 'Нет'}."
    )
# --- КОНЕЦ ФУНКЦИИ НАСТРОЙКИ ЛОГИРОВАНИЯ ---


def run_full_processing(
    config_path_or_obj: Union[str, Path, ConfigManager],
    log_callback: Callable[[str], None],
    progress_callback: Callable[[int, int], None],
) -> bool:
    """
    Выполняет полный цикл обработки: анализ, кластеризация, опциональные шаги.
    Возвращает True при успехе, False при критической ошибке.
    """
    processing_successful = False
    run_stage1_2 = False  # Флаг для выполнения анализа/кластеризации
    config = None  # Инициализируем

    # 1. Инициализация ConfigManager и проверка флага основного задания
    try:
        if isinstance(config_path_or_obj, (str, Path)):
            config = ConfigManager(str(config_path_or_obj))
        elif isinstance(config_path_or_obj, ConfigManager):
            config = config_path_or_obj
        else:
            raise TypeError("Неверный тип для config_path_or_obj")
        # Проверяем, нужно ли выполнять основные этапы
        run_stage1_2 = config.get("task", "run_image_analysis_and_clustering", True)
        logger.debug(f"Флаг task.run_image_analysis_and_clustering: {run_stage1_2}")

    except Exception as e:
        initial_error_msg = f"Критическая ошибка инициализации ConfigManager: {e}\n{traceback.format_exc()}"
        logger.critical(initial_error_msg, exc_info=False)
        log_callback(f"CRITICAL: {initial_error_msg}")
        return False  # Ошибка инициализации конфига - не можем продолжать

    # 2. Расчет общего прогресса (основан на конфиге)
    active_optional_steps = sum(
        [
            config.get("task", task_key, False)
            for task_key in [
                "keypoint_analysis",
                "create_xmp_file",
                "move_files_to_claster",
                "generate_html",
            ]
        ]
    )
    total_progress_steps = (2 if run_stage1_2 else 0) + active_optional_steps
    current_progress_step = 0
    logger.debug(f"Общее количество шагов прогресса: {total_progress_steps}")

    # 3. Основной блок обработки
    try:
        #log_callback("INFO: Инициализация основных компонентов...")

        # --- ИЗМЕНЕНИЕ: Создаем единый экземпляр ONNXModelManager ---
        onnx_manager = ONNXModelManager(config)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Инициализация обработчиков атрибутов лица (учитывая флаг run_stage1_2)
        # Теперь эта функция будет принимать onnx_manager, чтобы передать его дальше
        active_processors: List[FaceDataProcessorInterface] = (
            initialize_face_data_processors(config, run_stage1_2, onnx_manager)
        )

        # Инициализация менеджера JSON данных
        output_dir = Path(
            config.get("paths", "output_path")
        )  # Путь уже разрешен ConfigManager'ом
        output_dir.mkdir(parents=True, exist_ok=True)
        portrait_json_path = output_dir / "info_portrait_faces.json"
        group_json_path = output_dir / "info_group_faces.json"
        json_manager = JsonDataManager(
            portrait_json_path=portrait_json_path, group_json_path=group_json_path
        )
        logger.debug("JsonDataManager создан.")

        # --- Этапы 1 и 2: Анализ изображений и Кластеризация ---
        if run_stage1_2:
            # --- Этап 1: Обработка изображений и сбор данных ---
            stage_name = "Этап 1: Анализ изображений и сбор данных"
            print(f"")
            print(f"<b>{stage_name.upper()}</b>")
            #log_callback(f"INFO: {stage_name}...")
            
            # --- ИЗМЕНЕНИЕ: Передаем onnx_manager в FaceProcessor ---
            processor = FaceProcessor(
                config, json_manager, active_processors, onnx_manager, progress_callback
            )
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            logger.debug("FaceProcessor инициализирован.")
            # process_folder выполнит collect_and_process_images и сохранит JSON/Embeddings
            success_stage1 = processor.process_folder()
            
            # --- НОВЫЙ БЛОК: Явная очистка insightface сразу после использования ---
            if processor.detector is not None:
                logger.info("Освобождение основной модели insightface...")
                del processor.detector.analyzer
                del processor.detector
                processor.detector = None
                gc.collect()
            # --- КОНЕЦ НОВОГО БЛОКА ---
            
            if not success_stage1:
                msg = f"{stage_name} завершился с ошибкой."
                logger.error(msg)
                log_callback(f"ERROR: {msg}")
                raise RuntimeError(
                    msg
                )  # Прерываем, если основной сбор данных не удался
            #logger.info(f"{'-' * 20} {stage_name.upper()} ЗАВЕРШЕН {'-' * 20}")
            #log_callback(f"INFO: {stage_name} завершен.")
            current_progress_step = 1  # Шаг 1 пройден (прогресс обновлялся внутри)
            # Обновляем прогресс до конца шага 1 (если внутри не дошло до 100%)
            progress_callback(current_progress_step, total_progress_steps)

            # --- Этап 2: Кластеризация и сопоставление ---
            stage_name = "Этап 2: Кластеризация и сопоставление"
            print(f"")
            print(f"<b>{stage_name.upper()}</b>")
            #log_callback(f"INFO: {stage_name}...")
            # ClusterManager использует ConfigManager и JsonDataManager
            # Теперь он будет загружать эмбеддинги из файлов
            cluster_manager = ClusterManager(config, json_manager)
            success_stage2 = cluster_manager.run_clustering_and_matching()
            if not success_stage2:
                # Не прерываем выполнение, но логируем ошибку
                msg = f"{stage_name} завершился с ошибками."
                logger.error(msg)
                log_callback(f"ERROR: {msg}")
                # processing_successful останется False, если success_stage1 был True
            #logger.info(f"{'-' * 20} {stage_name.upper()} ЗАВЕРШЕН {'-' * 20}")
            #print(f"")
            #log_callback(f"INFO: {stage_name} завершен.")
            #print(f"")

            current_progress_step = 2  # Шаг 2 пройден
            progress_callback(current_progress_step, total_progress_steps)
        else:
            logger.info(
                "Этапы 1 и 2 (Анализ и Кластеризация) пропущены согласно конфигурации."
            )
            log_callback(
                f"INFO: Этапы 1 и 2 пропущены (task.{'run_image_analysis_and_clustering'} = false)."
            )
            # Если нужны опциональные шаги, пытаемся загрузить существующие JSON
            if active_optional_steps > 0 and (
                not json_manager.portrait_data and not json_manager.group_data
            ):
                log_callback(
                    "INFO: Загрузка существующих JSON данных для опциональных шагов..."
                )
                if not json_manager.load_data():
                    msg = "Не удалось загрузить JSON данные, необходимые для опциональных шагов."
                    logger.critical(msg)
                    log_callback(f"CRITICAL: {msg}")
                    return False  # Не можем выполнить опциональные шаги

        # --- Этап 3: Опциональные шаги ---
        print(f"")
        print(f"<b>ЭТАП 3: ОПЦИОНАЛЬНЫЕ ШАГИ</b>")
        #log_callback("INFO: Запуск опциональных шагов...")
        # Проверяем каждый шаг и выполняем, если включен
        if config.get("task", "keypoint_analysis", False):
            #log_callback("INFO: Анализ keypoints...")
            fc_keypoints.run_keypoint_analysis(config, json_manager)
            #log_callback("INFO: Анализ keypoints завершен.")
            current_progress_step += 1
            progress_callback(current_progress_step, total_progress_steps)
        #else:
            #log_callback("Анализ keypoints отключен.")

        if config.get("task", "create_xmp_file", False):
            #log_callback("INFO: Создание XMP...")
            fc_xmp_utils.run_xmp_creation(config, json_manager)
            #log_callback("INFO: Создание XMP завершено.")
            current_progress_step += 1
            progress_callback(current_progress_step, total_progress_steps)
        #else:
            #log_callback("Создание XMP отключено.")

        if config.get("task", "move_files_to_claster", False):
            log_callback("INFO: Перемещение/копирование файлов...")
            fc_file_manager.run_file_moving(config, json_manager)
            log_callback("INFO: Перемещение/копирование завершено.")
            current_progress_step += 1
            progress_callback(current_progress_step, total_progress_steps)
        #else:
            #log_callback("Перемещение/копирование отключено.")

        if config.get("task", "generate_html", False):
            #log_callback("INFO: Генерация HTML...")
            # ReportGenerator теперь загружает эмбеддинги сам
            fc_report.run_html_report_generation(config, json_manager)
            #log_callback("INFO: Генерация HTML завершена.")
            current_progress_step += 1
            progress_callback(current_progress_step, total_progress_steps)
        #else:
            #log_callback("INFO: Генерация HTML отключена.")

        #logger.info(f"{'-' * 20} ЭТАП 3 ЗАВЕРШЕН {'-' * 20}")
        #log_callback("INFO: Опциональные шаги завершены.")
        # Устанавливаем прогресс на 100%, если все шаги пройдены
        if total_progress_steps > 0:
            progress_callback(total_progress_steps, total_progress_steps)

        # --- Завершение ---
        final_message = get_message(
            "INFO_PROCESSING_DONE",
            portrait_file=json_manager.portrait_json_path.name,
            group_file=json_manager.group_json_path.name,
        )
        logger.info(final_message)
        print("")


        #log_callback(f"INFO: {final_message}")
        #log_callback("=" * 25 + " ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА " + "=" * 25)
        processing_successful = (
            True  # Считаем успехом, если дошли до конца без исключений
        )

    except Exception as e:
        # Ловим все исключения, которые могли произойти на любом этапе
        error_msg = f"Критическая ошибка во время выполнения run_full_processing: {e}\n{traceback.format_exc()}"
        logger.critical(
            error_msg, exc_info=False
        )  # exc_info=False, т.к. уже есть в traceback
        log_callback(f"CRITICAL: {error_msg}")
        processing_successful = False  # Явно указываем на ошибку
        # Не пробрасываем ошибку дальше, т.к. ее должен был обработать ProcessingWorker
        # Но возвращаем False
    
    # --- ИЗМЕНЕНИЕ: Добавляем блок finally для гарантированной очистки ---
    finally:
        # Проверяем, был ли создан onnx_manager, перед вызовом shutdown
        if 'onnx_manager' in locals() and hasattr(onnx_manager, 'shutdown'):
            onnx_manager.shutdown()
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    return processing_successful

