# analize/analyze_faces/run_analyze_faces.py

"""   
    TODO: construct_analysis_paths() - формирует пути для анализа на основе переменных контекста PySM.
    1. Путь к папке output_dir передавать в качества параметра командной строки (обязательный параметр).
    2. Путь к папке input_dir передавать в качества параметра командной строки (не обязательный параметр). Если параметр пустой или отсутствует, то input_dir = output_dir / "JPG"
"""

print("<b>ПОИСК ЛИЦ НА ФОТОГРАФИЯХ</b>")
print("<i>Инициализация...</i><br>")
import warnings

# Игнорируем специфичное FutureWarning от numpy
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`rcond` parameter will change to the default of machine precision",
    module="insightface.utils.transform"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Please use `SimilarityTransform.from_estimate` class constructor instead",
    module="insightface.utils.face_align"
)


# --- Блок 1: Импорты и настройка путей ---
# ==============================================================================
import argparse
import logging
import os
import sys
import cv2
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Добавляем корневую папку 'analize' в sys.path
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Импорт нового менеджера хранения
    from _common.face_storage import FaceStorageManager
    
    # Импорт компонентов локальной библиотеки
    from face_analysis.face_lib import ConfigManager, FaceAnalyzer
    from face_analysis.face_lib.result_writer import AnalysisResultWriter    
    
    # Импорт PySM
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder

    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    print("Убедитесь, что структура папок верна и все зависимости установлены.", file=sys.stderr)
    sys.exit(1)

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)



# Инициализируем глобальный логгер
logger = logging.getLogger(__name__)


# --- Блок 2: Вспомогательные функции ---
# ==============================================================================
def construct_analysis_paths() -> Dict[str, Optional[Path]]:
    """
    Формирует пути для анализа на основе переменных контекста PySM.
    TODO:
    1. Путь к папке output_dir передавать в качества параметра командной строки (обязательный параметр).
    2. Путь к папке input_dir передавать в качества параметра командной строки (не обязательный параметр). Если параметр пустой или отсутствует, то input_dir = output_dir / "JPG"
    """
    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical("Ошибка: Скрипт запущен без окружения PySM, автоматическое формирование путей невозможно.")
        return {"input": None, "output": None}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста не найдены.")
        return {"input": None, "output": None}

    base_path = Path(session_path_str) / session_name
    # Выходная папка для всего анализа
    output_dir = base_path / "Output" / f"Analysis_{photo_session}"
    # Входная папка - это подпапка JPG в общей выходной папке
    input_dir = output_dir / "JPG"

    return {"input": input_dir, "output": output_dir}


# --- Блок 3: Конфигурация и выполнение ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """Определяет CLI-аргументы и разрешает их с помощью ConfigResolver."""
    parser = argparse.ArgumentParser(description="Анализ лиц на изображениях.")
    
    default_config_path = Path(__file__).parent / "config.toml"
    parser.add_argument(
        "--a_af_config_file",
        type=str,
        dest="a_af_config_file",
        default=str(default_config_path),
        help="Путь к файлу конфигурации для этапа анализа."
    )
    # Основные пути
    parser.add_argument(f"--a_af_output_dir", type=str, required=True, help="Выходная папка")
    parser.add_argument(f"--a_af_input_dir", type=str, required=False, default=None, help="Папка с файлами JPG")

    # Динамические параметры
    parser.add_argument("--all_threads", type=int, dest="all_threads", default=0, help="Количество потоков (0=авто).")
    parser.add_argument(
        "--a_af_det_thresh",
        type=float,
        dest="a_af_det_thresh",
        default=None,
        help="Переопределить порог уверенности для детектора лиц."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def load_and_process_task(analyzer: FaceAnalyzer, image_path: Path) -> Tuple[str, Any]:
    """
    Функция-воркер для треда.
    1. Читает файл (I/O)
    2. Вызывает анализ (CPU/GPU)
    """
    try:
        # Чтение с поддержкой Unicode путей
        with open(image_path, "rb") as f:
            img_buffer = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Не удалось декодировать изображение: {image_path.name}")
            return image_path.name, (None, None, None)
            
        return image_path.name, analyzer.analyze_image(img, image_path.name)
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {image_path.name}: {e}")
        return image_path.name, (None, None, None)


def main():

    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN and pysm_context else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    cli_config = get_config()

    # 1. Загрузка конфига
    try:
        config_manager = ConfigManager(Path(cli_config.a_af_config_file))
    except Exception as e:
        logger.critical(f"Ошибка конфига: {e}")
        sys.exit(1)

    if cli_config.a_af_det_thresh is not None:
        config_manager.config['model']['det_thresh'] = cli_config.a_af_det_thresh

    # 2. Пути
    output_dir = Path(cli_config.a_af_output_dir)    
    if cli_config.a_af_input_dir is not None:
        input_dir = Path(cli_config.a_af_input_dir)
    else:    
        input_dir = output_dir / "JPG"

      
    # 2. Пути
    #paths = construct_analysis_paths()
    #input_dir, output_dir = paths.get("input"), paths.get("output")
    if not input_dir or not output_dir or not input_dir.is_dir():
        logger.critical(f"Проблема с путями: input={input_dir}, output={output_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"<br>Вход: <i>{input_dir.resolve()}</i>")
    logger.debug(f"Выход: <i>{output_dir.resolve()}</i><br>")

    # 3. Инициализация компонентов
    face_analyzer = FaceAnalyzer(config_manager, output_dir_override=output_dir)
    
    # --- ИЗМЕНЕНИЕ: Инициализация системы хранения ---
    storage_manager = FaceStorageManager(output_dir)
    result_writer = AnalysisResultWriter(storage_manager, batch_size=50) # Батч 50 фото
    
    # 4. Поиск файлов
    image_files = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    if not image_files:
        logger.warning("JPEG-файлы не найдены.")
        sys.exit(0)

    num_workers = cli_config.all_threads or (os.cpu_count() or 4)
    logger.info(f"Потоков: <b>{num_workers}</b>. Найдено <b>{len(image_files)}</b> изображений.")

    # 5. Основной цикл
    face_analyzer.prepare_models()
    
    processed_count = 0
    faces_found_total = 0
    skipped_files = []

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_and_process_task, face_analyzer, path): path for path in image_files}
            progress = tqdm(futures.items(), total=len(image_files), desc="Анализ изображений")
            
            for future, path in progress:
                filename, (result_meta, result_embeddings, original_shape) = future.result()
                
                if result_meta and result_embeddings:
                    # Делегируем буферизацию и сохранение в ResultWriter
                    result_writer.add_result(filename, result_meta, result_embeddings, original_shape)
                    faces_found_total += len(result_meta)
                else:
                    skipped_files.append(filename)
                
                processed_count += 1

        # 6. Финализация сохранения
        # Сначала сбрасываем остатки из буфера
        result_writer.close()

        # Затем собираем временные файлы в итоговые
        if storage_manager.finalize():
            if skipped_files:
                skipped_path = output_dir / "skipped_images.json"
                try:
                    with open(skipped_path, "w", encoding="utf-8") as f:
                        json.dump(skipped_files, f, ensure_ascii=False, indent=4)
                    logger.info(f"{icon_save_warning} файл <i>skipped_images.json</i> сохранен (пропущено <b>{len(skipped_files)}</b> изображений)<br>")
                except Exception as e:
                    logger.error(f"{icon_save_error} не удалось сохранить список пропущенных файлов: {e}<br>")

            logger.debug(f"<br>Анализ завершен. Обработано файлов: <b>{processed_count}</b>. Найденных лиц: <b>{faces_found_total}</b>.<br>")
        else:
            logger.error("<br>{icon_save_error} Анализ завершен с ошибками при сохранении данных<br>")

    except KeyboardInterrupt:
        logger.warning("\nПрерывание пользователем. Попытка сохранить обработанные данные...")
        result_writer.close()
        storage_manager.finalize()
        raise
    finally:
        # 7. Завершение и очистка ресурсов
        face_analyzer.shutdown()


       
    # 1. Инициализация
    script_dir = Path(__file__).resolve().parent

    tv_builder = StandardTreeBuilder(icon_size=28)


    root_node_config = ResourceNode("config.toml", Path(script_dir) / "config.toml", "file", "Файл конфигурации (дополнительные настройки скрипта)")

    root_node_target = ResourceNode("Исходная<br>папка", Path(input_dir), "folder", "Исходная папка с файлами JPG")

    # 2. Подготовка данных
    root_node = ResourceNode("Рабочая<br>папка", Path(output_dir), "folder", "Папка с результатами AI-анализа фотографий")
    root_node.children.append(ResourceNode("info_faces.json", Path(output_dir) / "info_faces.json", "code", "Подробная информация о всех лицах обнаруженных на фотографиях текущей фотосессии"))
    root_node.children.append(ResourceNode("skipped_images.json", Path(output_dir) / "skipped_images.json", "code", "Список необработанных фотографий"))
    

    tv_builder.add_section("<br>Рабочие папки и файлы", [root_node_config, root_node_target, root_node])


    # 4. Вывод
    pysm_context.log_html(tv_builder.get_html())

# --- Блок 4: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()