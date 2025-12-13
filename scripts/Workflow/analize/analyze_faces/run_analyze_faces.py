# analize/analyze_faces/run_analyze_faces.py
print("<b>ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ БИБЛИОТЕК...</b><br>")
import warnings

# Игнорируем специфичное FutureWarning от numpy, которое вызывается внутри insightface
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`rcond` parameter will change to the default of machine precision",
    module="insightface.utils.transform"
)
# --- Блок 1: Импорты и настройка путей ---
# ==============================================================================
import argparse
import logging
import os
import sys
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json

# Добавляем корневую папку 'analize' в sys.path
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Импорт общих утилит
    from _common.json_data_manager import JsonDataManager
    # Импорт компонентов локальной библиотеки
    from analyze_faces.face_lib import ConfigManager, FaceAnalyzer
    from analyze_faces.face_lib.result_writer import AnalysisResultWriter    
    
    # Импорт PySM
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    print("Убедитесь, что структура папок верна и все зависимости установлены.", file=sys.stderr)
    sys.exit(1)


# Инициализируем глобальный логгер, но пока не настраиваем его
logger = logging.getLogger(__name__)


# --- Блок 2: Вспомогательные функции ---
# ==============================================================================
def construct_analysis_paths() -> Dict[str, Optional[Path]]:
    """
    Формирует пути для анализа на основе переменных контекста PySM.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical("Ошибка: Скрипт запущен без окружения PySM, автоматическое формирование путей невозможно.")
        return {"input": None, "output": None}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_session_path, wf_session_name, wf_photo_session) не найдены.")
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
    
    # Путь к локальному файлу конфигурации
    default_config_path = Path(__file__).parent / "config.toml"
    parser.add_argument(
        "--a_af_config_file",
        type=str,
        dest="a_af_config_file",
        default=str(default_config_path),
        help="Путь к файлу конфигурации для этапа анализа."
    )
    # Динамические параметры
    parser.add_argument("--all_threads", type=int, dest="all_threads", default=0, help="Количество потоков (0=авто).")
    parser.add_argument(
        "--a_af_det_thresh",
        type=float,
        dest="a_af_det_thresh",
        default=None,
        help="Переопределить порог уверенности для детектора лиц (значение из config.toml)."
    )
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

# --- Новая вспомогательная функция ---
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

    logger.info("<b>ПОИСК ЛИЦ НА ФОТОГРАФИЯХ</b>")
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
    paths = construct_analysis_paths()
    input_dir, output_dir = paths.get("input"), paths.get("output")
    if not input_dir or not output_dir or not input_dir.is_dir():
        logger.critical(f"Проблема с путями: input={input_dir}, output={output_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"<br>Вход: <i>{input_dir.resolve()}</i>")
    logger.info(f"Выход: <i>{output_dir.resolve()}</i><br>")

    # 3. Инициализация компонентов
    face_analyzer = FaceAnalyzer(config_manager, output_dir_override=output_dir)
    result_writer = AnalysisResultWriter(output_dir) # Инициализируем наш новый класс
    
    # 4. Поиск файлов
    image_files = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    if not image_files:
        logger.warning("JPEG-файлы не найдены.")
        sys.exit(0)

    num_workers = cli_config.all_threads or (os.cpu_count() or 4)
    logger.info(f"Потоков: <b>{num_workers}</b>. Найдено <b>{len(image_files)}</b> изображений.")

    # 5. Основной цикл
    logger.info(f"\nКеширование моделей ONNX...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Запускаем задачи через новую функцию-обертку
        futures = {executor.submit(load_and_process_task, face_analyzer, path): path for path in image_files}
        progress = tqdm(futures.items(), total=len(image_files), desc="Анализ изображений")
        
        for future, path in progress:
            filename, (result_meta, result_embeddings, original_shape) = future.result()
            
            if result_meta and result_embeddings:
                # Делегируем сохранение в ResultWriter
                result_writer.add_result(filename, result_meta, result_embeddings, original_shape)

    # 6. Сохранение (через ResultWriter)
    json_manager = JsonDataManager(
        portrait_json_path=output_dir / "info_portrait_faces.json",
        group_json_path=output_dir / "info_group_faces.json"
    )
    result_writer.save_all(json_manager)
            
    # 7. Завершение
    face_analyzer.shutdown()

    logger.info(f"<br>Завершено. Портретов: <b>{len(result_writer.portrait_meta)}</b>, Групповых: <b>{len(result_writer.group_meta)}</b>.<br>")
    pysm_context.log_link(str(input_dir), "Исходные файлы")
    pysm_context.log_link(str(output_dir), "Результаты")
    print(" ", file=sys.stderr)

   

# --- Блок 4: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()