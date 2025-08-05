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
from typing import Dict, List, Any, Optional
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
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def main():
    log_level = "INFO" 
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout
    )

    """Основная функция-оркестратор."""
    logger.info("<b>ПОИСК ЛИЦ НА ФОТОГРАФИЯХ</b>")
    cli_config = get_config()

    # 1. Загрузка статичной конфигурации моделей
    config_path = Path(cli_config.a_af_config_file)
    try:
        config_manager = ConfigManager(config_path)
    except (FileNotFoundError, ValidationError) as e:
        logger.critical(f"Не удалось загрузить или провалидировать конфигурацию: {e}")
        sys.exit(1)

    # 2. Получение и валидация динамических путей из контекста
    paths = construct_analysis_paths()
    input_dir = paths.get("input")
    output_dir = paths.get("output")
    
    if not input_dir or not output_dir:
        # Сообщение об ошибке уже выведено в construct_analysis_paths
        sys.exit(1)

    if not input_dir.is_dir():
        logger.critical(f"Входная директория не найдена: {input_dir}")
        sys.exit(1)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Не удалось создать выходную папку {output_dir}: {e}")
        sys.exit(1)

    # 3. Инициализация компонентов с разделенными конфигурациями
    logger.info(f"<br>Папка с исходными файлами (JPG):")
    logger.info(f"<i>{input_dir.resolve()}</i>")
    logger.info(f"Папка с результатами анализа:")
    logger.info(f"<i>{output_dir.resolve()}</i><br>")

    face_analyzer = FaceAnalyzer(config_manager, output_dir_override=output_dir)
    
    # 4. Поиск файлов и запуск обработки
    image_files = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    if not image_files:
        logger.warning(f"В папке {input_dir} не найдено JPEG-файлов для анализа.")
        sys.exit(0)

    logger.info(f"Найдено <b>{len(image_files)} изображений</b> для анализа.")

    num_workers = cli_config.all_threads
    if not num_workers or num_workers <= 0:
        num_workers = os.cpu_count() or 4
        logger.info(f"Используется автоматическое количество потоков: <b>{num_workers}</b>")

    # 5. Основной цикл обработки
    portrait_meta: Dict[str, Any] = {}
    group_meta: Dict[str, Any] = {}
    portrait_embeddings: List[np.ndarray] = []
    group_embeddings: List[np.ndarray] = []
    portrait_index: Dict[str, int] = {}
    group_index: Dict[str, int] = {} 

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(face_analyzer.analyze_image, path): path for path in image_files}
        progress = tqdm(futures.items(), total=len(image_files), desc="Анализ изображений")
        
        for future, path in progress:
            result_meta, result_embeddings, original_shape = future.result()
            
            if not result_meta or not result_embeddings:
                continue

            is_portrait = len(result_meta) == 1
            file_data = {
                "filename": path.name,
                "faces": result_meta,
                "original_shape": original_shape
            }

            if is_portrait:
                portrait_meta[path.name] = file_data
                portrait_index[path.name] = len(portrait_embeddings)
                portrait_embeddings.append(result_embeddings[0])
            else:
                group_meta[path.name] = file_data
                for i, embedding in enumerate(result_embeddings):
                    key = f"{path.name}::{i}"
                    group_index[key] = len(group_embeddings)
                    group_embeddings.append(embedding)

    # 6. Сохранение результатов
    embeddings_dir = output_dir / "_Embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    json_manager = JsonDataManager(
        portrait_json_path=output_dir / "info_portrait_faces.json",
        group_json_path=output_dir / "info_group_faces.json"
    )
    json_manager.portrait_data = portrait_meta
    json_manager.group_data = group_meta
    json_manager.save_data()

    if portrait_embeddings:
        np.save(embeddings_dir / "portrait_embeddings.npy", np.array(portrait_embeddings))
        with open(embeddings_dir / "portrait_index.json", "w") as f:
            json.dump(portrait_index, f, indent=2)

    if group_embeddings:
        np.save(embeddings_dir / "group_embeddings.npy", np.array(group_embeddings))
        with open(embeddings_dir / "group_index.json", "w") as f:
            json.dump(group_index, f, indent=2)
            
    # 7. Освобождение ресурсов
    face_analyzer.shutdown()

    logger.info(f"<br>Поиск лиц на фотографиях завершен. Найдено <b>{len(portrait_meta)}</b> портретных и <b>{len(group_meta)}</b> групповых фотографий.<br>")
    
    pysm_context.log_link(url_or_path=str(input_dir), text="Открыть папку с исходными файлами (файлы JPG)")
    pysm_context.log_link(url_or_path=str(output_dir), text="Открыть папку с обработанными данными (файлы JSON)<br>")
    print(" ", file=sys.stderr)

    

# --- Блок 4: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()