print("<b>ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ БИБЛИОТЕК...</b><br>")

try:
    import onnxruntime as ort
    # Устанавливаем глобальный уровень логирования для всей библиотеки onnxruntime.
    # 2 соответствует уровню WARNING. Это подавит подробные INFO и VERBOSE логи
    # от C++ бэкенда, включая TensorRT.
    ort.set_default_logger_severity(2)
except ImportError:
    print("<small><i>Критическая ошибка: библиотека onnxruntime не найдена.</i></small>")
    pass

import argparse
import logging
import os
import sys
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent 
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from _common.json_data_manager import JsonDataManager
    from cluster_locations.location_lib import ConfigManager, LocationAnalyzer
    
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def construct_paths() -> Dict[str, Optional[Path]]:
    if not IS_MANAGED_RUN:
        logger.critical("Скрипт запущен без окружения PySM, авто-формирование путей невозможно.")
        return {}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Одна или несколько переменных контекста (wf_session_path, wf_session_name, wf_photo_session) не найдены.")
        return {}

    base_path = Path(session_path_str) / session_name
    data_dir = base_path / "Output" / f"Analysis_{photo_session}"
    
    return {
        "input_dir": data_dir / "JPG",
        "masks_dir": data_dir / "JPG" / "Masks",
        "output_dir": data_dir,
    }

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кластеризация фотографий по локациям.")
    default_config_path = Path(__file__).parent / "config.toml"
    parser.add_argument("--a_cl_config_file", type=str, default=str(default_config_path))
    parser.add_argument("--a_cl_location_prompts", type=str, nargs='*', default=[])
    parser.add_argument("--a_cl_cluster_eps", type=float, default=0.14)
    parser.add_argument("--a_cl_input_dir", type=str)
    parser.add_argument("--a_cl_masks_dir", type=str)
    parser.add_argument("--a_cl_output_dir", type=str)
    parser.add_argument("--all_threads", type=int, default=0)
    # --- НОВЫЙ АРГУМЕНТ ---
    parser.add_argument("--a_cl_mask_suffix", type=str, default=None, 
                        help="Суффикс файлов масок (например, _BiRefNet-portrait_output.jpg)")
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()

def run_clustering_mode(location_analyzer: LocationAnalyzer, mask_files: List[Path], cli_config) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    logger.info(f"\n<b>Режим работы: Автоматическая кластеризация</b>")
    num_workers = cli_config.all_threads or os.cpu_count() or 4

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(location_analyzer.get_image_embedding, path): path for path in mask_files}
        progress = tqdm(futures.items(), total=len(mask_files), desc="Вычисление эмбеддингов локаций")
        for future, path in progress:
            result = future.result()
            if result:
                results.append(result)

    if not results:
        logger.error("Не удалось вычислить ни одного эмбеддинга. Проверьте логи."); sys.exit(1)

    paths, embeddings = zip(*results)
    embeddings_matrix = np.vstack(embeddings)

    clustering_params = location_analyzer.config_manager.get("clustering")
    clustering_params['eps'] = cli_config.a_cl_cluster_eps
    
    logger.info(f"<br>Вычислено <b>{len(results)} эмбеддингов</b>. Запуск кластеризации с параметрами: <i>{clustering_params}</i>")
    clusterer = DBSCAN(**clustering_params)
    labels = clusterer.fit_predict(embeddings_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    logger.info(f"Найдено <b>{n_clusters}</b> локаций (кластеров). Шумовых/уникальных фото: <b>{n_noise}</b><br>")

    return list(paths), labels, embeddings_matrix

def run_classification_mode(location_analyzer: LocationAnalyzer, mask_files: List[Path], cli_config) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    logger.info(f"\n<b>Режим работы: Классификация по текстовым описаниям</b>")
    prompts = cli_config.a_cl_location_prompts
    
    try:
        text_embeddings = location_analyzer.get_text_embeddings(prompts)
    except (ImportError, RuntimeError) as e:
        logger.critical(f"Ошибка при обработке текста: {e}"); sys.exit(1)
    
    num_workers = cli_config.all_threads or os.cpu_count() or 4
    image_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(location_analyzer.get_image_embedding, path): path for path in mask_files}
        progress = tqdm(futures.items(), total=len(mask_files), desc="Вычисление эмбеддингов локаций")
        for future, path in progress:
            result = future.result()
            if result:
                image_results.append(result)

    if not image_results:
        logger.error("Не удалось вычислить ни одного эмбеддинга. Проверьте логи."); sys.exit(1)
        
    image_paths, image_embeddings_list = zip(*image_results)
    image_embeddings_matrix = np.vstack(image_embeddings_list)
    
    similarity_matrix = 1 - cdist(image_embeddings_matrix, text_embeddings, metric='cosine')
    
    best_prompt_indices = np.argmax(similarity_matrix, axis=1)
    best_prompt_scores = np.max(similarity_matrix, axis=1)
    
    threshold = location_analyzer.config_manager.get("classification.match_threshold", 0.25)
    labels = np.where(best_prompt_scores >= threshold, best_prompt_indices, -1)
    
    logger.info("Результаты классификации:")
    for i, prompt in enumerate(prompts):
        count = np.sum(labels == i)
        logger.info(f" - Локация {i} ('{prompt[:40]}...'): <b>{count}</b> фото")
    
    n_noise = np.sum(labels == -1)
    logger.info(f" - Не классифицировано (ниже порога {threshold}): <b>{n_noise}</b> фото<br>")
    
    return list(image_paths), labels, image_embeddings_matrix

def main():
    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    logger.info("<b>КЛАСТЕРИЗАЦИЯ ФОТОГРАФИЙ ПО ЛОКАЦИЯМ</b>")
    cli_config = get_config()
    
    # 1. Загрузка конфига
    try:
        config_manager = ConfigManager(Path(cli_config.a_cl_config_file))
    except Exception:
        sys.exit(1)

    # 2. Применение CLI переопределений
    # Передаем суффикс из CLI в конфиг, если он указан
    if cli_config.a_cl_mask_suffix:
        config_manager.config['model_params']['mask_suffix'] = cli_config.a_cl_mask_suffix

    # 3. Пути
    auto_paths = construct_paths()
    input_dir = Path(cli_config.a_cl_input_dir or auto_paths.get("input_dir"))
    masks_dir = Path(cli_config.a_cl_masks_dir or auto_paths.get("masks_dir"))
    output_dir = Path(cli_config.a_cl_output_dir or auto_paths.get("output_dir"))
    
    # ... (Проверки путей и вывод логов без изменений) ...
    
    location_analyzer = LocationAnalyzer(config_manager)

    # 4. Поиск файлов масок с использованием суффикса из конфига
    mask_suffix = config_manager.get("model_params.mask_suffix")
    mask_glob = f"*{mask_suffix}"
    
    mask_files = sorted([p for p in masks_dir.glob(mask_glob) if p.is_file()])
    if not mask_files:
        logger.warning(f"В папке {masks_dir} не найдено файлов масок по шаблону '{mask_glob}'.")
        sys.exit(0)



    prompts = cli_config.a_cl_location_prompts
    if prompts and any(p.strip() for p in prompts):
        paths, labels, embeddings_matrix = run_classification_mode(location_analyzer, mask_files, cli_config)
    else:
        paths, labels, embeddings_matrix = run_clustering_mode(location_analyzer, mask_files, cli_config)
    
    location_analyzer.shutdown()

    json_manager = JsonDataManager(
        portrait_json_path=output_dir / "info_portrait_faces.json",
        group_json_path=output_dir / "info_group_faces.json"
    )

    logger.info("<br>Обновление JSON файлов с метками кластеров...")
    if not json_manager.load_data(): sys.exit(1)
    

    # Блок 2: Цикл обновления JSON теперь не содержит копирования файлов
    for path, label in tqdm(zip(paths, labels), total=len(paths), desc="Обновление метаданных"):
        filename = path.name
        label_int = int(label)

        # Логика обновления JSON остается без изменений
        target_dict = json_manager.portrait_data if filename in json_manager.portrait_data else json_manager.group_data
        if filename in target_dict:
            target_dict[filename]['location_cluster'] = label_int
            target_dict[filename]['location_name'] = str(label_int) # Сохраняем как строку для консистентности
        
        # Логика, отвечающая за создание папок и копирование, полностью удалена.
    
    # Сохраняем обновленные JSON файлы
    json_manager.save_data()
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
    
    embeddings_dir = output_dir / "_Embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    np.save(embeddings_dir / "location_embeddings.npy", embeddings_matrix)
    location_index = {Path(p).name: i for i, p in enumerate(paths)}
    with open(embeddings_dir / "location_index.json", "w", encoding="utf-8") as f:
        json.dump(location_index, f, indent=2)
    
    logger.info("✅ эмбеддинги локаций: <i>_Embeddings/location_embeddings.npy</i>")
    logger.info("✅ индекс эмбеддингов: <i>_Embeddings/location_index.json</i><br>")
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Блок 3: Удалена ссылка на несуществующую папку
    if IS_MANAGED_RUN:
        pysm_context.log_link(url_or_path=str(output_dir), text="Открыть папку с результатами (JSON-файлы)")
        logger.info("<br>")    
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

if __name__ == "__main__":
    main()