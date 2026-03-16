# analize/cluster_locations/run_cluster_locations.py

print("<b>КЛАСТЕРИЗАЦИЯ ФОТОГРАФИЙ</b>")
print(f"<i>Инициализация...</i>")

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(2)
except ImportError:
    print("<small><i>Критическая ошибка: библиотека onnxruntime не найдена.</i></small>")
    pass

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Generator, Any
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent 
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from cluster_locations.location_lib import ConfigManager, LocationAnalyzer
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder
    
    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    sys.exit(1)
    
from _common import (
    icon_ok, icon_warning, icon_error, icon_info, icon_save, icon_save_warning, icon_save_error
)    

logger = logging.getLogger(__name__)

def chunked_iterable(iterable: List[Any], size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

# --- Блок 1: Конфигурация ---

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кластеризация фотографий по локациям.")
    default_config_path = Path(__file__).parent / "config.toml"
    p = "a_cl_" 

    parser.add_argument(f"--{p}config_file", type=str, default=str(default_config_path))
    parser.add_argument(f"--{p}data_dir", type=str, required=True, help="Путь к папке с результатами анализа")
    
    parser.add_argument("--mode", type=str, default="clustering", choices=["clustering", "classification"])
    parser.add_argument("--match_threshold", type=float, default=0.25)
    parser.add_argument("--use_originals", action="store_true", help="Анализировать оригинальные фото вместо масок")

    parser.add_argument(f"--{p}location_prompts", type=str, nargs='*', default=[])
    parser.add_argument(f"--{p}cluster_eps", type=float, default=0.14)
    parser.add_argument(f"--{p}mask_suffix", type=str, default=None)
    
    parser.add_argument("--all_threads", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()


# --- Блок 2: Режимы работы ---

def run_clustering_mode(
    location_analyzer: LocationAnalyzer, 
    files_to_process: List[Path], 
    cli_config,
    input_is_mask: bool
) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """Режим автоматической кластеризации (DBSCAN)."""
    logger.info(f"\n<b>Режим работы: Автоматическая кластеризация</b>")
    
    batch_size = cli_config.batch_size
    num_workers = cli_config.all_threads or os.cpu_count() or 4
    
    if not input_is_mask:
         logger.warning(f"{icon_warning} Кластеризация по оригиналам может группировать людей, а не фоны.")

    results = []
    
    print("", file=sys.stdout); sys.stdout.flush()
    with tqdm(total=len(files_to_process), desc="Вычисление эмбеддингов") as progress:
        for batch_paths in chunked_iterable(files_to_process, batch_size):
            batch_results = location_analyzer.get_image_embeddings_batch(
                batch_paths, 
                max_workers=num_workers,
                input_is_mask=input_is_mask
            )
            if batch_results:
                results.extend(batch_results)
            progress.update(len(batch_paths)); sys.stdout.flush()

    if not results:
        logger.error("Не удалось вычислить эмбеддинги.")
        sys.exit(1)

    paths, embeddings = zip(*results)
    embeddings_matrix = np.vstack(embeddings)

    clustering_params = location_analyzer.config_manager.get("clustering")
    clustering_params['eps'] = cli_config.a_cl_cluster_eps
    
    logger.info(f"{icon_info} Параметры: <i>{clustering_params}</i>")
    
    clusterer = DBSCAN(**clustering_params)
    labels = clusterer.fit_predict(embeddings_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"{icon_ok} Найдено <b>{n_clusters}</b> локаций")

    return list(paths), labels, embeddings_matrix


def run_classification_mode(
    location_analyzer: LocationAnalyzer, 
    files_to_process: List[Path], 
    cli_config,
    input_is_mask: bool
) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """Режим классификации по промптам."""
    logger.info(f"\n<b>Режим работы: Классификация по описаниям</b>")
    
    prompts = cli_config.a_cl_location_prompts
    if not prompts:
        logger.critical("Не заданы промпты для классификации.")
        sys.exit(1)

    if input_is_mask:
        logger.info(f"{icon_info} Источник: <b>Маски</b>")
    else:
        logger.info(f"{icon_info} Источник: <b>Оригинальные фото</b> (рекомендуется)")

    try:
        text_embeddings = location_analyzer.get_text_embeddings(prompts)
    except Exception as e:
        logger.critical(f"Ошибка текста: {e}")
        sys.exit(1)
    
    batch_size = cli_config.batch_size
    num_workers = cli_config.all_threads or os.cpu_count() or 4
    image_results = []
    
    print("", file=sys.stdout); sys.stdout.flush()
    with tqdm(total=len(files_to_process), desc="Вычисление эмбеддингов") as progress:
        for batch_paths in chunked_iterable(files_to_process, batch_size):
            batch_results = location_analyzer.get_image_embeddings_batch(
                batch_paths, 
                max_workers=num_workers,
                input_is_mask=input_is_mask
            )
            if batch_results:
                image_results.extend(batch_results)
            progress.update(len(batch_paths)); sys.stdout.flush()

    if not image_results:
        logger.error("Не удалось вычислить эмбеддинги.")
        sys.exit(1)
        
    image_paths, image_embeddings_list = zip(*image_results)
    image_embeddings_matrix = np.vstack(image_embeddings_list)
    
    similarity_matrix = 1 - cdist(image_embeddings_matrix, text_embeddings, metric='cosine')
    
    best_prompt_indices = np.argmax(similarity_matrix, axis=1)
    best_prompt_scores = np.max(similarity_matrix, axis=1)
    threshold = cli_config.match_threshold
    
    labels = np.where(best_prompt_scores >= threshold, best_prompt_indices, -1)
    
    logger.info("Результаты:")
    for i, p in enumerate(prompts):
        logger.info(f" - {p[:20]}: <b>{np.sum(labels == i)}</b>")
    logger.info(f" - Неклассифицировано: <b>{np.sum(labels == -1)}</b>")
    
    return list(image_paths), labels, image_embeddings_matrix


# --- Блок 3: Точка входа ---

def main():
    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    logger.info("<br><b>РЕЖИМ РАБОТЫ: LOCATION</b>")
    cli_config = get_config()
    
    try:
        config_manager = ConfigManager(Path(cli_config.a_cl_config_file))
    except Exception:
        sys.exit(1)

    if cli_config.a_cl_mask_suffix:
        config_manager.config['model_params']['mask_suffix'] = cli_config.a_cl_mask_suffix

    data_dir = Path(cli_config.a_cl_data_dir)
    json_path = data_dir / "info_faces.json"
    
    input_dir = data_dir / "JPG"
    masks_dir = input_dir / "Masks"

    if not data_dir.exists():
        logger.critical(f"Нет данных: {data_dir}")
        sys.exit(1)

    location_analyzer = LocationAnalyzer(config_manager)

    # ЛОГИКА ВЫБОРА ФАЙЛОВ (Умный поиск)
    files_to_process = []
    input_is_mask = False

    if cli_config.use_originals:
        # Если попросили оригиналы - ищем сразу в JPG
        logger.info(f"Поиск оригинальных изображений в: {input_dir}")
        # Ищем jpg, jpeg, JPG, JPEG
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
        for ext in extensions:
            files_to_process.extend(input_dir.glob(ext))
        input_is_mask = False
        files_to_process = sorted(list(set(files_to_process))) # Убираем дубли
    else:
        # Иначе ищем маски
        logger.info(f"Поиск масок в: {masks_dir}")
        if masks_dir.exists():
            mask_suffix = config_manager.get("model_params.mask_suffix")
            mask_glob = f"*{mask_suffix}"
            files_to_process = sorted([p for p in masks_dir.glob(mask_glob) if p.is_file()])
            input_is_mask = True
        else:
            logger.warning(f"Папка масок не найдена! Переключаюсь на поиск оригиналов.")
            files_to_process = sorted(list(input_dir.glob("*.jpg")))
            input_is_mask = False

    if not files_to_process:
        logger.error("Файлы для обработки не найдены.")
        sys.exit(0)

    logger.info(f"Найдено файлов для анализа: <b>{len(files_to_process)}</b>")

    if cli_config.mode == "classification":
        paths, labels, embeddings_matrix = run_classification_mode(location_analyzer, files_to_process, cli_config, input_is_mask)
    else:
        paths, labels, embeddings_matrix = run_clustering_mode(location_analyzer, files_to_process, cli_config, input_is_mask)
    
    logger.info("<b>Освобождение ресурсов...</b>")
    location_analyzer.shutdown()

    # Сохранение (без изменений)
    logger.info("<br><b>Сохранение результатов...</b>")
    try:
        with json_path.open("r", encoding="utf-8") as f:
            faces_data = json.load(f)
        updated_count = 0
        for path, label in zip(paths, labels):
            filename = path.name
            label_int = int(label)
            if filename in faces_data:
                faces_data[filename]['location_cluster'] = label_int
                faces_data[filename]['location_name'] = str(label_int)
                updated_count += 1
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(faces_data, f, ensure_ascii=False, indent=2)
        logger.info(f"{icon_info} обновлено <b>{updated_count}</b> записей в JSON")
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")

    embeddings_dir = data_dir / "_Embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    np.save(embeddings_dir / "location_embeddings.npy", embeddings_matrix)
    location_index = {Path(p).name: i for i, p in enumerate(paths)}
    with open(embeddings_dir / "location_index.json", "w", encoding="utf-8") as f:
        json.dump(location_index, f, indent=2)
    
    logger.info(f"{icon_save} location_embeddings.npy сохранён")           

    logger.info("<br>")    
    tv_builder = StandardTreeBuilder(icon_size=28)
    root_node1 = ResourceNode("config.toml", Path(cli_config.a_cl_config_file), "txt", "Конфиг")
    root_node = ResourceNode("Data", Path(cli_config.a_cl_data_dir), "folder", "Данные")
    tv_builder.add_section("Ресурсы", [root_node1, root_node])
    pysm_context.log_html(tv_builder.get_html())    

if __name__ == "__main__":
    main()