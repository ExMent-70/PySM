# analize/sort_files/run_sort_files.py

# --- Блок 1: Импорты и настройка путей ---
# ==============================================================================
import argparse
import logging
import sys
import shutil
import os
from pathlib import Path
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from _common.json_data_manager import JsonDataManager
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    JsonDataManager = None
    ConfigResolver = None
    pysm_context = None
    tqdm = lambda x, **kwargs: x

# --- Блок 2: Настройка логирования ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Блок 3: Класс FileSorter ---
# ==============================================================================
class FileSorter:
    def __init__(self, operation: str, extensions: List[str], source_paths: List[Path], output_root: Path):
        self.operation_func = shutil.copy2 if operation == 'copy' else shutil.move
        self.operation_name = "Копирование" if operation == 'copy' else "Перемещение"
        self.extensions: Set[str] = {ext.lower() for ext in extensions}
        self.source_paths: List[Path] = [p for p in source_paths if p.is_dir()]
        self.output_root: Path = output_root
        
        logger.info(f"Режим работы: {self.operation_name}")
        logger.info(f"Обрабатываемые расширения: {sorted(list(self.extensions))}")
        logger.info(f"Поиск исходных файлов в: {[str(p) for p in self.source_paths]}")
        logger.info(f"Папка для результатов: {self.output_root}")

    def _get_target_folder_name(self, file_data: Dict) -> str:
        if len(file_data.get("faces", [])) > 1:
            return "_Group_Photos"
        
        face_data = file_data.get("faces", [{}])[0]
        child_name = face_data.get("child_name")
        cluster_label = face_data.get("cluster_label")

        if child_name == "Noise":
            return "99-Noise"
        if child_name and child_name.startswith("Unknown"):
            return f"98-{child_name}"
        if child_name and cluster_label is not None:
            clean_name = child_name.split('-', 1)[-1] if '-' in child_name else child_name
            return f"{cluster_label:02d}-{clean_name}"
        
        return "97-Unsorted"

    def _process_file_set(self, source_file_stem: str, target_subfolder: str) -> List[str]:
        target_dir = self.output_root / target_subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        found_files = []
        
        for source_dir in self.source_paths:
            for ext in self.extensions:
                source_file = source_dir / f"{source_file_stem}{ext}"
                if source_file.is_file():
                    found_files.append(source_file)
        
        if not found_files:
            processed_files.append(f"[ПРЕДУПРЕЖДЕНИЕ] Для основы '{source_file_stem}' не найдено файлов с нужными расширениями.")
            return processed_files

        for source_file in found_files:
            target_file = target_dir / source_file.name
            try:
                if source_file.resolve() != target_file.resolve():
                    self.operation_func(source_file, target_file)
                    processed_files.append(f"{source_file.name} -> {target_subfolder}/")
                else:
                    processed_files.append(f"{source_file.name} уже в целевой папке.")
            except Exception as e:
                processed_files.append(f"[ОШИБКА] при обработке {source_file.name}: {e}")
        
        return processed_files

    def sort_all(self, json_manager: JsonDataManager, max_workers: int):
        all_files_data = {**json_manager.portrait_data, **json_manager.group_data}
        if not all_files_data:
            logger.warning("Нет данных в JSON для сортировки файлов.")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = {}
            for filename, file_data in all_files_data.items():
                file_stem = Path(filename).stem
                target_subfolder = self._get_target_folder_name(file_data)
                future = executor.submit(self._process_file_set, file_stem, target_subfolder)
                tasks[future] = filename

            for future in tqdm(as_completed(tasks), total=len(tasks), desc=self.operation_name):
                filename = tasks[future]
                try:
                    results = future.result()
                    for msg in results:
                        if "[ОШИБКА]" in msg or "[ПРЕДУПРЕЖДЕНИЕ]" in msg:
                            logger.warning(f"{filename}: {msg}")
                        else:
                            logger.debug(f"{filename}: {msg}")
                except Exception as e:
                    logger.error(f"Критическая ошибка в потоке для {filename}: {e}", exc_info=True)

# --- Блок 4: Конфигурация и `main` ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """Определяет и разрешает CLI-аргументы."""
    parser = argparse.ArgumentParser(description="Сортировка файлов по папкам кластеров.")
    parser.add_argument("--a_sf_operation", type=str, choices=['copy', 'move'], default='copy')
    parser.add_argument("--a_sf_extensions", type=str, nargs='+', default=['.jpg', '.xmp'])
    parser.add_argument("--all_threads", type=int, default=0)
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()

def main():
    logger.info("="*10 + " ЗАПУСК СОРТИРОВКИ ФАЙЛОВ " + "="*10)
    
    if not IS_MANAGED_RUN:
        logger.critical("Этот скрипт требует запуска из среды PySM для доступа к контексту.")
        sys.exit(1)

    config = get_config()

    # 1. Формирование путей на основе контекста
    session_path_str = pysm_context.get("wf_session_path")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")
    
    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_...) не найдены.")
        sys.exit(1)
    
    base_path = Path(session_path_str) / session_name
    analysis_dir = base_path / "Output" / f"Analysis_{photo_session}"
    capture_dir = base_path / "Capture" / photo_session
    jpg_dir = analysis_dir / "JPG"
    output_root = base_path / "Output" / f"Claster_{photo_session}"
    
    source_paths = [capture_dir, jpg_dir]

    # 2. Инициализация JsonDataManager
    json_manager = JsonDataManager(
        portrait_json_path=analysis_dir / "info_portrait_faces.json",
        group_json_path=analysis_dir / "info_group_faces.json"
    )
    if not json_manager.load_data():
        logger.critical("Не удалось загрузить JSON-данные. Завершение работы.")
        sys.exit(1)
        
    num_workers = config.all_threads
    if not num_workers or num_workers <= 0:
        num_workers = os.cpu_count() or 4
        logger.info(f"Параметр 'all_threads' установлен в 0 или не задан. Используется автоматическое количество потоков: {num_workers}")
        
    # 3. Инициализация и запуск Sorter
    sorter = FileSorter(
        operation=config.a_sf_operation,
        extensions=config.a_sf_extensions,
        source_paths=source_paths,
        output_root=output_root
    )
    sorter.sort_all(json_manager, num_workers)

    logger.info("="*10 + " СОРТИРОВКА ФАЙЛОВ ЗАВЕРШЕНА " + "="*10)

    if IS_MANAGED_RUN:
        pysm_context.log_link(url_or_path=str(output_root), text="<br>Открыть папку с отсортированными файлами")


if __name__ == "__main__":
    main()