# analize/tools/migrate_json_format.py
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Константы ---
# Переносим только тяжелую геометрию. kps оставляем в основном файле.
KEYS_TO_MIGRATE = ["landmark_2d_106", "landmark_3d_68"]

# --- Попытка импорта PySM (для совместимости с ConfigResolver) ---
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Миграция JSON файлов в новый формат с разделением ландмарков.")
    parser.add_argument("--path", type=str, required=True, help="Путь к папке Analysis_{session}, где лежат JSON файлы.")
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def split_data(original_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    Разделяет исходные данные на Main (легкие) и Landmarks (тяжелые).
    """
    main_data = {}
    landmarks_data = {}
    
    count_faces = 0
    count_landmarks = 0

    for filename, content in original_data.items():
        # 1. Структура для Main файла
        main_entry = content.copy()
        main_entry["faces"] = []
        
        # 2. Структура для Landmarks файла
        land_entry = {
            "filename": filename,
            "faces": []
        }
        
        has_landmarks_in_file = False
        
        # Обработка лиц
        for face in content.get("faces", []):
            count_faces += 1
            main_face = face.copy()
            land_face = {}
            
            # Переносим тяжелые ключи
            for key in KEYS_TO_MIGRATE:
                if key in main_face:
                    land_face[key] = main_face.pop(key)
                    has_landmarks_in_file = True
            
            main_entry["faces"].append(main_face)
            land_entry["faces"].append(land_face)
        
        main_data[filename] = main_entry
        
        if has_landmarks_in_file:
            landmarks_data[filename] = land_entry
            count_landmarks += 1
            
    return main_data, landmarks_data, count_faces

def process_file(file_path: Path):
    if not file_path.exists():
        logger.warning(f"Файл не найден, пропуск: {file_path.name}")
        return

    logger.info(f"Обработка файла: {file_path.name}")

    # 1. Бэкап (переименование)
    backup_path = file_path.parent / f"{file_path.stem}_OLD{file_path.suffix}"
    if backup_path.exists():
        logger.warning(f"  Бэкап уже существует ({backup_path.name}). Пропуск во избежание перезаписи.")
        return

    try:
        # Читаем оригинал
        with open(file_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        # Переименовываем оригинал в _OLD
        shutil.move(str(file_path), str(backup_path))
        logger.info(f"  Оригинал переименован в: {backup_path.name}")

        # 2. Разделение данных
        main_data, landmarks_data, count = split_data(original_data)
        
        # 3. Определение имени файла для ландмарков
        stem = file_path.stem
        if "faces" in stem:
            land_stem = stem.replace("faces", "landmarks")
        else:
            land_stem = stem + "_landmarks"
        
        landmarks_path = file_path.parent / f"{land_stem}.json"

        # 4. Сохранение
        # Сохраняем новый Main (с тем же именем, что был оригинал)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(main_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Сохранен основной файл: {file_path.name} (Лиц: {count})")

        # Сохраняем Landmarks
        if landmarks_data:
            with open(landmarks_path, "w", encoding="utf-8") as f:
                json.dump(landmarks_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  Сохранен файл ландмарков: {landmarks_path.name}")
        else:
            logger.info("  Ландмарки не найдены, дополнительный файл не создан.")

    except Exception as e:
        logger.error(f"  Ошибка при обработке {file_path.name}: {e}")
        # Пытаемся восстановить файл из бэкапа
        if backup_path.exists() and not file_path.exists():
            shutil.move(str(backup_path), str(file_path))
            logger.info("  Откат изменений выполнен.")

def main():
    args = get_args()
    
    if not args.path:
        logger.critical("Не указан путь к папке (--path).")
        sys.exit(1)

    target_dir = Path(args.path)

    if not target_dir.is_dir():
        logger.critical(f"Директория не найдена: {target_dir}")
        sys.exit(1)

    logger.info(f"Запуск миграции в папке: {target_dir}")
    
    process_file(target_dir / "info_portrait_faces.json")
    process_file(target_dir / "info_group_faces.json")
    
    logger.info("Миграция завершена.")

if __name__ == "__main__":
    main()