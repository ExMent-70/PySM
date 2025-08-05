# analize/_common/json_data_manager.py
"""
Этот модуль предоставляет класс JsonDataManager для унифицированной работы
с JSON-файлами, содержащими метаданные лиц.
Является общей утилитой для всех этапов конвейера анализа.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class JsonDataManager:
    """
    Класс для управления чтением, записью и обновлением данных
    в JSON-файлах info_portrait_faces.json и info_group_faces.json.
    """

    def __init__(self, portrait_json_path: Path, group_json_path: Path):
        """
        Инициализирует менеджер путями к JSON-файлам.

        Args:
            portrait_json_path: Путь к файлу с данными портретных фото.
            group_json_path: Путь к файлу с данными групповых фото.
        """
        if not isinstance(portrait_json_path, Path) or not isinstance(group_json_path, Path):
            raise TypeError("Пути к JSON должны быть объектами Path.")

        self.portrait_json_path = portrait_json_path.resolve()
        self.group_json_path = group_json_path.resolve()
        self.portrait_data: Dict[str, Dict[str, Any]] = {}
        self.group_data: Dict[str, Dict[str, Any]] = {}

        logger.debug("<br>JsonDataManager инициализирован.")

    def load_data(self) -> bool:
        """
        Загружает данные из JSON-файлов в память.
        Если файлы не существуют, инициализирует пустыми словарями.

        Returns:
            True, если загрузка прошла успешно, False при ошибке.
        """
        logger.debug(f"Загрузка данных из {self.portrait_json_path.name} и {self.group_json_path.name}")
        try:
            self.portrait_data = self._load_single_file(self.portrait_json_path)
            self.group_data = self._load_single_file(self.group_json_path)
            logger.info(f"Загружено <b>{len(self.portrait_data)}</b> портретных и <b>{len(self.group_data)}</b> групповых записей<br>")
            return True
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Критическая ошибка при загрузке JSON-данных: {e}", exc_info=True)
            return False

    def _load_single_file(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Загружает данные из одного JSON-файла."""
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"Файл {file_path.name} не содержит JSON-объект (словарь).")
            return data
        else:
            logger.warning(f"Файл {file_path} не найден. Будет создан новый.")
            return {}

    def save_data(self) -> bool:
        """
        Сохраняет данные из памяти в JSON-файлы.

        Returns:
            True, если сохранение прошло успешно, False при ошибке.
        """
        try:
            logger.info(f"<b>Сохранение результатов работы...</b>")
            self._save_single_file("портретных", self.portrait_json_path, self.portrait_data)
            self._save_single_file("групповых", self.group_json_path, self.group_data)
            logger.info(f"  - портретные фотографии: <i>{self.portrait_json_path}</i>")
            logger.info(f"  - групповые  фотографии: <i>{self.group_json_path}</i>")
            return True
        except IOError as e:
            logger.error(f"Критическая ошибка при сохранении JSON-данных: {e}", exc_info=True)
            return False

    def _save_single_file(self, description: str, file_path: Path, data: Dict[str, Dict[str, Any]]):
        """Сохраняет данные в один JSON-файл."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Файл {description} данных сохранен: {file_path}")

    def get_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Возвращает полный словарь данных для указанного имени файла."""
        return self.portrait_data.get(filename) or self.group_data.get(filename)

    def get_face(self, filename: str, face_index: int) -> Optional[Dict[str, Any]]:
        """Возвращает словарь данных для конкретного лица в указанном файле."""
        file_data = self.get_data(filename)
        if file_data and isinstance(file_data.get("faces"), list):
            if 0 <= face_index < len(file_data["faces"]):
                return file_data["faces"][face_index]
        return None

    def get_all_filenames(self, data_type: str = "all") -> List[str]:
        """Возвращает список имен файлов (ключей) из хранимых данных."""
        filenames = []
        if data_type in ["portrait", "all"]:
            filenames.extend(self.portrait_data.keys())
        if data_type in ["group", "all"]:
            filenames.extend(self.group_data.keys())
        return filenames

    def update_face(self, filename: str, face_index: int, updates: Dict[str, Any]) -> bool:
        """Обновляет (добавляет или перезаписывает) поля для указанного лица."""
        target_dict = None
        if filename in self.portrait_data:
            target_dict = self.portrait_data
        elif filename in self.group_data:
            target_dict = self.group_data
        else:
            logger.warning(f"Файл '{filename}' не найден для обновления данных лица.")
            return False

        file_entry = target_dict.get(filename, {})
        faces = file_entry.get("faces")

        if not isinstance(faces, list) or not (0 <= face_index < len(faces)):
            logger.warning(f"Некорректный индекс лица {face_index} или структура 'faces' для файла '{filename}'.")
            return False

        if isinstance(faces[face_index], dict):
            faces[face_index].update(updates)
            logger.debug(f"Обновлены данные для лица {face_index} в файле '{filename}': {list(updates.keys())}")
            return True
        else:
            logger.warning(f"Запись лица {face_index} в файле '{filename}' не является словарем.")
            return False

    def add_file_data(self, filename: str, file_data: Dict[str, Any], is_portrait: bool):
        """Добавляет или перезаписывает данные для целого файла."""
        if not (isinstance(file_data, dict) and "faces" in file_data and isinstance(file_data["faces"], list)):
            logger.error(f"Попытка добавить некорректные данные для файла '{filename}'.")
            return

        if is_portrait:
            self.portrait_data[filename] = file_data
        else:
            self.group_data[filename] = file_data

    def clear_data(self, data_type: str = "all"):
        """Очищает внутренние словари данных."""
        if data_type in ["portrait", "all"]:
            self.portrait_data.clear()
            logger.info("Данные портретных файлов очищены из памяти.")
        if data_type in ["group", "all"]:
            self.group_data.clear()
            logger.info("Данные групповых файлов очищены из памяти.")

    def get_portrait_filenames_with_children(self) -> Tuple[List[str], List[str]]:
        """Возвращает списки имен портретных файлов и соответствующих им имен детей."""
        filenames = list(self.portrait_data.keys())
        child_names = []
        for filename in filenames:
            face_data = self.get_face(filename, 0)
            child_name = "N/A"
            if face_data:
                child_name = face_data.get("child_name") or "N/A"
            child_names.append(child_name)
        return filenames, child_names