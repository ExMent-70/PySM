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
    Также поддерживает загрузку "тяжелых" данных ландмарков по требованию.
    """

    # --- Блок 1: Инициализация ---
    # ==============================================================================
    def __init__(self, portrait_json_path: Optional[Path] = None, group_json_path: Optional[Path] = None):
        """
        Инициализирует менеджер путями к JSON-файлам.
        Хотя бы один путь должен быть предоставлен.
        """
        if portrait_json_path and not isinstance(portrait_json_path, Path):
            raise TypeError("Путь к JSON портретов должен быть объектом Path.")
        if group_json_path and not isinstance(group_json_path, Path):
            raise TypeError("Путь к JSON групп должен быть объектом Path.")
        if not portrait_json_path and not group_json_path:
            raise ValueError("Необходимо предоставить хотя бы один путь к JSON-файлу.")

        # Основные файлы
        self.portrait_json_path = portrait_json_path.resolve() if portrait_json_path else None
        self.group_json_path = group_json_path.resolve() if group_json_path else None
        
        # Данные в памяти
        self.portrait_data: Dict[str, Dict[str, Any]] = {}
        self.group_data: Dict[str, Dict[str, Any]] = {}

        # --- НОВОЕ: Поддержка файлов ландмарков ---
        # Вычисляем пути к файлам ландмарков автоматически
        self.portrait_landmarks_path: Optional[Path] = None
        self.group_landmarks_path: Optional[Path] = None
        
        if self.portrait_json_path:
            self.portrait_landmarks_path = self.portrait_json_path.parent / "info_portrait_landmarks.json"
        if self.group_json_path:
            self.group_landmarks_path = self.group_json_path.parent / "info_group_landmarks.json"
            
        # Хранилище для ландмарков (загружаются только по требованию)
        self.portrait_landmarks: Dict[str, Dict[str, Any]] = {}
        self.group_landmarks: Dict[str, Dict[str, Any]] = {}

        logger.debug("<br>JsonDataManager инициализирован.")

    # --- Блок 2: Загрузка и сохранение данных ---
    # ==============================================================================
    def load_data(self) -> bool:
        """
        Загружает ОСНОВНЫЕ данные из JSON-файлов в память.

        Returns:
            True, если загрузка прошла успешно, False при ошибке.
        """
        try:
            loaded_messages = []
            if self.portrait_json_path:
                self.portrait_data = self._load_single_file(self.portrait_json_path)
                loaded_messages.append(f"<b>{len(self.portrait_data)}</b> портретных")
            
            if self.group_json_path:
                self.group_data = self._load_single_file(self.group_json_path)
                loaded_messages.append(f"<b>{len(self.group_data)}</b> групповых")

            if loaded_messages:
                print(f"Загружено {' и '.join(loaded_messages)} записей<br>")
            return True
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Критическая ошибка при загрузке JSON-данных: {e}", exc_info=True)
            return False

    def load_landmarks(self, data_type: str = "all") -> bool:
        """
        Загружает файлы ЛАНДМАРКОВ в память по требованию.
        
        Args:
            data_type: 'portrait', 'group' или 'all'.
        """
        try:
            loaded_msgs = []
            
            # Загрузка портретных ландмарков
            if data_type in ["portrait", "all"] and self.portrait_landmarks_path:
                if self.portrait_landmarks_path.exists():
                    self.portrait_landmarks = self._load_single_file(self.portrait_landmarks_path)
                    loaded_msgs.append(f"ландмарки портретов ({len(self.portrait_landmarks)})")
                else:
                    logger.warning(f"Файл ландмарков портретов не найден: {self.portrait_landmarks_path}")

            # Загрузка групповых ландмарков
            if data_type in ["group", "all"] and self.group_landmarks_path:
                if self.group_landmarks_path.exists():
                    self.group_landmarks = self._load_single_file(self.group_landmarks_path)
                    loaded_msgs.append(f"ландмарки групп ({len(self.group_landmarks)})")
                else:
                    logger.warning(f"Файл ландмарков групп не найден: {self.group_landmarks_path}")

            if loaded_msgs:
                logger.info(f"Дополнительно загружены: {', '.join(loaded_msgs)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке ландмарков: {e}", exc_info=True)
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
        Сохраняет данные из памяти в JSON-файлы, указанные при инициализации.

        Returns:
            True, если сохранение прошло успешно, False при ошибке.
        """
        try:
            print("<b>Сохранение результатов работы...</b>")
            saved_files_messages = []
            
            if self.portrait_json_path:
                self._save_single_file(self.portrait_json_path, self.portrait_data)
                saved_files_messages.append(f"- портретные фотографии: <i>{self.portrait_json_path.name}</i>")
            
            if self.group_json_path:
                self._save_single_file(self.group_json_path, self.group_data)
                saved_files_messages.append(f"- групповые  фотографии: <i>{self.group_json_path.name}</i>")
            
            if saved_files_messages:
                base_path = self.portrait_json_path.parent if self.portrait_json_path else (self.group_json_path.parent if self.group_json_path else None)
                if base_path:
                    print(f"Рабочая папка:\n<i>{base_path}</i>")
                for msg in saved_files_messages:
                    print(msg)
            return True
        except IOError as e:
            logger.error(f"Критическая ошибка при сохранении JSON-данных: {e}", exc_info=True)
            return False

    def _save_single_file(self, file_path: Path, data: Dict[str, Dict[str, Any]]):
        """Сохраняет данные в один JSON-файл."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Файл данных сохранен: {file_path}")

    # --- Блок 3: Методы доступа ---
    # ==============================================================================
    def get_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Возвращает полный словарь данных (основных) для указанного имени файла."""
        return self.portrait_data.get(filename) or self.group_data.get(filename)
        
    def get_data_with_type(self, filename: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Возвращает данные для файла и его тип источника ('portrait' или 'group').
        """
        if filename in self.portrait_data:
            return self.portrait_data[filename], "portrait"
        if filename in self.group_data:
            return self.group_data[filename], "group"
        return None  

    def get_landmarks_data(self, filename: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает данные ландмарков для файла, если они были загружены.
        """
        if data_type == "portrait":
            return self.portrait_landmarks.get(filename)
        elif data_type == "group":
            return self.group_landmarks.get(filename)
        return None

    def get_face(self, filename: str, face_index: int) -> Optional[Dict[str, Any]]:
        """Возвращает словарь данных для конкретного лица в указанном файле (из основных данных)."""
        file_data = self.get_data(filename)
        if file_data and isinstance(file_data.get("faces"), list):
            if 0 <= face_index < len(file_data["faces"]):
                return file_data["faces"][face_index]
        return None

    def get_all_filenames(self, data_type: str = "all") -> List[str]:
        """Возвращает список имен файлов (ключей) из хранимых данных."""
        filenames = []
        if data_type in ["portrait", "all"] and self.portrait_data:
            filenames.extend(self.portrait_data.keys())
        if data_type in ["group", "all"] and self.group_data:
            filenames.extend(self.group_data.keys())
        return filenames

    def update_face(
        self, filename: str, face_index: int, updates: Dict[str, Any], data_type: str
    ) -> bool:
        """
        Обновляет поля для указанного лица в словаре заданного типа (в основных данных).

        Args:
            filename: Имя файла для обновления.
            face_index: Индекс лица в списке "faces".
            updates: Словарь с обновляемыми полями.
            data_type: Тип данных для обновления ('portrait' или 'group').
        """
        target_dict = None
        
        if data_type == "portrait":
            target_dict = self.portrait_data
        elif data_type == "group":
            target_dict = self.group_data
        else:
            logger.warning(f"Неизвестный тип данных '{data_type}' для обновления лица.")
            return False

        if filename not in target_dict:
            logger.warning(
                f"Файл '{filename}' не найден в данных типа '{data_type}' "
                "при попытке обновления лица."
            )
            return False

        file_entry = target_dict.get(filename, {})
        faces = file_entry.get("faces")

        if not isinstance(faces, list) or not (0 <= face_index < len(faces)):
            logger.warning(
                f"Некорректный индекс лица {face_index} или структура "
                f"'faces' для файла '{filename}'."
            )
            return False

        if isinstance(faces[face_index], dict):
            faces[face_index].update(updates)
            return True
        else:
            logger.warning(
                f"Запись лица {face_index} в файле '{filename}' не является словарем."
            )
            return False

    def add_file_data(self, filename: str, file_data: Dict[str, Any], is_portrait: bool):
        """Добавляет или перезаписывает данные для целого файла (в основные данные)."""
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
            self.portrait_landmarks.clear()
            print("Данные портретных файлов очищены из памяти.")
        if data_type in ["group", "all"]:
            self.group_data.clear()
            self.group_landmarks.clear()
            print("Данные групповых файлов очищены из памяти.")

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