# analize/_common/json_data_manager.py
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

    def __init__(self, portrait_json_path: Optional[Path] = None, group_json_path: Optional[Path] = None):
        if portrait_json_path and not isinstance(portrait_json_path, Path):
            raise TypeError("Путь к JSON портретов должен быть объектом Path.")
        if group_json_path and not isinstance(group_json_path, Path):
            raise TypeError("Путь к JSON групп должен быть объектом Path.")
        if not portrait_json_path and not group_json_path:
            raise ValueError("Необходимо предоставить хотя бы один путь к JSON-файлу.")

        self.portrait_json_path = portrait_json_path.resolve() if portrait_json_path else None
        self.group_json_path = group_json_path.resolve() if group_json_path else None
        
        self.portrait_data: Dict[str, Dict[str, Any]] = {}
        self.group_data: Dict[str, Dict[str, Any]] = {}

        # Автоматическое определение путей к ландмаркам
        self.portrait_landmarks_path: Optional[Path] = None
        self.group_landmarks_path: Optional[Path] = None
        
        if self.portrait_json_path:
            self.portrait_landmarks_path = self.portrait_json_path.parent / "info_portrait_landmarks.json"
        if self.group_json_path:
            self.group_landmarks_path = self.group_json_path.parent / "info_group_landmarks.json"
            
        self.portrait_landmarks: Dict[str, Dict[str, Any]] = {}
        self.group_landmarks: Dict[str, Dict[str, Any]] = {}

        logger.debug("JsonDataManager инициализирован.")

    def load_data(self) -> bool:
        """Загружает основные данные из JSON-файлов."""
        try:
            loaded_messages = []
            if self.portrait_json_path:
                self.portrait_data = self._load_single_file(self.portrait_json_path)
                loaded_messages.append(f"<b>{len(self.portrait_data)}</b> портретных")
            
            if self.group_json_path:
                self.group_data = self._load_single_file(self.group_json_path)
                loaded_messages.append(f"<b>{len(self.group_data)}</b> групповых")

            if loaded_messages:
                print(f"ℹ️ Загружено {' и '.join(loaded_messages)} записей<br>")
            return True
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Критическая ошибка при загрузке JSON-данных: {e}", exc_info=True)
            return False

    def load_landmarks(self, data_type: str = "all") -> bool:
        """Загружает файлы ландмарков по требованию."""
        try:
            loaded_msgs = []
            if data_type in ["portrait", "all"] and self.portrait_landmarks_path:
                if self.portrait_landmarks_path.exists():
                    self.portrait_landmarks = self._load_single_file(self.portrait_landmarks_path)
                    loaded_msgs.append(f"{len(self.portrait_landmarks)} landmarks для портретных фотографий")
                else:
                    logger.warning(f"Файл landmarks не найден: {self.portrait_landmarks_path}")

            if data_type in ["group", "all"] and self.group_landmarks_path:
                if self.group_landmarks_path.exists():
                    self.group_landmarks = self._load_single_file(self.group_landmarks_path)
                    loaded_msgs.append(f"{len(self.group_landmarks)} landmarks для групповых фотографий")
                else:
                    logger.warning(f"Файл landmarks не найден: {self.group_landmarks_path}")

            if loaded_msgs:
                logger.info(f"Загружено {', '.join(loaded_msgs)}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке landmarks: {e}", exc_info=True)
            return False

    def _load_single_file(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
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
        """Сохраняет основные данные."""
        try:
            logger.info("<b>Сохранение JSON-файлов в папку:</b>")
            saved_files_messages = []
            
            if self.portrait_json_path:
                self._save_single_file(self.portrait_json_path, self.portrait_data)
                saved_files_messages.append(f"✅ портретные фотографии: <i>{self.portrait_json_path.name}</i>")
            
            if self.group_json_path:
                self._save_single_file(self.group_json_path, self.group_data)
                saved_files_messages.append(f"✅ групповые фотографии: <i>{self.group_json_path.name}</i>")
            
            if saved_files_messages:
                base_path = self.portrait_json_path.parent if self.portrait_json_path else (self.group_json_path.parent if self.group_json_path else None)
                if base_path:
                    logger.info(f"<i>{base_path}</i>")
                for msg in saved_files_messages:
                    logger.info(msg)
            return True
        except IOError as e:
            logger.error(f"Критическая ошибка при сохранении JSON-данных: {e}", exc_info=True)
            return False

    def _save_single_file(self, file_path: Path, data: Dict[str, Dict[str, Any]]):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Файл данных сохранен: {file_path}")

    # --- Методы доступа ---
    def get_data(self, filename: str) -> Optional[Dict[str, Any]]:
        return self.portrait_data.get(filename) or self.group_data.get(filename)
        
    def get_data_with_type(self, filename: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Возвращает данные для файла и его тип ('portrait' или 'group')."""
        if filename in self.portrait_data:
            return self.portrait_data[filename], "portrait"
        if filename in self.group_data:
            return self.group_data[filename], "group"
        return None  

    def get_landmarks_data(self, filename: str, data_type: str) -> Optional[Dict[str, Any]]:
        if data_type == "portrait":
            return self.portrait_landmarks.get(filename)
        elif data_type == "group":
            return self.group_landmarks.get(filename)
        return None

    def get_face(self, filename: str, face_index: int) -> Optional[Dict[str, Any]]:
        file_data = self.get_data(filename)
        if file_data and isinstance(file_data.get("faces"), list):
            if 0 <= face_index < len(file_data["faces"]):
                return file_data["faces"][face_index]
        return None

    def get_all_filenames(self, data_type: str = "all") -> List[str]:
        filenames = []
        if data_type in ["portrait", "all"] and self.portrait_data:
            filenames.extend(self.portrait_data.keys())
        if data_type in ["group", "all"] and self.group_data:
            filenames.extend(self.group_data.keys())
        return filenames

    def update_face(self, filename: str, face_index: int, updates: Dict[str, Any], data_type: str) -> bool:
        """Обновляет данные лица, сохраняя существующие поля."""
        target_dict = None
        if data_type == "portrait": target_dict = self.portrait_data
        elif data_type == "group": target_dict = self.group_data
        
        if not target_dict or filename not in target_dict:
            logger.warning(f"Файл {filename} не найден в {data_type}")
            return False

        file_entry = target_dict[filename]
        faces = file_entry.get("faces")

        if not isinstance(faces, list) or not (0 <= face_index < len(faces)):
            return False

        if isinstance(faces[face_index], dict):
            # ВАЖНО: update обновляет словарь in-place, сохраняя старые ключи (cluster_label и т.д.)
            faces[face_index].update(updates)
            return True
        return False

    def add_file_data(self, filename: str, file_data: Dict[str, Any], is_portrait: bool):
        if is_portrait: self.portrait_data[filename] = file_data
        else: self.group_data[filename] = file_data

    def clear_data(self, data_type: str = "all"):
        if data_type in ["portrait", "all"]:
            self.portrait_data.clear()
            self.portrait_landmarks.clear()
        if data_type in ["group", "all"]:
            self.group_data.clear()
            self.group_landmarks.clear()

    def get_portrait_filenames_with_children(self) -> Tuple[List[str], List[str]]:
        filenames = list(self.portrait_data.keys())
        child_names = []
        for filename in filenames:
            face_data = self.get_face(filename, 0)
            child_name = face_data.get("child_name") or "N/A" if face_data else "N/A"
            child_names.append(child_name)
        return filenames, child_names