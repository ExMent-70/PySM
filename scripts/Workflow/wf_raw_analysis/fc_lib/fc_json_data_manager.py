# fc_lib/fc_json_data_manager.py
# --- ИЗМЕНЕНИЙ НЕ ТРЕБУЕТСЯ ---

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Используем относительный импорт для messages
from .fc_messages import get_message

logger = logging.getLogger(__name__)


class JsonDataManager:
    """
    Класс для управления чтением, записью и обновлением данных
    в JSON-файлах info_portrait_faces.json и info_group_faces.json.
    (Предполагается, что эмбеддинги здесь НЕ хранятся).
    """

    def __init__(self, portrait_json_path: Path, group_json_path: Path):
        """
        Инициализирует менеджер путями к JSON-файлам.
        Пути нормализуются с помощью .resolve().

        Args:
            portrait_json_path: Путь к файлу с данными портретных фото.
            group_json_path: Путь к файлу с данными групповых фото.
        """
        if not isinstance(portrait_json_path, Path):
            raise TypeError("portrait_json_path должен быть объектом Path")
        if not isinstance(group_json_path, Path):
            raise TypeError("group_json_path должен быть объектом Path")

        try:
            self.portrait_json_path = portrait_json_path.resolve()
            self.group_json_path = group_json_path.resolve()
            logger.debug(f"Resolved portrait JSON path: {self.portrait_json_path}")
            logger.debug(f"Resolved group JSON path: {self.group_json_path}")
        except Exception as e:
            logger.error(
                f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось разрешить пути к JSON: {e}. Используются исходные пути.",
                exc_info=True,
            )
            self.portrait_json_path = portrait_json_path
            self.group_json_path = group_json_path

        self.portrait_data: Dict[str, Dict[str, Any]] = {}
        self.group_data: Dict[str, Dict[str, Any]] = {}
        print(" ")
        logger.info(
            f"<b>JsonDataManager инициализирован с путями:</b>"
        )
        logger.info(
            f"  - Портретные фотографии: <i>{self.portrait_json_path.name}</i>"
        )
        logger.info(
            f"  - Групповые фотографии: <i>{self.group_json_path.name}</i>"
        )
        print(" ")


    def load_data(self) -> bool:
        """
        Загружает данные из JSON-файлов в память.
        Если файлы не существуют, инициализирует пустыми словарями.
        Returns:
            bool: True если загрузка прошла успешно, False при ошибке декодирования.
        """
        logger.debug(
            f"Загрузка данных из {self.portrait_json_path} и {self.group_json_path}"
        )
        portrait_load_result = self._load_single_file(self.portrait_json_path)
        if portrait_load_result is None:
            return False
        self.portrait_data = portrait_load_result
        logger.debug(f"Загружено {len(self.portrait_data)} записей из файла портретов.")

        group_load_result = self._load_single_file(self.group_json_path)
        if group_load_result is None:
            return False
        self.group_data = group_load_result
        logger.debug(f"Загружено {len(self.group_data)} записей из файла групп.")
        logger.info(f"Загружены данные для <b>{len(self.portrait_data)}</b> портретных и <b>{len(self.group_data)}</b> групповых фотографий.")
        print(" ")

        return True

    def _load_single_file(self, file_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
        """Загружает данные из одного JSON-файла."""
        if file_path.exists():
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(
                        get_message(
                            "ERROR_LOADING_JSON",
                            file_path=file_path,
                            exc="Файл не содержит JSON-объект (словарь).",
                        )
                    )
                    return None
                logger.debug(f"Успешно загружен JSON из {file_path}")
                return data
            except json.JSONDecodeError as e:
                logger.error(
                    get_message(
                        "ERROR_LOADING_JSON",
                        file_path=file_path,
                        exc=f"Ошибка декодирования JSON: {e}",
                    )
                )
                return None
            except Exception as e:
                logger.error(
                    get_message("ERROR_LOADING_JSON", file_path=file_path, exc=e),
                    exc_info=True,
                )
                return None
        else:
            logger.warning(
                f"Файл {file_path} не найден. Инициализирован пустой словарь."
            )
            return {}


    def save_data(self, data_type: str = "all") -> bool:
        """
        Сохраняет данные из памяти в JSON-файлы.
        Может сохранять все файлы ("all"), только портретные ("portrait")
        или только групповые ("group").
        """
        logger.debug(f"Запрос на сохранение данных типа: {data_type}")
        success = True
        if data_type in ["portrait", "all"]:
            if not self._save_single_file(
                self.portrait_json_path, self.portrait_data, "портретных"
            ):
                success = False
        
        if data_type in ["group", "all"]:
            if not self._save_single_file(
                self.group_json_path, self.group_data, "групповых"
            ):
                success = False

        if success and data_type == "all":
            logger.info("Сохранение JSON файлов после обновления данных.")
        elif not success:
             logger.error("Ошибка при сохранении одного или нескольких JSON файлов.")
             
        return success


    def _save_single_file(
        self, file_path: Path, data: Dict[str, Dict[str, Any]], data_description: str
    ) -> bool:
        """Сохраняет данные в один JSON-файл."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # --- ИЗМЕНЕНИЕ: Используем ключ сообщения ---
            if "портретн" in data_description.lower():
                message_key = "INFO_JSON_PORTRAIT_SAVED"
            elif "группов" in data_description.lower():
                message_key = "INFO_JSON_GROUP_SAVED"
            else:
                # Fallback на старое сообщение, если описание неожиданное
                message_key = "INFO_FACES_SAVED"
                logger.warning(
                    f"Не удалось определить ключ сообщения для сохранения JSON: '{data_description}'"
                )

            logger.debug(get_message(message_key, output_file=file_path.resolve()))
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            return True
        except Exception as e:
            logger.error(
                get_message(
                    "ERROR_SAVING_JSON", output_file=file_path.resolve(), exc=e
                ),
                exc_info=True,
            )  # Используем resolve()
            return False

    def get_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Возвращает полный словарь данных для указанного имени файла."""
        if filename in self.portrait_data:
            return self.portrait_data[filename]
        elif filename in self.group_data:
            return self.group_data[filename]
        else:
            return None

    def get_face(self, filename: str, face_index: int) -> Optional[Dict[str, Any]]:
        """Возвращает словарь данных для конкретного лица в указанном файле."""
        file_data = self.get_data(filename)
        if file_data:
            faces = file_data.get("faces")
            if isinstance(faces, list):
                if 0 <= face_index < len(faces):
                    if isinstance(faces[face_index], dict):
                        return faces[face_index]
                    else:
                        logger.warning(
                            f"Элемент с индексом {face_index} в 'faces' файла '{filename}' не словарь."
                        )
                        return None
                else:
                    logger.warning(
                        f"Некорректный индекс лица {face_index} для файла '{filename}' (всего {len(faces)})."
                    )
                    return None
            else:
                logger.warning(
                    f"Поле 'faces' в файле '{filename}' не является списком."
                )
                return None
        return None

    def get_all_filenames(self, data_type: str = "all") -> List[str]:
        """Возвращает список имен файлов (ключей) из хранимых данных."""
        filenames = []
        if data_type == "portrait" or data_type == "all":
            filenames.extend(list(self.portrait_data.keys()))
        if data_type == "group" or data_type == "all":
            filenames.extend(list(self.group_data.keys()))
        if not filenames and data_type not in ["portrait", "group", "all"]:
            logger.warning(f"Неизвестный тип данных '{data_type}' в get_all_filenames.")
        return filenames

    def update_face(
        self, filename: str, face_index: int, updates: Dict[str, Any]
    ) -> bool:
        """Обновляет (добавляет или перезаписывает) поля для указанного лица."""
        target_data = None
        if filename in self.portrait_data:
            target_data = self.portrait_data
        elif filename in self.group_data:
            target_data = self.group_data
        else:
            logger.warning(f"Файл '{filename}' не найден для обновления данных лица.")
            return False

        if target_data:
            file_entry = target_data.get(filename)
            if isinstance(file_entry, dict):
                faces = file_entry.get("faces")
                if isinstance(faces, list):
                    if 0 <= face_index < len(faces):
                        if isinstance(faces[face_index], dict):
                            faces[face_index].update(updates)
                            logger.debug(
                                f"Обновлены данные для лица {face_index} в файле '{filename}': {list(updates.keys())}"
                            )
                            return True
                        else:
                            logger.warning(
                                f"Запись лица {face_index} в файле '{filename}' не словарь."
                            )
                            return False
                    else:
                        logger.warning(
                            f"Некорректный индекс лица {face_index} для обновления '{filename}'."
                        )
                        return False
                else:
                    logger.warning(f"Поле 'faces' в файле '{filename}' не список.")
                    return False
            else:
                logger.error(
                    f"Внутренняя ошибка: запись для файла '{filename}' не словарь."
                )
                return False
        return False

    def add_file_data(
        self, filename: str, file_data: Dict[str, Any], is_portrait: bool
    ) -> None:
        """Добавляет или перезаписывает данные для целого файла."""
        if (
            not isinstance(file_data, dict)
            or "filename" not in file_data
            or "faces" not in file_data
        ):
            logger.error(
                f"Попытка добавить некорректные данные для файла '{filename}'."
            )
            return
        if not isinstance(file_data["faces"], list):
            logger.error(
                f"Попытка добавить некорректные данные для файла '{filename}'. 'faces' не список."
            )
            return

        if is_portrait:
            if filename in self.group_data:
                logger.warning(
                    f"Файл '{filename}' удален из групповых перед добавлением в портретные."
                )
                del self.group_data[filename]
            self.portrait_data[filename] = file_data
            logger.debug(
                f"Добавлены/обновлены портретные данные для файла '{filename}'."
            )
        else:
            if filename in self.portrait_data:
                logger.warning(
                    f"Файл '{filename}' удален из портретных перед добавлением в групповые."
                )
                del self.portrait_data[filename]
            self.group_data[filename] = file_data
            logger.debug(
                f"Добавлены/обновлены групповые данные для файла '{filename}'."
            )

    def clear_data(self, data_type: str = "all") -> None:
        """Очищает внутренние словари данных (portrait_data, group_data)."""
        cleared = False
        if data_type == "portrait" or data_type == "all":
            if self.portrait_data:
                self.portrait_data = {}
                logger.info("Данные портретных файлов очищены из памяти.")
                cleared = True
        if data_type == "group" or data_type == "all":
            if self.group_data:
                self.group_data = {}
                logger.info("Данные групповых файлов очищены из памяти.")
                cleared = True
        if not cleared and data_type not in ["portrait", "group", "all"]:
            logger.warning(f"Неизвестный тип данных '{data_type}' для очистки.")
        elif not cleared:
            logger.debug("Данные для очистки уже были пусты.")

    # get_portrait_filenames_with_children остается без изменений
    def get_portrait_filenames_with_children(self) -> Tuple[List[str], List[str]]:
        filenames = list(self.portrait_data.keys())
        child_names = []
        if not filenames:
            logger.warning("Нет портретных данных для получения имен файлов и детей.")
            return [], []
        logger.debug(f"Получение имен детей для {len(filenames)} портретных файлов...")
        missing_data_count = 0
        for filename in filenames:
            face_data = self.get_face(filename, 0)
            child_name = "N/A"
            if face_data:
                child_name = face_data.get("child_name") or "N/A"
            else:
                missing_data_count += 1
            child_names.append(child_name)
        if missing_data_count > 0:
            logger.warning(
                f"Не найдены данные лица для {missing_data_count} портретных файлов при получении имен детей."
            )
        logger.debug(
            f"Получены имена файлов ({len(filenames)}) и детей ({len(child_names)})."
        )
        return filenames, child_names
