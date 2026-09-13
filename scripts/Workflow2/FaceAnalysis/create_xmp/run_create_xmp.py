"""Создание и обновление XMP-файлов по данным анализа Workflow2."""

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import concurrent.futures
import logging
import os
import pathlib
import sys
import json
import re
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Настройка путей для импорта проекта и локальных модулей.
try:
    current_script_path = pathlib.Path(__file__).resolve()
    script_dir = current_script_path.parent
    face_analysis_dir = script_dir.parent
    project_root = script_dir.parents[3]
    for import_path in (project_root, face_analysis_dir, script_dir):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from _common.xmp_editor import XmpEditor
    from _lib.student_roster import (
        StudentRoster,
        load_student_roster,
        normalize_student_id,
    )
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ИМПОРТА: {e}", file=sys.stderr)
    sys.exit(1)

# Попытка импорта PySM контекста
try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    try:
        from tqdm import tqdm
    except ImportError:
        class TqdmMock:
            def __init__(self, iterable, *args, **kwargs): self.iterable = iterable
            def __iter__(self): return iter(self.iterable)
            @staticmethod
            def write(msg, *args, **kwargs): print(msg)
        tqdm = TqdmMock

# Настройка логгера
class _MaxLevelFilter(logging.Filter):
    """Пропускает в обработчик сообщения не выше заданного уровня."""

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stdout_handler)
    
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(stderr_handler)

# Константы Unified Storage
PYSM_PREFIX = "PySM_"
SCRIPT_LOG_TITLE = "СОЗДАНИЕ XMP-ФАЙЛОВ"
IGNORED_SCAN_DIRECTORIES = frozenset({"backup", "xmp"})
FACES_JSON_FILENAME = "info_faces.json"
LANDMARKS_JSON_FILENAME = "info_faces_landmarks.json"
TEMPLATE_FILENAME = "template.xmp"

# 2. БЛОК: Структуры данных и Enum
# ==============================================================================
class PhotoType(str, Enum):
    PORTRAIT = "portrait"
    GROUP = "group_photo"

class JsonKeys(str, Enum):
    FACES = "faces"
    LOCATION_NAME = "location_name"
    ORIGINAL_BBOX = "original_bbox"
    ORIGINAL_SHAPE = "original_shape"
    BBOX = "bbox"
    POSE = "pose"
    CHILD_NAME = "child_name"
    MATCHED_CHILD_NAME = "matched_child_name"
    TEMP_CHILD_NAME = "temp_child_name"
    STUDENT_ID = "student_id"
    CLUSTER_LABEL = "cluster_label"
    MATCHED_PORTRAIT_CLUSTER_LABEL = "matched_portrait_cluster_label"
    EMOTION = "emotion_faceonnx"
    GENDER = "gender_faceonnx"
    EYE_LEFT = "eye_left_state"
    EYE_RIGHT = "eye_right_state"
    KEYPOINT_ANALYSIS = "keypoint_analysis"
    MOUTH_STATE = "mouth_state"
    LANDMARK_2D_106 = "landmark_2d_106"
    LANDMARK_3D_68 = "landmark_3d_68"
    EMBEDDING = "embedding"
    MATCH_DISTANCE = "match_distance"
    GENDER_INSIGHT = "gender_insight"
    AGE_INSIGHT = "age_insight"
    FACE_COUNT = "face_count"

class SpecialValues(str, Enum):
    EYES_OPEN = "Eyes_Open"
    EYES_CLOSED = "Eyes_Closed"
    EYE_STATE_OPEN = "Open"
    EYE_STATE_CLOSED = "Closed"

EXCLUDED_XMP_FIELDS = {
    JsonKeys.EMBEDDING.value, JsonKeys.CHILD_NAME.value, JsonKeys.MATCHED_CHILD_NAME.value,
    JsonKeys.CLUSTER_LABEL.value, JsonKeys.MATCHED_PORTRAIT_CLUSTER_LABEL.value,
    JsonKeys.MATCH_DISTANCE.value, JsonKeys.LANDMARK_3D_68.value,
    JsonKeys.GENDER_INSIGHT.value, JsonKeys.AGE_INSIGHT.value,
    JsonKeys.STUDENT_ID.value,
}

@dataclass(frozen=True)
class SubjectCode:
    """Структура для представления технического кода Iptc4xmpCore:SubjectCode."""
    key: str
    value: str
    prefix: str

    def __str__(self) -> str:
        return f"{self.prefix}_{self.key}:{self.value}"

# 3. БЛОК: Конфигурация и загрузка шаблона
# ==============================================================================
def get_config() -> Namespace:
    """Определяет CLI-параметры и возвращает конфигурацию запуска."""

    parser = argparse.ArgumentParser(description="Creates or updates XMP metadata files based on JSON data.")
    
    parser.add_argument(
        "--all_threads",
        type=int,
        default=12,
        help="Number of processing threads.",
    )
    
    parser.add_argument(
        "--analysis_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing info_faces.json."
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing images and where XMPs will be saved."
    )
    parser.add_argument(
        "--student_list_file",
        type=str,
        required=False,
        default=None,
        help=(
            "Optional path to the *.list file used as the source of student "
            "names. Names are omitted when the file is unavailable."
        ),
    )

    parser.add_argument(
        "--landmark_enable", 
        action="store_true",
        help="Enable writing 2D landmarks to XMP."
    )

    # Новые аргументы для улучшенной работы с папками
    parser.add_argument(
        "--scan_folder_mode", 
        action="store_true",
        help=(
            "Рекурсивно сканировать image_dir и сопоставлять файлы с JSON "
            "по последней группе цифр в основе имени."
        )
    )
    parser.add_argument(
        "--xmp_subfolder", 
        action="store_true",
        help="Сохранять XMP файлы в подпапке 'XMP' относительно целевого изображения."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def load_template_content(script_path: pathlib.Path) -> Optional[str]:
    template_path = script_path.parent / TEMPLATE_FILENAME
    if template_path.is_file():
        try:
            return template_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Ошибка чтения шаблона {template_path}: {e}")
            return None
    else:
        logger.warning(f"Шаблон {TEMPLATE_FILENAME} не найден. Будет использован встроенный базовый шаблон.")
        return None

# ==============================================================================
# 4. БЛОК: Класс MetadataProcessor (Бизнес-логика)
# ==============================================================================
class MetadataProcessor:
    """
    Отвечает за подготовку данных для записи в XMP.
    Преобразует JSON-структуры в списки ключевых слов и атрибутов.
    """
    def __init__(self, template_content: Optional[str], landmark_enable: bool,
                 student_roster: Optional[StudentRoster]):
        self.template_content = template_content
        self.landmark_enable = landmark_enable
        self.student_roster = student_roster

    def process_file(
        self,
        xmp_path: pathlib.Path,
        image_filename: str,
        file_data: Dict[str, Any],
        landmarks_data: Dict[str, Any],
        photo_type: PhotoType,
        session_name: Optional[str],
        photo_session: Optional[str] = None,
    ) -> bool:
        """
        Основной метод обработки одного файла.
        """
        # Слияние данных: основные + ландмарки (если они были загружены)
        merged_file_data = self._merge_landmarks(file_data, landmarks_data)

        # Инициализация редактора XMP (включает загрузку или создание файла)
        editor = XmpEditor(xmp_path, self.template_content)

        # 1. Заполнение базовых полей
        self._set_base_metadata(
            editor,
            image_filename,
            merged_file_data,
            photo_type,
            session_name,
            photo_session,
        )

        # 2. Обработка лиц (генерация ключевых слов и SubjectCode)
        faces = merged_file_data.get(JsonKeys.FACES.value, list())
        is_portrait = (photo_type == PhotoType.PORTRAIT)
        
        face_keywords, subject_codes, persons = self._extract_face_info(
            faces, photo_type, is_portrait
        )

        # Добавление глобальных данных изображения в SubjectCode
        if original_shape := merged_file_data.get(JsonKeys.ORIGINAL_SHAPE.value):
            formatted_shape = self._format_coordinates(JsonKeys.ORIGINAL_SHAPE.value, original_shape)
            if formatted_shape:
                shape_code = SubjectCode(
                    key=JsonKeys.ORIGINAL_SHAPE.value, 
                    value=formatted_shape, 
                    prefix="F0" 
                )
                subject_codes.insert(0, str(shape_code))

        # 3. Обновление списков (Bag)
        base_keywords = self._get_base_keywords(merged_file_data, photo_type)
        all_keywords = base_keywords.union(face_keywords)

        editor.update_bag("dc", "subject", list(all_keywords), sort=True)
        editor.update_bag("lightroom", "hierarchicalSubject", list(all_keywords), sort=True)
        editor.update_bag("Iptc4xmpCore", "SubjectCode", subject_codes, sort=False)

        transmission_reference = ", ".join(persons) if is_portrait else ""
        editor.set_simple_field(
            "photoshop", "TransmissionReference", transmission_reference
        )

        return editor.save()

    def _merge_landmarks(self, file_data: Dict[str, Any], landmarks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создает копию данных файла и безопасно объединяет их с ландмарками для каждого лица."""
        if not landmarks_data:
            return file_data
        
        merged = file_data.copy()
        merged_faces = list()
        land_faces = landmarks_data.get("faces", list())
        
        for i, face in enumerate(file_data.get("faces", list())):
            face_merged = face.copy()
            if i < len(land_faces):
                face_merged.update(land_faces[i])
            merged_faces.append(face_merged)
        merged["faces"] = merged_faces
        return merged

    def _set_base_metadata(
        self,
        editor: XmpEditor,
        image_filename: str,
        file_data: Dict[str, Any],
        photo_type: PhotoType,
        session_name: Optional[str],
        photo_session: Optional[str],
    ) -> None:
        """Устанавливает базовые глобальные метаданные для всего изображения."""
        editor.set_simple_field("photoshop", "Source", "1")
        editor.set_simple_field("photoshop", "Credit", "1")
        editor.set_simple_field("photoshop", "Headline", session_name)
        editor.set_simple_field("photoshop", "Category", photo_session)
        editor.set_simple_field("GettyImagesGIFT", "OriginalFilename", image_filename)
        editor.set_simple_field("Iptc4xmpCore", "IntellectualGenre", photo_type.value)

        location_name = file_data.get(JsonKeys.LOCATION_NAME.value)
        if isinstance(location_name, str) and location_name.strip():
            editor.set_simple_field(
                "Iptc4xmpCore", "Location", location_name.strip()
            )

        faces = file_data.get(JsonKeys.FACES.value, list())
        if faces and isinstance(faces[0], dict):
            first_face = faces[0]
            bbox = first_face.get(JsonKeys.ORIGINAL_BBOX.value) or first_face.get(JsonKeys.BBOX.value)
            pose = first_face.get(JsonKeys.POSE.value)
            
            if bbox:
                editor.set_simple_field("photoshop", "Instructions", self._format_coordinates(JsonKeys.BBOX.value, bbox))
            if pose:
                editor.set_localized_text(
                    "xmpRights",
                    "UsageTerms",
                    self._format_coordinates(JsonKeys.POSE.value, pose),
                )

    def _get_base_keywords(self, file_data: Dict[str, Any], photo_type: PhotoType) -> Set[str]:
        """Формирует базовый набор ключевых слов (жанр, локация)."""
        keywords = {f"{PYSM_PREFIX}GENRE_{photo_type.value}"}
        location_name = file_data.get(JsonKeys.LOCATION_NAME.value)
        if isinstance(location_name, str) and location_name.strip():
            keywords.add(f"{PYSM_PREFIX}LOCATION_{location_name.strip()}")
        return keywords

    def _identify_person(self, face: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Возвращает проверенный ``student_id`` и доступное ФИО из реестра."""

        raw_student_id = face.get(JsonKeys.STUDENT_ID.value)
        if raw_student_id is None or not str(raw_student_id).strip():
            return None, ""
        expected_list_id = (
            self.student_roster.list_id if self.student_roster is not None else None
        )
        student_id = normalize_student_id(raw_student_id, expected_list_id)
        if self.student_roster is None:
            return student_id, ""
        return student_id, self.student_roster.name_for(student_id)

    def _extract_portrait_keywords(self, face_attributes: Dict[str, Any]) -> Set[str]:
        """
        Анализирует атрибуты лица и генерирует ключевые слова, специфичные для портретов
        (эмоции, пол, состояние глаз и рта).
        """
        keywords = set()
        
        if emotion := face_attributes.get(JsonKeys.EMOTION.value):
            keywords.add(f"{PYSM_PREFIX}EMOTION_{str(emotion).strip()}")
            
        if gender := face_attributes.get(JsonKeys.GENDER.value):
            keywords.add(f"{PYSM_PREFIX}GENDER_{str(gender)}")
            
        eye_left = face_attributes.get(JsonKeys.EYE_LEFT.value)
        eye_right = face_attributes.get(JsonKeys.EYE_RIGHT.value)
        if eye_left == SpecialValues.EYE_STATE_CLOSED.value and eye_right == SpecialValues.EYE_STATE_CLOSED.value:
            keywords.add(f"{PYSM_PREFIX}EYE_{SpecialValues.EYES_CLOSED.value}")
        elif eye_left == SpecialValues.EYE_STATE_OPEN.value and eye_right == SpecialValues.EYE_STATE_OPEN.value:
            keywords.add(f"{PYSM_PREFIX}EYE_{SpecialValues.EYES_OPEN.value}")
            
        flattened_mouth_key = f"{JsonKeys.KEYPOINT_ANALYSIS.value}_{JsonKeys.MOUTH_STATE.value}"
        if mouth_state := face_attributes.get(flattened_mouth_key):
            keywords.add(f"{PYSM_PREFIX}MOUTH_{str(mouth_state).strip()}")
            
        return keywords

    def _generate_subject_codes(self, face_attributes: Dict[str, Any], face_idx: int) -> List[str]:
        """
        Преобразует плоский словарь атрибутов лица в список строк Iptc4xmpCore:SubjectCode.
        Ландмарки (если включены) добавляются в конец списка для сохранения предсказуемого порядка.
        """
        prefix = f"F{face_idx}"
        codes_for_face = list()
        landmark_entry = None

        for key, value in face_attributes.items():
            clean_val = str(value).strip()
            if not clean_val: 
                continue
            
            code = SubjectCode(key=key, value=clean_val, prefix=prefix)
            
            if key == JsonKeys.LANDMARK_2D_106.value:
                if self.landmark_enable:
                    landmark_entry = str(code)
            else:
                codes_for_face.append(str(code))
        
        # Сортируем все атрибуты по алфавиту для красоты в XML, кроме тяжелых ландмарков
        codes_for_face.sort()
        if landmark_entry:
            codes_for_face.append(landmark_entry)
            
        return codes_for_face

    def _extract_face_info(
        self,
        faces: List[Dict[str, Any]],
        photo_type: PhotoType,
        is_portrait: bool
    ) -> Tuple[Set[str], List[str], List[str]]:
        """
        Главный оркестратор сбора данных о лицах.
        Возвращает ключевые слова, SubjectCode и найденные ФИО.
        """
        keywords = set()
        subject_codes_final = list()
        persons_found = list()

        for face_idx, face in enumerate(faces):
            if not isinstance(face, dict): 
                continue

            student_id, person_identifier = self._identify_person(face)
            
            # Базовый словарь атрибутов для формирования XMP-тегов
            face_attributes = {'genre': photo_type.value}
            
            if person_identifier:
                face_attributes['person'] = person_identifier
                keywords.add(f"{PYSM_PREFIX}PERSON_{person_identifier}")
                persons_found.append(person_identifier)

            raw_temp_child_name = face.get(JsonKeys.TEMP_CHILD_NAME.value)
            if raw_temp_child_name is not None:
                temp_child_name = str(raw_temp_child_name).strip()
                if temp_child_name:
                    keywords.add(f"{PYSM_PREFIX}{temp_child_name}")
            
            # Разворачиваем вложенные словари (чистая функция, без мутации)
            flat_data = self._flatten_face_data(face)
            face_attributes.update(flat_data)

            if is_portrait:
                portrait_keywords = self._extract_portrait_keywords(face_attributes)
                keywords.update(portrait_keywords)

            # Формируем и добавляем SubjectCodes
            face_codes = self._generate_subject_codes(face_attributes, face_idx)
            if student_id:
                face_codes.append(f"F{face_idx}:{student_id}")
            subject_codes_final.extend(face_codes)

        return keywords, subject_codes_final, persons_found

    def _flatten_face_data(self, data: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Рекурсивно разворачивает вложенные словари JSON в плоский вид.
        Чистая функция (Pure Function): возвращает новый словарь, не изменяя исходный.
        """
        result = dict()
        for k, v in data.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            
            if new_key in EXCLUDED_XMP_FIELDS:
                continue
            
            if isinstance(v, dict):
                # Рекурсивный вызов и слияние результатов
                result.update(self._flatten_face_data(v, new_key))
            elif isinstance(v, list):
                formatted = self._format_coordinates(k, v)
                if formatted is not None:
                    result[new_key] = formatted
            else:
                result[new_key] = v
                
        return result

    def _format_coordinates(self, key: str, data: List[Any]) -> Optional[str]:
        """Форматирует списки координат и bounding box'ов в строковое представление."""
        try:
            if not isinstance(data, list): 
                return str(data)
            p = 3
            if key == JsonKeys.ORIGINAL_SHAPE.value:
                return ",".join(str(x) for x in data)
            
            if key in (JsonKeys.BBOX.value, JsonKeys.ORIGINAL_BBOX.value):
                return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 4 else None
            if key in ("kps", JsonKeys.LANDMARK_2D_106.value):
                return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f}" for pt in data if len(pt) >= 2)
            if key == JsonKeys.LANDMARK_3D_68.value:
                return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f},{float(pt[2]):.{p}f}" for pt in data if len(pt) >= 3)
            if key == JsonKeys.POSE.value:
                return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 3 else None
            return str(data)
        except (ValueError, TypeError, IndexError):
            return str(data)

# 5. БЛОК: Функция-оркестратор и Хелперы
# ==============================================================================
def extract_digits(filename: str) -> str:
    """Извлекает последнюю непрерывную группу цифр из основы имени файла."""

    matches = re.findall(r"\d+", pathlib.Path(filename).stem)
    return matches[-1] if matches else ""


def _is_in_ignored_scan_directory(
    file_path: pathlib.Path,
    image_folder_path: pathlib.Path,
) -> bool:
    """Проверяет, находится ли файл в служебной папке XMP или backup."""

    try:
        relative_parts = file_path.relative_to(image_folder_path).parts[:-1]
    except ValueError:
        return False
    return any(
        part.casefold() in IGNORED_SCAN_DIRECTORIES
        for part in relative_parts
    )


def _has_non_noise_identity_cluster(face: Dict[str, Any]) -> bool:
    """Проверяет, должно ли лицо уже иметь student_id."""

    for key in (
        JsonKeys.CLUSTER_LABEL.value,
        JsonKeys.MATCHED_PORTRAIT_CLUSTER_LABEL.value,
    ):
        value = face.get(key)
        if value is None:
            continue
        try:
            if int(value) >= 0:
                return True
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Поле {key} содержит неверное значение {value!r}.") from exc
    return False


def validate_xmp_tasks(tasks: List[Tuple[pathlib.Path, str, Dict[str, Any], str]],
                       student_roster: Optional[StudentRoster]) -> None:
    """Проверяет идентичность всех задач до создания первого XMP."""

    output_sources: Dict[str, str] = {}
    for xmp_path, json_key, file_data, _image_filename in tasks:
        output_key = os.path.normcase(os.path.abspath(xmp_path))
        previous_source = output_sources.get(output_key)
        if previous_source is not None:
            raise ValueError(
                f"Один XMP-файл {xmp_path} назначен нескольким источникам: "
                f"{previous_source!r} и {json_key!r}."
            )
        output_sources[output_key] = json_key

        faces = file_data.get(JsonKeys.FACES.value, list())
        if not isinstance(faces, list):
            raise ValueError(f"{json_key}: поле faces должно быть массивом.")
        for face_index, face in enumerate(faces):
            if not isinstance(face, dict):
                raise ValueError(
                    f"{json_key}: faces[{face_index}] должно быть JSON-объектом."
                )
            raw_student_id = face.get(JsonKeys.STUDENT_ID.value)
            if raw_student_id is not None and str(raw_student_id).strip():
                try:
                    expected_list_id = (
                        student_roster.list_id
                        if student_roster is not None
                        else None
                    )
                    student_id = normalize_student_id(
                        raw_student_id, expected_list_id
                    )
                    if student_roster is not None:
                        student_roster.name_for(student_id)
                except ValueError as exc:
                    raise ValueError(
                        f"{json_key}, лицо {face_index}: {exc}"
                    ) from exc
            else:
                try:
                    requires_identity = _has_non_noise_identity_cluster(face)
                except ValueError as exc:
                    raise ValueError(
                        f"{json_key}, лицо {face_index}: {exc}"
                    ) from exc
                if requires_identity:
                    raise ValueError(
                        f"{json_key}, лицо {face_index}: идентифицированный "
                        "портретный/matched-кластер не содержит student_id. "
                        "Исправьте данные в cluster_editor."
                    )


def run_xmp_creation(
    faces_data: Dict[str, Any],
    landmarks_data_full: Dict[str, Any],
    image_folder_path: pathlib.Path,
    session_name: Optional[str],
    max_workers: int,
    template_content: Optional[str],
    landmark_enable: bool,
    scan_folder_mode: bool,
    xmp_subfolder: bool,
    student_roster: Optional[StudentRoster],
    photo_session: Optional[str] = None,
):
    logger.debug(f"ℹ️ Целевая папка (корневая): {image_folder_path}")
    logger.info(f"ℹ️ Сохранение Landmark 2D: {'✅' if landmark_enable else '❌'}")
    
    processor = MetadataProcessor(
        template_content, landmark_enable, student_roster
    )
    tasks = list()  # Список кортежей: (путь_к_XMP, оригинальный_ключ_JSON, данные_файла, имя_изображения_для_метаданных)

    if scan_folder_mode:
        logger.info("ℹ️ Режим 2: Рекурсивное сканирование папки и сопоставление по цифрам.")
        
        # 1. Строим цифровой индекс из JSON
        numeric_index = dict()
        for json_key, file_data in faces_data.items():
            digits = extract_digits(json_key)
            if digits:
                previous = numeric_index.get(digits)
                if previous is not None:
                    raise ValueError(
                        "Неоднозначный номер кадра "
                        f"{digits!r} в info_faces.json: "
                        f"{previous[0]!r} и {json_key!r}."
                    )
                numeric_index[digits] = (json_key, file_data)
        
        # 2. Сканируем папку
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".psd", ".psb"}
        for file_path in image_folder_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in valid_extensions
                and not _is_in_ignored_scan_directory(
                    file_path, image_folder_path
                )
            ):
                digits = extract_digits(file_path.name)
                
                if digits and digits in numeric_index:
                    orig_key, file_data = numeric_index[digits]
                    
                    if xmp_subfolder:
                        xmp_path = file_path.parent / "XMP" / f"{file_path.stem}.xmp"
                    else:
                        xmp_path = file_path.parent / f"{file_path.stem}.xmp"
                        
                    tasks.append((xmp_path, orig_key, file_data, file_path.name))
                    
    else:
        logger.info("ℹ️ Режим 1: Генерация XMP по списку из JSON.")
        for fname, file_data in faces_data.items():
            if xmp_subfolder:
                xmp_path = image_folder_path / "XMP" / f"{pathlib.Path(fname).stem}.xmp"
            else:
                xmp_path = image_folder_path / f"{pathlib.Path(fname).stem}.xmp"
                
            tasks.append((xmp_path, fname, file_data, fname))

    if not tasks:
        logger.info("Нет файлов для обработки.")
        return

    validate_xmp_tasks(tasks, student_roster)
    if student_roster is None:
        logger.info(
            f"ℹ️ Проверен формат student_id для <b>{len(tasks)}</b> задач; "
            "список учеников не используется."
        )
    else:
        logger.info(
            f"ℹ️ Проверены student_id для <b>{len(tasks)}</b> задач; "
            f"list_id=<b>{student_roster.list_id}</b>."
        )

    logger.info(f"ℹ️ Сформировано задач: <b>{len(tasks)}</b>. Обработка в <b>{max_workers}</b> потоках...")
    
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = dict()
        for xmp_path, orig_key, file_data, real_image_filename in tasks:
            
            face_count = file_data.get(JsonKeys.FACE_COUNT.value, len(file_data.get("faces", list())))
            photo_type = PhotoType.PORTRAIT if face_count == 1 else PhotoType.GROUP
            file_landmarks = landmarks_data_full.get(orig_key, dict())
            
            future = executor.submit(
                processor.process_file,
                xmp_path,
                real_image_filename,
                file_data,
                file_landmarks,
                photo_type,
                session_name,
                photo_session,
            )
            futures[future] = real_image_filename

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обновление XMP"):
            try:
                if not future.result():
                    errors += 1
            except Exception as e:
                logger.error(f"❌ Ошибка в потоке для {futures[future]}: {e}", exc_info=True)
                errors += 1
                
    if errors > 0:
        raise RuntimeError(
            f"Ошибок при сохранении XMP-файлов: {errors}."
        )
    else:
        logger.info("\n")


# 6. БЛОК: Точка входа
# ==============================================================================
def load_optional_student_roster(
    raw_path: Optional[str],
) -> Optional[StudentRoster]:
    """Загружает список учеников либо сообщает, что ФИО будут пропущены."""

    path_value = str(raw_path or "").strip()
    if not path_value:
        logger.warning(
            "⚠️ Параметр --student_list_file не указан. "
            "Фамилии и имена учеников не будут записаны в XMP."
        )
        return None

    student_list_path = pathlib.Path(path_value)
    if not student_list_path.is_file():
        logger.warning(
            f"⚠️ Файл списка учеников не найден: {student_list_path}. "
            "Фамилии и имена учеников не будут записаны в XMP."
        )
        return None

    student_roster = load_student_roster(student_list_path)
    logger.info(
        f"ℹ️ Загружен список учеников: <i>{student_roster.path}</i>, "
        f"list_id=<b>{student_roster.list_id}</b>, "
        f"записей=<b>{len(student_roster.students)}</b>."
    )
    return student_roster


def main():
    config = get_config()
    logger.info(f"<b>{SCRIPT_LOG_TITLE}</b><br>")

    if not IS_MANAGED_RUN:
        logger.critical("❌ Требуется запуск в среде PySM.")
        sys.exit(1)

    template_content = load_template_content(current_script_path)
    
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")
    
    analysis_path = pathlib.Path(config.analysis_dir)
    image_folder = pathlib.Path(config.image_dir)

    if not analysis_path.exists():
        logger.error(f"❌ Папка анализа не найдена: {analysis_path}")
        sys.exit(1)

    try:
        student_roster = load_optional_student_roster(config.student_list_file)
    except Exception as exc:
        logger.critical(f"❌ Ошибка списка учеников: {exc}")
        sys.exit(1)
    
    # В классическом режиме (генерация в папку) создаем ее, если нет
    if not config.scan_folder_mode and not image_folder.exists():
        try:
            image_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"ℹ️ Папка назначения не найдена. Создана новая: {image_folder}")
        except Exception as e:
            logger.critical(f"❌ Не удалось создать папку назначения {image_folder}: {e}")
            sys.exit(1)
    # В режиме сканирования папка обязательно должна существовать
    elif config.scan_folder_mode and not image_folder.exists():
        logger.error(f"❌ Папка для сканирования не найдена: {image_folder}")
        sys.exit(1)

    # 1. Загрузка основного JSON
    faces_json_path = analysis_path / FACES_JSON_FILENAME
    if not faces_json_path.exists():
        logger.error(f"❌ Файл {FACES_JSON_FILENAME} не найден в {analysis_path}")
        sys.exit(1)

    try:
        with open(faces_json_path, 'r', encoding='utf-8') as f:
            faces_data = json.load(f)
        logger.info(f"ℹ️ Загружено <b>{len(faces_data)}</b> записей из {FACES_JSON_FILENAME}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {FACES_JSON_FILENAME}: {e}")
        sys.exit(1)
        
    # 2. Условная загрузка ландмарков
    landmarks_data = dict()
    if config.landmark_enable:
        landmarks_json_path = analysis_path / LANDMARKS_JSON_FILENAME
        if landmarks_json_path.exists():
            try:
                with open(landmarks_json_path, 'r', encoding='utf-8') as f:
                    landmarks_data = json.load(f)
                logger.info(f"ℹ️ Загружено ландмарков для <b>{len(landmarks_data)}</b> файлов.")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения ландмарков: {e}")
        else:
            logger.warning(f"⚠️ Файл ландмарков не найден: {landmarks_json_path}")
    else:
        logger.debug("⚠️ Загрузка ландмарков пропущена (настройка отключена).")

    # Запуск
    try:
        run_xmp_creation(
            faces_data=faces_data,
            landmarks_data_full=landmarks_data,
            image_folder_path=image_folder,
            session_name=session_name,
            max_workers=config.all_threads or (os.cpu_count() or 4),
            template_content=template_content,
            landmark_enable=config.landmark_enable,
            scan_folder_mode=config.scan_folder_mode,
            xmp_subfolder=config.xmp_subfolder,
            student_roster=student_roster,
            photo_session=photo_session,
        )
    except (ValueError, RuntimeError) as exc:
        logger.critical(f"❌ Создание XMP остановлено: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
