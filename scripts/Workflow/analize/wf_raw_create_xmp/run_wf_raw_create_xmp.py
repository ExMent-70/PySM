# run_wf_raw_create_xmp.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import concurrent.futures
import logging
import os
import pathlib
import sys
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Настройка путей для импорта локальных модулей
try:
    current_script_path = pathlib.Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from _common.json_data_manager import JsonDataManager
    from _common.xmp_editor import XmpEditor
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stdout_handler)
    
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(stderr_handler)

# Константы
ANALYSIS_SUBFOLDER_TEMPLATE = "Output/Analysis_{photo_session}"
IMAGE_SUBFOLDER_TEMPLATE = "Capture/{photo_session}"
PORTRAIT_JSON_FILENAME = "info_portrait_faces.json"
GROUP_JSON_FILENAME = "info_group_faces.json"
TEMPLATE_FILENAME = "template.xmp"

# 2. БЛОК: Структуры данных и Enum
# ==============================================================================
class PhotoType(str, Enum):
    PORTRAIT = "portrait"
    GROUP = "group_photo"

class RawPhotoType(str, Enum):
    PORTRAIT = "portrait"
    GROUP = "group"

class JsonKeys(str, Enum):
    FACES = "faces"
    LOCATION_NAME = "location_name"
    ORIGINAL_BBOX = "original_bbox"
    BBOX = "bbox"
    POSE = "pose"
    CHILD_NAME = "child_name"
    MATCHED_CHILD_NAME = "matched_child_name"
    CLUSTER_LABEL = "cluster_label"
    MATCHED_PORTRAIT_CLUSTER_LABEL = "matched_portrait_cluster_label"
    EMOTION = "emotion_faceonnx"
    GENDER = "gender_faceonnx"
    LEFT_EYE = "left_eye_state"
    RIGHT_EYE = "right_eye_state"
    LANDMARK_2D_106 = "landmark_2d_106"
    LANDMARK_3D_68 = "landmark_3d_68"
    EMBEDDING = "embedding"
    MATCH_DISTANCE = "match_distance"
    GENDER_INSIGHT = "gender_insight"
    AGE_INSIGHT = "age_insight"

class SpecialValues(str, Enum):
    NOISE = "Noise"
    NO_MATCH = "No Match"
    UNKNOWN = "Unknown"
    EYES_OPEN = "Eyes_Open"
    EYES_CLOSED = "Eyes_Closed"
    EYE_STATE_OPEN = "Open"
    EYE_STATE_CLOSED = "Closed"

EXCLUDED_XMP_FIELDS = {
    JsonKeys.EMBEDDING.value, JsonKeys.CHILD_NAME.value, JsonKeys.MATCHED_CHILD_NAME.value,
    JsonKeys.CLUSTER_LABEL.value, JsonKeys.MATCHED_PORTRAIT_CLUSTER_LABEL.value,
    JsonKeys.MATCH_DISTANCE.value, JsonKeys.LANDMARK_3D_68.value,
    JsonKeys.GENDER_INSIGHT.value, JsonKeys.AGE_INSIGHT.value,
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
    parser = argparse.ArgumentParser(description="Creates or updates XMP metadata files based on JSON data.")
    parser.add_argument("--all_threads", type=int, default=os.cpu_count() or 4, help="Number of processing threads.")
    
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

# 4. БЛОК: Класс MetadataProcessor (Бизнес-логика)
# ==============================================================================
class MetadataProcessor:
    """
    Отвечает за подготовку данных для записи в XMP.
    Преобразует JSON-структуры в списки ключевых слов и атрибутов.
    """
    def __init__(self, image_folder: pathlib.Path, template_content: Optional[str]):
        self.image_folder = image_folder
        self.template_content = template_content

    def process_file(
        self,
        image_filename: str,
        file_data: Dict[str, Any],
        landmarks_data: Dict[str, Any],  # Новое: Передаем ландмарки
        photo_type: PhotoType,
        session_name: Optional[str]
    ) -> bool:
        """
        Основной метод обработки одного файла.
        """
        xmp_path = self.image_folder / f"{pathlib.Path(image_filename).stem}.xmp"
        
        # Слияние данных: основные + ландмарки
        merged_file_data = self._merge_landmarks(file_data, landmarks_data)

        # Инициализация редактора XMP (включает загрузку или создание файла)
        editor = XmpEditor(xmp_path, self.template_content)

        # 1. Заполнение базовых полей
        self._set_base_metadata(editor, image_filename, merged_file_data, photo_type, session_name)

        # 2. Обработка лиц (генерация ключевых слов и SubjectCode)
        faces = merged_file_data.get(JsonKeys.FACES.value, [])
        is_portrait = (photo_type == PhotoType.PORTRAIT)
        
        face_keywords, subject_codes, persons = self._extract_face_info(faces, photo_type, is_portrait)

        # 3. Обновление списков (Bag)
        base_keywords = self._get_base_keywords(merged_file_data, photo_type)
        all_keywords = base_keywords.union(face_keywords)

        editor.update_bag("dc", "subject", list(all_keywords), sort=True)
        editor.update_bag("lightroom", "hierarchicalSubject", list(all_keywords), sort=True)
        
        # SubjectCode сохраняем без сортировки, чтобы landmark был в конце
        editor.update_bag("Iptc4xmpCore", "SubjectCode", subject_codes, sort=False)
        
        if is_portrait and persons:
            valid_persons = [p for p in persons if not p.startswith("Кластер_")]
            if not valid_persons: valid_persons = persons
            if valid_persons:
                editor.set_simple_field("photoshop", "TransmissionReference", ", ".join(valid_persons))

        return editor.save()

    def _merge_landmarks(self, file_data: Dict[str, Any], landmarks_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создает копию данных файла и объединяет их с ландмарками для каждого лица.
        """
        if not landmarks_data:
            return file_data
            
        merged = file_data.copy()
        merged_faces = []
        
        land_faces = landmarks_data.get("faces", [])
        
        for i, face in enumerate(file_data.get("faces", [])):
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
        session_name: Optional[str]
    ):
        editor.set_simple_field("photoshop", "Source", "1")
        editor.set_simple_field("photoshop", "Credit", "1")
        editor.set_simple_field("photoshop", "Headline", session_name)
        editor.set_simple_field("GettyImagesGIFT", "OriginalFilename", image_filename)
        editor.set_simple_field("Iptc4xmpCore", "IntellectualGenre", photo_type.value)

        location_name = file_data.get(JsonKeys.LOCATION_NAME.value)
        if isinstance(location_name, str) and location_name.strip():
             editor.set_simple_field("Iptc4xmpCore", "Location", location_name.strip())

        faces = file_data.get(JsonKeys.FACES.value, [])
        if faces and isinstance(faces[0], dict):
            first_face = faces[0]
            bbox = first_face.get(JsonKeys.ORIGINAL_BBOX.value) or first_face.get(JsonKeys.BBOX.value)
            pose = first_face.get(JsonKeys.POSE.value)
            
            if bbox:
                editor.set_simple_field("photoshop", "Instructions", self._format_coordinates(JsonKeys.BBOX.value, bbox))
            if pose:
                editor.set_simple_field("xmpRights", "UsageTerms", self._format_coordinates(JsonKeys.POSE.value, pose))

    def _get_base_keywords(self, file_data: Dict[str, Any], photo_type: PhotoType) -> Set[str]:
        keywords = {photo_type.value}
        location_name = file_data.get(JsonKeys.LOCATION_NAME.value)
        if isinstance(location_name, str) and location_name.strip():
            keywords.add(location_name.strip())
        return keywords

    def _extract_face_info(
        self,
        faces: List[Dict[str, Any]],
        photo_type: PhotoType,
        is_portrait: bool
    ) -> Tuple[Set[str], List[str], List[str]]:
        """Возвращает (Ключевые слова, SubjectCodes, Список имен)."""
        keywords = set()
        subject_codes_final = []
        persons_found = []
        ignore_names = {SpecialValues.NOISE.value, SpecialValues.NO_MATCH.value, SpecialValues.UNKNOWN.value}

        for face_idx, face in enumerate(faces):
            if not isinstance(face, dict): continue

            # 1. Определение имени/кластера
            name = face.get(JsonKeys.CHILD_NAME.value) or face.get(JsonKeys.MATCHED_CHILD_NAME.value)
            cluster = face.get(JsonKeys.CLUSTER_LABEL.value) or face.get(JsonKeys.MATCHED_PORTRAIT_CLUSTER_LABEL.value)
            
            person_identifier = ""
            if name and name not in ignore_names and not name.startswith(SpecialValues.UNKNOWN.value):
                person_identifier = name
            elif cluster is not None:
                try:
                    person_identifier = f"Кластер_{int(cluster):02d}"
                except (ValueError, TypeError):
                    pass
            
            # Сбор данных для SubjectCode
            face_attributes = {'genre': photo_type.value}
            if person_identifier:
                face_attributes['person'] = person_identifier
                keywords.add(person_identifier)
                persons_found.append(person_identifier)
            
            # Рекурсивное уплощение данных лица
            self._flatten_face_data(face, face_attributes)

            # Дополнительные ключевые слова для портретов
            if is_portrait:
                if emotion := face_attributes.get(JsonKeys.EMOTION.value):
                    keywords.add(str(emotion).strip())
                if gender := face_attributes.get(JsonKeys.GENDER.value):
                    keywords.add(str(gender))
                
                left_eye = face_attributes.get(JsonKeys.LEFT_EYE.value)
                right_eye = face_attributes.get(JsonKeys.RIGHT_EYE.value)
                if left_eye == SpecialValues.EYE_STATE_CLOSED.value and right_eye == SpecialValues.EYE_STATE_CLOSED.value:
                    keywords.add(SpecialValues.EYES_CLOSED.value)
                elif left_eye == SpecialValues.EYE_STATE_OPEN.value and right_eye == SpecialValues.EYE_STATE_OPEN.value:
                    keywords.add(SpecialValues.EYES_OPEN.value)

            # Генерация SubjectCode строк
            prefix = f"F{face_idx}"
            codes_for_face = []
            landmark_entry = None

            for key, value in face_attributes.items():
                clean_val = str(value).strip()
                if not clean_val: continue
                
                code = SubjectCode(key=key, value=clean_val, prefix=prefix)
                if key == JsonKeys.LANDMARK_2D_106.value:
                    landmark_entry = str(code)
                else:
                    codes_for_face.append(str(code))
            
            codes_for_face.sort()
            if landmark_entry:
                codes_for_face.append(landmark_entry)
            
            subject_codes_final.extend(codes_for_face)

        return keywords, subject_codes_final, persons_found

    def _flatten_face_data(self, data: Dict[str, Any], target_dict: Dict[str, Any], parent_key: str = ''):
        for k, v in data.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if new_key in EXCLUDED_XMP_FIELDS:
                continue
            
            if isinstance(v, dict):
                self._flatten_face_data(v, target_dict, new_key)
            elif isinstance(v, list):
                formatted = self._format_coordinates(k, v)
                if formatted:
                    target_dict[new_key] = formatted
            else:
                target_dict[new_key] = v

    def _format_coordinates(self, key: str, data: List) -> Optional[str]:
        try:
            if not isinstance(data, list): return str(data)
            p = 3
            if key in (JsonKeys.BBOX.value, JsonKeys.ORIGINAL_BBOX.value):
                return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 4 else None
            if key in ("kps", JsonKeys.LANDMARK_2D_106.value):
                return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f}" for pt in data if len(pt) >= 2)
            if key == JsonKeys.LANDMARK_3D_68.value:
                # В XMP обычно пишут только 2D, но если вдруг, то формат такой же
                return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f},{float(pt[2]):.{p}f}" for pt in data if len(pt) >= 3)
            if key == JsonKeys.POSE.value:
                return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 3 else None
            return str(data)
        except (ValueError, TypeError, IndexError):
            return str(data)


# 5. БЛОК: Функция-оркестратор
# ==============================================================================
def run_xmp_creation(
    json_manager: JsonDataManager,
    image_folder_path: pathlib.Path,
    session_name: Optional[str],
    max_workers: int,
    template_content: Optional[str]
):
    logger.info(f"Запуск создания XMP. Папка: {image_folder_path}")
    
    processor = MetadataProcessor(image_folder_path, template_content)

    all_filenames = json_manager.get_all_filenames("all")
    if not all_filenames:
        logger.info("Нет файлов для обработки.")
        return

    logger.info(f"Обработка {len(all_filenames)} изображений в {max_workers} потоках...")
    
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fname in all_filenames:
            file_info = json_manager.get_data_with_type(fname)
            if file_info:
                file_data, raw_photo_type = file_info
                
                # Получаем ландмарки из менеджера
                data_type_str = "portrait" if raw_photo_type == RawPhotoType.PORTRAIT else "group"
                landmarks_data = json_manager.get_landmarks_data(fname, data_type_str) or {}
                
                photo_type = PhotoType.GROUP if raw_photo_type == RawPhotoType.GROUP else PhotoType.PORTRAIT
                
                future = executor.submit(
                    processor.process_file,
                    fname,
                    file_data,
                    landmarks_data,
                    photo_type,
                    session_name
                )
                futures[future] = fname
            else:
                logger.warning(f"Данные не найдены: {fname}")

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обновление XMP"):
            try:
                if not future.result():
                    errors += 1
            except Exception as e:
                logger.error(f"Ошибка в потоке для {futures[future]}: {e}", exc_info=True)
                errors += 1

    logger.info(f"Завершено. Ошибок: {errors}.")


# 6. БЛОК: Точка входа
# ==============================================================================
def main():
    if not IS_MANAGED_RUN:
        logger.critical("Требуется запуск в среде PySM.")
        sys.exit(1)

    config = get_config()
    template_content = load_template_content(current_script_path)

    session_path_str = pysm_context.get("wf_session_path")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Отсутствуют переменные контекста wf_*.")
        sys.exit(1)

    base_path = pathlib.Path(session_path_str) / session_name
    analysis_path = base_path / ANALYSIS_SUBFOLDER_TEMPLATE.format(photo_session=photo_session)
    image_folder = base_path / IMAGE_SUBFOLDER_TEMPLATE.format(photo_session=photo_session)

    portrait_json = analysis_path / PORTRAIT_JSON_FILENAME
    group_json = analysis_path / GROUP_JSON_FILENAME

    if not portrait_json.exists() or not group_json.exists():
        logger.error(f"JSON файлы не найдены в {analysis_path}")
        sys.exit(1)

    # 1. Загрузка основных данных
    json_manager = JsonDataManager(portrait_json, group_json)
    if not json_manager.load_data():
        logger.error("Ошибка загрузки JSON данных.")
        sys.exit(1)
        
    # 2. Загрузка ландмарков (Тяжелые данные)
    if not json_manager.load_landmarks("all"):
        logger.warning("Ландмарки не загружены или отсутствуют. XMP будут созданы без детальной геометрии.")

    # Запуск
    run_xmp_creation(
        json_manager=json_manager,
        image_folder_path=image_folder,
        session_name=session_name,
        max_workers=config.all_threads,
        template_content=template_content
    )

    if image_folder.exists():
        pysm_context.log_link(str(image_folder), "Открыть папку с XMP-файлами")
    print(" ")

if __name__ == "__main__":
    main()