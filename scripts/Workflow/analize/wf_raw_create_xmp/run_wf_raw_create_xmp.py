# 1. БЛОК: Импорты и глобальные константы
# ==============================================================================
import argparse
import concurrent.futures
import logging
import os
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET
from argparse import Namespace
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Set, Union

print(f"<b>ЗАГРУЗКА ДАННЫХ И БИБЛИОТЕК...</b>")

try:
    current_script_path = pathlib.Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from _common.json_data_manager import JsonDataManager
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать JsonDataManager. {e}", file=sys.stderr)
    sys.exit(1)

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    print("PySM INFO: PySM libraries not found. Running in standalone mode.", file=sys.stderr)
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    try:
        from tqdm import tqdm
    except ImportError:
        print("PySM WARNING: tqdm library not found. Progress bar will not be shown.", file=sys.stderr)
        class TqdmMock:
            def __init__(self, iterable, *args, **kwargs): self.iterable = iterable
            def __iter__(self): return iter(self.iterable)
            @staticmethod
            def write(msg, *args, **kwargs): print(msg)
            def set_postfix(self, *args, **kwargs): pass
        tqdm = TqdmMock

logger = logging.getLogger()
logger.setLevel(logging.INFO) 
if logger.hasHandlers():
    logger.handlers.clear()

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO
stdout_handler.addFilter(InfoFilter())
stdout_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(stderr_handler)

ANALYSIS_SUBFOLDER_TEMPLATE = "Output/Analysis_{photo_session}"
IMAGE_SUBFOLDER_TEMPLATE = "Capture/{photo_session}"
PORTRAIT_JSON_FILENAME = "info_portrait_faces.json"
GROUP_JSON_FILENAME = "info_group_faces.json"
TEMPLATE_FILENAME = "template.xmp"
_template_content: Optional[str] = None

EXCLUDED_XMP_FIELDS = [
    "embedding", "child_name", "matched_child_name", "cluster_label",
    "matched_portrait_cluster_label", "match_distance", "landmark_3d_68",
    "gender_insight", "age_insight",
]

# 2. БЛОК: Инкапсулированные сообщения, пространства имен и ЗАГРУЗКА ШАБЛОНА
# ==============================================================================
MESSAGES = {
    "ERROR_SAVING_XMP": "Ошибка сохранения XMP для {file_path}: {exc}",
    "INFO_XMP_UPDATE_START": "Запуск создания/обновления XMP-файлов",
    "INFO_XMP_UPDATE_COMPLETE": "Создание/обновление XMP файлов завершено. Ошибок: {errors}.",
}

def get_message(key: str, **kwargs) -> str:
    message = MESSAGES.get(key, f"[Сообщение '{key}' не найдено]")
    try:
        return message.format(**kwargs)
    except KeyError:
        return message

NS = {
    "x": "adobe:ns:meta/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/", "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/", "lightroom": "http://ns.adobe.com/lightroom/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
}
for prefix, uri in NS.items():
    try:
        ET.register_namespace(prefix, uri)
    except ValueError:
        pass

try:
    _template_path = pathlib.Path(__file__).parent / TEMPLATE_FILENAME
    if not _template_path.is_file():
        logger.error(f"XMP Template file '{TEMPLATE_FILENAME}' не найден в {pathlib.Path(__file__).parent.resolve()}")
        _template_content = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="python-xmp-utils"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:lightroom="http://ns.adobe.com/lightroom/1.0/" xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"><dc:subject><rdf:Bag/></dc:subject><lightroom:hierarchicalSubject><rdf:Bag/></lightroom:hierarchicalSubject><Iptc4xmpCore:SubjectCode><rdf:Bag/></Iptc4xmpCore:SubjectCode></rdf:Description></rdf:RDF></x:xmpmeta>"""
        logger.warning("Используется базовый XMP шаблон.")
    else:
        _template_content = _template_path.read_text(encoding="utf-8")
        logger.info(f"Загружен XMP шаблон: {_template_path.resolve()}")
except Exception as e:
    logger.error(f"Критическая ошибка загрузки XMP шаблона '{TEMPLATE_FILENAME}': {e}")
    _template_content = None

# 3. БЛОК: Конфигурация скрипта
# ==============================================================================
def get_config() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Creates or updates XMP metadata files based on JSON data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--all_threads", type=int, default=os.cpu_count() or 4, help="Number of processing threads.")

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()
        
# 4. БЛОК: Класс XmpManager
# ==============================================================================
class XmpManager:
    def __init__(self, image_folder_path: pathlib.Path):
        self.image_folder_path = image_folder_path
        self.excluded_xmp_fields: Set[str] = set(EXCLUDED_XMP_FIELDS)
        logger.info(f"<b>Эти поля НЕ будут сохранены в XMP-файл:</b>")
        logger.info(f"{sorted(list(self.excluded_xmp_fields))}")

    def _get_xmp_path(self, image_filename: str) -> pathlib.Path:
        return self.image_folder_path / f"{pathlib.Path(image_filename).stem}.xmp"

    def _parse_xmp(self, xmp_path: pathlib.Path) -> Optional[ET.ElementTree]:
        if xmp_path.exists() and xmp_path.stat().st_size > 0:
            try:
                with xmp_path.open("r", encoding="utf-8") as f: tree = ET.parse(f)
                return tree
            except Exception as e:
                logger.warning(f"Ошибка парсинга XMP {xmp_path}: {e}. Создание нового.")
                return self._create_tree_from_template()
        else:
            return self._create_tree_from_template()

    def _create_tree_from_template(self) -> Optional[ET.ElementTree]:
        if _template_content is None:
            logger.error("XMP шаблон не загружен.")
            return None
        try:
            return ET.ElementTree(ET.fromstring(_template_content))
        except Exception as e:
            logger.error(f"Ошибка создания дерева из шаблона: {e}")
            return None

    def _find_or_create(self, parent: ET.Element, tag_name: str, ns_prefix: str) -> Optional[ET.Element]:
        ns_uri = NS.get(ns_prefix)
        if not ns_uri: return None
        element = parent.find(f"./{ns_prefix}:{tag_name}", namespaces=NS)
        if element is None:
            element = ET.SubElement(parent, ET.QName(ns_uri, tag_name))
        return element

    def _add_keywords_to_bag(self, tree: ET.ElementTree, ns_prefix: str, tag_name: str, keywords: List[str]) -> bool:
        description = tree.getroot().find(".//rdf:Description", NS)
        if description is None: return False
        
        unique_keywords = sorted(list(set(kw.strip() for kw in keywords if kw and kw.strip())))
        
        container_element = self._find_or_create(description, tag_name, ns_prefix)
        if container_element is None: return False
        bag_element = self._find_or_create(container_element, "Bag", "rdf")
        if bag_element is None: return False
        
        for li in bag_element.findall(f"{{{NS['rdf']}}}li"): bag_element.remove(li)
        for kw in unique_keywords: ET.SubElement(bag_element, ET.QName(NS["rdf"], "li")).text = kw
        return True
    
    def _update_keyword_sections(self, tree: ET.ElementTree, keywords: List[str]) -> bool:
        sections_to_update = [("dc", "subject"), ("lightroom", "hierarchicalSubject")]
        success = True
        for ns_prefix, tag_name in sections_to_update:
            if not self._add_keywords_to_bag(tree, ns_prefix, tag_name, keywords):
                success = False
        return success

    def _update_text_field(self, description: ET.Element, tag_name: str, ns_prefix: str, text: Optional[str]):
        if text is None: return
        element = self._find_or_create(description, tag_name, ns_prefix)
        if element is None: return
        container = element.find(f"{{{NS['rdf']}}}Alt") or element.find(f"{{{NS['rdf']}}}Seq")
        if container is not None:
            li = container.find(f"{{{NS['rdf']}}}li") or self._find_or_create(container, "li", "rdf")
            if li is not None: li.text = str(text)
        else:
            element.text = str(text)

    def _format_coordinates(self, key: str, data: List) -> Optional[str]:
        try:
            if not isinstance(data, list): return str(data)
            p = 3
            if key in ("bbox", "original_bbox"): return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 4 else None
            if key in ("kps", "landmark_2d_106"): return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f}" for pt in data if isinstance(pt, (list, tuple)) and len(pt) >= 2)
            if key == "landmark_3d_68": return ";".join(f"{float(pt[0]):.{p}f},{float(pt[1]):.{p}f},{float(pt[2]):.{p}f}" for pt in data if isinstance(pt, (list, tuple)) and len(pt) >= 3)
            if key == "pose": return ",".join(f"{float(c):.{p}f}" for c in data) if len(data) == 3 else None
            return str(data)
        except (ValueError, TypeError, IndexError):
            return str(data)
    
    def _save_xmp_tree(self, tree: ET.ElementTree, xmp_path: pathlib.Path) -> bool:
        try:
            ET.indent(tree, space="  ", level=0)
            
            xml_string = ET.tostring(tree.getroot(), encoding="utf-8", method="xml")
            
            if not xml_string.startswith(b"<?xml"):
                xml_string = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
            
            with NamedTemporaryFile("wb", delete=False, dir=xmp_path.parent, suffix=".xmp~") as tmp:
                tmp_path_obj = pathlib.Path(tmp.name)
                tmp.write(xml_string)
            
            if os.name == 'nt': os.replace(tmp_path_obj, xmp_path)
            else: shutil.move(str(tmp_path_obj), xmp_path)
            
            return True
        except Exception as e:
            logger.error(get_message("ERROR_SAVING_XMP", file_path=xmp_path.name, exc=e), exc_info=True)
            return False

    def create_or_update_from_json(self, image_filename: str, json_manager: JsonDataManager, session_name: Optional[str]) -> bool:
        xmp_path = self._get_xmp_path(image_filename)
        file_data = json_manager.get_data(image_filename)
        tree = self._parse_xmp(xmp_path)
        if tree is None: return False

        description = tree.getroot().find(".//rdf:Description", NS)
        if description is None: return False
        
        faces = (file_data or {}).get("faces", [])
        
        first_face = faces[0] if faces and isinstance(faces[0], dict) else {}
        self._update_text_field(description, "Source", "photoshop", "1")
        self._update_text_field(description, "Credit", "photoshop", "1")

        self._update_text_field(description, "Headline", "photoshop", session_name)
        self._update_text_field(description, "IntellectualGenre", "Iptc4xmpCore", "Группа" if len(faces) > 1 else "Портрет")
        bbox_val = first_face.get("original_bbox") or first_face.get("bbox")
        pose_val = first_face.get("pose")
        if bbox_val: self._update_text_field(description, "Instructions", "photoshop", self._format_coordinates("bbox", bbox_val))
        if pose_val: self._update_text_field(description, "UsageTerms", "xmpRights", self._format_coordinates("pose", pose_val))

        general_keywords = set()
        subject_codes_final = []
        is_portrait = len(faces) == 1
        photo_type = "Портрет" if is_portrait else "Группа"
        general_keywords.add(photo_type)
        
        for face_idx, face in enumerate(faces):
            if not isinstance(face, dict): continue
            
            face_attributes = {}
            face_attributes['genre'] = photo_type
            
            name = face.get("child_name") or face.get("matched_child_name")
            cluster = face.get("cluster_label") or face.get("matched_portrait_cluster_label")
            person_identifier = ""
            if name and name not in ["Noise", "No Match", "Unknown"] and not name.startswith("Unknown"):
                person_identifier = name
            elif cluster is not None:
                try: person_identifier = f"Кластер_{int(cluster):02d}"
                except (ValueError, TypeError): pass
            
            if person_identifier:
                face_attributes['person'] = person_identifier
                general_keywords.add(person_identifier)
                if is_portrait:
                    self._update_text_field(description, "TransmissionReference", "photoshop", person_identifier)     

            def flatten_dict_to_temp_storage(d, temp_storage, parent_key=''):
                for k, v in d.items():
                    new_key = f"{parent_key}_{k}" if parent_key else k
                    if new_key in self.excluded_xmp_fields: continue
                    if isinstance(v, dict):
                        flatten_dict_to_temp_storage(v, temp_storage, new_key)
                    elif isinstance(v, list):
                        formatted_val = self._format_coordinates(k, v)
                        if formatted_val: temp_storage[new_key] = formatted_val
                    else:
                        temp_storage[new_key] = v
            
            flatten_dict_to_temp_storage(face, face_attributes)
            
            if is_portrait:
                if face_attributes.get("emotion_faceonnx"): general_keywords.add(f"{face_attributes['emotion_faceonnx']}".strip())
                if face_attributes.get("gender_faceonnx"): general_keywords.add(f"{face_attributes['gender_faceonnx']}")
                left_eye, right_eye = face_attributes.get("left_eye_state"), face_attributes.get("right_eye_state")
                if left_eye == "Closed" and right_eye == "Closed": general_keywords.add("Eyes_Closed")
                elif left_eye == "Open" and right_eye == "Open": general_keywords.add("Eyes_Open")

            prefix = f"F{face_idx}"
            subject_codes_for_face = []
            landmark_entry = None
            
            for key, value in face_attributes.items():
                clean_value = str(value).strip()
                if not clean_value: continue
                
                if key == 'landmark_2d_106':
                    landmark_entry = f"{prefix}_{key}:{clean_value}"
                else:
                    subject_codes_for_face.append(f"{prefix}_{key}:{clean_value}")
            
            subject_codes_for_face.sort()
            
            if landmark_entry:
                subject_codes_for_face.append(landmark_entry)
            
            subject_codes_final.extend(subject_codes_for_face)

        self._update_keyword_sections(tree, list(general_keywords))
        self._add_keywords_to_bag(tree, "Iptc4xmpCore", "SubjectCode", subject_codes_final)
            
        return self._save_xmp_tree(tree, xmp_path)

# 5. БЛОК: Функция-оркестратор
# ==============================================================================
def run_xmp_creation(json_manager: JsonDataManager, image_folder_path: pathlib.Path, session_name: Optional[str], max_workers: int):
     
    logger.debug(get_message("INFO_XMP_UPDATE_START"))
    xmp_manager = XmpManager(image_folder_path)
    
    all_filenames = json_manager.get_all_filenames("all")
    if not all_filenames:
        logger.info("Нет файлов для обработки XMP.")
        return
        
    print(" ")
    logger.info(f"Создание/обновление XMP файлов для {len(all_filenames)} изображений ({max_workers} потоков)...")
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(xmp_manager.create_or_update_from_json, fname, json_manager, session_name): fname for fname in all_filenames}
        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(all_filenames), desc="Обновление XMP")
        
        for future in progress_bar:
            try:
                if not future.result(): errors += 1
            except Exception as e:
                logger.error(f"Исключение в потоке XMP для {futures[future]}: {e}", exc_info=True)
                errors += 1
    print(" ")   
    logger.info(get_message("INFO_XMP_UPDATE_COMPLETE", errors=errors))
    print(" ")


# 6. БЛОК: Точка входа
# ==============================================================================
def main():
    if not IS_MANAGED_RUN:
        logger.critical("Этот скрипт требует запуска из среды PySM для доступа к контексту.")
        sys.exit(1)

    if _template_content is None:
        logger.critical("XMP шаблон не был загружен. Работа скрипта невозможна.")
        sys.exit(1)
        
    config = get_config()

    # Получаем пути из контекста
    session_path_str = pysm_context.get("wf_session_path")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_...) не найдены.")
        sys.exit(1)

    # Конструируем пути
    base_path = pathlib.Path(session_path_str) / session_name
    analysis_path = base_path / ANALYSIS_SUBFOLDER_TEMPLATE.format(photo_session=photo_session)
    image_folder_for_xmp = base_path / IMAGE_SUBFOLDER_TEMPLATE.format(photo_session=photo_session)

    portrait_json_path = analysis_path / PORTRAIT_JSON_FILENAME
    group_json_path = analysis_path / GROUP_JSON_FILENAME

    if not portrait_json_path.is_file() or not group_json_path.is_file():
        logger.error(f"Ошибка: Один или оба JSON-файла не найдены по ожидаемым путям:")
        logger.error(f" - {portrait_json_path}")
        logger.error(f" - {group_json_path}")
        sys.exit(1)

    logger.info("Инициализация менеджера данных JSON...")
    json_manager = JsonDataManager(portrait_json_path, group_json_path)
    if not json_manager.load_data():
        logger.error("Не удалось загрузить данные из JSON-файлов. Отмена.")
        sys.exit(1)
        
    run_xmp_creation(
        json_manager=json_manager,
        image_folder_path=image_folder_for_xmp,
        session_name=session_name,
        max_workers=config.all_threads
    )

if __name__ == "__main__":
    main()