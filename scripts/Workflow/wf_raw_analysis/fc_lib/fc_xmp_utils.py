# fc_lib/fc_xmp_utils.py

from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import logging
from typing import Optional, Dict, Any, List, Union, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

# --- Импорты ---
from .fc_messages import get_message
from .fc_config import ConfigManager
from .fc_json_data_manager import JsonDataManager

logger = logging.getLogger(__name__)

# --- Пространства имен XMP ---
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "lightroom": "http://ns.adobe.com/lightroom/1.0/",
    "custom": "http://example.com/myapp/1.0/",
    "xml": "http://www.w3.org/XML/1998/namespace",
    # --- НОВАЯ СТРОКА ---
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
} 
for prefix, uri in NS.items():
    try:
        ET.register_namespace(prefix, uri)
    except ValueError:
        pass

# --- Загрузка шаблона ---
TEMPLATE_FILENAME = "face_xmp_template.xmp"
_template_content: Optional[str] = None

try:
    _template_path = Path(__file__).parent / TEMPLATE_FILENAME
    if not _template_path.is_file():
        logger.error(
            f"XMP Template file '{TEMPLATE_FILENAME}' не найден в {Path(__file__).parent.resolve()}"
        )
        _template_content = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="python-xmp-utils"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/" xmlns:lightroom="http://ns.adobe.com/lightroom/1.0/" xmlns:custom="http://example.com/myapp/1.0/"><dc:subject><rdf:Bag/></dc:subject><lightroom:hierarchicalSubject><rdf:Bag/></lightroom:hierarchicalSubject></rdf:Description></rdf:RDF></x:xmpmeta>"""
        logger.warning("Используется базовый XMP шаблон.")
    else:
        _template_content = _template_path.read_text(encoding="utf-8")
        logger.info(f"Загружен XMP шаблон: {_template_path.resolve()}")
except Exception as e:
    logger.error(f"Ошибка загрузки XMP шаблона '{TEMPLATE_FILENAME}': {e}")
    _template_content = None

# Список ключей координат
COORDINATE_KEYS = {
    "kps",
    "bbox",
    "landmark_2d_106",
    "landmark_3d_68",
    "pose",
    "original_bbox",
}


class XmpManager:
    """Класс для управления созданием и обновлением XMP-файлов метаданных."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.image_folder_path = Path(self.config.get("paths", "folder_path"))
        self.custom_ns_prefix = "custom"
        self.custom_ns_uri = NS.get(
            self.custom_ns_prefix, "http://example.com/myapp/1.0/"
        )
        try:
            ET.register_namespace(self.custom_ns_prefix, self.custom_ns_uri)
        except ValueError:
            pass
        xmp_config = self.config.get("xmp", default={})
        self.excluded_xmp_fields: Set[str] = set(xmp_config.get("exclude_fields", []))
        if "embedding" not in self.excluded_xmp_fields:
            logger.debug(
                "Поле 'embedding' добавлено в xmp.exclude_fields по умолчанию."
            )
            self.excluded_xmp_fields.add("embedding")
        logger.info(
            f"XMP поля, исключаемые из custom:*: {sorted(list(self.excluded_xmp_fields))}"
        )

    def _get_xmp_path(self, image_filename: str) -> Path:
        """Возвращает путь к XMP файлу рядом с исходным изображением."""
        image_path = Path(image_filename)
        return self.image_folder_path / f"{image_path.stem}.xmp"

    def _parse_xmp(self, xmp_path: Path) -> Optional[ET.ElementTree]:
        """Парсит существующий XMP или создает новый из шаблона."""
        if xmp_path.exists() and xmp_path.stat().st_size > 0:
            try:
                with xmp_path.open("r", encoding="utf-8") as f:
                    tree = ET.parse(f)
                rdf_root = tree.getroot().find(".//rdf:RDF", NS)
                if rdf_root is None or rdf_root.find(".//rdf:Description", NS) is None:
                    raise ET.ParseError("Invalid RDF structure")
                return tree
            except Exception as e:
                logger.warning(f"Ошибка парсинга XMP {xmp_path}: {e}. Создание нового.")
                return self._create_tree_from_template()
        else:
            return self._create_tree_from_template()

    def _create_tree_from_template(self) -> Optional[ET.ElementTree]:
        """Создает XML дерево из строки шаблона."""
        if _template_content is None:
            logger.error("XMP шаблон не загружен.")
            return None
        try:
            root = ET.fromstring(_template_content)
            tree = ET.ElementTree(root)
            return tree
        except Exception as e:
            logger.error(f"Ошибка создания дерева из шаблона: {e}")
            return None

    def _find_or_create(
        self, parent: ET.Element, tag_name: str, ns_prefix: str
    ) -> Optional[ET.Element]:
        """Находит существующий или создает новый дочерний элемент."""
        ns_uri = NS.get(ns_prefix)
        if not ns_uri:
            logger.error(f"Не найден URI для префикса '{ns_prefix}'.")
            return None
        xpath_query = f"./{ns_prefix}:{tag_name}"
        element = parent.find(xpath_query, namespaces=NS)
        if element is None:
            try:
                qname = ET.QName(ns_uri, tag_name)
                element = ET.SubElement(parent, qname)
            except ValueError as e:
                logger.error(
                    f"Ошибка создания элемента '{tag_name}' с URI '{ns_uri}': {e}"
                )
                return None
        return element

    def _update_text_field(
        self,
        description: ET.Element,
        tag_name: str,
        ns_prefix: str,
        text: Optional[str],
    ):
        """Обновляет или создает текстовое поле в XMP, обрабатывая rdf:Alt/Seq."""
        if text is None:
            return
        element: Optional[ET.Element] = None
        try:
            element = self._find_or_create(description, tag_name, ns_prefix)
            if element is None:
                logger.error(
                    f"Не удалось найти или создать элемент {ns_prefix}:{tag_name}"
                )
                return
            text_str = str(text)
            alt_elem = element.find(f"{{{NS['rdf']}}}Alt")
            seq_elem = element.find(f"{{{NS['rdf']}}}Seq")
            container = alt_elem if alt_elem is not None else seq_elem
            if container is not None:
                li_elem = container.find(f"{{{NS['rdf']}}}li")
                if li_elem is None:
                    li_elem = self._find_or_create(container, "li", "rdf")
                if li_elem is not None:
                    li_elem.text = text_str
                    if (
                        ns_prefix == "dc"
                        and tag_name in ["title", "rights", "creator", "description"]
                    ) or (ns_prefix == "xmpRights" and tag_name == "UsageTerms"):
                        try:
                            xml_lang_qname = ET.QName(NS["xml"], "lang")
                            li_elem.set(xml_lang_qname, "x-default")
                        except Exception:
                            li_elem.set(
                                "{http://www.w3.org/XML/1998/namespace}lang",
                                "x-default",
                            )
            else:
                element.text = text_str
        except Exception as e:
            logger.error(
                f"Ошибка при обновлении поля {ns_prefix}:{tag_name}: {e}", exc_info=True
            )

    # --- ВОЗВРАЩАЕМ ЛОГИКУ СОЗДАНИЯ НОВЫХ LI ---
    def update_keywords(self, tree: ET.ElementTree, keywords_to_add: List[str]) -> bool:
        """Добавляет уникальные ключевые слова в dc:subject и lightroom:hierarchicalSubject."""
        if tree is None:
            logger.error("Передано пустое XMP дерево в update_keywords.")
            return False
        root = tree.getroot()
        if root is None:
            logger.error("Не удалось получить корневой элемент XMP в update_keywords.")
            return False

        description = root.find(".//rdf:Description", NS)
        if description is None:
            rdf_node = root.find(f"{{{NS['rdf']}}}RDF")
            if rdf_node is None:
                logger.error("Не найдена структура rdf:RDF в XMP.")
                return False
            description = self._find_or_create(rdf_node, "Description", "rdf")
            if description is None:
                logger.error("Не удалось создать rdf:Description.")
                return False
            logger.warning(
                "Основной rdf:Description не найден в update_keywords, был создан новый."
            )

        processed_sections = 0
        tags_to_update = [("dc", "subject"), ("lightroom", "hierarchicalSubject")]
        unique_valid_keywords = sorted(
            list(set(kw.strip() for kw in keywords_to_add if kw and kw.strip()))
        )

        if not unique_valid_keywords:
            logger.debug("update_keywords: нет валидных слов для добавления.")
            return True

        logger.debug(
            f"update_keywords: попытка добавить слова: {unique_valid_keywords}"
        )

        for ns_prefix, tag_name in tags_to_update:
            section_processed_successfully = False
            try:
                subject_element = self._find_or_create(description, tag_name, ns_prefix)
                if subject_element is None:
                    logger.error(f"Не удалось найти/создать {ns_prefix}:{tag_name}")
                    continue

                bag_element = self._find_or_create(subject_element, "Bag", "rdf")
                if bag_element is None:
                    logger.error(
                        f"Не удалось найти/создать rdf:Bag в {ns_prefix}:{tag_name}"
                    )
                    continue

                existing_keywords_set = {
                    li.text.strip()
                    for li in bag_element.findall(f"{{{NS['rdf']}}}li")
                    if li.text and li.text.strip()
                }
                logger.debug(
                    f"Существующие слова в {ns_prefix}:{tag_name}: {existing_keywords_set}"
                )

                added_count = 0
                # --- Используем ET.SubElement для каждого нового слова ---
                for cleaned_keyword in unique_valid_keywords:
                    if cleaned_keyword not in existing_keywords_set:
                        try:
                            li_qname = ET.QName(NS["rdf"], "li")
                            # Создаем НОВЫЙ элемент li как дочерний для bag_element
                            new_li = ET.SubElement(bag_element, li_qname)
                            new_li.text = cleaned_keyword
                            added_count += 1
                            logger.debug(
                                f"  -> Добавлено '{cleaned_keyword}' в {ns_prefix}:{tag_name}"
                            )
                        except Exception as add_err:
                            logger.error(
                                f"     ... ОШИБКА: Не удалось создать/добавить rdf:li для '{cleaned_keyword}': {add_err}"
                            )
                # --- Конец изменения ---

                logger.debug(
                    f"В секцию {ns_prefix}:{tag_name} добавлено {added_count} новых слов."
                )
                # Отладочный вывод содержимого Bag
                try:
                    bag_content_str = ET.tostring(
                        bag_element, encoding="unicode", method="xml"
                    )
                    logger.debug(
                        f"Итоговое содержимое {ns_prefix}:{tag_name}/rdf:Bag:\n{bag_content_str}"
                    )
                except Exception as dump_err:
                    logger.error(f"Ошибка при отладочном выводе rdf:Bag: {dump_err}")

                section_processed_successfully = True

            except Exception as e:
                logger.error(
                    f"Ошибка обработки секции keywords {ns_prefix}:{tag_name}: {e}",
                    exc_info=True,
                )
                section_processed_successfully = False

            if section_processed_successfully:
                processed_sections += 1

        return processed_sections == len(tags_to_update)

    def _format_coordinates(self, key: str, data: List) -> Optional[str]:
        """Форматирует список координат в строку."""
        try:
            if not isinstance(data, list):
                raise TypeError("Ожидался список")
            precision = 3
            if key in ("bbox", "original_bbox"):
                if len(data) == 4:
                    return ",".join(f"{float(c):.{precision}f}" for c in data)
                else:
                    raise ValueError("bbox должен содержать 4 элемента")
            elif key in ("kps", "landmark_2d_106"):
                return ";".join(
                    f"{float(p[0]):.{precision}f},{float(p[1]):.{precision}f}"
                    for p in data
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                )
            elif key == "landmark_3d_68":
                return ";".join(
                    f"{float(p[0]):.{precision}f},{float(p[1]):.{precision}f},{float(p[2]):.{precision}f}"
                    for p in data
                    if isinstance(p, (list, tuple)) and len(p) >= 3
                )
            elif key == "pose":
                if len(data) == 3:
                    return ",".join(f"{float(c):.{precision}f}" for c in data)
                else:
                    raise ValueError("pose должен содержать 3 элемента")
            logger.warning(
                f"Ключ '{key}' не распознан как координатный тип для форматирования."
            )
            return str(data)
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(
                f"Не удалось форматировать координаты для ключа '{key}': {e}. Данные: {data}"
            )
            return str(data)

    def _dict_to_xml(
        self, parent: ET.Element, data: Union[Dict, List, Any], element_name: str
    ):
        """Рекурсивно преобразует словарь/список Python в XML XMP."""
        if element_name in self.excluded_xmp_fields:
            return
        qname = ET.QName(self.custom_ns_uri, element_name)
        if element_name in COORDINATE_KEYS and isinstance(data, list):
            formatted_coords = self._format_coordinates(element_name, data)
            if formatted_coords is not None:
                element = ET.SubElement(parent, qname)
                element.text = formatted_coords
            else:
                logger.warning(f"Тег для координат '{element_name}' не создан.")
            return
        if isinstance(data, dict):
            dict_container_elem = ET.SubElement(parent, qname)
            desc_qname = ET.QName(NS["rdf"], "Description")
            desc_element = ET.SubElement(dict_container_elem, desc_qname)
            for key, value in data.items():
                if value is None or key in self.excluded_xmp_fields:
                    continue
                safe_key = "".join(
                    c if c.isalnum() or c in ["_", "-"] else "_" for c in str(key)
                )
                if not safe_key or (not safe_key[0].isalpha() and safe_key[0] != "_"):
                    safe_key = "_" + safe_key
                if not safe_key:
                    continue
                try:
                    ET.fromstring(f"<{safe_key}>test</{safe_key}>")
                except ET.ParseError:
                    logger.warning(
                        f"Ключ '{key}' ('{safe_key}') невалиден для XML, пропуск."
                    )
                    continue
                self._dict_to_xml(desc_element, value, safe_key)
            return
        if isinstance(data, list):
            list_container_elem = ET.SubElement(parent, qname)
            seq_qname = ET.QName(NS["rdf"], "Seq")
            seq_element = ET.SubElement(list_container_elem, seq_qname)
            li_qname = ET.QName(NS["rdf"], "li")
            for item_idx, item in enumerate(data):
                li_element = ET.SubElement(seq_element, li_qname)
                if isinstance(item, (dict, list)):
                    item_desc_qname = ET.QName(NS["rdf"], "Description")
                    item_desc = ET.SubElement(li_element, item_desc_qname)
                    self._dict_to_xml(item_desc, item, "item")
                else:
                    li_element.text = str(item) if item is not None else ""
            return
        element = ET.SubElement(parent, qname)
        element.text = str(data) if data is not None else ""

    def update_custom_data(
        self, tree: ET.ElementTree, data_dict: Dict[str, Any], section_name: str
    ) -> bool:
        """Обновляет или создает кастомную секцию в XMP."""
        if tree is None:
            logger.error("Передано пустое XMP дерево в update_custom_data.")
            return False
        root = tree.getroot()
        if root is None:
            logger.error(
                "Не удалось получить корневой элемент XMP в update_custom_data."
            )
            return False
        description = root.find(".//rdf:Description", NS)
        if description is None:
            rdf_node = root.find(f"{{{NS['rdf']}}}RDF")
            if rdf_node is None:
                logger.error("Не найдена структура rdf:RDF в XMP.")
                return False
            description = self._find_or_create(rdf_node, "Description", "rdf")
            if description is None:
                logger.error("Не удалось создать rdf:Description.")
                return False
            logger.warning(
                "Основной rdf:Description не найден в update_custom_data, был создан новый."
            )
        xpath_query = f"./{self.custom_ns_prefix}:{section_name}"
        existing_section = description.find(xpath_query, namespaces=NS)
        if existing_section is not None:
            description.remove(existing_section)
            logger.debug(f"Удалена существующая секция {section_name}")
        try:
            logger.debug(f"Добавление/Обновление секции {section_name}")
            self._dict_to_xml(description, data_dict, section_name)
            return True
        except Exception as e:
            logger.error(
                f"Ошибка при добавлении custom данных '{section_name}': {e}",
                exc_info=True,
            )
            return False

    def _update_basic_fields(
        self,
        tree: ET.ElementTree,
        filename_str: str,
        file_data: Optional[Dict[str, Any]],
    ):
        """Обновляет стандартные поля XMP."""
        if tree is None:
            logger.error("Передано пустое XMP дерево в _update_basic_fields.")
            return
        root = tree.getroot()
        if root is None:
            logger.error(
                "Не удалось получить корневой элемент XMP в _update_basic_fields."
            )
            return
        description = root.find(".//rdf:Description", NS)
        if description is None:
            rdf_node = root.find(f"{{{NS['rdf']}}}RDF")
            if rdf_node is not None:
                description = self._find_or_create(rdf_node, "Description", "rdf")
            if description is None:
                logger.error(
                    "rdf:Description не найден и не может быть создан для базовых полей."
                )
                return
            else:
                logger.warning(
                    "Основной rdf:Description не найден в _update_basic_fields, был создан новый."
                )
        _file_data = file_data if file_data else {}
        faces = _file_data.get("faces", [])
        first_face = faces[0] if faces and isinstance(faces[0], dict) else {}
               
        # --- НАЧАЛО НОВОЙ ЛОГИКИ ---
        # Блок 2.1: Определение жанра
        genre = "Портрет"
        if len(faces) > 1:
            genre = "Группа"
        
        self._update_text_field(description, "IntellectualGenre", "Iptc4xmpCore", genre)
        if ConfigManager.session_name_str is not None:
            session_name = ConfigManager.session_name_str # Заданное значение
        else:
            session_name = "КЛАСС"
        self._update_text_field(description, "Headline", "photoshop", session_name)
        
        
        child_name_candidate = first_face.get("child_name") or first_face.get(
            "matched_child_name"
        )
        cluster_label = first_face.get("cluster_label") or first_face.get(
            "matched_portrait_cluster_label"
        )
        child_name_ref = "Unknown"
        if (
            child_name_candidate
            and child_name_candidate not in ["Noise", "No Match", "Unknown"]
            and not child_name_candidate.startswith("Unknown")
        ):
            child_name_ref = child_name_candidate
        elif cluster_label is not None:
            try:
                child_name_ref = f"Кластер_{int(cluster_label):02d}"
            except (ValueError, TypeError):
                pass
        bbox_val = first_face.get("original_bbox") or first_face.get("bbox")
        bbox_str = self._format_coordinates("bbox", bbox_val) if bbox_val else None
        pose_val = first_face.get("pose")
        pose_str = self._format_coordinates("pose", pose_val) if pose_val else None
        credit_provider = "PugachevA"
        if bbox_str:
            self._update_text_field(description, "Instructions", "photoshop", bbox_str)
        if child_name_ref:
            self._update_text_field(
                description, "TransmissionReference", "photoshop", child_name_ref
            )
        if credit_provider:
            self._update_text_field(description, "Credit", "photoshop", credit_provider)
        if pose_str:
            self._update_text_field(description, "UsageTerms", "xmpRights", pose_str)

    def _save_xmp_tree(self, tree: ET.ElementTree, xmp_path: Path) -> bool:
        """Сохраняет XML дерево в файл XMP с форматированием."""
        try:
            # Отладочный вывод перед сохранением
            try:
                xml_string_debug = ET.tostring(
                    tree.getroot(), encoding="unicode", method="xml"
                )
                subject_start_index = xml_string_debug.find("<dc:subject")
                if subject_start_index != -1:
                    subject_end_index = xml_string_debug.find(
                        "</dc:subject>", subject_start_index
                    )
                    if subject_end_index != -1:
                        debug_snippet = xml_string_debug[
                            subject_start_index : subject_end_index
                            + len("</dc:subject>")
                        ]
                    else:
                        debug_snippet = (
                            xml_string_debug[
                                subject_start_index : subject_start_index + 300
                            ]
                            + "..."
                        )
                    logger.debug(
                        f"--- Фрагмент dc:subject перед сохранением {xmp_path.name} ---\n{debug_snippet}\n-----------------------------------------"
                    )
                else:
                    logger.debug(
                        f"Секция dc:subject не найдена в XML перед сохранением {xmp_path.name}"
                    )
            except Exception as e_debug:
                logger.error(
                    f"Ошибка при отладочном выводе XML перед сохранением: {e_debug}"
                )

            # Основная логика сохранения
            xml_string = ET.tostring(tree.getroot(), encoding="utf-8", method="xml")
            if not xml_string.startswith(b"<?xml"):
                xml_string = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
            try:
                dom = minidom.parseString(xml_string)
                pretty_xml_bytes = dom.toprettyxml(indent="  ", encoding="utf-8")
                lines = pretty_xml_bytes.splitlines()
                non_empty_lines = [line for line in lines if line.strip()]
                final_xml_bytes = b"\n".join(non_empty_lines)
            except Exception as pretty_err:
                logger.warning(
                    f"Не удалось отформатировать XML: {pretty_err}. Сохранение без форматирования."
                )
                final_xml_bytes = xml_string
            temp_dir = xmp_path.parent
            temp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path_obj: Optional[Path] = None
            try:
                with NamedTemporaryFile(
                    "wb", delete=False, dir=temp_dir, suffix=".xmp~"
                ) as tmp:
                    tmp_path_obj = Path(tmp.name)
                    tmp.write(final_xml_bytes)
                if tmp_path_obj:
                    if os.name == "nt":
                        os.replace(tmp_path_obj, xmp_path)
                    else:
                        shutil.move(str(tmp_path_obj), xmp_path)
                    logger.debug(f"XMP файл успешно сохранен: {xmp_path.resolve()}")
                    return True
                else:
                    logger.error(
                        f"Не удалось создать временный файл для {xmp_path.name}"
                    )
                    return False
            finally:
                if tmp_path_obj and tmp_path_obj.exists():
                    try:
                        tmp_path_obj.unlink()
                    except OSError:
                        pass
        except Exception as e:
            logger.error(
                get_message("ERROR_SAVING_XMP", file_path=xmp_path.name, exc=e),
                exc_info=True,
            )
            return False

    def create_or_update_from_json(
        self, image_filename: str, json_manager: JsonDataManager
    ) -> bool:
        """Создает или обновляет XMP из данных JSON."""
        xmp_path = self._get_xmp_path(image_filename)
        logger.debug(f"Обработка XMP для: {image_filename} -> {xmp_path.name}")

        file_data = json_manager.get_data(image_filename)
        tree = self._parse_xmp(xmp_path)
        if tree is None:
            logger.error(f"Не удалось получить/создать XMP дерево для {image_filename}")
            return False

        self._update_basic_fields(tree, image_filename, file_data if file_data else {})

        if (
            file_data
            and "faces" in file_data
            and isinstance(file_data["faces"], list)
            and file_data["faces"]
        ):
            faces = file_data["faces"]
            keywords_to_add_for_file = set()
            all_processed_faces_for_details = []
            logger.debug(
                f"--- Начало сбора keywords для файла {image_filename} ({len(faces)} лиц) ---"
            )

            for face_idx, face_dict_orig in enumerate(faces):
                if not isinstance(face_dict_orig, dict):
                    continue
                face_dict = face_dict_orig.copy()
                all_processed_faces_for_details.append(face_dict)

                logger.debug(f"  --- Анализ лица {face_idx} для keywords ---")
                current_face_keywords_found = set()
                # Сбор атрибутов
                gender = face_dict.get("gender_faceonnx")
                logger.debug(f"    Пол: {gender}")
                if gender:
                    current_face_keywords_found.add(f"kg_{gender}")
                emotion = face_dict.get("emotion_faceonnx")
                logger.debug(f"    Эмоция: {emotion}")
                if emotion and isinstance(emotion, str) and emotion.strip():
                    current_face_keywords_found.add(f"ke_{emotion.strip()}")
                left_state = face_dict.get("left_eye_state")
                right_state = face_dict.get("right_eye_state")
                logger.debug(f"    Глаза: L={left_state}, R={right_state}")
                if left_state == "Closed" and right_state == "Closed":
                    current_face_keywords_found.add("k_eyes_Closed")
                elif left_state == "Open" and right_state == "Open":
                    current_face_keywords_found.add("k_eyes_Open")
                elif left_state == "Closed" and right_state != "Closed":
                    current_face_keywords_found.add("k_left_eyes_Closed")
                elif right_state == "Closed" and left_state != "Closed":
                    current_face_keywords_found.add("k_right_eyes_Closed")
                mouth_s = face_dict.get("keypoint_analysis", {}).get("mouth_state")
                logger.debug(f"    Рот: {mouth_s}")
                if mouth_s and isinstance(mouth_s, str) and mouth_s.strip():
                    current_face_keywords_found.add(f"km_mouth_{mouth_s.strip()}")
                # Сбор имени/кластера
                name_source = None
                cluster_source = None
                if len(faces) == 1:  # Портрет
                    name_source = face_dict.get("child_name")
                    cluster_source = face_dict.get("cluster_label")
                    logger.debug(
                        f"    Портрет: Имя={name_source}, Кластер={cluster_source}"
                    )
                    if (
                        name_source
                        and name_source != "Noise"
                        and not name_source.startswith("Unknown")
                    ):
                        current_face_keywords_found.add(name_source)
                    elif cluster_source is not None:
                        try:
                            current_face_keywords_found.add(
                                f"Кластер_{int(cluster_source):02d}"
                            )
                        except:
                            pass
                else:  # Группа
                    name_source = face_dict.get("matched_child_name")
                    cluster_source = face_dict.get("matched_portrait_cluster_label")
                    logger.debug(
                        f"    Группа: Совп.Имя={name_source}, Совп.Кластер={cluster_source}"
                    )
                    if (
                        name_source
                        and name_source != "No Match"
                        and not name_source.startswith("Unknown")
                    ):
                        current_face_keywords_found.add(name_source)
                    elif cluster_source is not None:
                        try:
                            current_face_keywords_found.add(
                                f"Кластер_{int(cluster_source):02d}"
                            )
                        except:
                            pass
                logger.debug(
                    f"    Найденные слова для лица {face_idx}: {current_face_keywords_found}"
                )
                keywords_to_add_for_file.update(current_face_keywords_found)

            if len(faces) > 1:
                keywords_to_add_for_file.add("kf_Group")
            else:
                keywords_to_add_for_file.add("kf_Portrait")                
            unique_keywords = sorted(list(keywords_to_add_for_file))
            logger.debug(
                f"Итоговые ключевые слова для {image_filename}: {unique_keywords}"
            )
            if unique_keywords:
                logger.debug(
                    f"Вызов update_keywords для {image_filename} со словами: {unique_keywords}"
                )
                update_success = self.update_keywords(tree, unique_keywords)
            if not update_success:
                logger.warning(
                    f"Функция update_keywords не смогла успешно обработать обе секции для {xmp_path.name}"
                )
            else:
                logger.debug(f"Нет ключевых слов для добавления в {image_filename}")
            if all_processed_faces_for_details:
                custom_data_root = {"faces": all_processed_faces_for_details}
                self.update_custom_data(tree, custom_data_root, "faceDetails")
            else:
                logger.debug(
                    f"Нет данных лиц для custom:faceDetails в {image_filename}"
                )
        else:
            logger.warning(
                f"Нет данных о лицах для '{image_filename}' для записи в XMP."
            )
        return self._save_xmp_tree(tree, xmp_path)


# --- run_xmp_creation (без изменений) ---
def run_xmp_creation(config: ConfigManager, json_manager: JsonDataManager) -> None:
    if not config.get("task", "create_xmp_file", default=False):
        print(" ", file=sys.stderr)
        logger.info("Создание XMP отключено.")
        return
    logger.info("=" * 10 + " Запуск создания/обновления XMP-файлов " + "=" * 10)
    xmp_manager = XmpManager(config)
    if _template_content is None:
        logger.error("XMP шаблон не загружен. Отмена.")
        return
    if not json_manager.portrait_data and not json_manager.group_data:
        if not json_manager.load_data():
            logger.error("Не удалось загрузить JSON. Отмена.")
            return
    all_filenames = json_manager.get_all_filenames("all")
    if not all_filenames:
        logger.info("Нет файлов для обработки XMP.")
        return
    max_workers_config = config.get(
        "processing", "max_workers", default=os.cpu_count() or 4
    )
    max_workers_limit = config.get("processing", "max_workers_limit", default=16)
    max_concurrent_xmp = config.get(
        "processing", "max_concurrent_xmp_tasks", default=50
    )
    max_workers = min(max_workers_config, max_workers_limit, max_concurrent_xmp)
    logger.info(
        f"Обработка XMP для {len(all_filenames)} файлов ({max_workers} потоков)..."
    )
    xmp_creation_errors = 0
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="XmpWorker"
    ) as executor:
        futures = {
            executor.submit(
                xmp_manager.create_or_update_from_json, filename_str, json_manager
            ): filename_str
            for filename_str in all_filenames
        }
        for future in tqdm(
            futures.keys(), total=len(all_filenames), desc="Обновление XMP"
        ):
            filename_orig = futures[future]
            try:
                if not future.result():
                    xmp_creation_errors += 1
                    logger.error(f"Ошибка создания/обновления XMP для: {filename_orig}")
            except Exception as e:
                logger.error(
                    f"Исключение в потоке XMP для {filename_orig}: {e}", exc_info=True
                )
                xmp_creation_errors += 1
    logger.info(f"Создание/обновление XMP завершено. Ошибок: {xmp_creation_errors}.")
    logger.info("=" * 10 + " Создание/обновление XMP-файлов завершено " + "=" * 10)
