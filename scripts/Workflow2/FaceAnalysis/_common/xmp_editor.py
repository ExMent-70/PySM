# _common/xmp_editor.py

import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from tempfile import NamedTemporaryFile
from typing import List, Optional, Dict, Union

# Настройка логгера для модуля
logger = logging.getLogger(__name__)
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"

# Глобальные пространства имен XMP
NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "lightroom": "http://ns.adobe.com/lightroom/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "GettyImagesGIFT": "http://xmp.gettyimages.com/gift/1.0/",
}

# Старые XMP PySM использовали URI, который ExifTool считал неизвестным tmp0.
# При следующем обновлении такого файла переносим свойство в корректный namespace.
LEGACY_NAMESPACE_URIS = {
    "GettyImagesGIFT": ("http://ns.gettyimages.com/gift/1.0/",),
}

# Регистрация пространств имен в ET, чтобы при сохранении теги выглядели красиво (dc:subject, а не ns0:subject)
for prefix, uri in NAMESPACES.items():
    try:
        ET.register_namespace(prefix, uri)
    except ValueError:
        pass

# Базовый шаблон, если файл шаблона не передан или не найден
DEFAULT_XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="python-xmp-utils">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" 
        xmlns:dc="http://purl.org/dc/elements/1.1/" 
        xmlns:lightroom="http://ns.adobe.com/lightroom/1.0/" 
        xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/" 
        xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">
        <dc:subject><rdf:Bag/></dc:subject>
        <lightroom:hierarchicalSubject><rdf:Bag/></lightroom:hierarchicalSubject>
        <Iptc4xmpCore:SubjectCode><rdf:Bag/></Iptc4xmpCore:SubjectCode>
    </rdf:Description>
</rdf:RDF>
</x:xmpmeta>"""


class XmpEditor:
    """
    Класс для редактирования метаданных XMP.
    Обеспечивает загрузку, создание, модификацию полей и безопасное сохранение.
    """

    def __init__(
        self,
        file_path: Union[str, pathlib.Path],
        template_content: Optional[str] = None,
    ):
        """
        Инициализация редактора.

        :param file_path: Путь к .xmp файлу (существующему или новому).
        :param template_content: XML-строка шаблона, используемая если файл не существует.
                                 Если None, используется DEFAULT_XMP_TEMPLATE.
        """
        self.file_path = pathlib.Path(file_path)
        self.template_content = (
            template_content if template_content else DEFAULT_XMP_TEMPLATE
        )
        self.tree: Optional[ET.ElementTree] = None
        self.root: Optional[ET.Element] = None
        self.description: Optional[ET.Element] = None

        self._load_or_create()

    def _load_or_create(self):
        """Загружает существующий XMP или создает новый из шаблона."""
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    self.tree = ET.parse(f)
            except ET.ParseError as e:
                logger.warning(f"Ошибка парсинга XMP {self.file_path}: {e}. Пересоздание из шаблона.")
                self._create_from_template()
        else:
            self._create_from_template()

        if self.tree:
            self.root = self.tree.getroot()
            # Находим основной блок Description. Если его нет — файл битый/пустой.
            self.description = self.root.find(".//rdf:Description", NAMESPACES)
            if self.description is None:
                # Пытаемся восстановить структуру, если Description потерян, но проще пересоздать
                logger.warning(f"В файле {self.file_path} не найден rdf:Description. Сброс к шаблону.")
                self._create_from_template()
                self.root = self.tree.getroot()
                self.description = self.root.find(".//rdf:Description", NAMESPACES)

    def _create_from_template(self):
        """Создает структуру XML из шаблона."""
        try:
            self.tree = ET.ElementTree(ET.fromstring(self.template_content))
        except ET.ParseError as e:
            logger.error(f"Критическая ошибка шаблона XMP: {e}")
            # Аварийный откат к минимальному шаблону, чтобы не падать
            self.tree = ET.ElementTree(ET.fromstring(DEFAULT_XMP_TEMPLATE))

    def _find_or_create_element(
        self,
        parent: ET.Element,
        tag: str,
        ns_prefix: str,
    ) -> Optional[ET.Element]:
        """Находит или создает элемент с учетом namespace."""
        ns_uri = NAMESPACES.get(ns_prefix)
        if not ns_uri:
            logger.error(f"Неизвестный префикс пространства имен: {ns_prefix}")
            return None

        # Для поиска через find (удобнее использовать префиксы, если они зарегистрированы)
        search_path = f"./{ns_prefix}:{tag}"
        
        element = parent.find(search_path, namespaces=NAMESPACES)
        for legacy_uri in LEGACY_NAMESPACE_URIS.get(ns_prefix, ()):
            legacy_elements = parent.findall(f"./{{{legacy_uri}}}{tag}")
            for legacy_element in legacy_elements:
                if element is None:
                    legacy_element.tag = ET.QName(ns_uri, tag)
                    element = legacy_element
                else:
                    parent.remove(legacy_element)
        if element is None:
            element = ET.SubElement(parent, ET.QName(ns_uri, tag))
        return element

    def set_simple_field(self, ns_prefix: str, tag: str, value: str):
        """
        Устанавливает значение простого текстового поля (например, photoshop:Headline).
        Если значение None или пустая строка, поле не удаляется (поведение можно изменить),
        но обычно обновляется.
        """
        if self.description is None or value is None:
            return

        element = self._find_or_create_element(self.description, tag, ns_prefix)
        if element is None:
            return

        # Проверяем, не является ли поле контейнером (Alt/Seq).
        # Некоторые поля (например, dc:description) требуют внутри rdf:Alt -> rdf:li
        # Здесь мы предполагаем простую структуру для полей типа photoshop:Source, Iptc4xmpCore:Location
        
        # Для совместимости с XMP спецификацией, если это simple property
        element.text = str(value)

    def set_localized_text(
        self,
        ns_prefix: str,
        tag: str,
        value: str,
        lang: str = "x-default",
    ) -> None:
        """
        Устанавливает значение для полей типа Lang Alt (например, dc:description, dc:title).
        """
        if self.description is None or value is None:
            return

        container = self._find_or_create_element(self.description, tag, ns_prefix)
        if container is None:
            return
        container.text = None

        alt = self._find_or_create_element(container, "Alt", "rdf")
        if alt is None:
            return
        
        # Очищаем существующие li (упрощение: перезаписываем всё одним языком)
        # В идеале нужно искать li с xml:lang='x-default'
        for li in alt.findall(f"{{{NAMESPACES['rdf']}}}li"):
            alt.remove(li)
        
        li = ET.SubElement(alt, ET.QName(NAMESPACES['rdf'], "li"))
        li.set(f"{{{XML_NAMESPACE}}}lang", lang)
        li.text = str(value)

    def update_bag(self, ns_prefix: str, tag: str, items: List[str], sort: bool = True, append: bool = False):
        """
        Обновляет список элементов (rdf:Bag). Используется для ключевых слов, subject codes и т.д.
        
        :param ns_prefix: Префикс (например, 'dc', 'Iptc4xmpCore').
        :param tag: Имя тега (например, 'subject', 'SubjectCode').
        :param items: Список строк.
        :param sort: Нужно ли сортировать элементы (для keywords - да, для SubjectCode - нет).
        :param append: Если True, добавляет к существующим. Если False - заменяет список целиком.
        """
        if self.description is None:
            return

        # Фильтрация пустых значений и strip
        clean_items = [str(x).strip() for x in items if x and str(x).strip()]
        if not clean_items and not append:
            # Если список пуст и мы не добавляем -> можно очистить поле, но пока оставим пустой Bag
            pass

        container = self._find_or_create_element(self.description, tag, ns_prefix)
        if container is None: return

        bag = self._find_or_create_element(container, "Bag", "rdf")
        if bag is None: return

        current_items = set()
        if append:
            # Считываем текущие
            for li in bag.findall(f"{{{NAMESPACES['rdf']}}}li"):
                if li.text:
                    current_items.add(li.text.strip())
            
            # Добавляем новые
            current_items.update(clean_items)
            final_list = list(current_items)
        else:
            final_list = clean_items

        if sort:
            final_list.sort()

        # Полная перезапись содержимого Bag
        for li in bag.findall(f"{{{NAMESPACES['rdf']}}}li"):
            bag.remove(li)

        for item in final_list:
            li = ET.SubElement(bag, ET.QName(NAMESPACES['rdf'], "li"))
            li.text = item

    def save(self) -> bool:
        """Атомарно сохраняет XMP и очищает временный файл при ошибке."""

        if self.tree is None:
            return False

        temporary_path: pathlib.Path | None = None
        try:
            ET.indent(self.tree, space="  ", level=0)
            xml_string = ET.tostring(
                self.tree.getroot(),
                encoding="utf-8",
                method="xml",
            )
            if not xml_string.startswith(b"<?xml"):
                xml_string = (
                    b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
                )

            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "wb",
                delete=False,
                dir=self.file_path.parent,
                suffix=".xmp~",
            ) as temporary_file:
                temporary_file.write(xml_string)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = pathlib.Path(temporary_file.name)

            os.replace(temporary_path, self.file_path)
            temporary_path = None
            return True
        except Exception as error:
            logger.error(
                f"Ошибка сохранения XMP {self.file_path}: {error}",
                exc_info=True,
            )
            return False
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
