# pysm_lib/pysm_context.py

"""
Этот модуль предоставляет основной API для взаимодействия пользовательских скриптов
с исполняющей средой PyScriptManager.

Ключевым элементом этого модуля является глобальный объект `pysm_context`,
который представляет собой экземпляр класса `PySMContext`.

Основной принцип работы с этим модулем:
1. Импортировать глобальный объект: `from pysm_lib import pysm_context`
2. Вызывать его методы для чтения и записи данных:
   - `value = pysm_context.get("my_variable")`
   - `pysm_context.set("my_variable", new_value)`
   - `pysm_context.set_next_script("instance_id_123")`

Глобальные функции-обертки были удалены в пользу этого единого,
объектно-ориентированного подхода для повышения чистоты и предсказуемости API.
"""

import argparse
import base64
import os
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union




try:
    from pythoncom import com_error as COMError
except ImportError:

    class COMError(Exception):
        pass


from .locale_manager import LocaleManager


locale_manager = LocaleManager()

_RESERVED_KEYS = {"pysm_info", "pysm_set_instance_ids", "pysm_next_script"}
FIELD_MAP: Dict[str, Tuple[str, str]] = {
    "CaptionWriter": ("photoshop:CaptionWriter", "simple"),
    "Headline": ("photoshop:Headline", "simple"),
    "City": ("photoshop:City", "simple"),
    "State": ("photoshop:State", "simple"),
    "Country": ("photoshop:Country", "simple"),
    "Source": ("photoshop:Source", "simple"),
    "Instructions": ("photoshop:Instructions", "simple"),
    "Category": ("photoshop:Category", "simple"),
    "TransmissionReference": ("photoshop:TransmissionReference", "simple"),
    "Credit": ("photoshop:Credit", "simple"),
    "SupplementalCategories": ("photoshop:SupplementalCategories/rdf:Bag", "array"),
    "Location": ("Iptc4xmpCore:Location", "simple"),
    "IntellectualGenre": ("Iptc4xmpCore:IntellectualGenre", "simple"),
    "Scene": ("Iptc4xmpCore:Scene/rdf:Bag", "array"),
    "SubjectCode": ("Iptc4xmpCore:SubjectCode/rdf:Bag", "structure"),
    "Label": ("xmp:Label", "simple"),
    "Rating": ("xmp:Rating", "simple"),
    "Personality": ("GettyImagesGIFT:Personality", "simple"),
    "Description": ("dc:description/rdf:Alt/rdf:li", "simple"),
    "Copyright": ("dc:rights/rdf:Alt/rdf:li", "simple"),
    "Creator": ("dc:creator/rdf:Seq", "array"),
    "Keywords": ("dc:subject/rdf:Bag", "array"),
}
# XML неймспейсы для парсинга метаданных
XML_NAMESPACES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "GettyImagesGIFT": "http://xmp.gettyimages.com/gift/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
}


def _safe_get_attr(obj: Any, attr_name: str, attr_type: type, default: Any = None):
    try:
        value = getattr(obj, attr_name, default)
        return attr_type(value) if value is not None else default
    except (ValueError, TypeError, COMError):
        return default


# 2. БЛОК: Основной класс для управления контекстом выполнения
# ==============================================================================


class PySMContext:
    """
    Класс, инкапсулирующий логику чтения, записи и управления данными
    в общем файле контекста (`pysm_context.json`).
    """

    def __init__(self):
        """Инициализирует объект, находит путь к файлу контекста и кэширует его."""
        self._context_file_path: Optional[pathlib.Path] = None
        self._raw_context_data_cache: Optional[Dict[str, Any]] = None
        self._initialize()

    def _initialize(self):
        """
        Парсит аргументы командной строки при запуске, чтобы найти
        аргумент `--pysm-context-file` и сохранить путь к нему.
        Очищает `sys.argv`, чтобы пользовательский скрипт не видел этот аргумент.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--pysm-context-file", type=str, dest="pysm_context_file_path"
        )
        args, remaining_argv = parser.parse_known_args(args=sys.argv[1:])
        if args.pysm_context_file_path:
            path = pathlib.Path(args.pysm_context_file_path)
            if path.is_file():
                self._context_file_path = path
        sys.argv = [sys.argv[0]] + remaining_argv
        self._read_data()

    def _read_data(self) -> Dict[str, Any]:
        """Читает данные из файла контекста и кэширует их."""
        if self._raw_context_data_cache is not None:
            return self._raw_context_data_cache
        path = self._context_file_path
        if not path or not path.is_file():
            self._raw_context_data_cache = {}
            return self._raw_context_data_cache
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._raw_context_data_cache = json.load(f)
                return self._raw_context_data_cache
        except (json.JSONDecodeError, FileNotFoundError):
            self._raw_context_data_cache = {}
            return self._raw_context_data_cache

    def _write_data(self, data: Dict[str, Any]) -> None:
        """Записывает данные в файл контекста и обновляет кэш."""
        path = self._context_file_path
        if not path:
            raise IOError("Путь к файлу контекста не определен.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._raw_context_data_cache = data

    def _infer_type_from_value(self, value: Any) -> str:
        """Определяет тип переменной по ее значению."""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "json"
        if isinstance(value, str) and "\n" in value:
            return "string_multiline"
        return "string"

    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение переменной из контекста по ключу."""
        variable_data = self._read_data().get(key)
        if variable_data and isinstance(variable_data, dict):
            return variable_data.get("value", default)
        return default

    def get_structured(self, key: str, default: Any = None) -> Any:
        """Получает вложенное значение, используя точечную нотацию (например, 'a.b.c')."""
        keys = key.split(".")
        base_key = keys[0]
        value = self.get(base_key, default if len(keys) == 1 else {})
        if len(keys) > 1 and isinstance(value, dict):
            for sub_key in keys[1:]:
                try:
                    value = value.get(sub_key, default)
                except (AttributeError, TypeError):
                    return default
        return value

    # Заменить в классе PySMContext
    # ==============================================================================

    def set(self, key: str, value: Any, var_type: Optional[str] = None) -> None:
        """
        Устанавливает или создает переменную в контексте.

        Этот метод реализует логику "UPSERT" (UPDATE or INSERT).

        - Если переменная с ключом `key` существует, он обновляет ее значение.
          Если при этом передан `var_type`, тип переменной также будет обновлен.
        - Если переменная не существует, она будет создана.
          Тип будет взят из `var_type`, если он указан, иначе будет определен
          автоматически по значению.

        Args:
            key (str): Ключ (имя) переменной.
            value (Any): Новое значение переменной.
            var_type (Optional[str], optional): Явное указание типа переменной
                                                (например, "dir_path", "int").
        """
        data = self._read_data()
        variable_data = data.get(key)

        # Сценарий 1: Переменная уже существует
        if variable_data and isinstance(variable_data, dict):
            # Проверяем, не защищена ли она от записи
            if variable_data.get("read_only", False):
                print(
                    f"PySM Context Warning: Переменная '{key}' защищена от записи.",
                    file=sys.stderr,
                )
                return

            # Обновляем значение
            variable_data["value"] = value
            # Если тип указан явно, обновляем и его
            if var_type:
                variable_data["type"] = var_type

        # Сценарий 2: Переменная не существует, создаем новую
        else:
            # Если тип не указан явно, определяем его по значению
            final_type = var_type if var_type else self._infer_type_from_value(value)
            data[key] = {
                "type": final_type,
                "value": value,
                "description": "Auto-created by script",
                "read_only": False,
                "choices": None,
            }

        # Записываем обновленные данные в файл
        self._write_data(data)

    def update(self, update_dict: Dict[str, Any]) -> None:
        """Обновляет несколько переменных в контексте из словаря."""
        data = self._read_data()
        for key, value in update_dict.items():
            variable_data = data.get(key)
            if variable_data and isinstance(variable_data, dict):
                if variable_data.get("read_only", False):
                    print(
                        f"PySM Context Warning: Переменная '{key}' защищена от записи.",
                        file=sys.stderr,
                    )
                    continue
                variable_data["value"] = value
            else:
                inferred_type = self._infer_type_from_value(value)
                data[key] = {
                    "type": inferred_type,
                    "value": value,
                    "description": "Auto-created by script",
                    "read_only": False,
                    "choices": None,
                }
        self._write_data(data)

    def remove(self, keys_to_remove: Optional[Union[str, List[str]]] = None) -> None:
        """Удаляет переменные из контекста."""
        data = self._read_data()
        keys_for_deletion = []
        if keys_to_remove is None:
            keys_for_deletion = [k for k in data.keys() if k not in _RESERVED_KEYS]
        elif isinstance(keys_to_remove, str):
            keys_for_deletion = [keys_to_remove]
        elif isinstance(keys_to_remove, list):
            keys_for_deletion = keys_to_remove
        if not keys_for_deletion:
            return
        for key in keys_for_deletion:
            data.pop(key, None)
        self._write_data(data)

    def get_all(self) -> Dict[str, Any]:
        """Возвращает все переменные и их значения из контекста."""
        raw_data = self._read_data()
        return {k: v.get("value") for k, v in raw_data.items()}

    def resolve_template(self, template_string: Optional[str]) -> str:
        """Заменяет плейсхолдеры {key} в строке на значения из контекста."""
        if not template_string or "{" not in template_string:
            return template_string if template_string is not None else ""
        placeholders = re.findall(r"{([^}]+)}", template_string)
        resolved_string = template_string
        for key in set(placeholders):
            value = self.get_structured(key, default=f"{{{key}}}")
            resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        return resolved_string

    def resolve_path(self, path_str: str) -> pathlib.Path:
        """Преобразует относительный путь в абсолютный, используя директорию коллекции как базу."""
        input_path = pathlib.Path(path_str)
        if input_path.is_absolute():
            return input_path
        pysm_info = self.get("pysm_info", {})
        base_dir = pysm_info.get("collection_dir")
        if base_dir:
            return (pathlib.Path(base_dir) / input_path).resolve()
        else:
            return input_path.resolve()

    def get_variable(self, key: str) -> Optional[Dict[str, Any]]:
        """Возвращает полную модель переменной (словарь) по ключу."""
        return self._read_data().get(key)

    # 1. БЛОК: Метод set_next_script (ИЗМЕНЕН)
    def set_next_script(self, instance_id: str) -> None:
        """
        Указывает, какой скрипт должен быть запущен следующим,
        переопределяя стандартную последовательность.
        """
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        # Теперь мы сохраняем не просто ID, а словарь с ID и именем
        instance_name = instance_id  # Имя по умолчанию, если не найдено
        all_instances = self.list_instances()  # Используем новый метод

        # Ищем экземпляр с нужным ID в списке всех экземпляров
        for instance_data in all_instances:
            if (
                isinstance(instance_data, dict)
                and instance_data.get("id") == instance_id
            ):
                instance_name = instance_data.get("name", instance_id)
                break

        value_to_set = {"id": instance_id, "name": instance_name}
        self.set("pysm_next_script", value_to_set)

    def list_instances(self) -> List[Dict[str, str]]:
        """
        Возвращает список данных о всех экземплярах в текущем наборе.
        Каждый элемент - это словарь {"id": "...", "name": "..."}.
        """
        return self.get("pysm_set_instance_ids", [])


    def log_image(
        self,
        image_path: Union[str, pathlib.Path],
        width: int = 300,
        align: str = "left",
        margin: int = 5,
        img_desc: Optional[str] = None,
    ):
        try:
            path = pathlib.Path(image_path)
            if not path.is_file():
                print(
                    f"PySM API Error: Image file not found at '{path}'", file=sys.stderr
                )
                return

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            ext = path.suffix.lower().lstrip(".")
            mime_type = (
                f"image/{ext}"
                if ext in ["png", "jpg", "jpeg", "gif", "bmp"]
                else "image/png"
            )

            styles = f"text-align: {align}; margin-top: {margin}px; margin-bottom: {margin}px;"
            html_parts = [f'<div style="{styles}">']
            html_parts.append(
                f'<br><img src="data:{mime_type};base64,{encoded_string}" width="{width}">'
            )
            if img_desc:
                html_parts.append(f'<div style="{{theme.api_image_description}}">{img_desc}</div>')
            html_parts.append("</div>")
            html_tag = "".join(html_parts)

            print(" ")
            print(f"PYSM_HTML_BLOCK:{html_tag}", file=sys.stderr, flush=True)
            print(" ")

        except Exception as e:
            print(
                f"PySM API Error: Failed to log image '{image_path}'. Reason: {e}",
                file=sys.stderr,
            )

    def log_link(
        self,
        url_or_path: str,
        text: Optional[str] = None,
        align: str = "left",
        margin: int = 5,
    ):
        try:
            link_text = text or url_or_path
            href = url_or_path

            if not (href.startswith("http://") or href.startswith("https://")):
                path = pathlib.Path(href)
                href = path.resolve().as_uri()
            
            styles = f"text-align: {align}; margin-top: {margin}px; margin-bottom: {margin}px;"
            html_link_tag = (
                f'<div style="{styles}"><a href="{href}" style="{{theme.api_link}}">{link_text}</a></div>'
            )

            print(f"PYSM_HTML_BLOCK:{html_link_tag}", file=sys.stderr, flush=True)

        except Exception as e:
            print(
                f"PySM API Error: Failed to log link '{url_or_path}'. Reason: {e}",
                file=sys.stderr,
            )


    def get_available_metadata_fields(self) -> List[str]:
        """Возвращает список поддерживаемых полей метаданных."""
        return list(FIELD_MAP.keys())

    def get_document_metadata(
        self,
        doc_path: Optional[str] = None,
        fields: Union[str, List[str]] = "__all__",
        clear_before_write: bool = False,
        prefix: str = "psd_meta_",
    ) -> Dict[str, Any]:
        """
        Извлекает XMP и системные метаданные из документа Photoshop.
        """
        try:
            from photoshop import api
            from photoshop.api.enumerations import SaveOptions

            app = api.Application()
        except COMError as e:
            raise RuntimeError(
                f"Не удалось подключиться к Adobe Photoshop. Убедитесь, что приложение установлено. Системная ошибка: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Непредвиденная ошибка при подключении к Adobe Photoshop: {e}"
            )

        doc = None
        doc_was_opened_by_us = False
        final_doc_path = ""

        try:
            if doc_path:
                final_doc_path = str(pathlib.Path(doc_path).resolve())
                if not os.path.exists(final_doc_path):
                    raise FileNotFoundError(f"Файл не найден: {final_doc_path}")

                is_already_open = False
                if app.documents:
                    for open_doc in app.documents:
                        if (
                            str(pathlib.Path(open_doc.fullName).resolve())
                            == final_doc_path
                        ):
                            doc = open_doc
                            is_already_open = True
                            break

                if not is_already_open:
                    doc = app.open(final_doc_path)
                    doc_was_opened_by_us = True
            else:
                if not app.documents:
                    raise RuntimeError("Нет открытых документов в Adobe Photoshop.")
                doc = app.activeDocument
                final_doc_path = doc.fullName

            if not doc:
                raise RuntimeError("Не удалось получить доступ к документу Photoshop.")

            if clear_before_write:
                all_possible_vars_to_clear = [
                    f"{prefix}{key}" for key in FIELD_MAP.keys()
                ]
                all_possible_vars_to_clear.append(f"{prefix}doc")
                self.remove(all_possible_vars_to_clear)
                print(
                    f"Все переменные с префиксом '{prefix}' удалены из контекста.",
                    file=sys.stderr,
                )

            # --- БЛОК 1: ИЗВЛЕЧЕНИЕ СИСТЕМНЫХ МЕТАДАННЫХ (С ЯВНЫМ ПРИВЕДЕНИЕМ ТИПОВ) ---
            doc_info_dict = {}
            try:
                doc_info_dict["name"] = str(doc.name)
            except Exception:
                pass
            try:
                doc_info_dict["fullName"] = str(doc.fullName)
            except Exception:
                pass
            try:
                doc_info_dict["width"] = int(doc.width)
            except Exception:
                pass
            try:
                doc_info_dict["height"] = int(doc.height)
            except Exception:
                pass
            try:
                doc_info_dict["resolution"] = float(doc.resolution)
            except Exception:
                pass
            try:
                doc_info_dict["colorProfileName"] = str(doc.colorProfileName)
            except Exception:
                pass
            try:
                doc_info_dict["bitsPerChannel"] = str(doc.bitsPerChannel)
            except Exception:
                pass
            try:
                doc_info_dict["mode"] = str(doc.mode)
            except Exception:
                pass

            if final_doc_path and os.path.exists(final_doc_path):
                try:
                    doc_info_dict["file_size"] = os.path.getsize(final_doc_path)
                except Exception:
                    pass
                try:
                    doc_info_dict["creation_time"] = datetime.fromtimestamp(
                        os.path.getctime(final_doc_path)
                    ).isoformat()
                except Exception:
                    pass
                try:
                    doc_info_dict["modification_time"] = datetime.fromtimestamp(
                        os.path.getmtime(final_doc_path)
                    ).isoformat()
                except Exception:
                    pass

            # --- Логика извлечения XMP-данных ---
            results_to_update = {}
            raw_xmp_data = doc.xmpMetadata.rawData
            if not raw_xmp_data:
                print(
                    "Предупреждение: XMP метаданные в файле отсутствуют.",
                    file=sys.stderr,
                )
            else:
                clean_xml = raw_xmp_data.strip("\x00")
                root = ET.fromstring(clean_xml)
                description_node = root.find(".//rdf:Description", XML_NAMESPACES)
                if description_node is None:
                    print(
                        "Ошибка: Не удалось найти блок <rdf:Description> в XMP.",
                        file=sys.stderr,
                    )
                else:
                    fields_to_extract = (
                        list(FIELD_MAP.keys())
                        if fields == "__all__"
                        else (fields if isinstance(fields, list) else [fields])
                    )

                    for field_name in fields_to_extract:
                        if field_name not in FIELD_MAP:
                            continue
                        xpath, field_type = FIELD_MAP[field_name]
                        node = description_node.find(xpath, XML_NAMESPACES)
                        result_value: Any = None
                        if node is not None:
                            if field_type == "simple":
                                result_value = node.text
                            elif field_type in ["array", "structure"]:
                                items = [
                                    li.text
                                    for li in node.findall(".//rdf:li", XML_NAMESPACES)
                                    if li.text
                                ]
                                if field_type == "structure":
                                    structured_dict = {}
                                    for item in items:
                                        if ":" in item:
                                            key_part, val_part = item.split(":", 1)
                                            structured_dict[key_part.strip()] = (
                                                val_part.strip()
                                            )
                                    result_value = structured_dict
                                else:
                                    result_value = items
                        results_to_update[f"{prefix}{field_name}"] = result_value

            # --- БЛОК 2: Добавляем системные метаданные в общий результат ---
            if doc_info_dict:
                results_to_update[f"{prefix}doc"] = doc_info_dict

        finally:
            if doc and doc_was_opened_by_us:
                doc.close(SaveOptions.DoNotSaveChanges)

        if results_to_update:
            self.update(results_to_update)

        return results_to_update

# 3. БЛОК: Создание глобального экземпляра-синглтона
# ==============================================================================
# Этот объект является единственной точкой входа для пользовательских скриптов
# для взаимодействия с контекстом.
pysm_context = PySMContext()


# 1. БЛОК: Класс ConfigResolver (ИЗМЕНЕН)
# ==============================================================================
class ConfigResolver:
    """
    Универсальный помощник для получения конфигурации скрипта с учетом приоритетов
    и автоматической обработкой путей и шаблонов.

    Приоритеты получения значения:
    1. Аргумент командной строки (высший приоритет).
    2. Значение из контекста PySM.
    3. Значение по умолчанию, определенное в ArgumentParser.

    Соглашения об именах для автоматической обработки:
    ---------------------------------------------------
    - Имена аргументов, содержащие 'path', 'dir', 'file' или 'folder'
      (например, 'source_path', 'output_dir', 'config_file'),
      будут автоматически обработаны как пути. Если скрипт запущен под управлением
      PySM, относительные пути будут разрешены от папки коллекции.
      В автономном режиме они будут разрешены от текущей рабочей директории.
    - Все строковые аргументы (включая пути) перед разрешением путей
      проходят через обработчик шаблонов, который заменяет в них
      плейсхолдеры вида {имя_переменной_контекста}.
    """

    # 1. БЛОК: Конструктор __init__ (ИЗМЕНЕН)
    def __init__(
        self,
        parser: argparse.ArgumentParser,
        force_path_args: Optional[List[str]] = None,
    ):
        """
        Инициализирует резолвер с парсером argparse.

        Args:
            parser (argparse.ArgumentParser): Парсер аргументов для скрипта.
            force_path_args (Optional[List[str]]): Список имен аргументов, которые
                нужно принудительно обрабатывать как пути, независимо от их имени.
        """
        self._parser = parser
        self._cli_args, _ = parser.parse_known_args()
        self._context = pysm_context
        self._is_managed = self._context._context_file_path is not None
        self._arg_actions = {action.dest: action for action in self._parser._actions}
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Преобразуем список в множество для быстрой проверки 'in'
        self._force_path_args = set(force_path_args or [])
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    def _convert_to_expected_type(self, value: Any, param_name: str) -> Any:
        if value is None:
            return None
        action = self._arg_actions.get(param_name)
        if action and action.nargs in ("+", "*") and isinstance(value, str):
            return [line for line in value.splitlines() if line]
        return value

    # 2. БЛОК: Метод get (ИЗМЕНЕН)
    def get(self, param_name: str, default: Any = None) -> Any:
        cli_value = getattr(self._cli_args, param_name, None)
        default_value = self._parser.get_default(param_name)
        raw_value: Optional[Any] = None
        if cli_value is not None and cli_value != default_value:
            raw_value = cli_value
        elif self._is_managed:
            raw_value = self._context.get(param_name, default_value)
        else:
            raw_value = default_value

        processed_value = raw_value
        if isinstance(raw_value, str):
            if self._is_managed:
                processed_value = self._context.resolve_template(raw_value)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ---
            # КОММЕНТАРИЙ: Проверяем, является ли строка URL-адресом
            is_url = processed_value.lower().startswith(("http://", "https://"))

            param_name_lower = param_name.lower()
            is_path_like = any(
                keyword in param_name_lower
                for keyword in ["path", "dir", "file", "folder"]
            )
            force_as_path = param_name in self._force_path_args

            # КОММЕНТАРИЙ: Обрабатываем как путь, только если это НЕ URL
            if not is_url and (is_path_like or force_as_path) and processed_value:
                if self._is_managed:
                    processed_value = str(self._context.resolve_path(processed_value))
                else:
                    processed_value = str(pathlib.Path(processed_value).resolve())
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        final_value = self._convert_to_expected_type(processed_value, param_name)
        return final_value if final_value is not None else default_value

    def resolve_all(self) -> Namespace:
        config = Namespace()
        for action in self._parser._actions:
            if action.dest != "help":
                setattr(config, action.dest, self.get(action.dest))
        return config
