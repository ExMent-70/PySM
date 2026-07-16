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

Данные автоматически сохраняются в файл при завершении работы скрипта.
"""

import argparse
import atexit  # <-- НОВЫЙ ИМПОРТ ДЛЯ АВТОСОХРАНЕНИЯ
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
from .context_store import FileContextStore, ContextStoreError
from .context_shared_memory import SharedMemoryContextStore, SharedMemoryContextError


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


# ==============================================================================
# ОСНОВНОЙ КЛАСС УПРАВЛЕНИЯ КОНТЕКСТОМ ВЫПОЛНЕНИЯ
# ==============================================================================

class PySMContext:
    """
    Класс, инкапсулирующий логику чтения, записи и управления данными
    в общем файле контекста (`pysm_context.json`).
    Поддерживает кэширование записи (Lazy Write) для оптимизации I/O.
    """

    def __init__(self):
        """Инициализирует объект, находит путь к файлу контекста и настраивает кэш."""
        self._context_file_path: Optional[pathlib.Path] = None
        self._context_shm_name: Optional[str] = None
        self._context_store: Optional[Union[FileContextStore, SharedMemoryContextStore]] = None
        self._context_store_generation: int = -1
        self._context_mode: str = "file"
        self._raw_context_data_cache: Optional[Dict[str, Any]] = None
        self._is_dirty: bool = False  # Флаг наличия несохраненных изменений
        self._initialize()
        
        # Регистрируем автоматическое сохранение при нормальном завершении скрипта
        atexit.register(self.commit)

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
        parser.add_argument(
            "--pysm-context-shm-name", type=str, dest="pysm_context_shm_name"
        )
        parser.add_argument(
            "--pysm-context-mode", type=str, dest="pysm_context_mode"
        )
        args, remaining_argv = parser.parse_known_args(args=sys.argv[1:])
        context_file_arg = args.pysm_context_file_path or os.environ.get("PYSM_CONTEXT_FILE")
        context_shm_arg = args.pysm_context_shm_name or os.environ.get("PYSM_CONTEXT_SHM_NAME")
        context_mode_arg = args.pysm_context_mode or os.environ.get("PYSM_CONTEXT_MODE")
        self._context_mode = (
            context_mode_arg or ("shared_memory" if context_shm_arg else "file")
        ).strip().lower()

        if context_file_arg:
            self._context_file_path = pathlib.Path(context_file_arg)

        if context_shm_arg and self._context_mode == "shared_memory":
            self._context_shm_name = context_shm_arg
            try:
                self._context_store = SharedMemoryContextStore.open(context_shm_arg)
            except SharedMemoryContextError as e:
                print(
                    f"PySM Context Error: failed to open shared memory context: {e}",
                    file=sys.stderr,
                )
                self._context_store = None

        if self._context_store is None and self._context_file_path:
            self._context_store = FileContextStore(self._context_file_path)

        sys.argv = [sys.argv[0]] + remaining_argv
        self._read_data()

    def _atomic_write_json(self, target_path: pathlib.Path, data_to_dump: Dict[str, Any]) -> None:
        """Write JSON via a same-directory temp file so readers never see an empty file."""

        temp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.{id(data_to_dump)}.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, target_path)

    def _read_data(self) -> Dict[str, Any]:
        """Читает данные из файла контекста и кэширует их."""
        context_store = getattr(self, "_context_store", None)
        if context_store and context_store.backend_name == "shared_memory":
            try:
                current_generation = context_store.generation
                if (
                    self._raw_context_data_cache is None
                    or current_generation != self._context_store_generation
                ):
                    self._raw_context_data_cache = context_store.load()
                    self._context_store_generation = context_store.generation
                return self._raw_context_data_cache
            except ContextStoreError as e:
                print(
                    f"PySM Context Error: failed to read shared memory context: {e}",
                    file=sys.stderr,
                )
                self._raw_context_data_cache = self._raw_context_data_cache or {}
                return self._raw_context_data_cache

        if self._raw_context_data_cache is not None:
            return self._raw_context_data_cache
        if context_store:
            self._raw_context_data_cache = context_store.load()
            self._context_store_generation = context_store.generation
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

    def commit(self) -> None:
        """
        Принудительно записывает кэшированные данные в файл контекста,
        если были произведены изменения (is_dirty == True).
        """
        if not self._is_dirty:
            return

        context_store = getattr(self, "_context_store", None)
        if self._raw_context_data_cache is not None and context_store:
            try:
                context_store.save(self._raw_context_data_cache)
                self._context_store_generation = context_store.generation
                self._is_dirty = False
            except Exception as e:
                print(f"PySM Context Error: failed to save context store: {e}", file=sys.stderr)
            return

        path = self._context_file_path
        if not path:
            # Если файл не передан (запуск вне PySM), мы просто сбрасываем флаг,
            # позволяя скрипту штатно работать с контекстом в ОЗУ
            self._is_dirty = False
            return
            
        if self._raw_context_data_cache is None:
            return

        try:
            self._atomic_write_json(path, self._raw_context_data_cache)
            self._is_dirty = False
        except Exception as e:
            print(f"PySM Context Error: Не удалось сохранить файл контекста: {e}", file=sys.stderr)

    def _mark_dirty(self, data: Dict[str, Any], force_commit: bool = False) -> None:
        """Обновляет кэш в памяти и помечает его для отложенного сохранения."""
        self._raw_context_data_cache = data
        self._is_dirty = True
        context_store = getattr(self, "_context_store", None)
        if context_store and context_store.backend_name == "shared_memory":
            self.commit()
            return
        if force_commit:
            self.commit()

    @property
    def is_managed(self) -> bool:
        return getattr(self, "_context_store", None) is not None or self._context_file_path is not None

    @property
    def backend_name(self) -> str:
        context_store = getattr(self, "_context_store", None)
        if context_store:
            return context_store.backend_name
        return "memory"

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

    def _is_list_index_token(self, token: str) -> bool:
        """Проверяет, можно ли сегмент dot-notation использовать как индекс списка."""
        if not token:
            return False
        return token.isdigit()

    def _get_nested_value(self, root_value: Any, path_parts: List[str], default: Any = None) -> Any:
        """
        Получает вложенное значение из dict/list-структуры.

        Поддерживает:
        - ключи словаря: a.b.c
        - индексы списка: items.0.name
        """
        current_value = root_value

        for part in path_parts:
            if isinstance(current_value, dict):
                if part not in current_value:
                    return default
                current_value = current_value[part]
                continue

            if isinstance(current_value, list):
                if not self._is_list_index_token(part):
                    return default
                index = int(part)
                if index < 0 or index >= len(current_value):
                    return default
                current_value = current_value[index]
                continue

            return default

        return current_value

    def _nested_path_exists(self, root_value: Any, path_parts: List[str]) -> bool:
        """
        Проверяет существование вложенного пути внутри dict/list-структуры.

        Отличает отсутствующий путь от существующего значения None.
        """
        current_value = root_value

        for part in path_parts:
            if isinstance(current_value, dict):
                if part not in current_value:
                    return False
                current_value = current_value[part]
                continue

            if isinstance(current_value, list):
                if not self._is_list_index_token(part):
                    return False
                index = int(part)
                if index < 0 or index >= len(current_value):
                    return False
                current_value = current_value[index]
                continue

            return False

        return True

    def _resolve_nested_parent(
        self,
        root_value: Any,
        path_parts: List[str],
        create_missing_dicts: bool = False,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Возвращает родительский контейнер и последний сегмент пути.

        Для dict допускается автоматическое создание промежуточных словарей.
        Для list автоматическое расширение не выполняется намеренно, чтобы не
        создавать скрытых мутаций и не угадывать структуру данных.
        """
        if not path_parts:
            return None, None

        current_value = root_value

        for part in path_parts[:-1]:
            if isinstance(current_value, dict):
                if part not in current_value:
                    if create_missing_dicts:
                        current_value[part] = {}
                    else:
                        return None, None

                if not isinstance(current_value[part], (dict, list)):
                    if create_missing_dicts:
                        current_value[part] = {}
                    else:
                        return None, None

                current_value = current_value[part]
                continue

            if isinstance(current_value, list):
                if not self._is_list_index_token(part):
                    return None, None
                index = int(part)
                if index < 0 or index >= len(current_value):
                    return None, None
                current_value = current_value[index]
                continue

            return None, None

        return current_value, path_parts[-1]

    def exists(self, key: str) -> bool:
        """
        Проверяет существование переменной или вложенного значения.

        В отличие от get()/get_structured(), корректно отличает отсутствующий
        ключ от существующего значения None.

        Примеры:
        - exists("my_var")
        - exists("wf_school_info.city")
        - exists("templates.0.name")
        """
        if not key:
            return False

        data = self._read_data()
        keys = key.split(".")
        base_key = keys[0]

        if base_key not in data:
            return False

        if len(keys) == 1:
            return True

        variable_data = data.get(base_key)
        if not variable_data or not isinstance(variable_data, dict):
            return False

        root_value = variable_data.get("value")
        return self._nested_path_exists(root_value, keys[1:])

    def get_structured(self, key: str, default: Any = None) -> Any:
        """
        Получает значение переменной или вложенное значение по dot-notation.

        Поддерживает:
        - верхнеуровневые переменные;
        - вложенные dict-ключи: a.b.c;
        - индексы списков: items.0.name.

        Если путь отсутствует, возвращает default.
        """
        if not key:
            return default

        keys = key.split(".")
        base_key = keys[0]
        value = self.get(base_key, default if len(keys) == 1 else None)

        if len(keys) == 1:
            return value

        if value is None:
            return default

        return self._get_nested_value(value, keys[1:], default)

    def _send_ipc_update(self, action: str, **kwargs):
        """Отправляет мгновенную команду обновления контекста в процесс PySM."""
        try:
            payload = {"action": action}
            payload.update(kwargs)
            print(f"PYSM_CONTEXT_UPDATE:{json.dumps(payload)}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"PySM API Error: Failed to send context update. Reason: {e}", file=sys.stderr)

    def set(self, key: str, value: Any, var_type: Optional[str] = None, commit: bool = False) -> None:
        """Устанавливает или создает переменную в контексте (UPSERT)."""
        data = self._read_data()
        variable_data = data.get(key)

        final_type = var_type
        if variable_data and isinstance(variable_data, dict):
            if variable_data.get("read_only", False):
                print(f"PySM Context Warning: Переменная '{key}' защищена от записи.", file=sys.stderr)
                return
            variable_data["value"] = value
            if var_type:
                variable_data["type"] = var_type
            final_type = var_type or variable_data.get("type", "string")
        else:
            final_type = var_type if var_type else self._infer_type_from_value(value)
            data[key] = {
                "type": final_type,
                "value": value,
                "description": "Auto-created by script",
                "read_only": False,
                "choices": None,
            }

        self._mark_dirty(data, force_commit=commit)
        # Мгновенно отправляем изменения в оперативную память PySM
        self._send_ipc_update("set", key=key, value=value, var_type=final_type)
        
    def set_structured(self, key_path: str, value: Any, commit: bool = False) -> None:
        """
        Устанавливает значение переменной, поддерживая dot-notation.

        Поведение:
        - без точки работает как set();
        - для JSON-словарей создаёт отсутствующие промежуточные dict-узлы;
        - поддерживает существующие индексы списков, например: items.0.name;
        - не расширяет списки автоматически, чтобы не создавать скрытых структур.
        """
        if not key_path:
            print("PySM Context Error: Empty key path is not allowed.", file=sys.stderr)
            return

        if "." not in key_path:
            self.set(key_path, value, commit=commit)
            return

        keys = key_path.split(".")
        base_key = keys[0]

        data = self._read_data()
        variable_data = data.get(base_key)

        if variable_data and isinstance(variable_data, dict):
            if variable_data.get("read_only", False):
                print(f"PySM Context Warning: Переменная '{base_key}' защищена от записи.", file=sys.stderr)
                return

            if variable_data.get("type") != "json" or not isinstance(variable_data.get("value"), dict):
                print(
                    f"PySM Context Error: Невозможно применить точечную нотацию к '{base_key}', "
                    "так как это не 'json' объект.",
                    file=sys.stderr,
                )
                return

            root_value = variable_data["value"]
        else:
            root_value = {}
            variable_data = {
                "type": "json",
                "value": root_value,
                "description": "Auto-created by script via structured set",
                "read_only": False,
                "choices": None,
            }
            data[base_key] = variable_data

        parent_container, target_key = self._resolve_nested_parent(
            root_value,
            keys[1:],
            create_missing_dicts=True,
        )

        if parent_container is None or target_key is None:
            print(f"PySM Context Error: Не удалось разрешить путь '{key_path}'.", file=sys.stderr)
            return

        if isinstance(parent_container, dict):
            parent_container[target_key] = value
        elif isinstance(parent_container, list):
            if not self._is_list_index_token(target_key):
                print(
                    f"PySM Context Error: Сегмент '{target_key}' не является индексом списка.",
                    file=sys.stderr,
                )
                return
            index = int(target_key)
            if index < 0 or index >= len(parent_container):
                print(
                    f"PySM Context Error: Индекс '{index}' вне диапазона списка для пути '{key_path}'.",
                    file=sys.stderr,
                )
                return
            parent_container[index] = value
        else:
            print(f"PySM Context Error: Родительский контейнер пути '{key_path}' не поддерживает запись.", file=sys.stderr)
            return

        self._mark_dirty(data, force_commit=commit)
        self._send_ipc_update("set", key=base_key, value=root_value, var_type="json")        

    def update(self, update_dict: Dict[str, Any], commit: bool = False) -> None:
        """Обновляет несколько переменных в контексте из словаря."""
        data = self._read_data()
        for key, value in update_dict.items():
            variable_data = data.get(key)
            final_type = "string"
            if variable_data and isinstance(variable_data, dict):
                if variable_data.get("read_only", False):
                    print(f"PySM Context Warning: Переменная '{key}' защищена от записи.", file=sys.stderr)
                    continue
                variable_data["value"] = value
                final_type = variable_data.get("type", "string")
            else:
                final_type = self._infer_type_from_value(value)
                data[key] = {
                    "type": final_type,
                    "value": value,
                    "description": "Auto-created by script",
                    "read_only": False,
                    "choices": None,
                }
            self._send_ipc_update("set", key=key, value=value, var_type=final_type)
            
        self._mark_dirty(data, force_commit=commit)

    def remove(self, keys_to_remove: Optional[Union[str, List[str]]] = None, commit: bool = False) -> None:
        """
        Удаляет переменные из контекста.

        Поддерживает:
        - удаление верхнеуровневых переменных;
        - удаление вложенных dict-ключей по dot-notation;
        - удаление элементов list по числовому индексу в dot-notation.

        Примеры:
        - remove("my_var")
        - remove("wf_school_info.city")
        - remove("template.override_labels.0")
        """
        data = self._read_data()

        if keys_to_remove is None:
            keys_for_deletion = [k for k in data.keys() if k not in _RESERVED_KEYS]
        elif isinstance(keys_to_remove, str):
            keys_for_deletion = [keys_to_remove]
        elif isinstance(keys_to_remove, list):
            keys_for_deletion = keys_to_remove
        else:
            return

        if not keys_for_deletion:
            return

        removed_top_level_keys: List[str] = []
        updated_structured_keys: set[str] = set()

        for key in keys_for_deletion:
            if not key:
                continue

            if key in data:
                if key in _RESERVED_KEYS:
                    continue
                data.pop(key, None)
                removed_top_level_keys.append(key)
                continue

            if "." not in key:
                continue

            keys = key.split(".")
            base_key = keys[0]
            variable_data = data.get(base_key)

            if not variable_data or not isinstance(variable_data, dict):
                continue

            if variable_data.get("read_only", False):
                print(f"PySM Context Warning: variable '{base_key}' is read-only.", file=sys.stderr)
                continue

            root_value = variable_data.get("value")
            if variable_data.get("type") != "json" or not isinstance(root_value, dict):
                continue

            parent_container, target_key = self._resolve_nested_parent(
                root_value,
                keys[1:],
                create_missing_dicts=False,
            )

            if parent_container is None or target_key is None:
                continue

            removed = False

            if isinstance(parent_container, dict):
                if target_key in parent_container:
                    del parent_container[target_key]
                    removed = True

            elif isinstance(parent_container, list):
                if self._is_list_index_token(target_key):
                    index = int(target_key)
                    if 0 <= index < len(parent_container):
                        parent_container.pop(index)
                        removed = True

            if removed:
                updated_structured_keys.add(base_key)

        if not removed_top_level_keys and not updated_structured_keys:
            return

        self._mark_dirty(data, force_commit=commit)

        if removed_top_level_keys:
            self._send_ipc_update("remove", keys=removed_top_level_keys)

        for base_key in updated_structured_keys:
            root_value = data[base_key]["value"]
            self._send_ipc_update("set", key=base_key, value=root_value, var_type="json")


    def get_all(self) -> Dict[str, Any]:
        """Возвращает все переменные и их значения из контекста."""
        raw_data = self._read_data()
        return {k: v.get("value") for k, v in raw_data.items()}

    def resolve_template(self, template_string: Optional[str]) -> str:
        """
        Рекурсивно заменяет плейсхолдеры {key} в строке на значения из контекста.
        Выполняется до тех пор, пока все возможные плейсхолдеры не будут разрешены.
        """
        if not template_string or not isinstance(template_string, str) or "{" not in template_string:
            return template_string if template_string is not None else ""

        resolved_string = template_string
        max_depth = 10  # Защита от бесконечной рекурсии

        for _ in range(max_depth):
            placeholders = re.findall(r"{([^}]+)}", resolved_string)
            if not placeholders:
                break

            made_changes = False
            for key in set(placeholders):
                val = self.get_structured(key, default=None)

                if val is not None:
                    str_val = str(val)
                    if str_val != f"{{{key}}}":
                        resolved_string = resolved_string.replace(f"{{{key}}}", str_val)
                        made_changes = True

            if not made_changes:
                break

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

    def get_schema(self, key: str, default: Any = None) -> Any:
        """
        Возвращает JSON-схему для переменной контекста.

        Соглашение:
        - данные хранятся в переменной <key>;
        - схема хранится в переменной <key>_schema.

        Пример:
        - get_schema("wf_school_info") читает "wf_school_info_schema".
        """
        if not key:
            return default
        return self.get_structured(f"{key}_schema", default=default)

    def set_next_script(self, instance_id: str, commit: bool = False) -> None:
        """
        Отправляет команду маршрутизации (переход к другому скрипту)
        в главный процесс PyScriptManager через stderr.
        Параметр commit оставлен для обратной совместимости, но игнорируется.
        """
        try:
            routing_data = {"target_id": instance_id}
            # Отправляем специальный маркер, который перехватит ScriptRunner "на лету"
            print(f"PYSM_ROUTING_CMD:{json.dumps(routing_data)}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"PySM API Error: Failed to send routing command. Reason: {e}", file=sys.stderr)

    def list_instances(self) -> List[Dict[str, str]]:
        """
        Возвращает список данных о всех экземплярах в текущем наборе.
        Каждый элемент - это словарь {"id": "...", "name": "..."}.
        """
        return self.get("pysm_set_instance_ids",[])

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
                if ext in["png", "jpg", "jpeg", "gif", "bmp"]
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

    def log_html(
        self,
        html_content: str,
        align: str = "left",
        margin: int = 5,
        padding: int = 10,
    ) -> None:
        """
        Выводит произвольный HTML-контент в консоль PyScriptManager.
        """
        try:
            if not html_content:
                return

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ:
            # Удаляем символы переноса строки, так как передача через stderr
            # требует, чтобы сообщение с префиксом PYSM_HTML_BLOCK было одной строкой.
            # Заменяем их на пробелы, чтобы не склеить слова.
            clean_content = html_content.replace("\n", "<br>").replace("\r", "")

            styles = f"text-align: {align}; margin-top: {margin}px; margin-bottom: {margin}px; padding: {padding}px;"
            #wrapped_html = f'<div style="{styles}">{clean_content}</div>'
            wrapped_html = f'{clean_content}'

            print(f"PYSM_HTML_BLOCK:{wrapped_html}", file=sys.stderr, flush=True)

        except Exception as e:
            print(
                f"PySM API Error: Failed to log HTML content. Reason: {e}",
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
                all_possible_vars_to_clear =[
                    f"{prefix}{key}" for key in FIELD_MAP.keys()
                ]
                all_possible_vars_to_clear.append(f"{prefix}doc")
                self.remove(all_possible_vars_to_clear)
                print(
                    f"Все переменные с префиксом '{prefix}' удалены из контекста.",
                    file=sys.stderr,
                )

            doc_info_dict = {}
            try: doc_info_dict["name"] = str(doc.name)
            except Exception: pass
            try: doc_info_dict["fullName"] = str(doc.fullName)
            except Exception: pass
            try: doc_info_dict["width"] = int(doc.width)
            except Exception: pass
            try: doc_info_dict["height"] = int(doc.height)
            except Exception: pass
            try: doc_info_dict["resolution"] = float(doc.resolution)
            except Exception: pass
            try: doc_info_dict["colorProfileName"] = str(doc.colorProfileName)
            except Exception: pass
            try: doc_info_dict["bitsPerChannel"] = str(doc.bitsPerChannel)
            except Exception: pass
            try: doc_info_dict["mode"] = str(doc.mode)
            except Exception: pass

            if final_doc_path and os.path.exists(final_doc_path):
                try: doc_info_dict["file_size"] = os.path.getsize(final_doc_path)
                except Exception: pass
                try: doc_info_dict["creation_time"] = datetime.fromtimestamp(os.path.getctime(final_doc_path)).isoformat()
                except Exception: pass
                try: doc_info_dict["modification_time"] = datetime.fromtimestamp(os.path.getmtime(final_doc_path)).isoformat()
                except Exception: pass

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
                                items =[
                                    li.text
                                    for li in node.findall(".//rdf:li", XML_NAMESPACES)
                                    if li.text
                                ]
                                if field_type == "structure":
                                    structured_dict = {}
                                    for item in items:
                                        if ":" in item:
                                            key_part, val_part = item.split(":", 1)
                                            structured_dict[key_part.strip()] = val_part.strip()
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

# ==============================================================================
# СИНГЛТОН КОНТЕКСТА
# ==============================================================================
pysm_context = PySMContext()


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
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
    PATH_KEYWORDS = {"path", "dir", "file", "folder"}
    _MISSING = object()  # Уникальный маркер для проверки отсутствия значения

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        force_path_args: Optional[List[str]] = None,
    ):
        self._parser = parser
        self._cli_args, _ = parser.parse_known_args()
        self._context = pysm_context
        self._is_managed = self._context.is_managed
        self._arg_actions = {action.dest: action for action in self._parser._actions}
        self._force_path_args = set(force_path_args or[])
        
        # Предварительно определяем аргументы, явно переданные через CLI
        self._explicit_cli_args = self._detect_explicit_cli_args()

    def _detect_explicit_cli_args(self) -> set[str]:
        """Определяет, какие аргументы были явно переданы через командную строку."""
        explicit_args = set()
        for action in self._parser._actions:
            if action.option_strings:
                # Опциональные аргументы: ищем совпадения напрямую в sys.argv
                for opt in action.option_strings:
                    # Поддерживает форматы --arg value и --arg=value
                    if any(arg == opt or arg.startswith(f"{opt}=") for arg in sys.argv):
                        explicit_args.add(action.dest)
                        break
            else:
                # Позиционные аргументы: полагаемся на то, что они не равны дефолту
                cli_val = getattr(self._cli_args, action.dest, None)
                if cli_val != self._parser.get_default(action.dest):
                    explicit_args.add(action.dest)
        return explicit_args

    def _get_raw_value(self, param_name: str) -> Any:
        """Извлекает сырое значение с учетом приоритетов (CLI -> Context -> Default)."""
        # 1. Явно переданный CLI аргумент
        if param_name in self._explicit_cli_args:
            return getattr(self._cli_args, param_name)

        # 2. Значение из контекста PySM
        default_value = self._parser.get_default(param_name)
        if self._is_managed:
            context_value = self._context.get(param_name, default=self._MISSING)
            if context_value is not self._MISSING:
                return context_value

        # 3. Дефолтное значение argparse
        return default_value

    def _process_string_value(self, param_name: str, value: Any) -> Any:
        """Обрабатывает строковые значения: применяет шаблоны, URL и пути."""
        if not isinstance(value, str) or not value:
            return value

        # 1. Рекурсивное разрешение шаблонов PySM
        if self._is_managed:
            value = self._context.resolve_template(value)

        # 2. Игнорируем URL-адреса для обработки путей
        if value.lower().startswith(("http://", "https://")):
            return value

        # 3. Разрешение путей
        param_name_lower = param_name.lower()
        is_path_like = any(kw in param_name_lower for kw in self.PATH_KEYWORDS)
        force_as_path = param_name in self._force_path_args

        if is_path_like or force_as_path:
            if self._is_managed:
                return str(self._context.resolve_path(value))
            return str(pathlib.Path(value).resolve())

        return value

    def _convert_to_expected_type(self, param_name: str, value: Any) -> Any:
        """Приводит значение к ожидаемому типу на основе конфигурации argparse."""
        if value is None:
            return None
            
        action = self._arg_actions.get(param_name)
        
        # Преобразование многострочного текста в список, если argparse ожидает список
        if action and action.nargs in ("+", "*") and isinstance(value, str):
            return [line for line in value.splitlines() if line]
            
        return value

    def get(self, param_name: str, default: Any = None) -> Any:
        """Получает итоговое значение параметра, пропуская его через все стадии обработки."""
        raw_value = self._get_raw_value(param_name)
        processed_value = self._process_string_value(param_name, raw_value)
        final_value = self._convert_to_expected_type(param_name, processed_value)
        return final_value if final_value is not None else default

    def resolve_all(self) -> Namespace:
        """Разрешает все аргументы парсера и возвращает итоговый Namespace."""
        config = Namespace()
        for action in self._parser._actions:
            if action.dest != "help":
                setattr(config, action.dest, self.get(action.dest))
        return config
