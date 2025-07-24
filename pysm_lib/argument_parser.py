# pysm_lib/argument_parser.py

import ast
import logging
import os
from typing import Dict, Any, Optional, List, Tuple

from .models import ScriptArgMetaDetailModel
from .locale_manager import LocaleManager

locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")


# 1. БЛОК: Эвристические функции для определения путей с системой приоритетов
# --------------------------------------------------------------------------

# Список кортежей, определяющий приоритет и тип для ключевых слов.
# Формат: (Приоритет, Тип результата, Список ключевых слов)
PATH_HINT_PRIORITY_LIST: List[Tuple[int, str, List[str]]] = [
    # Приоритет 1: Самые точные слова для директорий
    (1, "dir_path", ["dir", "folder", "directory"]),
    # Приоритет 2: Самые точные слова для файлов
    (2, "file_path", ["file", "log", "executable"]),
    # Приоритет 3: Менее однозначные слова, которые чаще указывают на файл
    (3, "file_path", ["input", "output", "source", "target"]),
    # Приоритет 4: Самое общее слово. Если ничего другого не подошло
    (4, "dir_path", ["path"]),
]


def _infer_type_from_path_string(value: Any) -> Optional[str]:
    """
    Эвристика для определения, является ли строка путем к файлу или директории.
    Возвращает 'file_path', 'dir_path' или None.
    """
    if not isinstance(value, str) or (
        "/" not in value and "\\" not in value and not value.startswith((".", ".."))
    ):
        return None

    if value.endswith(("/", "\\")):
        return "dir_path"

    basename = os.path.basename(value)
    if "." in basename and basename not in (".", ".."):
        return "file_path"

    return "dir_path"


def _infer_type_from_name(arg_name: str) -> Optional[str]:
    """
    Эвристика для определения типа пути по ключевым словам в имени аргумента.
    Использует систему приоритетов для разрешения конфликтов.
    """
    arg_name_lower = arg_name.lower().replace("_", "-")

    # Сортируем список по приоритету (первое число в кортеже)
    sorted_hints = sorted(PATH_HINT_PRIORITY_LIST, key=lambda x: x[0])

    # Проходим по списку в порядке приоритета
    for priority, type_name, keywords in sorted_hints:
        for keyword in keywords:
            # Ищем ключевое слово в имени аргумента
            if f"-{keyword}" in arg_name_lower or arg_name_lower.endswith(keyword):
                return type_name  # Нашли совпадение - сразу возвращаем тип

    return None  # Ничего не нашли


# 2. БЛОК: Улучшенный AST-визитор для парсинга аргументов
# -----------------------------------------------------


class ArgumentVisitor(ast.NodeVisitor):
    """
    AST-визитор для поиска вызовов parser.add_argument, извлекающий
    максимум метаданных.
    """

    def __init__(self):
        self.arguments: Dict[str, ScriptArgMetaDetailModel] = {}
        # Список имен, которые могут использоваться для парсера
        self.parser_names = {"parser", "arg_parser", "argument_parser"}

    def _literal_eval_safer(self, node: Optional[ast.expr]) -> Any:
        """Безопасное вычисление значения AST-узла."""
        if node is None:
            return None
        try:
            return ast.literal_eval(ast.unparse(node))
        except (ValueError, SyntaxError, TypeError):
            if isinstance(node, ast.Name):
                return node.id
            return ast.unparse(node)

    def visit_Assign(self, node: ast.Assign):
        """Находит переменные, которым присваивается ArgumentParser."""
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "argparse"
            and node.value.func.attr == "ArgumentParser"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.parser_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Находит вызовы метода add_argument у известных парсеров."""
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.parser_names
            and node.func.attr == "add_argument"
        ):
            self._parse_add_argument_call(node)
        self.generic_visit(node)

    def _parse_add_argument_call(self, node: ast.Call):
        """Основная логика парсинга вызова add_argument."""
        # 1. Извлекаем имена аргументов (например, '-f', '--file')
        arg_names = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        if not arg_names:
            return

        # 2. Находим "длинное" имя (например, 'file' из '--file')
        long_name = next(
            (name.lstrip("-") for name in arg_names if name.startswith("--")), None
        )
        if not long_name:
            long_name = arg_names[0].lstrip("-")

        # 3. Собираем все именованные аргументы в словарь
        kwargs = {kw.arg: kw.value for kw in node.keywords}

        # 4. Извлекаем и анализируем значения
        action = self._literal_eval_safer(kwargs.get("action"))
        default_val = self._literal_eval_safer(kwargs.get("default"))
        choices = self._literal_eval_safer(kwargs.get("choices"))
        nargs = self._literal_eval_safer(kwargs.get("nargs"))

        # 5. Определяем тип аргумента с учетом эвристики
        arg_type_node = kwargs.get("type")
        arg_type_str = "string"  # По умолчанию тип - строка

        # 5.1. Точное определение
        if isinstance(arg_type_node, ast.Name):
            type_map = {"int": "int", "float": "float", "str": "string"}
            arg_type_str = type_map.get(arg_type_node.id, "string")

        if action == "store_true":
            arg_type_str = "bool"
            if default_val is None:
                default_val = False
        elif action == "store_false":
            arg_type_str = "bool"
            if default_val is None:
                default_val = True

        if isinstance(choices, list) and choices:
            arg_type_str = "choice"

        if nargs in ("*", "+"):
            arg_type_str = "list"

        # 5.2. Эвристическое определение (только если тип остался 'string')
        if arg_type_str == "string":
            # Приоритет 2: Анализ значения default
            inferred_type_from_val = _infer_type_from_path_string(default_val)
            if inferred_type_from_val:
                arg_type_str = inferred_type_from_val
            else:
                # Приоритет 3: Анализ имени аргумента
                inferred_type_from_name = _infer_type_from_name(long_name)
                if inferred_type_from_name:
                    arg_type_str = inferred_type_from_name

        # 6. Создаем модель метаданных
        meta = ScriptArgMetaDetailModel(
            type=arg_type_str,
            description=self._literal_eval_safer(kwargs.get("help")),
            required=self._literal_eval_safer(
                kwargs.get("required", ast.Constant(False))
            ),
            default=default_val,
            choices=choices if isinstance(choices, list) else None,
        )

        self.arguments[long_name] = meta


# 3. БЛОК: Основная функция сканирования (без изменений)
# ----------------------------------------------------


def scan_for_arguments(file_path: str) -> Dict[str, ScriptArgMetaDetailModel]:
    """
    Сканирует Python-файл и извлекает из него аргументы argparse.

    :param file_path: Путь к .py файлу.
    :return: Словарь с метаданными найденных аргументов.
    """
    logger.info(
        locale_manager.get("argument_parser.log_info.scan_started", path=file_path)
    )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        tree = ast.parse(source_code)
        visitor = ArgumentVisitor()
        visitor.visit(tree)
        logger.info(
            locale_manager.get(
                "argument_parser.log_info.scan_finished", count=len(visitor.arguments)
            )
        )
        return visitor.arguments
    except FileNotFoundError:
        logger.error(
            locale_manager.get(
                "argument_parser.log_error.file_not_found", path=file_path
            )
        )
    except SyntaxError as e:
        logger.error(
            locale_manager.get(
                "argument_parser.log_error.syntax_error", path=file_path, error=e
            )
        )
    except Exception as e:
        logger.error(
            locale_manager.get(
                "argument_parser.log_error.unexpected_error", path=file_path, error=e
            ),
            exc_info=True,
        )

    return {}
