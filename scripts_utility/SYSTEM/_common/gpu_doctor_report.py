"""Единое оформление консольных отчётов для диагностик GPU."""

from __future__ import annotations

import argparse
import os
import sys
from html import escape
from typing import Any

try:
    from pysm_lib.pysm_icons import icons
except ImportError:
    icons = None


class DiagnosticReport:
    """Формирует одинаковый отчёт для PySM и обычной консоли."""

    _TEXT_PREFIXES = {
        "success": "[УСПЕХ]",
        "warning": "[ПРЕДУПРЕЖДЕНИЕ]",
        "error": "[ОШИБКА]",
        "info": "[ИНФОРМАЦИЯ]",
    }
    _ICON_NAMES = {
        "success": "OK",
        "warning": "WARNING",
        "error": "ERROR",
        "info": "INFO",
    }

    def __init__(self, title: str) -> None:
        self.title = title
        self.html_enabled = os.environ.get("PY_SCRIPT_MANAGER_ACTIVE") == "1"
        self.success_count = 0
        self.warning_count = 0
        self.error_count = 0

    def _icon(self, name: str, size: int = 18) -> str:
        if not self.html_enabled or icons is None:
            return ""
        try:
            return getattr(icons, name)(size=size)
        except (AttributeError, RuntimeError, TypeError):
            return ""

    def _safe(self, value: Any) -> str:
        text = str(value)
        return escape(text) if self.html_enabled else text

    def begin(self) -> None:
        """Печатает заголовок отчёта."""
        if self.html_enabled:
            print(f"<h2>{self._icon('REPORT', 22)} {self._safe(self.title)}</h2>")
        else:
            print(f"\n=== {self.title} ===")

    def section(self, number: int, title: str) -> None:
        """Открывает нумерованный раздел."""
        if self.html_enabled:
            print(f"\n<h3>{number}. {self._safe(title)}</h3>")
        else:
            print(f"\n--- {number}. {title} ---")

    def detail(self, label: str, value: Any, *, indent: int = 0) -> None:
        """Печатает строку с именованным значением."""
        padding = "&nbsp;" * (indent * 4) if self.html_enabled else " " * (indent * 2)
        if self.html_enabled:
            print(f"{padding}<b>{self._safe(label)}:</b> {self._safe(value)}")
        else:
            print(f"{padding}{label}: {value}")

    def line(self, message: str, *, indent: int = 0) -> None:
        """Печатает поясняющую строку без статуса."""
        padding = "&nbsp;" * (indent * 4) if self.html_enabled else " " * (indent * 2)
        print(f"{padding}{self._safe(message)}")

    def _status(self, kind: str, message: str) -> None:
        if kind == "success":
            self.success_count += 1
        elif kind == "warning":
            self.warning_count += 1
        elif kind == "error":
            self.error_count += 1

        icon = self._icon(self._ICON_NAMES[kind])
        prefix = icon or self._TEXT_PREFIXES[kind]
        print(f"{prefix} {self._safe(message)}")

    def success(self, message: str) -> None:
        self._status("success", message)

    def warning(self, message: str) -> None:
        self._status("warning", message)

    def error(self, message: str) -> None:
        self._status("error", message)

    def info(self, message: str) -> None:
        self._status("info", message)

    def exception(self, context: str, error: BaseException) -> None:
        """Печатает русское описание и технический текст исключения."""
        self.error(context)
        self.detail("Технические сведения", f"{type(error).__name__}: {error}", indent=1)

    def finish(self, conclusion: str) -> None:
        """Печатает единый итог и счётчики статусов."""
        self.section(5, "Итоговое заключение")
        if self.error_count:
            self.error(conclusion)
        elif self.warning_count:
            self.warning(conclusion)
        else:
            self.success(conclusion)
        self.detail("Успешных проверок", self.success_count)
        self.detail("Предупреждений", self.warning_count)
        self.detail("Ошибок", self.error_count)
        if self.html_enabled:
            print(f"\n<b>{self._icon('OK')} Диагностика завершена.</b>\n")
        else:
            print("\nДиагностика завершена.\n")


class _RussianHelpFormatter(argparse.HelpFormatter):
    """Переводит стандартные заголовки справки argparse."""

    def start_section(self, heading: str | None) -> None:
        translations = {
            "options": "параметры",
            "optional arguments": "необязательные параметры",
            "positional arguments": "позиционные параметры",
        }
        super().start_section(translations.get(heading, heading))


class RussianArgumentParser(argparse.ArgumentParser):
    """ArgumentParser с полностью русской стандартной справкой."""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "использование:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "использование:", 1)

    def error(self, message: str) -> None:
        translations = {
            "unrecognized arguments:": "неизвестные параметры:",
            "the following arguments are required:": "не указаны обязательные параметры:",
        }
        for source, target in translations.items():
            message = message.replace(source, target)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка: {message}\n")


def build_help_parser(description: str) -> argparse.ArgumentParser:
    """Создаёт парсер без англоязычных служебных сообщений."""
    parser = RussianArgumentParser(
        description=description,
        add_help=False,
        formatter_class=_RussianHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать эту справку и завершить работу",
    )
    return parser


def format_bytes(value: int | float | None) -> str:
    """Возвращает объём памяти в удобной для чтения форме."""
    if value is None:
        return "недоступно"
    gibibytes = float(value) / (1024**3)
    return f"{gibibytes:.2f} ГиБ"


def compact_python_version(version: str) -> str:
    """Убирает переносы строк из полного описания версии Python."""
    return " ".join(version.split())
