#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cli_var_increment.py
========================
Утилита для увеличения или уменьшения числовой переменной контекста без GUI.

Особенности:
- Читает текущее значение переменной из контекста.
- Увеличивает или уменьшает значение на заданное число.
- Поддерживает int и float.
- При ошибке преобразования завершает работу с кодом 1.
"""

# 1. БЛОК: Импорты и настройки
# ==============================================================================
import argparse
import logging
import sys
from argparse import Namespace
from decimal import Decimal, InvalidOperation


try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import format_error, format_success, read_context_value, write_context_value
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"
    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"
    IS_MANAGED_RUN = False


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# 2. БЛОК: Константы
# ==============================================================================
VALUE_TYPES = ["auto", "int", "float"]
OPERATIONS = ["add", "subtract"]


# 3. БЛОК: Работа с контекстом
# ==============================================================================
class ContextHandler:
    """
    Отвечает за чтение и сохранение переменной контекста PySM.
    """

    def __init__(self, var_name: str):
        self.var_name = var_name

    def read_value(self) -> str:
        """
        Читает текущее значение переменной из контекста.

        Если переменная отсутствует —
        автоматически используется значение 0.
        """

        if IS_MANAGED_RUN and pysm_context:

            try:
                result = read_context_value(pysm_context, self.var_name)
                value = result.value if result.exists else None

                if value is None:

                    logger.debug(
                        f"Переменная контекста "
                        f"'{self.var_name}' "
                        f"не найдена. "
                        f"Используется значение по умолчанию: 0"
                    )

                    return "0"

                return str(value)

            except Exception as e:

                logger.critical(
                    f"Ошибка при чтении переменной контекста "
                    f"'{self.var_name}': {e}"
                )

                sys.exit(1)

        logger.info(
            "Запуск в автономном режиме. "
            "Чтение из контекста недоступно."
        )

        sys.exit(1)

    def save_value(self, value: str) -> None:
        """
        Сохраняет новое значение в контекст.
        """

        if IS_MANAGED_RUN and pysm_context:
            try:
                write_context_value(pysm_context, self.var_name, value)
                logger.info(format_success(self.var_name, value) + "\n")

            except Exception as e:
                logger.critical(
                    f"Ошибка при сохранении переменной контекста "
                    f"'{self.var_name}': {e}"
                )
                sys.exit(1)

        else:
            logger.info(
                "Запуск в автономном режиме. "
                "Запись в контекст недоступна."
            )
            sys.exit(1)


# 4. БЛОК: Числовая обработка
# ==============================================================================
class NumericProcessor:
    """
    Выполняет числовое преобразование и изменение значения.
    """

    def __init__(
        self,
        current_value: str,
        delta_value: str,
        value_type: str,
        operation: str,
    ):
        self.current_value = current_value
        self.delta_value = delta_value
        self.value_type = value_type
        self.operation = operation

    def process(self) -> str:
        """
        Возвращает новое значение в виде строки.
        """

        current_number = self._parse_decimal(
            self.current_value,
            "текущее значение переменной",
        )
        delta_number = self._parse_decimal(
            self.delta_value,
            "значение изменения",
        )

        if self.operation == "add":
            result = current_number + delta_number
        elif self.operation == "subtract":
            result = current_number - delta_number
        else:
            raise ValueError(f"Неизвестная операция: {self.operation}")

        return self._format_result(result)

    @staticmethod
    def _parse_decimal(value: str, value_label: str) -> Decimal:
        """
        Преобразует строку в Decimal.
        """

        prepared_value = str(value).strip().replace(",", ".")

        if not prepared_value:
            raise ValueError(f"Пустое {value_label}.")

        try:
            return Decimal(prepared_value)

        except InvalidOperation as e:
            raise ValueError(
                f"Не удалось преобразовать {value_label} "
                f"в число: '{value}'"
            ) from e

    def _format_result(self, value: Decimal) -> str:
        """
        Форматирует результат согласно выбранному типу.
        """

        if self.value_type == "int":
            if value != value.to_integral_value():
                raise ValueError(
                    "Результат не является целым числом: "
                    f"{value}"
                )

            return str(int(value))

        if self.value_type == "float":
            return self._format_decimal_as_float(value)

        if self.value_type == "auto":
            if value == value.to_integral_value():
                return str(int(value))

            return self._format_decimal_as_float(value)

        raise ValueError(f"Неизвестный тип значения: {self.value_type}")

    @staticmethod
    def _format_decimal_as_float(value: Decimal) -> str:
        """
        Форматирует Decimal как строковое float-значение без лишних нулей.
        """

        normalized = value.normalize()

        if normalized == normalized.to_integral_value():
            return f"{normalized:.1f}"

        return format(normalized, "f")


# 5. БЛОК: Конфигурация
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Увеличивает или уменьшает числовую переменную контекста."
        )
    )

    parser.add_argument(
        "--inc_var_name",
        type=str,
        required=True,
        help="Имя переменной контекста. Поддерживается точечная нотация, например project.counter.",
    )

    parser.add_argument(
        "--inc_delta",
        type=str,
        default="1",
        help=(
            "Число, на которое нужно изменить "
            "значение переменной."
        ),
    )

    parser.add_argument(
        "--inc_operation",
        type=str,
        default="add",
        choices=OPERATIONS,
        help="Операция: add или subtract.",
    )

    parser.add_argument(
        "--inc_value_type",
        type=str,
        default="auto",
        choices=VALUE_TYPES,
        help="Тип результата: auto, int или float.",
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()

    return parser.parse_args()


# 6. БЛОК: Главная функция
# ==============================================================================
def main():
    """
    Оркестратор процесса.
    """

    config = get_config()

    context_handler = ContextHandler(config.inc_var_name)

    current_value = context_handler.read_value()

    logger.debug(
        f"Текущее значение <b>{config.inc_var_name}</b>: "
        f"<i>{current_value}</i>"
    )

    logger.debug(
        f"Операция: <b>{config.inc_operation}</b>, "
        f"изменение: <i>{config.inc_delta}</i>"
    )

    try:
        processor = NumericProcessor(
            current_value=current_value,
            delta_value=config.inc_delta,
            value_type=config.inc_value_type,
            operation=config.inc_operation,
        )

        new_value = processor.process()

    except Exception as e:
        logger.error(format_error(str(e)))
        sys.exit(1)

    context_handler.save_value(new_value)

    sys.exit(0)


# 7. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()
