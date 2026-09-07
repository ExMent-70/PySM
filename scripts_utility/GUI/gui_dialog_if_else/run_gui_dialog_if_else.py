#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show an HTML choice dialog and route PySM execution by the result."""

from __future__ import annotations

import argparse
import html
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


GUI_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(GUI_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_SCRIPTS_DIR))


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

IS_MANAGED_RUN = False

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.input_processor import InputProcessor
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = pysm_context.is_managed
except ImportError as import_error:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    InputProcessor = None

    def format_success(var_name: str, value: Any) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"

    logger.debug("PySM API недоступен: %s", import_error)

try:
    from _common.html_dialog import (
        build_html_document,
        load_html_sources,
        log_html_to_console,
        show_html_message_dialog,
        validate_html_layout,
    )
except ImportError:
    print(
        "Ошибка: для работы этого скрипта требуется PySide6.",
        file=sys.stderr,
    )
    sys.exit(1)


OUTPUT_DIALOG = "dialog"
OUTPUT_CONSOLE_DIALOG = "console_dialog"
CONSOLE_OUTPUT_MODES = {OUTPUT_CONSOLE_DIALOG}
DIALOG_TYPES = ("yes_no", "yes_no_cancel")


def get_config() -> Namespace:
    """Resolve command-line and PySM collection-context parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Показывает HTML-диалог и выбирает следующий скрипт "
            "по ответу Yes или No."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--html_content",
        type=str,
        default="",
        help="Строка с HTML-разметкой для вывода.",
    )
    parser.add_argument(
        "--html_file",
        type=str,
        help="Путь к UTF-8 HTML-файлу, содержимое которого нужно вывести.",
    )
    parser.add_argument(
        "--html_output",
        type=str,
        choices=[OUTPUT_DIALOG, OUTPUT_CONSOLE_DIALOG],
        default=OUTPUT_CONSOLE_DIALOG,
        help="Показывать HTML только в диалоге или также в консоли PySM.",
    )
    parser.add_argument(
        "--html_align",
        type=str,
        choices=["left", "center", "right"],
        default="left",
        help="Горизонтальное выравнивание HTML-контента.",
    )
    parser.add_argument(
        "--html_margin",
        type=int,
        default=0,
        help="Вертикальный внешний отступ HTML-блока в пикселях.",
    )
    parser.add_argument(
        "--html_padding",
        type=int,
        default=10,
        help="Внутренний отступ HTML-блока в пикселях.",
    )
    parser.add_argument(
        "--html_style",
        type=str,
        default="script_description",
        help="Имя HTML-стиля из активной темы PySM.",
    )
    parser.add_argument(
        "--dlg_msg_var",
        type=str,
        default="dlg_go_var",
        help="Переменная Контекста Коллекции для результата диалога.",
    )
    parser.add_argument(
        "--dlg_msg_type",
        type=str,
        choices=DIALOG_TYPES,
        default="yes_no",
        help="Набор кнопок: yes_no или yes_no_cancel.",
    )
    parser.add_argument(
        "--dlg_msg_text_ok",
        type=str,
        default="Продолжить",
        help="Пользовательская подпись кнопки Yes.",
    )
    parser.add_argument(
        "--dlg_msg_text_no",
        type=str,
        default="Остановить",
        help="Пользовательская подпись кнопки No.",
    )
    parser.add_argument(
        "--dlg_msg_text_cancel",
        type=str,
        default="Отменить",
        help="Пользовательская подпись кнопки Cancel.",
    )
    parser.add_argument(
        "--dlg_msg_title",
        type=str,
        default="Информационное сообщение",
        help="Заголовок диалогового окна.",
    )
    parser.add_argument(
        "--dlg_msg_size_width",
        type=int,
        default=700,
        help="Начальная ширина диалогового окна в пикселях.",
    )
    parser.add_argument(
        "--dlg_msg_size_height",
        type=int,
        default=500,
        help="Начальная высота диалогового окна в пикселях.",
    )
    parser.add_argument(
        "--instance-id-yes",
        type=str,
        required=True,
        help="ID экземпляра скрипта для перехода после выбора Yes.",
    )
    parser.add_argument(
        "--instance-id-no",
        type=str,
        required=True,
        help="ID экземпляра скрипта для перехода после выбора No.",
    )

    if IS_MANAGED_RUN:
        resolver = ConfigResolver(parser, force_path_args=["html_file"])
        return resolver.resolve_all()

    return parser.parse_args()


def ensure_required_text(value: Any, field_name: str) -> str:
    """Return a stripped required value or raise a configuration error."""
    normalized = "" if value is None else str(value).strip()
    if not normalized:
        raise ValueError(f"Параметр '{field_name}' не задан.")
    return normalized


def validate_config(config: Namespace) -> tuple[str, str]:
    """Validate dialog values and return normalized branch targets."""
    validate_html_layout(
        align=config.html_align,
        margin=config.html_margin,
        padding=config.html_padding,
    )
    ensure_required_text(config.dlg_msg_var, "dlg_msg_var")
    instance_id_yes = ensure_required_text(
        config.instance_id_yes,
        "instance-id-yes",
    )
    instance_id_no = ensure_required_text(
        config.instance_id_no,
        "instance-id-no",
    )
    return instance_id_yes, instance_id_no


def save_dialog_choice(config: Namespace, result: str) -> None:
    """Persist the dialog result in the Collection Context."""
    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)
    processor.process(
        raw_value=result,
        var_name=config.dlg_msg_var,
        value_type="string",
    )


def log_dialog_choice(config: Namespace, result: str) -> None:
    """Report the saved choice without interpreting user-controlled HTML."""
    safe_result = html.escape(result.upper())
    safe_var_name = html.escape(config.dlg_msg_var)
    logger.info("<b>Выбор пользователя сохранён:</b>")
    logger.info(format_success(safe_var_name, safe_result))


def branch_target(
    result: str,
    instance_id_yes: str,
    instance_id_no: str,
) -> tuple[str, str] | None:
    """Return the selected branch name and target for Yes or No."""
    if result == "yes":
        return "YES", instance_id_yes
    if result == "no":
        return "NO", instance_id_no
    return None


def log_selected_branch(branch_name: str, target_id: str) -> None:
    """Report the branch selected by the dialog result."""
    logger.info("<b>Выбрана ветка:</b> %s", html.escape(branch_name))
    logger.info("Следующий instance_id: <i>%s</i>", html.escape(target_id))


def main() -> int:
    """Show the dialog, save its result, and configure the next script."""
    config = get_config()

    if not IS_MANAGED_RUN or pysm_context is None:
        logger.error(format_error("Скрипт предназначен для запуска в среде PySM."))
        return 1

    try:
        instance_id_yes, instance_id_no = validate_config(config)
        blocks, base_dir = load_html_sources(
            config.html_content,
            config.html_file,
        )
        html_document = build_html_document(
            blocks,
            theme_api=theme_api,
            style_name=config.html_style,
            align=config.html_align,
            margin=config.html_margin,
            padding=config.html_padding,
        )

        if config.html_output in CONSOLE_OUTPUT_MODES:
            log_html_to_console(
                blocks,
                pysm_context=pysm_context,
                theme_api=theme_api,
                style_name=config.html_style,
                align=config.html_align,
                margin=config.html_margin,
                padding=config.html_padding,
            )

        result = show_html_message_dialog(
            theme_api=theme_api,
            title=config.dlg_msg_title,
            html_document=html_document,
            dialog_type=config.dlg_msg_type,
            width=config.dlg_msg_size_width,
            height=config.dlg_msg_size_height,
            base_dir=base_dir,
            button_texts={
                "ok": config.dlg_msg_text_ok,
                "no": config.dlg_msg_text_no,
                "cancel": config.dlg_msg_text_cancel,
            },
        )
        save_dialog_choice(config, result)

        if config.html_output in CONSOLE_OUTPUT_MODES:
            log_dialog_choice(config, result)

        selected_branch = branch_target(
            result,
            instance_id_yes,
            instance_id_no,
        )
        if selected_branch is None:
            return 1

        branch_name, target_id = selected_branch
        pysm_context.set_next_script(target_id)

        if config.html_output in CONSOLE_OUTPUT_MODES:
            log_selected_branch(branch_name, target_id)

        return 0
    except Exception as error:
        logger.error(format_error(str(error)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
