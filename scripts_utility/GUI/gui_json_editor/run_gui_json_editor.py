#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI wrapper for the public PySM visual JSON editor API."""

import argparse
import json
import logging
import sys
from pathlib import Path


GUI_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(GUI_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_SCRIPTS_DIR))

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_json_editor import (
        JsonEditorStatus,
        edit_json_variable,
        show_json_editor_error,
    )
    from _common.html_dialog import (
        build_html_document,
        log_html_to_console,
        validate_html_layout,
    )

    IS_MANAGED_RUN = getattr(pysm_context, "_context_file_path", None) is not None
except ImportError:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    JsonEditorStatus = None
    edit_json_variable = None
    show_json_editor_error = None
    build_html_document = None
    log_html_to_console = None
    validate_html_layout = None
    IS_MANAGED_RUN = False

    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

OUTPUT_DIALOG = "dialog"
OUTPUT_CONSOLE_DIALOG = "console_dialog"


def get_config() -> argparse.Namespace:
    """Define arguments and return the fully resolved script configuration."""

    parser = argparse.ArgumentParser(
        description="Visual editor for PySM context variables of type json."
    )
    parser.add_argument("--var_name", required=True)
    parser.add_argument("--title", default="Редактор JSON")
    parser.add_argument(
        "--html_content",
        default="",
        help="HTML-текст над таблицей редактора.",
    )
    parser.add_argument(
        "--html_output",
        choices=[OUTPUT_DIALOG, OUTPUT_CONSOLE_DIALOG],
        default=OUTPUT_DIALOG,
        help="Показывать HTML только в редакторе или также в консоли PySM.",
    )
    parser.add_argument(
        "--html_align",
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
        default="script_description",
        help="Имя HTML-стиля из активной темы PySM.",
    )

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def main() -> int:
    """Open the API editor and translate its result to the CLI contract."""

    config = get_config()

    if not IS_MANAGED_RUN or edit_json_variable is None:
        logger.error(format_error("Этот скрипт предназначен для запуска внутри PySM."))
        return 1

    try:
        validate_html_layout(
            align=config.html_align,
            margin=config.html_margin,
            padding=config.html_padding,
        )
        blocks = (
            [config.html_content]
            if config.html_content and config.html_content.strip()
            else []
        )
        html_document = ""
        if blocks:
            html_document = build_html_document(
                blocks,
                theme_api=theme_api,
                style_name=config.html_style,
                align=config.html_align,
                margin=config.html_margin,
                padding=config.html_padding,
            )

        if config.html_output == OUTPUT_CONSOLE_DIALOG and blocks:
            log_html_to_console(
                blocks,
                pysm_context=pysm_context,
                theme_api=theme_api,
                style_name=config.html_style,
                align=config.html_align,
                margin=config.html_margin,
                padding=config.html_padding,
            )

        result = edit_json_variable(
            config.var_name,
            title=config.title,
            message=html_document,
            context=pysm_context,
        )
    except Exception as exc:
        if show_json_editor_error is not None:
            show_json_editor_error(str(exc), apply_theme=False)
        logger.error(format_error(str(exc)))
        return 1

    value = json.dumps(result.value, ensure_ascii=False, indent=2)
    logger.info(format_success(config.var_name, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
