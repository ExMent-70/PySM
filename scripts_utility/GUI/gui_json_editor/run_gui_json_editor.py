#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI wrapper for the public PySM visual JSON editor API."""

import argparse
import json
import logging
import sys

try:
    from pysm_lib import pysm_context
    from pysm_lib.context_variable_ops import format_error, format_success
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_json_editor import (
        JsonEditorStatus,
        edit_json_variable,
        show_json_editor_error,
    )

    IS_MANAGED_RUN = getattr(pysm_context, "_context_file_path", None) is not None
except ImportError:
    pysm_context = None
    ConfigResolver = None
    JsonEditorStatus = None
    edit_json_variable = None
    show_json_editor_error = None
    IS_MANAGED_RUN = False

    def format_success(var_name: str, value) -> str:
        return f"✅ <b>{var_name}</b> = <i>{value}</i>"

    def format_error(message: str) -> str:
        return f"❌ ОШИБКА: {message}"


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_config() -> argparse.Namespace:
    """Define arguments and return the fully resolved script configuration."""

    parser = argparse.ArgumentParser(
        description="Visual editor for PySM context variables of type json."
    )
    parser.add_argument("--var_name", required=True)
    parser.add_argument("--title", default="Редактор JSON")
    parser.add_argument("--msg", default="")

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
        result = edit_json_variable(
            config.var_name,
            title=config.title,
            message=config.msg,
            context=pysm_context,
        )
    except Exception as exc:
        if show_json_editor_error is not None:
            show_json_editor_error(str(exc), apply_theme=False)
        logger.error(format_error(str(exc)))
        return 1

    if result.status == JsonEditorStatus.SAVED:
        value = json.dumps(result.value, ensure_ascii=False, indent=2)
        logger.info(format_success(config.var_name, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
